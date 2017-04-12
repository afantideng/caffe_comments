#include <algorithm>
#include <vector>

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 进行 softmax 的维度坐标(通常为1, 即channel这个维度)
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  // sum_multiplier 先全部置为1
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  // 通常为 batch_size (softmax_axis_ == 1) ????
  outer_num_ = bottom[0]->count(0, softmax_axis_); 
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;        // 变成了[1,1,w,h] 即 1*1*inner_num
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  // 通道数即为类别个数
  int channels = bottom[0]->shape(softmax_axis_); 
  // 每一个 batch 的维度总数
  int dim = bottom[0]->count() / outer_num_;      
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // 对每一个 batch 分别操作
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    // scale_data : (1, inner_num)
    // 对 inner_num 的每一个位置，选出所有channel中的最大值
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction 做减法(目的是将分子和分母同时放大, 避免出现异常数值)
    // sum_multiplier_:  (channels, 1)
    // scale_data: (1, inner_num)
    // top_data: (channels, inner_num_)
    // top_data = top_data - sum_multiplier * scale_data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation 求指数
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp 
    // 在 channel 的方向上求和，便于用作分母
    // top_data(转置前): (channels, inner_num_)
    // sum_multiplier_:  (channels, 1)
    // scale_data: (inner_num, 1)
    // scale_data = (Trans)top_data * sum_multiplier_
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division 做除法
    // top_data: (channels, inner_num_)
    // scale_data: (1, inner_num)
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data(); // 1 * inner_num
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  // 先把 top_diff 复制给 bottom_diff
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  // 依次对每一个 batch 操作
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    // bottom_diff = bottom_diff - sum_multiplier * scale_data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  // bottom_diff = top_data *. bottom_diff
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace caffe
