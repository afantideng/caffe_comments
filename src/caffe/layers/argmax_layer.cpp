#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/argmax_layer.hpp"

namespace caffe {

template <typename Dtype>
void ArgMaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 从 prototxt 中读取 argmax 层的参数
  const ArgMaxParameter& argmax_param = this->layer_param_.argmax_param();
  out_max_val_ = argmax_param.out_max_val();
  top_k_ = argmax_param.top_k();
  has_axis_ = argmax_param.has_axis();
  CHECK_GE(top_k_, 1) << "top k must not be less than 1.";
  if (has_axis_) {
    // 读取进行 argmax 的维度
    axis_ = bottom[0]->CanonicalAxisIndex(argmax_param.axis());
    CHECK_GE(axis_, 0) << "axis must not be less than 0.";
    CHECK_LE(axis_, bottom[0]->num_axes()) <<
      "axis must be less than or equal to the number of axis.";
    // top_k 不能大于 axis 维度上的 shape 总数（总类别数）
    CHECK_LE(top_k_, bottom[0]->shape(axis_))
      << "top_k must be less than or equal to the dimension of the axis.";
  } else {
    // 默认情况下, bottom blob 是一个flatten的形态(全连接层), 此时 top_k 应该小于 bottom[0]->count(1)
    CHECK_LE(top_k_, bottom[0]->count(1))
      << "top_k must be less than or equal to"
        " the dimension of the flattened bottom blob per instance.";
  }
}

template <typename Dtype>
void ArgMaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int num_top_axes = bottom[0]->num_axes();
  if ( num_top_axes < 3 ) num_top_axes = 3;
  std::vector<int> shape(num_top_axes, 1);
  if (has_axis_) {
    // Produces max_ind or max_val per axis
    shape = bottom[0]->shape();
    shape[axis_] = top_k_;
  } else {
    shape[0] = bottom[0]->shape(0);
    // Produces max_ind
    shape[2] = top_k_;
    if (out_max_val_) {
      // Produces max_ind and max_val
      shape[1] = 2;
    }
  }
  // top[0] 最常见的情况下, 尺寸为: (N, 1, 1, 1)
  //                    gerneral: (N, 1(2), top_k, 1)
  top[0]->Reshape(shape);
}

template <typename Dtype>
void ArgMaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int dim, axis_dist;
  if (has_axis_) {
    // dim 为类别数
    dim = bottom[0]->shape(axis_);
    // Distance between values of axis in blob
    // axis 维度上每一个单元起始位置在 blob 里的点数间隔 (通常为channel之间的点数间隔)
    axis_dist = bottom[0]->count(axis_) / dim;
  } else {
    dim = bottom[0]->count(1);
    axis_dist = 1;
  }
  // num 是 N * H * W (通常来讲就是N, 即 batch_size)
  int num = bottom[0]->count() / dim;
  std::vector<std::pair<Dtype, int> > bottom_data_vector(dim);
  // 依次处理每个 batch (flatten 情形下)
  for (int i = 0; i < num; ++i) {
    // 将每个类别的预测概率值写入 bottom_data_vector
    for (int j = 0; j < dim; ++j) {
      bottom_data_vector[j] = std::make_pair(
        bottom_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
    }
    std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
        bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
    for (int j = 0; j < top_k_; ++j) {
      if (out_max_val_) {
        if (has_axis_) {
          // Produces max_val per axis
          top_data[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist]
            = bottom_data_vector[j].first;
        } else {
          // Produces max_ind and max_val
          top_data[2 * i * top_k_ + j] = bottom_data_vector[j].second;
          top_data[2 * i * top_k_ + top_k_ + j] = bottom_data_vector[j].first;
        }
      } else {
        // Produces max_ind per axis
        top_data[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist]
          = bottom_data_vector[j].second;
      }
    }
  }
}

INSTANTIATE_CLASS(ArgMaxLayer);
REGISTER_LAYER_CLASS(ArgMax);

}  // namespace caffe
