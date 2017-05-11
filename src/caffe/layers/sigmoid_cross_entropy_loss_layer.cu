#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void SigmoidCrossEntropyLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* target, Dtype* loss,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    if (has_ignore_label_ && target_value == ignore_label_) {
      loss[i] = 0;
      counts[i] = 0;
    } else {
      loss[i] = input_data[i] * (target[i] - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] *
          (input_data[i] >= 0)));
      counts[i] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossIgnoreDiffGPU(const int count,
    const int ignore_label, const Dtype* target, Dtype* diff) {
  CUDA_KERNEL_LOOP(i, count) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value == ignore_label) {
      diff[i] = 0;
    }
  }
}


template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  // 内嵌的 sigmoid 函数进行前向传播 (放入了 sigmoid_output_ 里)
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // 获取输入数据的像素点总数
  const int count = bottom[0]->count();
  // Stable version of loss computation from input data
  // 获取只读输入数据和标签
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  // 获取可写入输入数据和标签
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  Dtype* count_data = bottom[1]->mutable_gpu_diff();
  Dtype valid_count;

  // NOLINT_NEXT_LINE(whitespace/operators)
  // ----- 前向传播计算loss -----
  SigmoidCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, input_data, target, loss_data,
      has_ignore_label_, ignore_label_, count_data);
  // Only launch another CUDA kernel if we actually need the valid count.
  // 如果normlize使用VALID模式,且有label无效的情况下, 才去认真地计算有效点数
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(count, count_data, &valid_count);
  } else {
    valid_count = count;
  }
  Dtype loss;

  // 把每一个点的loss加和
  caffe_gpu_asum(count, loss_data, &loss);
  // 归一化loss, 放入top[0]
  normalizer_ = get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    // 获取内嵌 sigmoid 函数的输出值
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    // 获取标签
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);

    // 点对点(element-wise)相减的操作
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);

    // Zero out gradient of ignored targets.
    if (has_ignore_label_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      SigmoidCrossEntropyLossIgnoreDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, ignore_label_, target, bottom_diff);
    }

    // Scale down gradient
    // 归一化 loss 并放入 bottom_diff
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    caffe_gpu_scal(count, loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidCrossEntropyLossLayer);

}  // namespace caffe
