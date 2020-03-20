#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

const int CUDA_NUM_THREADS = 1024;

inline int CUDA_GET_BLOCKS(const array_size_t n){
	return (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

__global__ void array_set_kernel(array_size_t size, float *input, float value){

  array_size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size){
    input[index] = value;
  }
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  array_size_t size = arr->size();
  array_set_kernel<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS>>>(size, (float*)arr->data, value);
  return 0;
}

__global__ void broad_cast_kernel(array_size_t size, 
                                  array_size_t bc_dim, 
                                  const float *input, 
                                  float *output)
{
  extern __shared__ float value[];
  if (threadIdx.x == 0){
    *value = input[blockIdx.x];
  }
  __syncthreads();
  array_size_t index = size * threadIdx.x + blockIdx.x;
  output[index] = *value;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim + 1 == output->ndim);
  for (int i = 0; i < input->ndim; i++){
    assert(input->shape[i] == output->shape[i+1]);
  }
  array_size_t input_size = input->size();
  array_size_t bc_dim = output->shape[0];

  broad_cast_kernel<<<input_size, bc_dim, sizeof(float)>>>(input_size, bc_dim, (const float *)input->data, (float *)output->data);

  return 0;
}

__global__ void reduce_sum_axis_zero_kernel(array_size_t reduce_dim, 
                                            array_size_t output_size,
                                            const float *input,
                                            float *output)
{
  float sum = 0;
  array_size_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i=0; i < reduce_dim; i++){
    array_size_t index = i * output_size + output_index;
    sum += input[index];
  }
  output[output_index] = sum;
}


int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim - 1 == output->ndim);
  for (int i = 0; i < output->ndim; i++){
    assert(output->shape[i] == input->shape[i+1]);
  }
  array_size_t output_size = output->size();
  array_size_t reduce_dim = input->shape[0];

  dim3 threads;
  dim3 blocks;

  if (output_size <= 1024){
    threads.x = output_size;
    blocks.x = 1;
  }else{
    threads.x = 1024;
    blocks.x = (output_size + 1023) / 1024;
  }

  reduce_sum_axis_zero_kernel<<<blocks, threads>>>(reduce_dim, output_size, (const float*)input->data, (float *)output->data);

  return 0;
}


__global__ void matrix_elementwise_add_kernel(array_size_t size,
                                        const float *input_a,
                                        const float *input_b,
                                        float *output)
{
  array_size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  output[index] = input_a[index] + input_b[index];
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(matA->ndim == 2);
  assert(matB->ndim == 2);
  assert(output->ndim == 2);
  for (int i = 0; i < output->ndim; i++){
    assert(matA->shape[i] == matB->shape[i]);
    assert(matA->shape[i] == output->shape[i]);
  }

  array_size_t size = output->size();

  dim3 threads;
  dim3 blocks;
  if (size <= 1024){
    threads.x = size;
    blocks.x = 1;
  }else{
    threads.x = 1024;
    blocks.x = (size + 1023) / 1024;
  }
  matrix_elementwise_add_kernel<<<blocks, threads>>>(size, (const float *)matA->data, (const float *)matB->data, (float *)output->data);

  return 0;
}

__global__ void matrix_elementwise_add_by_const_kernel(
  array_size_t size,
  const float *input,
  float value,
  float *output
){
  array_size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  output[index] = input[index] + value;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  for (int i = 0; i < output->ndim; i++){
    assert(output->shape[i] == input->shape[i]);
  }

  array_size_t size = output->size();

  dim3 threads;
  dim3 blocks;
  if (size <= 1024){
    threads.x = size;
    blocks.x = 1;
  }else{
    threads.x = 1024;
    blocks.x = (size + 1023) / 1024;
  }
  matrix_elementwise_add_by_const_kernel<<<blocks, threads>>>(size, (const float *)input->data, val, (float *)output->data);

  return 0;
}

__global__ void matrix_elementwise_multiply_kernel(array_size_t size,
  const float *input_a,
  const float *input_b,
  float *output)
{
  array_size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  output[index] = input_a[index] * input_b[index];
}


int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  assert(matA->ndim == 2);
  assert(matB->ndim == 2);
  assert(output->ndim == 2);
  for (int i = 0; i < output->ndim; i++){
    assert(matA->shape[i] == matB->shape[i]);
    assert(matA->shape[i] == output->shape[i]);
  }

  array_size_t size = output->size();

  dim3 threads;
  dim3 blocks;
  if (size <= 1024){
    threads.x = size;
    blocks.x = 1;
  }else{
    threads.x = 1024;
    blocks.x = (size + 1023) / 1024;
  }
  matrix_elementwise_multiply_kernel<<<blocks, threads>>>(size, (const float *)matA->data, (const float *)matB->data, (float *)output->data);

  return 0;
}

__global__ void matrix_elementwise_multiply_by_const_kernel(
  array_size_t size,
  const float *input,
  float value,
  float *output
){
  array_size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  output[index] = input[index] * value;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  for (int i = 0; i < output->ndim; i++){
    assert(output->shape[i] == input->shape[i]);
  }

  array_size_t size = output->size();

  dim3 threads;
  dim3 blocks;
  if (size <= 1024){
    threads.x = size;
    blocks.x = 1;
  }else{
    threads.x = 1024;
    blocks.x = (size + 1023) / 1024;
  }
  matrix_elementwise_multiply_by_const_kernel<<<blocks, threads>>>(size, (const float *)input->data, val, (float *)output->data);

  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
