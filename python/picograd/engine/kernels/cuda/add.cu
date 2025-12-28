extern "C" __global__ void add(float *z, const float *x, const float *y, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) z[i] = x[i] + y[i];
}