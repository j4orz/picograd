// matmul_f32.hip
extern "C"
__global__ void matmul_f32(const float* __restrict__ A,  // [M,K]
                const float* __restrict__ B,  // [K,N]
                float* __restrict__ C,        // [M,N]
                int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row*K + k] * B[k*N + col];
    C[row*N + col] = acc;
  }
}