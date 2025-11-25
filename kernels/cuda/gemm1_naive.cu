__global__ void sgemm(
  int M, int N, int K,
  float alpha, const float *A, const float *B, float beta, float *C
) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) { // `if` condition is necessary for when M or N aren't multiples of 32.
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) { tmp += A[x * K + i] * B[i * N + y]; }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y]; // C = α*(A@B)+β*C
  }
}