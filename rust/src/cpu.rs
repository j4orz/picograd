pub fn sgemm(
  m: usize, n: usize, k: usize,
  alpha: f32, beta: f32,
  a: &[f32], b: &[f32], c: &mut [f32]
) {
  for j in 0..n {
    for p in 0..k {
      for i in 0..m {
        let idx = i * n + j;
        let prod = alpha * a[i * k + p] * b[p * n + j];
        if p == 0 {
          c[idx] = beta * c[idx] + prod;
        } else {
          c[idx] += prod;
        }
      }
    }
  }
} 