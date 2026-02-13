pub fn sgemm(
  m: usize, n: usize, p: usize,
  alpha: f32, beta: f32,
  a: &[f32], b: &[f32], c: &mut [f32] ) {
  for i in 0..m {
    for j in 0..n {
      let mut ab = 0.0f32;
      for k in 0..p { ab += a[i*p+k] * b[k*n+j] }
      let idx = i*n+j;
      c[idx] = alpha * ab + beta * c[idx];
    }
  }
}

pub fn sgemm1( // temporal locality (loads/store): loop order
  m: usize, n: usize, p: usize,
  alpha: f32, beta: f32,
  a: &[f32], b: &[f32], c: &mut [f32] ) {
  for i in 0..m {
    for j in 0..n {
      let mut ab = 0.0f32;
      for k in 0..p { ab += a[i*p+k] * b[k*n+j] }
      let idx = i*n+j;
      c[idx] = alpha * ab + beta * c[idx];
    }
  }
}

pub fn sgemm2( // spatial locality (loads/store): block on register and caches
  m: usize, n: usize, p: usize,
  alpha: f32, beta: f32,
  a: &[f32], b: &[f32], c: &mut [f32] ) {
  for i in 0..m {
    for j in 0..n {
      let mut ab = 0.0f32;
      for k in 0..p { ab += a[i*p+k] * b[k*n+j] }
      let idx = i*n+j;
      c[idx] = alpha * ab + beta * c[idx];
    }
  }
}

pub fn sgemm3( // ilp loop unroll
  m: usize, n: usize, p: usize,
  alpha: f32, beta: f32,
  a: &[f32], b: &[f32], c: &mut [f32] ) {
  for i in 0..m {
    for j in 0..n {
      let mut ab = 0.0f32;
      for k in 0..p { ab += a[i*p+k] * b[k*n+j] }
      let idx = i*n+j;
      c[idx] = alpha * ab + beta * c[idx];
    }
  }
}

// core::arch/asm! intrinsics/inline assembly
// std::simd portable simd
// autov11n
pub fn sgemm4( // dlp microkernel (avx/sve) matrix extensions??
  m: usize, n: usize, p: usize,
  alpha: f32, beta: f32,
  a: &[f32], b: &[f32], c: &mut [f32] ) {
  for i in 0..m {
    for j in 0..n {
      let mut ab = 0.0f32;
      for k in 0..p { ab += a[i*p+k] * b[k*n+j] }
      let idx = i*n+j;
      c[idx] = alpha * ab + beta * c[idx];
    }
  }
}

pub fn sgemm5( // tlp todo!
  m: usize, n: usize, p: usize,
  alpha: f32, beta: f32,
  a: &[f32], b: &[f32], c: &mut [f32] ) {
  for i in 0..m {
    for j in 0..n {
      let mut ab = 0.0f32;
      for k in 0..p { ab += a[i*p+k] * b[k*n+j] }
      let idx = i*n+j;
      c[idx] = alpha * ab + beta * c[idx];
    }
  }
}