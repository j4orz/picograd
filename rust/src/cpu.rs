use pyo3::{buffer::PyBuffer, prelude::*};

// level 1
#[pyfunction]
#[pyo3(name = "saxpy")]
pub fn saxpypy(n: usize, alpha: f32, x: PyBuffer<f32>, y: PyBuffer<f32>) -> PyResult<()> {
  let x = unsafe { std::slice::from_raw_parts(x.buf_ptr() as *const f32, n) };
  let y = unsafe { std::slice::from_raw_parts_mut(y.buf_ptr() as *mut f32, n) };
  saxpy(n, alpha, x, y);
  Ok(())
}

pub fn saxpy(n: usize, alpha: f32, x: &[f32], y: &mut [f32]) {
  for i in 0..n { y[i] = alpha * x[i] + y[i] }
}

#[pyfunction]
#[pyo3(name = "smul")]
pub fn smulpy(n: usize, x: PyBuffer<f32>, y: PyBuffer<f32>, z: PyBuffer<f32>) -> PyResult<()> {
  let x = unsafe { std::slice::from_raw_parts(x.buf_ptr() as *const f32, n) };
  let y = unsafe { std::slice::from_raw_parts(y.buf_ptr() as *mut f32, n) };
  let z = unsafe { std::slice::from_raw_parts_mut(z.buf_ptr() as *mut f32, n) };
  smul(n, x, y, z);
  Ok(())
}

pub fn smul(n: usize, x: &[f32], y: &[f32], z: &mut [f32]) {
  for i in 0..n { z[i] = x[i] * y[i] }
}

#[pyfunction]
#[pyo3(name = "stanh")]
pub fn stanhpy(n: usize, x: PyBuffer<f32>, y: PyBuffer<f32>) -> PyResult<()> {
  // SAFETY: x, y are array.array('f') buffers from Python with length n.
  let x = unsafe { std::slice::from_raw_parts(x.buf_ptr() as *const f32, n) };
  let y = unsafe { std::slice::from_raw_parts_mut(y.buf_ptr() as *mut f32, n) };
  stanh(n, x, y);
  Ok(())
}

pub fn stanh(n: usize, x: &[f32], y: &mut [f32]) {
  for i in 0..n { y[i] = x[i].tanh() }
}

// level 2

// level 3
#[pyfunction]
#[pyo3(name = "sgemm")]
pub fn sgemmpy(
  m: usize, n: usize, p: usize,
  alpha: f32, beta: f32,
  a: PyBuffer<f32>, b: PyBuffer<f32>, c: PyBuffer<f32>
) -> PyResult<()> {
  // SAFETY: a, b, c are array.array('f') buffers from Python with lengths m*p, p*n, m*n respectively.
  // The buffer protocol guarantees the pointer is valid and the data is contiguous f32.
  let a = unsafe { std::slice::from_raw_parts(a.buf_ptr() as *const f32, m * p) };
  let b = unsafe { std::slice::from_raw_parts(b.buf_ptr() as *const f32, p * n) };
  let c = unsafe { std::slice::from_raw_parts_mut(c.buf_ptr() as *mut f32, m * n) };
  sgemmrs(m, n, p, alpha, beta, a, b, c);
  Ok(())
}

pub fn sgemmrs(
  m: usize, n: usize, p: usize,
  alpha: f32, beta: f32,
  a: &[f32], b: &[f32], c: &mut [f32] ) {
  for i in 0..m {
    for j in 0..n {
      let mut ab = 0.0f32;
      for k in 0..p { ab += a[i*p+k] * b[k*n+j] }
      c[i*n+j] = alpha * ab + beta * c[i*n+j];
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