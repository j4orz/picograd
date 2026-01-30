#[rustfmt::skip]
pub fn gemv(
    m: usize, n: usize,
    alpha: f32, beta: f32,
    a: &[f32], x: &[f32], y: &mut [f32]) { // (MN)(N)+M // y = aAx+by
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..n { sum += a[i * n + j] * x[j]; }
        y[i] = alpha * sum + beta * y[i];
    }
}

#[allow(non_snake_case)]
#[rustfmt::skip]
pub fn gemm(
    m: usize, n: usize, k: usize,
    alpha: f32, beta: f32,
    A: &[f32], B: &[f32], C: &mut [f32] // (MK)@(KN) + (MN) //C = aAB + bC
) {
    for i in 0..m {
        for j in 0..n { C[i * n + j] *= beta; }
    }

    // C += alpha * A * B
    for i in 0..m {
        for p in 0..k {
            let a_ip = alpha * A[i * k + p];
            for j in 0..n { C[i * n + j] += a_ip * B[p * n + j]; }
        }
    }
}

pub fn sgemm(m: usize, n: usize, k: usize, alpha: f32, a: &[f32], lda: usize, b: &[f32], ldb: usize, beta: f32, c: &mut [f32], ldc: usize) {
    assert!(m > 0 && n > 0 && k > 0, "mat dims must be non-zero");
    assert!(lda >= k && a.len() >= m * lda);
    assert!(ldb >= n && b.len() >= k * ldb);
    assert!(ldc >= n && c.len() >= m * ldc);

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * lda + p] * b[p * ldb + j];
            }
            let idx = i * ldc + j;
            c[idx] = alpha * acc + beta * c[idx];
        }
    }
}
