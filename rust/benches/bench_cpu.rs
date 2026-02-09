#![feature(test)]
extern crate test;

#[cfg(test)]
#[rustfmt::skip]
mod benches {
  use std::time::Instant;
  use test::Bencher;

  fn bench_gemm_size(size: usize) {
    let (m, n, k) = (size, size, size);
    let (a, b, mut c) = (vec![1.0f32; m * k], vec![1.0; k * n], vec![0.0; m * n]);

    let (warmup, iterations)  = (3, 10);
    for _ in 0..warmup { rs::cpu::sgemm(m, n, k, 1.0, 0.0, &a, &b, &mut c); }
    let start = Instant::now();
    for _ in 0..iterations { rs::cpu::sgemm(m, n, k, 1.0, 0.0, &a, &b, &mut c); }
    let elapsed = start.elapsed();

    let gflop_count = (2 * m * n * k * iterations) as f64 / 1e9;
    let gflops = gflop_count / elapsed.as_secs_f64();
    println!("GEMM {m}x{n}x{k}: {gflops:.2} GFLOPS");
  }

  #[bench] fn bench_gemm_256(_b: &mut Bencher) { bench_gemm_size(256) }
  #[bench] fn bench_gemm_512(_b: &mut Bencher) { bench_gemm_size(512) }
  #[bench] fn bench_gemm_1024(_b: &mut Bencher) { bench_gemm_size(1024) }

  // #[bench]
  // fn bench_gemm_2048(_b: &mut Bencher) {
  //     bench_gemm_size(2048);
  // }
  // #[bench]
  // fn bench_gemm_4096(_b: &mut Bencher) {
  //     bench_gemm_size(4096);
  // }
}
