#[cfg(feature = "cuda")] use rs::gpu;

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  #[cfg(feature = "cuda")] gpu::cudars_helloworld()?;

  #[cfg(not(feature = "cuda"))] {
    println!("teenygradrs built without CUDA support.");
    println!("To enable CUDA, rebuild with: cargo run --features cuda");

    for &n in &[16usize, 32, 64, 128, 256, 512] {
      let (m, k) = (n, n);
      let (a, b, mut c) = (vec![1.0f32; m * k], vec![1.0f32; k * n], vec![0.0f32; m * n]);

      let (warmup, iterations) = (3, 10);
      for _ in 0..warmup { rs::cpu::sgemm(m, n, k, 1.0, 0.0, &a, &b, &mut c) }
      let start = Instant::now();
      for _ in 0..iterations { rs::cpu::sgemm(m, n, k, 1.0, 0.0, &a, &b, &mut c); }
      let elapsed = start.elapsed();

      let gflop_count = (2 * m * n * k * iterations) as f64 / 1e9;
      let gflops = gflop_count / elapsed.as_secs_f64();
      println!("GEMM {m}x{n}x{k}: {gflops:.2} GFLOPS");
    }
  }

  Ok(())
}