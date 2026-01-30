#[cfg(feature = "cuda")]
use rs::gpu;

use rs::cpu;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")]
    gpu::cudars_helloworld()?;

    #[cfg(not(feature = "cuda"))]
    {
        println!("teenygradrs built without CUDA support.");
        println!("To enable CUDA, rebuild with: cargo run --features cuda");

        // Test GEMM: 2x3 * 3x2 = 2x2
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let mut c = [0.0; 4]; // 2x2
        cpu::gemm(2, 2, 3, 1.0, 0.0, &a, &b, &mut c);
        println!("GEMM result: {:?}", c);

        // Test GEMV: 2x3 * 3 = 2
        let mut y = [0.0; 2];
        let x = [1.0, 2.0, 3.0];
        cpu::gemv(2, 3, 1.0, 0.0, &a, &x, &mut y);
        println!("GEMV result: {:?}", y);

        for &n in &[16usize, 32, 64, 128, 256, 512] {
            let (m, k) = (n, n);
            let (a, b, mut c) = (vec![1.0f32; m * k], vec![1.0f32; k * n], vec![0.0f32; m * n]);

            let t0 = std::time::Instant::now();
            cpu::sgemm(m, n, k, 1.0, &a, k, &b, n, 0.0, &mut c, n);
            let secs = t0.elapsed().as_secs_f64().max(std::f64::MIN_POSITIVE);
            let gflop = 2.0 * (m as f64) * (n as f64) * (k as f64) / 1e9;
            let gflops = gflop / secs;

            println!("m=n=k={n:4} | {:7.3} ms | {:6.2} GFLOP/s", secs * 1e3, gflops);
        }

        for &n in &[16usize, 32, 64, 128, 256, 512] {
            let (m, k) = (n, n);
            let (a, b, mut c) = (vec![1.0f32; m * k], vec![1.0f32; k * n], vec![0.0f32; m * n]);

            let t0 = std::time::Instant::now();
            cpu::gemm(m, n, k, 1.0, 0.0, &a, &b, &mut c);
            let secs = t0.elapsed().as_secs_f64().max(std::f64::MIN_POSITIVE);
            let gflop = 2.0 * (m as f64) * (n as f64) * (k as f64) / 1e9;
            let gflops = gflop / secs;

            println!("m=n=k={n:4} | {:7.3} ms | {:6.2} GFLOP/s", secs * 1e3, gflops);
        }
    }

    Ok(())
}
