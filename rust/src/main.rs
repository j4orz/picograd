#[cfg(feature = "cuda")]
use teenygradrs::device_kernels;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")] { device_kernels::cudars_helloworld()?; }

    #[cfg(not(feature = "cuda"))] {
        println!("teenygradrs built without CUDA support.");
        println!("To enable CUDA, rebuild with: cargo run --features cuda");
    }

    Ok(())
}
