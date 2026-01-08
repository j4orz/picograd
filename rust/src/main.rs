use teenygradrs::device_kernels;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("hello from teenygrad._rs binary crate!");
    device_kernels::cudars_helloworld()?;
    Ok(())
}