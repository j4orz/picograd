use teenygradrs::device_kernels;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    device_kernels::cudars_helloworld()?;
    Ok(())
}