use teenygradrs::foo;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    foo()?;
    Ok(())
}