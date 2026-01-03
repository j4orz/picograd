use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction] fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule] fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    println!("Hello from Rust!");
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    let ten = PyModule::new(m.py(), "ten")?;
    let linalg = PyModule::new(m.py(), "linalg")?;

    m.add_submodule(&ten)?;
    m.add_submodule(&linalg)?;
    
    Ok(())
}
