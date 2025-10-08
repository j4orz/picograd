pub mod ten;
pub mod eval;
pub mod eagker;
pub mod graph;
pub mod amruntime;

use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction] fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pymodule] fn linalg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule] fn pgrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    let ten = PyModule::new(m.py(), "ten")?;
    let linalg = PyModule::new(m.py(), "linalg")?;

    m.add_submodule(&ten)?;
    m.add_submodule(&linalg)?;
    
    Ok(())
}
