pub mod nn;
pub mod optim;
pub mod linalg;
pub mod ten;
pub mod engine;
pub mod eagker;
pub mod graph;
pub mod runtime;

use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction] fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule] fn pgrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
