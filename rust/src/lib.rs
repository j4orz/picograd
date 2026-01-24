use pyo3::prelude::*;
pub mod host_kernels;

#[cfg(feature = "cuda")]
pub mod device_kernels;

#[cfg(feature = "cuda")]
#[pyfunction]
fn cuda_smoke_test() -> PyResult<()> {
    device_kernels::cudars_helloworld()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
fn cuda_smoke_test() -> PyResult<()> {
    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        "CUDA support not compiled. Rebuild with --features cuda",
    ))
}

/// A Python module implemented in Rust.
#[pymodule] fn rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    println!("hello from teenygrad._rs's library crate!");

    let host_kernels_pymodule = PyModule::new(m.py(), "cpu_kernels")?;
    m.add_submodule(&host_kernels_pymodule)?;

    #[cfg(feature = "cuda")]
    {
        let device_kernels_pymodule = PyModule::new(m.py(), "gpu_kernels")?;
        m.add_submodule(&device_kernels_pymodule)?;
    }

    m.add_function(wrap_pyfunction!(cuda_smoke_test, m)?)?;

    Ok(())
}
