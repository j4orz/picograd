use pyo3::prelude::*;
pub mod host_kernels;
pub mod device_kernels;

#[pyfunction] fn cuda_smoke_test() -> PyResult<()> {
  device_kernels::cudars_helloworld().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
  Ok(())
}

/// A Python module implemented in Rust.
#[pymodule] fn teenygradrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    println!("hello from teenygrad._rs!");
    let (host_kernels_pymodule, device_kernels_pymodule) = (PyModule::new(m.py(), "cpu_kernels")?, PyModule::new(m.py(), "gpu_kernels")?);
    let (_, _) = (m.add_submodule(&host_kernels_pymodule)?, m.add_submodule(&device_kernels_pymodule)?);

    m.add_function(wrap_pyfunction!(cuda_smoke_test, m)?)?;
    
    Ok(())
}