use pyo3::prelude::*;
pub mod cpu;
#[cfg(feature = "cuda")] pub mod gpu;

#[pymodule] /// A Python module implemented in Rust. todo: rename to rustykernels
fn rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
  println!("initializing teenygrad.eagkers python module from rust");

  let cpu = PyModule::new(m.py(), "cpu")?;
  cpu.add_function(wrap_pyfunction!(cpu::saxpypy, &cpu)?)?;
  cpu.add_function(wrap_pyfunction!(cpu::sgemmpy, &cpu)?)?;
  m.add_submodule(&cpu)?;

  Ok(())
}