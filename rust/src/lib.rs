use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction] fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction] fn cuda_smoke_test() -> PyResult<()> {
  foo().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
  Ok(())
}

/// A Python module implemented in Rust.
#[pymodule] fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    println!("hello from teenygrad._rs!");
    let (cpu_kernels_pymodule, gpu_kernels_pymodule) = (PyModule::new(m.py(), "cpu_kernels")?, PyModule::new(m.py(), "gpu_kernels")?);
    let (_, _) = (m.add_submodule(&cpu_kernels_pymodule)?, m.add_submodule(&gpu_kernels_pymodule)?);

    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_smoke_test, m)?)?;
    
    Ok(())
}

use cust::prelude::*;
use kernels::T;
use std::error::Error;

// Embed the PTX code as a static string.
static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

pub fn foo() -> Result<(), Box<dyn Error>> {
    let _ctx = cust::quick_init()?; // Initialize the CUDA Driver API. `_ctx` must be kept alive until the end.
    let module = Module::from_ptx(PTX, &[])?; // Create a module from the PTX code compiled by `cuda_builder`.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?; // Create a stream, which is like a thread for dispatching GPU calls.

    // Initialize input and output buffers in CPU memory.
    let (a, b): ([T; _], [T; _]) = ([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]);
    let mut c: Vec<T> = vec![0.0 as T; a.len()];

    // Allocate memory on the GPU and copy the contents from the CPU memory.
    let (a_gpu, b_gpu, c_gpu) = (a.as_dbuf()?, b.as_dbuf()?, c.as_slice().as_dbuf()?);

    // Launch the kernel on the GPU.
    // - The first two parameters between the triple angle brackets specify 1
    //   block of 4 threads.
    // - The third parameter is the number of bytes of dynamic shared memory.
    //   This is usually zero.
    // - These threads run in parallel, so each kernel invocation must modify
    //   separate parts of `c_gpu`. It is the kernel author's responsibility to
    //   ensure this.
    // - Immutable slices are passed via pointer/length pairs. This is unsafe
    //   because the kernel function is unsafe, but also because, like an FFI
    //   call, any mismatch between this call and the called kernel could
    //   result in incorrect behaviour or even uncontrolled crashes.
    let add_kernel = module.get_function("add")?;
    unsafe {
        launch!(
            add_kernel<<<1, 4, 0, stream>>>(
                a_gpu.as_device_ptr(),
                a_gpu.len(),
                b_gpu.as_device_ptr(),
                b_gpu.len(),
                c_gpu.as_device_ptr(),
            )
        )?;
    }

    // Synchronize all threads, i.e. ensure they have all completed before continuing.
    stream.synchronize()?;

    c_gpu.copy_to(&mut c)?; // Copy the GPU memory back to the CPU.
    println!("c from cuda iss = {:?}", c);

    Ok(())
}