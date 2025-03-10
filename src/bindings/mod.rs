mod tpy;

use crate::{
    DtypeVal, nn,
    trs::{self, Tensor},
};
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyList};

#[pyfunction]
fn bla<'py>(data: Bound<'py, PyAny>) -> PyResult<Tensor> {
    fn alloc_from_np<T: numpy::Element, 'py>(
        np: &Bound<'py, numpy::PyArrayDyn<T>>,
    ) -> PyResult<Tensor>
    where
        T: Into<DtypeVal> + Copy,
    {
        let np = np.try_readonly()?;
        let (data, shape) = (
            np.as_slice()?
                .iter()
                .map(|x| (*x).into())
                .collect::<Vec<DtypeVal>>(),
            np.shape().to_vec(),
        );
        Ok(trs::alloc(&shape, data))
    }

    if let Ok(np) = data.downcast::<numpy::PyArrayDyn<i32>>() {
        return alloc_from_np(np);
    } else if let Ok(np) = data.downcast::<numpy::PyArrayDyn<i64>>() {
        return alloc_from_np(np);
    } else if let Ok(np) = data.downcast::<numpy::PyArrayDyn<f32>>() {
        return alloc_from_np(np);
    } else if let Ok(pylist) = data.downcast::<PyList>() {
        let (mut data, mut shape) = (Vec::new(), Vec::new());
        flatten_pylist(pylist.as_any(), &mut data, &mut shape, 1)?;
        // println!("moose {:?}", data);
        // println!("moose {:?}", shape);
        Ok(trs::alloc(&shape, data))
    } else {
        Err(PyRuntimeError::new_err(
            "Unsupported type: Expected NumPy ndarray or list",
        ))
    }
}

fn flatten_pylist<'py>(
    pyobj: &Bound<'py, PyAny>,
    data: &mut Vec<DtypeVal>,
    shape: &mut Vec<usize>,
    i: usize, // 1-based
) -> PyResult<()> {
    match pyobj.downcast::<PyList>() {
        Ok(pylist) => {
            if shape.len() < i {
                shape.push(pylist.len());
            }
            for item in pylist.iter() {
                flatten_pylist(&item, data, shape, i + 1)?;
            }
        }
        Err(_) => data.push(pyobj.extract::<DtypeVal>()?),
    };

    Ok(())
}

#[pyfunction]
fn tanh(x: Tensor) -> PyResult<Tensor> {
    x.tanh().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn exp(x: Tensor) -> PyResult<Tensor> {
    x.exp().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn log(x: Tensor) -> PyResult<Tensor> {
    x.log().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn sum(x: Tensor, dim: usize, keepdim: bool) -> PyResult<Tensor> {
    x.sum(dim, keepdim)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// A Python module implemented in Rust.
#[pymodule]
fn picograd(py: Python, pg_m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_class::<Tensor>()?;

    // constructors
    // pg_m.add_function(wrap_pyfunction!(tensor::tensor, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(trs::zeros, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(trs::ones, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(trs::randn, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(trs::arange, pg_m)?)?;

    // ops
    pg_m.add_function(wrap_pyfunction!(tanh, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(exp, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(log, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(sum, pg_m)?)?;

    // nn.functional
    let ff_m = PyModule::new(py, "functional")?;
    ff_m.add_function(wrap_pyfunction!(nn::cross_entropy, &ff_m)?)?;

    // nn
    let nn_m = PyModule::new(py, "nn")?;
    nn_m.add_submodule(&ff_m)?;
    pg_m.add_submodule(&nn_m)?;

    // BUG: see pyo3/issues/759: https://github.com/PyO3/pyo3/issues/759#issuecomment-977835119
    py.import("sys")?
        .getattr("modules")?
        .set_item("picograd.nn", nn_m)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("picograd.nn.functional", ff_m)?;

    Ok(())
}

// fn register_nn_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
//     let nn_m = PyModule::new(parent_module.py(), "nn")?;
//     register_nn_functional_module(&nn_m)?;
//     parent_module.add_submodule(&nn_m)
// }

// fn register_nn_functional_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
//     let functional_m = PyModule::new(parent_module.py(), "functional")?;
//     functional_m.add_function(wrap_pyfunction!(nn::cross_entropy, &functional_m)?)?;
//     parent_module.add_submodule(&functional_m)
// }
