use crate::{
    ops::cpu_ops::OpForwardError,
    trs::{self, Tensor, ViewOpError},
    {Device, Dtype, DtypeVal, Layout, nn},
};
use numpy::{PY_ARRAY_API, PyArrayMethods, PyUntypedArrayMethods, dtype, npyffi};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyList, types::PyTuple};
use rand::{Rng, distr::StandardUniform};
use std::{
    ops::{Add, Div, Mul, Sub},
    os::raw::{c_int, c_void},
    ptr,
};

impl From<OpForwardError> for PyErr {
    fn from(e: OpForwardError) -> Self {
        PyRuntimeError::new_err(e.to_string())
    }
}

impl From<ViewOpError> for PyErr {
    fn from(e: ViewOpError) -> Self {
        PyRuntimeError::new_err(e.to_string())
    }
}

// todo: requires_grad: bool
#[pyfunction]
pub fn new(data: Vec<DtypeVal>) -> Tensor {
    trs::alloc(&vec![data.len()], data)
}

#[pyfunction]
pub fn zeros(shape: Vec<usize>, dtype: Dtype) -> Tensor {
    let n = shape.iter().product();
    match dtype {
        Dtype::Float32 => trs::alloc(&shape, vec![DtypeVal::Float32(0.0); n]),
        _ => todo!(),
    }
}

#[pyfunction]
pub fn ones(shape: Vec<usize>) -> Tensor {
    let n = shape.iter().product();
    let data = vec![DtypeVal::Float32(1.0); n];
    trs::alloc(&shape, data)
}

#[pyfunction]
pub fn randn(shape: Vec<usize>) -> Tensor {
    let n: usize = shape.iter().product::<usize>();
    let data = (0..n)
        .map(|_| DtypeVal::Float32(rand::rng().sample(StandardUniform)))
        .collect::<Vec<_>>();

    trs::alloc(&shape, data)
}

#[pyfunction]
pub fn arange(start: usize, end: usize) -> Tensor {
    let shape = vec![end - start];
    let data = (start..end)
        .map(|x| DtypeVal::Int32(x.try_into().unwrap()))
        .collect::<Vec<_>>();

    trs::alloc(&shape, data)
}

#[pymethods]
impl Tensor {
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
        let data = self
            .storage
            .borrow()
            .data
            .iter()
            .map(|&x| x.into())
            .collect::<Vec<f32>>();

        let dims_ptr = self.shape.as_slice().as_ptr();
        let strides_ptr = self.stride.as_slice().as_ptr();
        let data_ptr = self.storage.borrow().data.as_slice().as_ptr();
        let foo = unsafe {
            pyo3::Python::with_gil(|py| {
                let ptr = PY_ARRAY_API.PyArray_NewFromDescr(
                    py,
                    PY_ARRAY_API.get_type_object(py, npyffi::NpyTypes::PyArray_Type),
                    dtype::<f32>(py), // T::get_dtype(py).into_dtype_ptr(),
                    self.ndim as c_int,
                    dims_ptr as *mut npyffi::npy_intp, // libc isize
                    strides_ptr as *mut npyffi::npy_intp, // strides
                    data_ptr as *mut c_void,           // data
                    npyffi::NPY_ARRAY_WRITEABLE,       // flag
                    ptr::null_mut(),                   // obj
                );

                PY_ARRAY_API.PyArray_SetBaseObject(
                    py,
                    ptr as *mut npyffi::PyArrayObject,
                    container as *mut pyo3::ffi::PyObject,
                );

                Bound::from_owned_ptr(py, ptr).downcast_into_unchecked()
            })
        };

        Ok(foo)
    }

    pub fn detach(&self) -> Self {
        Self {
            ndim: self.ndim,
            shape: self.shape.clone(),
            stride: self.stride.clone(),
            input_op: None, // detach
            requires_grad: self.requires_grad,
            storage: self.storage.clone(),
            device: self.device.clone(),
            layout: self.layout.clone(),
            dtype: self.dtype.clone(),
        }
    }

    // fn view(&self, shape: Bound<'_, PyList>) -> PyResult<Tensor> {
    //     Ok(self.no_alloc(shape))
    // }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self))
    }

    fn __getitem__(&self, I: Tensor) -> PyResult<Self> {
        todo!()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.ndim
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.shape)
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    #[setter]
    fn set_requires_grad(&mut self, value: bool) {
        self.requires_grad = value;
    }

    #[getter]
    fn stride(&self) -> Vec<usize> {
        self.stride.clone()
    }

    #[getter]
    fn device(&self) -> Device {
        self.device.clone()
    }

    #[getter]
    fn layout(&self) -> Layout {
        self.layout.clone()
    }

    #[getter]
    fn dtype(&self) -> Dtype {
        self.dtype.clone()
    }

    // VIEW OPS
    fn reshape(&self, shape: Bound<'_, PyTuple>) -> PyResult<Tensor> {
        let shape = shape.extract::<Vec<i32>>()?;
        Ok(self._reshape(&shape)?)
    }

    fn permute(&self, shape: Bound<'_, PyTuple>) -> PyResult<Tensor> {
        let shape = shape.extract::<Vec<usize>>()?;
        Ok(self._permute(&shape))
    }

    // POINTWISE OPS
    fn __add__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(val) = other.extract::<f32>() {
            Ok(self.add(val)?)
        } else if let Ok(t2) = other.extract::<Tensor>() {
            Ok(self.add(&t2)?)
        } else {
            Err(PyRuntimeError::new_err("expected a tensor or scalar"))
        }
    }

    fn __sub__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(val) = other.extract::<f32>() {
            Ok(self.sub(val)?)
        } else if let Ok(t2) = other.extract::<Tensor>() {
            Ok(self.sub(&t2)?)
        } else {
            Err(PyRuntimeError::new_err("expected a tensor or scalar"))
        }
    }

    fn __mul__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(val) = other.extract::<f32>() {
            Ok(self.mul(val)?)
        } else if let Ok(t2) = other.extract::<Tensor>() {
            Ok(self.mul(&t2)?)
        } else {
            Err(PyRuntimeError::new_err("expected a tensor or scalar"))
        }
    }

    fn __truediv__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(val) = other.extract::<f32>() {
            Ok(self.div(val)?)
        } else if let Ok(t2) = other.extract::<Tensor>() {
            Ok(self.div(&t2)?)
        } else {
            Err(PyRuntimeError::new_err("expected a tensor or scalar"))
        }
    }

    // PROCESSING OPS
    fn __matmul__(&self, other: &Tensor) -> PyResult<Tensor> {
        Ok(self.matmul(other)?)
    }
}

#[pyfunction]
pub fn tensor<'py>(data: Bound<'py, PyAny>) -> PyResult<Tensor> {
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
    pg_m.add_function(wrap_pyfunction!(tensor, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(zeros, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(ones, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(randn, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(arange, pg_m)?)?;

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
