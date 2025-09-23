use crate::{
    Device, Dtype, DtypeVal, Layout, tensor::nn,
    kernels::cpu::{OpForwardError, ReduceDimInput},
    tensor::rsten::{self, Tensor, ViewOpError},
};
use numpy::{IntoPyArray, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::types::PyInt;
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyList, types::PyTuple};
use std::ops::{Add, Div, Mul, Sub};

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

#[pymodule]
fn picograd(py: Python, pg_m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_class::<Tensor>()?;

    // constructors
    pg_m.add_function(wrap_pyfunction!(tensor, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(zeros, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(ones, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(randint, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(randn, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(arange, pg_m)?)?;

    // samplers
    pg_m.add_function(wrap_pyfunction!(multinomial, pg_m)?)?;

    // ops
    pg_m.add_function(wrap_pyfunction!(tanh, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(exp, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(log, pg_m)?)?;

    // nn.functional/nn
    let ff_m = PyModule::new(py, "functional")?;
    ff_m.add_function(wrap_pyfunction!(nn::cross_entropy, &ff_m)?)?;
    ff_m.add_function(wrap_pyfunction!(softmax, &ff_m)?)?;
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

fn register_nn_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let nn_m = PyModule::new(parent_module.py(), "nn")?;
    register_nn_functional_module(&nn_m)?;
    parent_module.add_submodule(&nn_m)
}

fn register_nn_functional_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let functional_m = PyModule::new(parent_module.py(), "functional")?;
    functional_m.add_function(wrap_pyfunction!(nn::cross_entropy, &functional_m)?)?;
    parent_module.add_submodule(&functional_m)
}

// todo: requires_grad: bool
#[pyfunction]
pub fn new(data: Vec<DtypeVal>) -> Tensor {
    rsten::alloc(&vec![data.len()], data)
}

#[pyfunction] // TODO: &[usize, &Dtype]
pub fn zeros(shape: Vec<usize>, dtype: Dtype) -> Tensor {
    let n = shape.iter().product();
    match dtype {
        Dtype::Float32 => rsten::alloc(&shape, vec![DtypeVal::Float32(0.0); n]),
        _ => todo!(),
    }
}

#[pyfunction]
pub fn ones(shape: Vec<usize>) -> Tensor {
    let n = shape.iter().product();
    let data = vec![DtypeVal::Float32(1.0); n];
    rsten::alloc(&shape, data)
}

#[pyfunction] pub fn randint(low: i32, high: i32, shape: Vec<usize>) -> Tensor { Tensor::randint(low, high, &shape) }
#[pyfunction] pub fn randn(shape: Vec<usize>) -> Tensor { Tensor::randn(&shape) }

#[pyfunction]
pub fn multinomial(dist: &Tensor, num_samples: usize, replacement: bool) -> Tensor {
    Tensor::multinomial(dist, num_samples, replacement)
}

#[pyfunction]
pub fn arange(start: usize, end: usize) -> Tensor {
    let shape = vec![end - start];
    let data = (start..end)
        .map(|x| DtypeVal::Int32(x.try_into().unwrap()))
        .collect::<Vec<_>>();

    rsten::alloc(&shape, data)
}

fn decode_pyi(pyi: i32, n: usize) -> usize {
    if pyi < 0 {
        if n == 0 {
            0
        } else {
            let ifromn = (pyi * -1) as usize;
            n - ifromn
        }
    } else {
        pyi as usize
    }
}

#[pymethods]
impl Tensor {
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
        // todo: why does pytorch need to detatch first?
        let num_elements = self.numel();
        let mut data_vec = Vec::with_capacity(num_elements);
        let mut logical_index = vec![0; self.ndim];
        let storage_borrow = self.storage.borrow(); // Borrow outside the loop

        for _ in 0..num_elements {
            // 1. Calculate physical index from logical index and stride
            let physical_index = Self::decode(&logical_index, &self.stride);

            // 2. Get data from storage at physical index
            let value: f32 = storage_borrow.data[physical_index].into(); // Assuming .into() exists for DtypeVal -> f32
            data_vec.push(value);

            // 3. Increment logical index (like odometer)
            for i in (0..self.ndim).rev() {
                logical_index[i] += 1;
                if logical_index[i] < self.shape[i] {
                    // No carry-over needed
                    break;
                }
                // Carry-over: reset current dimension index and continue to the next
                logical_index[i] = 0;
            }
        }

        // Convert the correctly ordered Vec to a NumPy array and reshape
        let np_flat = data_vec.into_pyarray(py).to_dyn().to_owned();
        let np_shaped = np_flat.reshape(self.shape.clone())?;
        Ok(np_shaped)
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

    fn __repr__(&self) -> PyResult<String> { Ok(format!("{}", self)) }

    fn __getitem__(&self, I: Tensor) -> PyResult<Self> { Ok(self.getitem_embedding(&I)) }

    #[getter] fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> { PyTuple::new(py, &self.shape) }
    #[getter] fn ndim(&self) -> usize { self.ndim }
    #[getter] fn requires_grad(&self) -> bool { self.requires_grad }
    #[setter] fn set_requires_grad(&mut self, value: bool) { self.requires_grad = value; }
    #[getter] fn stride(&self) -> Vec<usize> { self.stride.clone() }

    #[getter] fn device(&self) -> Device { self.device.clone() }
    #[getter] fn layout(&self) -> Layout { self.layout.clone() }
    #[getter] fn dtype(&self) -> Dtype { self.dtype.clone() }

    // VIEW OPS
    fn reshape(&self, shape: Bound<'_, PyTuple>) -> PyResult<Tensor> {
        let shape = shape.extract::<Vec<i32>>()?;
        Ok(self._reshape(&shape)?)
    }

    fn permute(&self, shape: Bound<'_, PyTuple>) -> PyResult<Tensor> {
        let shape = shape.extract::<Vec<usize>>()?;
        Ok(self._permute(&shape))
    }

    // fn transpose(&self, shape: Bound<'_, PyTuple>) -> PyResult<Tensor> {
    //     let shape = shape.extract::<Vec<usize>>()?;
    //     Ok(self._transpose(&shape))
    // }

    fn T(&self) -> PyResult<Tensor> {
        Ok(self._T())
    }

    #[pyo3(signature = (*args))]
    fn squeeze(&self, args: Bound<'_, PyTuple>) -> PyResult<Tensor> {
        let dims = args
            .extract::<Vec<i32>>()?
            .iter()
            .map(|&x| decode_pyi(x, self.ndim))
            .collect::<Vec<_>>();
        Ok(self._squeeze(&dims))
    }

    fn unsqueeze(&self, dim: Bound<'_, PyInt>) -> PyResult<Tensor> {
        let dim = decode_pyi(dim.extract::<i32>()?, self.ndim + 1); // unsqueeze(-1) == unsqueeze(n)
        Ok(self._unsqueeze(dim))
    }

    fn item(&self) -> PyResult<DtypeVal> {
        Ok(self._item())
    }

    // POINTWISE OPS
    fn __add__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(val) = other.extract::<f32>() { Ok(self.add(val)?) }
        else if let Ok(t2) = other.extract::<Tensor>() { Ok(self.add(&t2)?) }
        else { Err(PyRuntimeError::new_err("expected a tensor or scalar")) }
    }

    fn __sub__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(val) = other.extract::<f32>() { Ok(self.sub(val)?) }
        else if let Ok(t2) = other.extract::<Tensor>() { Ok(self.sub(&t2)?) }
        else { Err(PyRuntimeError::new_err("expected a tensor or scalar")) }
    }

    fn __mul__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        todo!()
        // if let Ok(val) = other.extract::<f32>() { Ok(self.mul(val)?) }
        // else if let Ok(t2) = other.extract::<Tensor>() { Ok(self.mul(&t2)?) }
        // else { Err(PyRuntimeError::new_err("expected a tensor or scalar")) }
    }

    fn __truediv__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(val) = other.extract::<f32>() { Ok(self.div(val)?) }
        else if let Ok(t2) = other.extract::<Tensor>() { Ok(self.div(&t2)?) }
        else { Err(PyRuntimeError::new_err("expected a tensor or scalar")) }
    }

    #[pyo3(signature=(dim=None, keepdim=None))]
    fn sum(&self, dim: Option<usize>, keepdim: Option<bool>) -> PyResult<Tensor> {
        match (dim, keepdim) {
            (None, None) => {
                let output = self
                    ._sum(None)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()));
                output
            }
            (Some(_), None) | (None, Some(_)) => unimplemented!(),
            (Some(dim), Some(keepdim)) => {
                let rdi = ReduceDimInput { dim, keepdim };
                let output = self
                    ._sum(Some(rdi))
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()));
                output
            }
        }
    }

    // PROCESSING OPS
    fn __matmul__(&self, other: &Tensor) -> PyResult<Tensor> {
        Ok(self.matmul(other)?)
    }
}

// nnops
#[pyfunction]
fn softmax(x: Tensor, dim: i32) -> PyResult<Tensor> {
    let rsi = decode_pyi(dim, x.ndim());
    nn::_softmax(x, rsi)
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
        Ok(rsten::alloc(&shape, data))
    }

    if let Ok(np) = data.downcast::<numpy::PyArrayDyn<i32>>() { return alloc_from_np(np); }
    else if let Ok(np) = data.downcast::<numpy::PyArrayDyn<i64>>() { return alloc_from_np(np); }
    else if let Ok(np) = data.downcast::<numpy::PyArrayDyn<f32>>() { return alloc_from_np(np); }
    else if let Ok(pylist) = data.downcast::<PyList>() {
        let (mut data, mut shape) = (Vec::new(), Vec::new());
        flatten_pylist(pylist.as_any(), &mut data, &mut shape, 1)?;
        // println!("moose {:?}", data);
        // println!("moose {:?}", shape);
        Ok(rsten::alloc(&shape, data))
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

#[pyfunction] fn tanh(x: Tensor) -> PyResult<Tensor> { x.tanh().map_err(|e| PyRuntimeError::new_err(e.to_string())) }
#[pyfunction] fn exp(x: Tensor) -> PyResult<Tensor> { x.exp().map_err(|e| PyRuntimeError::new_err(e.to_string())) }
#[pyfunction] fn log(x: Tensor) -> PyResult<Tensor> { x.log().map_err(|e| PyRuntimeError::new_err(e.to_string())) }