use crate::{kernels::cpu::ReduceDimInput, tensor::rsten::Tensor};
use pyo3::{PyResult, pyfunction};
use std::ops::{Div, Sub};

#[pyfunction]
pub fn cross_entropy(p: Tensor, q: Tensor) -> PyResult<Tensor> {
    todo!()
}

#[pyfunction]
pub fn _softmax(x: Tensor, dim: usize) -> PyResult<Tensor> {
    let shifted = x.sub(&x.max(dim, true)?)?;
    let exp = shifted.exp()?;
    let expsum = exp._sum(Some(ReduceDimInput { dim, keepdim: true }))?;
    let softmax = exp.div(&expsum)?;
    Ok(softmax)
}
