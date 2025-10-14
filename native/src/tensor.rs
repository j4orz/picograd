//! Tensor
//! ...

use std::ops::{Add, Sub, Mul, Div};

use crate::runtime::{Device, Storage};
use crate::engine::{MicroOp, eval};

pub struct Tensor {
    pub shape: Vec<usize>, pub stride: Vec<usize>, pub dtype: String,
    pub device: Device, pub storage: Storage,
    // forward (primal, tangent)
    // reverse (primal, adjoint)
    // -> backprop is a specific case of reverse mode?
    // Jacobian??
}

impl Tensor {
    pub fn new(dtype: String, shape: Vec<usize>, stride: Vec<usize>) -> Self {
        Self { dtype, shape, stride, device: todo!(), storage: todo!() }
    }   
}

impl Add for Tensor { type Output = Self; fn add(self, rhs: Self) -> Self::Output { eval(&MicroOp::Add, &vec![&self, &rhs]) } }
impl Sub for Tensor { type Output = Self; fn sub(self, rhs: Self) -> Self::Output { eval(&MicroOp::Sub, &vec![&self, &rhs]) } }
impl Mul for Tensor { type Output = Self; fn mul(self, rhs: Self) -> Self::Output { eval(&MicroOp::Sub, &vec![&self, &rhs]) } }
impl Div for Tensor { type Output = Self; fn div(self, rhs: Self) -> Self::Output { eval(&MicroOp::Sub, &vec![&self, &rhs]) } }
impl Tensor {
    fn dot(&self, other: Tensor) -> Self { todo!() }
    fn matmul(&self, other: Self) -> Self { todo!() }
}