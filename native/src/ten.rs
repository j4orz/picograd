//! Tensor
//! ...

use std::ops::{Add, Sub, Mul, Div};

use crate::eval::Device;

pub struct Tensor {
    pub device: Device,
    pub shape: Vec<usize>, stride: Vec<usize>,
}

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output { todo!() }
}

impl Sub for Tensor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output { todo!() }
}

impl Mul for Tensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output { todo!() }
}

impl Div for Tensor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output { todo!() }
}

impl Tensor {
    fn dot(&self, other: Tensor) -> Self { todo!() }
    fn matmul(&self, other: Self) -> Self { todo!() }
}