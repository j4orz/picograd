use pyo3::prelude::*;
use rand::distr::StandardUniform;
use rand::Rng;
use std::{
    fmt::{self, Display},
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
};

// pub mod nn;

// TODO
// - strides (read/write)
// - algebraic/transcendental ops (+, *, sin, cos, exp, log, etc.)
// - autograd
// - fuzzer

#[rustfmt::skip]
#[derive(Clone, Debug)]
enum Device { Cpu, Cuda, Mps }

#[rustfmt::skip]
#[derive(Clone, Debug)]
enum Layout { Strided  } // Sparse, // MklDnn

#[rustfmt::skip]
#[derive(Clone, Debug)]
enum Dtype { Bool, Float16, Float32, Float64, Int16, Int32, Int64 }

#[derive(Clone, Debug)]
pub struct Tensor {
    // logical
    shape: Vec<usize>,
    stride: Vec<usize>,
    grad: Option<Box<Tensor>>,

    // physical
    device: Device,
    layout: Layout,
    dtype: Dtype,
    data: Vec<f32>, // picograd fixed on fp32 to bootstrap
}

// TODO: impl .item() for pythonic pytorch api?
impl Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format(f, &self.shape, &self.stride, 0)
    }
}

fn tensor(input: Vec<usize>) -> Tensor {
    Tensor::new()
}

impl Tensor {
    // *****************************************************************************************************************
    // ************************************************** ALLOCATORS ***************************************************
    // *****************************************************************************************************************

    fn stride(shape: &[usize]) -> Vec<usize> {
        let stride = shape
            .iter()
            .rev()
            .fold((vec![], 1), |(mut strides, acc), &dim| {
                strides.push(acc);
                (strides, acc * dim)
            })
            .0
            .into_iter()
            .rev()
            .collect::<Vec<_>>();

        stride
    }

    pub fn new() -> Self {
        todo!()
    }

    /// randn returns a tensor filled with random numbers from a normal distribution
    /// with mean 0 and variance 1 (also called the standard normal distribution)
    ///
    /// Examples
    /// ```rust
    /// let x = Tensor::randn(4);
    /// println!("{:?}", x);
    ///
    /// let y = Tensor::randn(&[2, 3]);
    /// println!("{:?}", y);
    /// ```
    pub fn randn(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product::<usize>();
        let data = (0..size)
            .map(|_| rand::rng().sample(StandardUniform))
            .collect::<Vec<f32>>();

        Tensor {
            shape: shape.to_owned(),
            stride: Self::stride(shape),
            grad: None,
            device: Device::Cpu,
            layout: Layout::Strided,
            dtype: Dtype::Float32,
            data,
        }
    }

    /// returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size
    ///
    /// Examples
    /// ```rust
    /// let x = Tensor::zeros(&[2, 3]);
    /// println!("{:?}", x);
    ///
    /// let y = Tensor::zeros(5);
    /// println!("{:?}", y);
    /// ```
    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product::<usize>();
        Tensor {
            shape: shape.to_owned(),
            stride: Self::stride(shape),
            grad: None,
            device: Device::Cpu,
            layout: Layout::Strided,
            dtype: Dtype::Float32,
            data: vec![0.0; size],
        }
    }

    /// returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size
    ///
    /// Examples
    /// ```rust
    /// let x = Tensor::ones(&[2, 3]);
    /// println!("{:?}", x);
    ///
    /// let y = Tensor::ones(5);
    /// println!("{:?}", y);
    /// ```
    pub fn ones(shape: &[usize]) -> Self {
        let size = shape.iter().product::<usize>();
        Tensor {
            shape: shape.to_owned(),
            stride: Self::stride(shape),
            grad: None,
            device: Device::Cpu,
            layout: Layout::Strided,
            dtype: Dtype::Float32,
            data: vec![1.0; size],
        }
    }

    // *****************************************************************************************************************
    // ************************************************** VIEWS ***************************************************
    // *****************************************************************************************************************

    /// returns a new tensor with the same data as the self tensor but of a different shape.
    ///
    /// The returned tensor shares the same data and must have the same number of elements, but may have a different size.
    /// For a tensor to be viewed, the new view size must be compatible with its original size and stride
    /// i.e., each new view dimension must either be a subspace of an original dimension, or only span across original dimensions
    ///
    /// Examples
    /// ```rust
    /// let x = Tensor::randn(&[4, 4]);
    /// let y = x.view(&[16]);
    /// let z = y.view(&[-1, 8]);
    /// ```
    pub fn view(&self, shape: &[i32]) -> Self {
        todo!()
    }

    pub fn permute(&self) -> Self {
        todo!()
    }

    pub fn reshape(&self) -> Self {
        todo!()
    }

    // *****************************************************************************************************************
    // ************************************************** HELPERS ***************************************************
    // *****************************************************************************************************************
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        shape: &[usize],
        stride: &[usize],
        offset: usize,
    ) -> fmt::Result {
        match (shape, stride) {
            ([], []) => {
                write!(fmt, "{:.4}", self.data[offset])?;
                Ok(())
            }
            // basecase: indexed ndarray all the way through
            // ([_dimf], [_stridef]) => {
            //     write!(fmt, "{:.4}", self.data[offset])?;
            //     Ok(())
            // }
            ([D, dimr @ ..], [stridef, strider @ ..]) => {
                write!(fmt, "[")?;
                for d in 0..*D {
                    // rolling out dimf
                    self.format(fmt, dimr, strider, offset + stridef * d)?;
                    if d != *D - 1 {
                        write!(fmt, ", ")?;
                    }
                }
                write!(fmt, "]\n")?;
                Ok(())
            }
            _ => panic!(),
        }
    }
}

// *****************************************************************************************************************
// ************************************************ AUTOGRAD + OPS *************************************************
// *****************************************************************************************************************

impl Tensor {
    fn backward(mut self) -> Self {
        if self.grad.is_none() {
            let grad_tensor = Tensor {
                shape: self.shape.clone(),
                stride: self.stride.clone(),
                device: self.device.clone(),
                layout: self.layout.clone(),
                dtype: self.dtype.clone(),
                data: vec![1.0; self.data.len()],
                grad: None,
            };
            self.grad = Some(Box::new(grad_tensor));
        }

        todo!()
    }
}

#[rustfmt::skip]
enum Op {
    Add(Box<Tensor>, Box<Tensor>),
    Sub(Box<Tensor>, Box<Tensor>),
    Mul(Box<Tensor>, Box<Tensor>),
    Div(Box<Tensor>, Box<Tensor>),
    Sin(Box<Tensor>),
    Cos(Box<Tensor>),
    Exp(Box<Tensor>),
    Log(Box<Tensor>),
    Matmul(Box<Tensor>, Box<Tensor>),
    Tanh(Box<Tensor>),
    Mean(Box<Tensor>),
    Var(Box<Tensor>),
}

impl Op {
    fn forward(&self) -> Tensor {
        match self {
            Op::Add(a, b) => Self::apply_binary_op(a, b, |a, b| a + b),
            Op::Sub(a, b) => Self::apply_binary_op(a, b, |a, b| a - b),
            Op::Mul(a, b) => Self::apply_binary_op(a, b, |a, b| a * b),
            Op::Div(a, b) => Self::apply_binary_op(a, b, |a, b| a / b),
            Op::Sin(x) => todo!(),
            Op::Cos(x) => todo!(),
            Op::Exp(x) => todo!(),
            Op::Log(x) => todo!(),
            Op::Matmul(a, b) => todo!(),
            Op::Tanh(x) => todo!(),
            Op::Mean(x) => todo!(),
            Op::Var(x) => todo!(),
        }
    }

    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        match self {
            Op::Add(_, _) => todo!(),
            Op::Sub(_, _) => todo!(),
            Op::Mul(_, _) => todo!(),
            Op::Div(_, _) => todo!(),
            Op::Sin(x) => todo!(),
            Op::Cos(x) => todo!(),
            Op::Exp(x) => todo!(),
            Op::Log(x) => todo!(),
            Op::Matmul(a, b) => todo!(),
            Op::Tanh(x) => todo!(),
            Op::Mean(x) => todo!(),
            Op::Var(x) => todo!(),
        }
    }

    fn apply_binary_op<F>(a: &Tensor, b: &Tensor, op: F) -> Tensor
    where
        F: Fn(f32, f32) -> f32,
    {
        assert_eq!(a.shape, b.shape, "Shape mismatch in operation");
        let data = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&a_val, &b_val)| op(a_val, b_val))
            .collect();

        Tensor {
            shape: a.shape.clone(),
            stride: a.stride.clone(),
            grad: None,
            device: a.device.clone(),
            layout: a.layout.clone(),
            dtype: a.dtype.clone(),
            data,
        }
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Self::Output {
        Op::Add(Box::new(self), Box::new(other)).forward()
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Self::Output {
        Op::Sub(Box::new(self), Box::new(other)).forward()
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Self::Output {
        Op::Mul(Box::new(self), Box::new(other)).forward()
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Self::Output {
        Op::Div(Box::new(self), Box::new(other)).forward()
    }
}

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        todo!()
    }

    pub fn sin(&self) -> Tensor {
        todo!()
    }

    pub fn cos(&self) -> Tensor {
        todo!()
    }

    pub fn exp(&self) -> Tensor {
        todo!()
    }

    pub fn log(&self) -> Tensor {
        todo!()
    }

    pub fn tanh(&self) -> Tensor {
        todo!()
    }

    pub fn mean(&self) -> Tensor {
        todo!()
    }

    pub fn var(&self) -> Tensor {
        todo!()
    }
}

// impl Index<(usize, usize)> for Tensor {
//     type Output = f32; // fixed to fp32 for now

//     fn index(&self, index: (usize, usize)) -> &Self::Output {
//         let (i, j) = index;
//         let idx = i * self.shape[1] + j; // Row-major ordering
//         &self.data[idx]
//     }
// }

// impl IndexMut<(usize, usize)> for Tensor {
//     fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
//         let (i, j) = index;
//         let idx = i * self.shape[1] + j; // Row-major ordering
//         &mut self.data[idx]
//     }
// }

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn picograd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
