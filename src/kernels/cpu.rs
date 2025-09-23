use crate::{ tensor::pyten::{self}, tensor::rsten::{self, Op, Storage, Tensor, ViewOpError}, Dtype, DtypeVal };
use std::{ cell::RefCell, cmp, ops::{Add, AddAssign, Div, Mul, Neg, Sub}, rc::Rc };
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OpForwardError {
    #[error(transparent)] TensorError(#[from] ViewOpError),
    #[error("unknown operation error")] Unknown,
}

pub fn forward_cpu(op: &Op) -> Result<Tensor, OpForwardError> {
    match op {
        Op::Add(x, y) => zip_cpu(|xi, yi| xi + yi, x, y),
        Op::Sub(x, y) => zip_cpu(|xi, yi| xi - yi, x, y),
        Op::Mul(x, y) => zip_cpu(|xi, yi| xi * yi, x, y),
        Op::Div(x, y) => zip_cpu(|xi, yi| xi / yi, x, y),
        Op::Neg(x) => map_cpu(&op, |xi| -xi, x),
        // TODO: generalize transcental uops to other dtypes?
        Op::Exp(x) => map_cpu(&op, |xi| DtypeVal::Float32(f32::from(xi).exp()), x),
        Op::Log(x) => map_cpu(&op, |xi| DtypeVal::Float32(f32::from(xi).ln()), x),
        Op::Sinh(x) => map_cpu(&op, |xi| DtypeVal::Float32(f32::from(xi).sinh()), x),
        Op::Cosh(x) => map_cpu(&op, |xi| DtypeVal::Float32(f32::from(xi).cosh()), x),
        Op::Tanh(x) => map_cpu(&op, |xi| DtypeVal::Float32(f32::from(xi).tanh()), x),
        Op::Sum(x, rdi) => reduce_cpu(|xi, yi| xi + yi, x, rdi),
        Op::Max(x, rdi) => reduce_cpu(
            // TODO: maybe_reduce_cpu where F: DTypeVal, DTypeVal -> Option<DTypeVal>?
            // b/c of partialcmp.
            |xi, yi| match xi.partial_cmp(&yi).unwrap() {
                cmp::Ordering::Less => yi,
                cmp::Ordering::Equal => xi,
                cmp::Ordering::Greater => xi,
            },
            x,
            rdi,
        ),
        Op::Matmul(X, Y) => matmul_cpu(X, Y),
    }
}

fn map_cpu<F>(op: &Op, f: F, x: &Tensor) -> Result<Tensor, OpForwardError>
where
    F: Fn(DtypeVal) -> DtypeVal,
{
    Ok(Tensor {
        ndim: x.ndim,
        shape: x.shape.clone(),
        stride: x.stride.clone(),
        input_op: Some(Box::new(op.clone())),
        requires_grad: x.requires_grad,
        storage: Rc::new(RefCell::new(Storage {
            data: x.storage.borrow().data.iter().map(|&xi| f(xi)).collect(),
            grad: None,
        })),
        device: x.device.clone(),
        layout: x.layout.clone(),
        dtype: x.dtype.clone(),
    })
}

// 1. z_shape <- broadcast_shape(x.shape, y.shape)
// 2. for physz in z.numel():

//      **a. find the logical nd index of x and y**
//      3. logz <- encode(i) # lift physical z to logical z
//      4. (logx, logy) <- broadcast_logidx(logx, logz), broadcast_logidx(logy, logz) # map logical z over to logical x and y
//      5. physx, physy <- decode(logx), decode(logy) # lower x and y to physical

//      6. z[physz] <- f(x[physx], y[physy])
fn zip_cpu<F>(f: F, x: &Tensor, y: &Tensor) -> Result<Tensor, OpForwardError>
where
    F: Fn(DtypeVal, DtypeVal) -> DtypeVal,
{
    let (x, y, z) = if x.shape != y.shape {
        let z_shape = Tensor::broadcast_shape(&x.shape, &y.shape)?;
        let z = pyten::zeros(z_shape, Dtype::Float32); // clone because of python/rust memory mismatch

        let (x_stride, y_stride) = (
            Tensor::clamp_stride(&x.shape),
            Tensor::clamp_stride(&y.shape),
        );

        let (mut x, mut y) = (x.clone(), y.clone());
        x.stride = x_stride;
        y.stride = y_stride;

        (x, y, z)
    } else {
        let z = pyten::zeros(x.shape.clone(), Dtype::Float32); // clone because of python/rust memory mismatch
        (x.clone(), y.clone(), z) // TODO: possibly remove taking view just to satisfy typechecker
    };

    // println!("x.shape: {:?}, x.stride: {:?}", x.shape, x.stride);
    // println!("y.shape: {:?}, y.stride: {:?}", y.shape, y.stride);
    // println!("z.shape: {:?}, z.stride: {:?}", z.shape, z.stride);

    {
        let (x_storage, y_storage, mut z_storage) = (
            x.storage.borrow(),
            y.storage.borrow(),
            z.storage.borrow_mut(),
        );
        let (logx, logy) = (vec![0; x.ndim], vec![0; y.ndim]);

        for phyz in 0..z.numel() {
            let logz = Tensor::encode(phyz, &z.shape);

            let (logx, logy) = (
                Tensor::broadcast_logidx(&logx, &logz)?,
                Tensor::broadcast_logidx(&logy, &logz)?,
            );
            let (phyx, phyy) = (
                Tensor::decode(&logx, &x.stride),
                Tensor::decode(&logy, &y.stride),
            );

            // println!("phy: {:?}", phyz);
            // println!("logx {:?} -> physx: {:?}", logx, phyx);
            // println!("logy {:?} -> physy: {:?}", logy, phyy);
            // println!("logz {:?} -> physz: {:?}", logz, phyz);

            z_storage.data[phyz] = f(x_storage.data[phyx], y_storage.data[phyy]);
        }
    }

    Ok(z)
}

#[derive(Clone)]
pub struct ReduceDimInput {
    pub dim: usize,
    pub keepdim: bool,
}

fn reduce_cpu<F>(f: F, x: &Tensor, rdi: &Option<ReduceDimInput>) -> Result<Tensor, OpForwardError>
where
    F: Fn(DtypeVal, DtypeVal) -> DtypeVal,
{
    if x.ndim == 0 {
        return Ok(x.clone());
    }

    match rdi {
        None => {
            let init = match x.dtype {
                Dtype::Bool => DtypeVal::Bool(true),
                Dtype::Float32 => DtypeVal::Float32(0.0),
                Dtype::Float64 => DtypeVal::Float64(0.0),
                Dtype::Int32 => DtypeVal::Int32(0),
                Dtype::Int64 => DtypeVal::Int64(0),
            };
            let output_scalar = &x.storage.borrow().data.iter().fold(init, |acc, &x| acc + x);
            let output_tensor = rsten::alloc(&vec![], vec![*output_scalar]);
            Ok(output_tensor)
        }
        Some(rdi) => {
            let (dim, keepdim) = (rdi.dim, rdi.keepdim);
            let y_shape = x
                .shape
                .iter()
                .enumerate()
                .map(|(i, &dim_size)| if i == dim { 1 } else { dim_size })
                .collect();
            let mut y = pyten::zeros(y_shape, Dtype::Float32);

            {
                let (x_storage, mut y_storage) = (x.storage.borrow(), y.storage.borrow_mut());
                for physy in 0..y.numel() {
                    // reconstructing logx from phsy by undoing reduce via logx[dim]=d
                    let mut logx = Tensor::encode(physy, &y.shape);
                    for d in 0..x.shape[dim] {
                        logx[dim] = d;
                        let physx = Tensor::decode(&logx, &x.stride);

                        // y=y+x
                        y_storage.data[physy] = f(y_storage.data[physy], x_storage.data[physx]);
                    }
                }
            }

            if !keepdim {
                y.shape.remove(dim);
                y.ndim -= 1;
            }

            Ok(y)
        }
    }
}

fn matmul_cpu(X: &Tensor, Y: &Tensor) -> Result<Tensor, OpForwardError> {
    // promote 1D -> 2D via unsqueeze
    let (X, Y, unsqzd) = match (X.ndim, Y.ndim) {
        (2, 1) => (X.clone(), Y._unsqueeze(1), Some(1)), // matvec
        (1, 2) => (X._unsqueeze(0), Y.clone(), Some(0)), // vecmat
        (1, 1) => (X._unsqueeze(1), Y._unsqueeze(0), Some(0)), // vecvec
        (2, 2) => (X.clone(), Y.clone(), None),          // matmat
        _ => unimplemented!("matmul only supports 1D and 2D tensors"),
    };

    let (N, M1, M2, P) = (X.shape[0], X.shape[1], Y.shape[0], Y.shape[1]);
    assert_eq!(M1, M2, "Shape mismatch in operation");
    assert_eq!(X.ndim, 2, "X must be a 2D tensor");
    assert_eq!(Y.ndim, 2, "Y must be a 2D tensor");
    let Z = pyten::zeros(vec![N, P], Dtype::Float32);

    {
        let (X_storage, Y_storage, mut Z_storage) = (
            X.storage.borrow(),
            Y.storage.borrow(),
            Z.storage.borrow_mut(),
        );

        // p linear combinations of m basis vectors in R^n (linear combination p times)

        // (nxm)@(m@p)
        for n in 0..N {
            for p in 0..P {
                // NB: p linear combinations of m2 entries mapped to m1 basis vectors in R^n
                // the loops over n p are interchangeable (n*p or p*n).
                // wikpedia has n*p even though p*n is more geometically intuitive

                // dot product of m entries mapped to m basis vectors
                for m in 0..M1 {
                    // dot product:
                    let x = X_storage.data[n * X.stride[0] + m * X.stride[1]];
                    let y = Y_storage.data[m * Y.stride[0] + p * Y.stride[1]];
                    Z_storage.data[n * Z.stride[0] + p * Z.stride[1]] += x * y;
                }
            }
        }
    }

    if let Some(dim) = unsqzd {
        let mut Zsqzd = Z;
        Zsqzd.shape.remove(dim);
        Zsqzd.stride.remove(dim);
        Zsqzd.ndim -= 1;
        return Ok(Zsqzd);
    }

    Ok(Z)
}

impl Add for DtypeVal {
    type Output = DtypeVal;

    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (DtypeVal::Float32(x), DtypeVal::Float32(y)) => DtypeVal::Float32(x + y),
            (DtypeVal::Int32(x), DtypeVal::Int32(y)) => DtypeVal::Int32(x + y),
            _ => todo!(),
        }
    }
}

impl AddAssign for DtypeVal {
    fn add_assign(&mut self, other: Self) {
        *self = self.add(other);
    }
}

impl Sub for DtypeVal {
    type Output = DtypeVal;

    fn sub(self, other: Self) -> Self::Output {
        match (self, other) {
            (DtypeVal::Float32(x), DtypeVal::Float32(y)) => DtypeVal::Float32(x - y),
            (DtypeVal::Int32(x), DtypeVal::Int32(y)) => DtypeVal::Int32(x - y),
            _ => todo!(),
        }
    }
}

impl Mul for DtypeVal {
    type Output = DtypeVal;

    fn mul(self, other: Self) -> Self::Output {
        match (self, other) {
            (DtypeVal::Float32(x), DtypeVal::Float32(y)) => DtypeVal::Float32(x * y),
            (DtypeVal::Int32(x), DtypeVal::Int32(y)) => DtypeVal::Int32(x * y),
            _ => todo!(),
        }
    }
}

impl Div for DtypeVal {
    type Output = DtypeVal;

    fn div(self, other: Self) -> Self::Output {
        match (self, other) {
            (DtypeVal::Float32(x), DtypeVal::Float32(y)) => DtypeVal::Float32(x / y),
            (DtypeVal::Int32(x), DtypeVal::Int32(y)) => DtypeVal::Int32(x / y),
            _ => todo!(),
        }
    }
}

impl Neg for DtypeVal {
    type Output = DtypeVal;

    fn neg(self) -> Self::Output {
        match self {
            DtypeVal::Float32(x) => DtypeVal::Float32(-x),
            DtypeVal::Int32(x) => DtypeVal::Int32(-x),
            _ => todo!(),
        }
    }
}