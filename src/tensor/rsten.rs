use crate::{eagker::cpu::{forward_cpu, OpForwardError, ReduceDimInput}, engine, tensor::pyten, uop::UOp, Device, Dtype, DtypeVal, Layout};
use std::{ cell::RefCell, cmp::{Ordering, max}, collections::HashSet, fmt::{self, Display}, hash, iter, rc::Rc, ops::{Add, Div, Mul, Neg, Sub} };
use pyo3::prelude::*;
use rand::{Rng, distr::Uniform};
use thiserror::Error;

#[derive(Clone)] pub struct Storage<DtypeVal> {
    pub data: Vec<DtypeVal>, pub grad: Option<Tensor>,
}

#[pyclass(unsendable)]
pub struct Tensor {
    pub ndim: usize,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub input_op: Option<Box<UOp>>,
    pub requires_grad: bool,
    pub storage: Rc<RefCell<Storage<DtypeVal>>>,
    pub device: Device,
    pub layout: Layout,
    pub dtype: Dtype,
}

impl Neg for &Tensor { type Output = Result<Tensor, OpForwardError>; fn neg(self) -> Self::Output { unimplemented!() }}
impl Add for &Tensor { type Output = Result<Tensor, OpForwardError>; fn add(self, rhs: &Tensor) -> Self::Output { unimplemented!() }}
impl Add<f32> for &Tensor { type Output = Result<Tensor, OpForwardError>; fn add(self, rhs: f32) -> Self::Output { unimplemented!() }}
impl Sub for &Tensor { type Output = Result<Tensor, OpForwardError>; fn sub(self, rhs: &Tensor) -> Self::Output { unimplemented!() }}
impl Sub<f32> for &Tensor { type Output = Result<Tensor, OpForwardError>; fn sub(self, rhs: f32) -> Self::Output { unimplemented!() }}
impl Mul<f32> for &Tensor { type Output = Result<Tensor, OpForwardError>; fn mul(self, rhs: f32) -> Self::Output { unimplemented!() }}
impl Mul<&Tensor> for f32 { type Output = Result<Tensor, OpForwardError>; fn mul(self, rhs: &Tensor) -> Self::Output { unimplemented!()} }
impl Div for &Tensor { type Output = Result<Tensor, OpForwardError>; fn div(self, rhs: &Tensor) -> Self::Output { unimplemented!() }}
impl Div<f32> for &Tensor { type Output = Result<Tensor, OpForwardError>; fn div(self, rhs: f32) -> Self::Output { unimplemented!() }}

impl Tensor {
    // matmul and flashattention is all you need ;)
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, OpForwardError> { unimplemented!() }
    pub fn fa() -> () { unimplemented!( )}

    // unary
    pub fn neg() -> Self { unimplemented!()}
    pub fn contiguous() -> Self { unimplemented!()}
    pub fn log() -> Self { unimplemented!()}
    pub fn log2() -> Self { unimplemented!()}
    pub fn exp() -> Self { unimplemented!()}
    pub fn exp2() -> Self { unimplemented!()}
    pub fn sigmoid() -> Self { unimplemented!()}
    pub fn logsigmoid() -> Self { unimplemented!()}
    pub fn hardsigmoid() -> Self { unimplemented!()}
    pub fn sqrt() -> Self { unimplemented!()}
    pub fn rsqrt() -> Self { unimplemented!()}
    pub fn sin() -> Self { unimplemented!()}
    pub fn cos() -> Self { unimplemented!()}
    pub fn tan() -> Self { unimplemented!()}
    pub fn asin() -> Self { unimplemented!()}
    pub fn acos() -> Self { unimplemented!()}
    pub fn atan() -> Self { unimplemented!()}

    // binary
    pub fn sub() -> Self { unimplemented!()}
    pub fn div() -> Self { unimplemented!()}
    pub fn modd() -> Self { unimplemented!()}
    pub fn lshift() -> Self { unimplemented!()}
    pub fn rshift() -> Self { unimplemented!()}
    pub fn pow() -> Self { unimplemented!()}
    pub fn max() -> Self { unimplemented!()}
    pub fn min() -> Self { unimplemented!()}

    // math
    pub fn trunc() -> Self { unimplemented!()}
    pub fn ceil() -> Self { unimplemented!()}
    pub fn floor() -> Self { unimplemented!()}
    pub fn round() -> Self { unimplemented!()}
    pub fn isinf() -> Self { unimplemented!()}
    pub fn isnan() -> Self { unimplemented!()}
    pub fn isfinite() -> Self { unimplemented!()}
    pub fn lerp() -> Self { unimplemented!()}
    pub fn square() -> Self { unimplemented!()}
    pub fn clamp() -> Self { unimplemented!()}
    pub fn clip() -> Self { unimplemented!()}
    pub fn sign() -> Self { unimplemented!()}
    pub fn abs() -> Self { unimplemented!()}
    pub fn recip() -> Self { unimplemented!()}

    // activations
    pub fn elu() -> Self { unimplemented!() }
    pub fn celu() -> Self { unimplemented!() }
    pub fn gelu() -> Self { unimplemented!() }
    pub fn relu() -> Self { unimplemented!() }
    pub fn selu() -> Self { unimplemented!() }
    pub fn silu() -> Self { unimplemented!() }
    pub fn swish() -> Self { unimplemented!() }
    pub fn sinh(&self) -> Result<Tensor, OpForwardError> { unimplemented!() }
    pub fn cosh(&self) -> Result<Tensor, OpForwardError> { unimplemented!() }
    pub fn tanh(&self) -> Result<Tensor, OpForwardError> { unimplemented!() }
    pub fn asinh(&self) -> Result<Tensor, OpForwardError> { unimplemented!() }
    pub fn acosh(&self) -> Result<Tensor, OpForwardError> { unimplemented!() }
    pub fn atanh(&self) -> Result<Tensor, OpForwardError> { unimplemented!() }

    // tensor
    pub fn ndim() -> () { todo!() }
    // pub fn numel() -> () { todo!() }

    // reductions
    // pub fn _sum(&self, rdi: Option<ReduceDimInput>) -> Result<Tensor, OpForwardError> { self.forward(&UOp::Sum(self.clone(), rdi)) }
    // pub fn max(&self, dim: usize, keepdim: bool) -> Result<Tensor, OpForwardError> { let rdi = ReduceDimInput { dim, keepdim }; self.forward(&Op::Max(self.clone(), Some(rdi))) }
}

impl Tensor {
    // Tensor
    // ------
    // storage: two design decisions
    //      1. lifetimes Box<_> vs Rc<_>
    //      2. mutation: RefCell<_>(safe) vs UnsafeCell<_>(speed)j
    // impl Tensor:
    // - CONSTRUCTORS (alloc): alloc
    // - VIEWS (no alloc): continuous_view, noncontinuous_view, reshape, permute, getitem, squeeze, unsqueeze
    // - INDEXING (internal): shape_to_stride, encode, decode
    // - BROADCASTING: broadcast_shape, clamp_stride, broadcast_logidx
    // - SAMPLING: randn
    // - AUTODIFF: toposort, backward

// NB:
// - dtype not typed with storagek
    pub fn to(&self, d: &Device) -> Self {
        let foo = match d {
            Device::Cpu => todo!(),
            Device::Gpu => todo!(),
            Device::Amd => todo!(),
        };

        todo!()
    }

    pub fn numel(&self) -> usize { self.shape.iter().product::<usize>() }

    // *************************************************************************
    // *************************** VIEWS (no alloc) ****************************
    // *************************************************************************

    pub fn is_contiguous(&self) -> bool {
        todo!()
    }

    pub fn _view(&self, shape: &[usize], stride: &[usize]) -> Self {
        Self {
            ndim: shape.len(),
            shape: shape.to_vec(),
            stride: stride.to_vec(),
            input_op: self.input_op.clone(), // Box<_>.clone()?
            requires_grad: self.requires_grad,
            storage: self.storage.clone(),
            device: self.device.clone(),
            layout: self.layout.clone(),
            dtype: self.dtype.clone(),
        }
    }

    pub fn view(&self) -> Self {
        todo!()
    }

    pub fn _reshape(&self, new_shape: &[i32]) -> Result<Self, ViewOpError> {
        let old_size = self.shape.iter().product::<usize>();
        let negone_count = new_shape.iter().filter(|&&x| x == -1).count();

        let new_shape = match negone_count.cmp(&1) {
            Ordering::Greater => Err(ViewOpError::InvalidReshapeInput),
            Ordering::Less => Ok(new_shape
                .iter()
                .map(|&x| usize::try_from(x).map_err(|_| ViewOpError::InvalidReshapeInput))
                .collect::<Result<Vec<_>, _>>()?),
            Ordering::Equal => {
                let rest_shape = new_shape
                    .iter()
                    .filter(|&&x| x != -1)
                    .map(|&x| usize::try_from(x).map_err(|_| ViewOpError::InvalidReshapeInput))
                    .collect::<Result<Vec<_>, _>>()?;
                let rest_shape_size = rest_shape.iter().product::<usize>();

                if old_size % rest_shape_size != 0 {
                    Err(ViewOpError::ShapeMismatch)
                } else {
                    let inferred_dim = old_size / rest_shape_size;
                    Ok(new_shape
                        .iter()
                        .map(|&x| {
                            if x == -1 {
                                Ok(inferred_dim)
                            } else {
                                usize::try_from(x).map_err(|_| ViewOpError::InvalidReshapeInput)
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?)
                }
            }
        }?;

        Ok(self._view(&new_shape, &Self::shape_to_stride(&new_shape))) // TODO: alloc just to ref which allocs gain inside ._view?
    }

    // TODO: validate permutatation
    pub fn _permute(&self, permuted_shape: &[usize]) -> Self {
        let new_shape = permuted_shape
            .iter()
            .map(|&i| self.shape[i])
            .collect::<Vec<_>>();
        let new_stride = permuted_shape
            .iter()
            .map(|&i| self.stride[i])
            .collect::<Vec<_>>();

        self._view(&new_shape, &new_stride)
    }

    // pub fn _transpose(&self, indices: &[usize]) -> Self {
    //     todo!()
    // }

    pub fn _T(&self) -> Self {
        self._permute(&[1, 0])
    }

    pub fn _squeeze(&self, dim: &Vec<usize>) -> Self {
        let squeezed_indices = dim.iter().copied().collect::<HashSet<_>>();
        let noop_indices = self
            .shape
            .iter()
            .enumerate()
            .filter(|(i, x)| **x == 1)
            .map(|(i, x)| i)
            .collect::<HashSet<_>>();
        let squeezed_noop_indices = if squeezed_indices.len() == 0 {
            noop_indices
        } else {
            squeezed_indices
                .intersection(&noop_indices)
                .copied()
                .collect::<HashSet<_>>()
        };

        let (new_shape, new_stride) = (
            self.shape
                .iter()
                .enumerate()
                .filter(|(i, x)| !squeezed_noop_indices.contains(i))
                .map(|(i, x)| x)
                .copied()
                .collect::<Vec<_>>(),
            self.stride
                .iter()
                .enumerate()
                .filter(|(i, x)| !squeezed_noop_indices.contains(i))
                .map(|(i, x)| x)
                .copied()
                .collect::<Vec<_>>(),
        );
        self._view(&new_shape, &new_stride)
    }

    pub fn _unsqueeze(&self, dim: usize) -> Self {
        let (mut new_shape, mut new_stride) = (self.shape.clone(), self.stride.clone());
        new_shape.insert(dim, 1);
        new_stride.insert(dim, 1);
        self._view(&new_shape, &new_stride)
    }

    // Z_I1I2...IND2..DN = X_*D1*D2...DN[I_I1I2..IN]
    // I indexes into X's first dimension
    // so I ∈ 0..D1 must hold.
    // e.g C_VE[X_BT] for a character-level language model where |V| = 27 ==> X_BT ∈ 0..26
    pub fn getitem_embedding(&self, I: &Tensor) -> Self {
        let X = self;
        let shape_z = I
            .shape
            .iter()
            .chain(X.shape[1..].iter())
            .cloned()
            .collect::<Vec<_>>();
        let data_z = vec![DtypeVal::Float32(0.0); shape_z.iter().product()];
        let z = alloc(&shape_z, data_z);

        for phy in 0..I.storage.borrow().data.len() {
            let index = usize::from(I.storage.borrow().data[phy]);
            let plucked_phystart = index * X.stride[0]; // I's index used to index X's first dim
            let plucked_phyend = plucked_phystart + X.stride[0]; // .numel() in D2*D3*...DN = X.stride[0] = X.shape.iter().skip(1).product();
            let plucked = &X.storage.borrow().data[plucked_phystart..plucked_phyend];
            let dst_phystart = phy * plucked.len();
            let dst_phyend = dst_phystart + plucked.len();
            z.storage.borrow_mut().data[dst_phystart..dst_phyend].copy_from_slice(plucked);
        }

        z
    }

    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        shape: &[usize],
        stride: &[usize],
        offset: usize,
    ) -> fmt::Result {
        match (shape, stride) {
            ([], []) => {
                if offset < self.storage.borrow().data.len() {
                    write!(fmt, "{:?}", self.storage.borrow().data[offset])?;
                }
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
            (foo, bar) => {
                println!("moose: {:?}, {:?}", foo, bar);
                panic!()
            }
        }
    }

    // *************************************************************************
    // ******************************* INDEXING ********************************
    // *************************************************************************

    // note: best way to think about strides
    // is to conceptualize the indices of the nested for loop when unrolling
    // e.g: shape: [3, 4, 6] ==> stride: [24, 6, 1]
    //      - one step for outer is 24 physical indices
    //      - one step for middle is 6 physical indices
    //      - one step for inner is 1 physical index (always)

    // 1 is last because of C/C++/Rust row-major ordering over Fortran/IDL column-major ordering
    fn shape_to_stride(shape: &[usize]) -> Vec<usize> {
        let stride = shape
            .iter()
            .rev()
            .fold((vec![], 1), |(mut strides, acc), &dim| {
                strides.push(acc);
                (strides, acc * dim) // last acc does not push
            })
            .0
            .into_iter()
            .rev()
            .collect::<Vec<_>>();

        stride
    }

    // NB: Vec<usize> is used for internal indexing
    // Vec<DtypeVal=Int32> is used by library crate users

    // encode (lifting): phys(usize) -> log(Vec<usize>)
    pub fn encode(phys: usize, shape: &[usize]) -> Vec<usize> {
        let mut log = vec![0; shape.len()];
        let (stride, mut phys) = (Self::shape_to_stride(shape), phys);

        // expand the product into factorization
        // traversing from largest step to smallest step
        for i in 0..stride.len() {
            log[i] = phys / stride[i]; // how many stride[i] fit into phys
            phys %= stride[i]; // remainder
        }

        log
    }

    // decode (lowering): log(Vec<usize>) -> phys(usize)
    pub fn decode(log: &[usize], stride: &[usize]) -> usize {
        // collapse the factorization into a product
        log.iter()
            .zip(stride.iter())
            // .inspect(|(i, s)| println!("i: {:?}, s: {:?}", i, s))
            .fold(0, |acc, (i, s)| acc + i * s)
    }

    pub fn _item(&self) -> DtypeVal {
        let data = &self.storage.borrow().data;
        if data.len() > 1 { todo!() } else { data[0] }
    }

    // *************************************************************************
    // ***************************** BROADCASTING ******************************
    // *************************************************************************
    pub fn broadcast_shape(
        shape_x: &[usize],
        shape_y: &[usize],
    ) -> Result<Vec<usize>, ViewOpError> {
        let max_len = max(shape_x.len(), shape_y.len());
        let shape_x = iter::repeat(1)
            .take(max_len - shape_x.len())
            .chain(shape_x.iter().copied())
            .collect::<Vec<_>>();
        let shape_y = iter::repeat(1)
            .take(max_len - shape_y.len())
            .chain(shape_y.iter().copied())
            .collect::<Vec<_>>();

        let output = shape_x
            .iter()
            .zip(shape_y.iter())
            .map(|(&x, &y)| match (x, y) {
                (x, y) if x == y => Ok(x),
                (1, y) => Ok(y),
                (x, 1) => Ok(x),
                _ => Err(ViewOpError::BroadcastMismatch),
            })
            .collect::<Result<Vec<usize>, ViewOpError>>();

        output
    }

    pub fn clamp_stride(shape_x: &[usize]) -> Vec<usize> {
        // TODO: assumes input_shape successfully broadcasted with other input
        // TODO: ignore padded scenario when input_shape.len() < output_shape.len()
        let output = shape_x
            .iter()
            .rev()
            .fold((vec![], 1), |(mut stride, mut acc), &dim| {
                if dim == 1 {
                    stride.push(0);
                } else {
                    stride.push(acc);
                    acc *= dim;
                }
                (stride, acc)
            })
            .0
            .into_iter()
            .rev()
            .collect::<Vec<_>>();

        output
    }

    pub fn broadcast_logidx(log_x: &[usize], log_y: &[usize]) -> Result<Vec<usize>, ViewOpError> {
        let max_len = max(log_x.len(), log_y.len());
        let padded_x = iter::repeat(0)
            .take(max_len - log_x.len())
            .chain(log_x.iter().copied());
        let padded_y = iter::repeat(0)
            .take(max_len - log_y.len())
            .chain(log_y.iter().copied());

        let output = padded_x
            .zip(padded_y)
            .map(|(x, y)| match (x, y) {
                (x, y) if x == y => Ok(x),
                (0, y) => Ok(y),
                (x, 0) => Ok(x),
                _ => Err(ViewOpError::BroadcastMismatch),
            })
            .collect::<Result<Vec<usize>, ViewOpError>>();

        output
    }

    // *************************************************************************
    // ******************************** RANDOM *********************************
    // *************************************************************************
    pub fn randint(low: i32, high: i32, shape: &[usize]) -> Self {
        let mut rng = rand::rng();
        let dist = Uniform::new(low, high).unwrap();
        let n = shape.iter().product::<usize>();

        let data = (0..n)
            .map(|_| DtypeVal::Int32(rng.sample(dist)))
            .collect::<Vec<_>>();

        alloc(&shape, data)
    }

    pub fn randn(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product::<usize>();
        let mut rng = rand::rng();
        let data = (0..n)
            .map(|_| DtypeVal::Float32(rng.sample(Uniform::new(0.0, 1.0).unwrap())))
            .collect::<Vec<_>>();
        alloc(&shape, data)
    }

    // inverse (smirnov) transform sampling
    // 1. u ~ U(0, dist.iter().sum())
    // 2. x <- F⁻¹(u)
    // see: https://en.wikipedia.org/wiki/Inverse_transform_sampling
    pub fn multinomial(dist: &Tensor, samples: usize, replacement: bool) -> Tensor {
        let dist = &dist
            .storage
            .borrow()
            .data
            .iter()
            .map(|d| match d {
                DtypeVal::Bool(_) => todo!(),
                DtypeVal::Float32(x) => *x,
                DtypeVal::Float64(_) => todo!(),
                DtypeVal::Int16(_) => todo!(),
                DtypeVal::Int32(_) => todo!(),
                DtypeVal::Int64(_) => todo!(),
            })
            .collect::<Vec<_>>();

        let total = dist.iter().sum::<f32>();
        let mut output = Vec::with_capacity(samples);
        let mut rng = rand::rng();
        for _ in 0..samples {
            let u = rng.sample(Uniform::new(0.0, total).unwrap());
            let (mut cumsum, mut index) = (0.0, 0);
            for (i, p) in dist.iter().enumerate() {
                cumsum += p;
                if cumsum >= u {
                    index = i;
                    break; // walk until we hit the border of the area
                }
            }

            output.push(DtypeVal::Int32(index as i32)) // TODO: dtypes are messed up
        }
        alloc(&vec![samples], output)
    }
}


// NB: clones are not expensive because they implement view semantics. the
//     storage.clone() call is a call to Rc::clone() which simply increments
//     an internal counter. See Niko Matsakis' early thoughts on making these
//     cheap clones with a "Claim" trait:
//     - https://smallcultfollowing.com/babysteps/blog/2024/06/21/claim-auto-and-otherwise/
//     - https://smallcultfollowing.com/babysteps/blog/2024/06/26/claim-followup-1/

// TODO: detach?
impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            ndim: self.ndim,
            shape: self.shape.clone(),
            stride: self.stride.clone(),
            input_op: self.input_op.clone(),
            requires_grad: self.requires_grad,
            storage: self.storage.clone(),
            device: self.device.clone(),
            layout: self.layout.clone(),
            dtype: self.dtype.clone(),
        }
    }
}

// NB: reference comparison instead of value since f32 is not Eq.
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Eq for Tensor {}

impl hash::Hash for Tensor {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        (self as *const Self).hash(state);
    }
}

// TODO: impl .item() for pythonic pytorch api?
impl Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format(f, &self.shape, &self.stride, 0)
    }
}

#[derive(Error, Debug)]
pub enum ViewOpError {
    #[error("broadcast mismatch")] BroadcastMismatch,
    #[error("shape mismatch")] ShapeMismatch,
    #[error("invalid reshape input")] InvalidReshapeInput,
    #[error("unknown tensor error")] Unknown,
}

pub fn alloc(shape: &[usize], data: Vec<DtypeVal>) -> Tensor {
    Tensor {
        ndim: shape.len(), shape: shape.to_owned(), stride: Tensor::shape_to_stride(shape),
        input_op: None, requires_grad: false,
        storage: Rc::new(RefCell::new(Storage { data, grad: None })), device: Device::Cpu, layout: Layout::Strided, dtype: Dtype::Float32,
    }
}