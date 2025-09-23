use std::collections::HashSet;

use crate::{kernels::cpu::{forward_cpu, OpForwardError}, tensor::pyten, tensor::rsten::{Op, Tensor}, Device};

// The variant order preserves the intended toposort priority.
pub enum TinyOp {
    NoOp, Sink, Unique, Device, Kernel, Precast, RewriteError, // uops that aren't rendered
    Child, Children, // track children
    Copy, Buffer, BufferView, MSelect, MStack, // buffer ops 
    Bufferize, // create buffer
    Contiguous, ContiguousBackward, Detach, Fuse, Realize, // ops that adjust the behavior of the scheduler
    Block, BlockStart, BlockEnd, BlockFinal, // blocks in linearizer (only used there) 
    Reshape, Permute, Expand, Pad, Shrink, Flip, // movement ops! these only exist in the tensor graph 
    Multi, // MULTI is really a movement op
    View, // view is what all movement ops become
    Valid, // TODO: remove VALID with the VIEW(CONST(DEVICE)) refactor
    DefineGlobal, DefineLocal, DefineReg, // TODO: unify these ops into the levels of the memory hierarchy. depends on ASSIGN is STORE
    DefineVar, Bind, // this is for symbolic shapes
    Special, // this is a RANGE for GPU dimensions, similar to symbolic shapes but not exactly
    ReduceAxis, Reduce, AllReduce, // reduce
    Unroll, Contract, Gep, Vectorize, Cat, PtrCat, // optimization helper ops
    Cast, BitCast, Exp2, Log2, Sin, Sqrt, Recip, Neg, Trunc, // UnaryOps 
    Load, Store, // load/store before math
    Assign, // TODO: ASSIGN is STORE, remove ASSIGN
    Wmma, // tensor core math op, not elementwise
    Index, // INDEX is a BinaryOp similar to ADD, but it operates on pointers
    Add, Mul, Shl, Shr, IDiv, Max, Mod, CmpLt, CmpNe, CmpEq, Xor, Or, And, ThreeFry, Sub, FDiv, Pow, // binops
    Where, MulAcc, // ternops
    Barrier, Range, If, EndRange, EndIf, // controlflowops
    VConst, Const, // consts. VCONST is a vectorized const
    Custom, CustomI, // CUSTOM/CUSTOMI are used to output strings into codegen. the I makes the string inline
}

impl Op {
    pub fn inputs(&self) -> Vec<&Tensor> {
        // flatten inputs: tup -> vec
        match self {
            Op::Add(x, y) | Op::Sub(x, y) | Op::Mul(x, y) | Op::Div(x, y) | Op::Matmul(x, y) => { vec![x, y] }
            Op::Neg(x) | Op::Exp(x) | Op::Log(x) | Op::Sinh(x) | Op::Cosh(x) | Op::Tanh(x) => { vec![x] } // | Op::Mean(x) | Op::Var(x)
            Op::Sum(x, _) => vec![x],
            Op::Max(x, _) => vec![x]
      }
    }
}

impl Tensor {
    pub fn forward(&self, op: &Op) -> Result<Tensor, OpForwardError> {
        match self.device {
            Device::Cpu => forward_cpu(op),
            // Device::Gpu => forward_wgsl(op),
            // Device::Cuda => forward_cuda(op),
            _ => unimplemented!("picograd only supports cpu, gpu(opencl) or nv(cuda)"),
        }
    }

    pub fn backward(&self) -> () {
        self.storage.borrow_mut().grad = Some(pyten::ones(self.shape.clone()));
        self._backward();
        for tensor in self.topo().iter().rev() {
            tensor._backward();
        }
    }

    fn _backward(&self) -> () {
        // NB: autodifferentiation's .backward() is defined on Ops, not Tensors.
        // since the gradient gives us the perturbation sensitivty that a
        // function's input has on the final loss, it would be clearer mathematically if
        // .grad lived on Op, not Tensor.
        if self.input_op.is_none() { return }
        let op = self.input_op.as_ref().unwrap();
        let storage_ref = self.storage.borrow(); // lifetime needs to be extended
        let dfdx_cached = storage_ref.grad.as_ref().unwrap();

        // evaluate local derivatives via chain rule
        let local_grads = backward_cpu(&op, &self, dfdx_cached);

        // propagate derivative to inputs assuming grads.len() == op.inputs().len()
        for (x, dfdx_next) in op.inputs().into_iter().zip(local_grads.iter()) {
            let mut storage = x.storage.borrow_mut();
            match storage.grad {
                Some(ref mut dfdx_prev) => *dfdx_prev = (&*dfdx_prev + dfdx_next).unwrap(),
                None => storage.grad = Some(dfdx_next.clone()),
            }
        }
    }

    fn topo(&self) -> Vec<Tensor> {
        let (mut output, mut seen) = (Vec::new(), HashSet::new());
        Self::_visit(self, &mut output, &mut seen);
        output
    }

    fn _visit(tensor: &Tensor, output: &mut Vec<Tensor>, seen: &mut HashSet<Tensor>) {
        if seen.contains(&tensor) { return }
        seen.insert(tensor.clone());
        if let Some(ref op) = tensor.input_op {
            for input in op.inputs() { Self::_visit(input, output, seen) }
        }
        output.push(tensor.clone());
    }
}

// dFdx is the global derivative (backpropagated to the current op)
// dfdx is the local derivative
pub fn backward_cpu(op: &Op, opout: &Tensor, dFdx: &Tensor) -> Vec<Tensor> {
    match op {
        Op::Add(_x, _y) => vec![
            (1.0 * &dFdx.clone()).unwrap(),
            (1.0 * &dFdx.clone()).unwrap(),
        ],
        Op::Sub(x, y) => todo!(),
        Op::Mul(x, y) => todo!(), //vec![(y * &dFdx.clone()).unwrap(), (x * &dFdx.clone()).unwrap()],
        Op::Div(x, y) => todo!(),
        Op::Matmul(x, y) => todo!(),
        Op::Neg(x) => todo!(),
        Op::Exp(x) => todo!(),
        Op::Log(x) => todo!(),
        Op::Sinh(x) => todo!(),
        Op::Cosh(x) => todo!(),
        Op::Tanh(x) => {
            // // d(tanh)dx: 1-tanh(x)^2
            // let tanh_x = opout;
            // let (ones_tensor, tanh_squared) =
            //     (pyten::ones(tanh_x.shape.clone()), (tanh_x * tanh_x).unwrap());
            // let dfdx = (&ones_tensor - &tanh_squared).unwrap();

            // vec![(&dfdx * &dFdx.clone()).unwrap()] // chain rule
            todo!()
        }
        Op::Sum(x, reduce_dim_input) => todo!(),
        Op::Max(x, reduce_dim_input) => todo!(),
    }
}
