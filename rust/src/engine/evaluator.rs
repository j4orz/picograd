use crate::tensor::Tensor;
use crate::runtime::Device;

// autograd
// 1. add gradient tracking to tensor
// 2. update eval to record on foward pass
// 3. implement .backward() on tensor

#[derive(Clone)]
pub enum MicroOp {
    Exp2, Log2, Sin, Sqrt, Recip, Neg, Trunc, // uop
    Add, Mul, Shl, Shr, IDiv, Max, Mod, CmpLt, CmpNe, CmpEq, Xor, Or, And, ThreeFry, Sub, FDiv, Pow, // binops
    Where, MulAcc, // ternops
    ReduceAxis, Reduce, AllReduce, // reduce
}

pub fn eval(op: &MicroOp, inputs: &[&Tensor]) -> Tensor {
    // return the tensor
    // let foo = match &x.device {
    // Device::Cpu(cpulang) => {
    // },
    // Device::Gpu(gpulang) => todo!(),
    // };

    // record backward stuff. tape and transform
    todo!()
}

use std::{cell::RefCell, rc::Rc};

struct TapeNode {
    pub op: MicroOp, pub inputs: Vec<Rc<RefCell<TapeNode>>>, // pointer chase
}

pub fn backward(x: Tensor) -> Tensor {
    todo!()
}

pub fn grad(f: Box<dyn Fn() -> ()>) -> Box<dyn Fn() -> ()> {
    todo!()
}