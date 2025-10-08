use crate::ten::Tensor;
use crate::run::Device;

#[derive(Clone)]
pub enum MicroOp {
    Exp2, Log2, Sin, Sqrt, Recip, Neg, Trunc, // uop
    Add, Mul, Shl, Shr, IDiv, Max, Mod, CmpLt, CmpNe, CmpEq, Xor, Or, And, ThreeFry, Sub, FDiv, Pow, // binops
    Where, MulAcc, // ternops
    ReduceAxis, Reduce, AllReduce, // reduce
}

pub fn eval(op: &MicroOp, x: &Tensor, y: &Tensor) -> Tensor {
    // record backward stuff. tape and transform

    // return the tensor
    let foo = match &x.device {
        Device::Cpu(cpulang) => {
        },
        Device::Gpu(gpulang) => todo!(),
    };

    todo!()
}

pub fn backward(x: Tensor) -> Tensor {
    todo!()
}

pub fn grad(f: Box<dyn Fn() -> ()>) -> Box<dyn Fn() -> ()> {
    todo!()
}