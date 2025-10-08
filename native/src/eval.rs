//! OP
//! ...

use crate::ten::Tensor;

pub enum TenOp {
    Exp2, Log2, Sin, Sqrt, Recip, Neg, Trunc, // uop
    Add, Mul, Shl, Shr, IDiv, Max, Mod, CmpLt, CmpNe, CmpEq, Xor, Or, And, ThreeFry, Sub, FDiv, Pow, // binops    
    Where, MulAcc, // ternops
    ReduceAxis, Reduce, AllReduce, // reduce
}

type CpuBuf = Vec<f32>;
type GpuBuf = Vec<f32>;

pub enum Device {
    Cpu(CpuBuf),
    Gpu(GpuBuf),
}

fn eval(op: TenOp, x: Tensor, y: Tensor) -> Tensor {
    let device = x.device;
    match (device, op) {
        (Device::Cpu(buf), TenOp::Exp2) => todo!(),
        (Device::Cpu(buf), TenOp::Log2) => todo!(),
        (Device::Cpu(buf), TenOp::Sin) => todo!(),
        (Device::Cpu(buf), TenOp::Sqrt) => todo!(),
        (Device::Cpu(buf), TenOp::Recip) => todo!(),
        (Device::Cpu(buf), TenOp::Neg) => todo!(),
        (Device::Cpu(buf), TenOp::Trunc) => todo!(),
        (Device::Cpu(buf), TenOp::Add) => todo!(),
        (Device::Cpu(buf), TenOp::Mul) => todo!(),
        (Device::Cpu(buf), TenOp::Shl) => todo!(),
        (Device::Cpu(buf), TenOp::Shr) => todo!(),
        (Device::Cpu(buf), TenOp::IDiv) => todo!(),
        (Device::Cpu(buf), TenOp::Max) => todo!(),
        (Device::Cpu(buf), TenOp::Mod) => todo!(),
        (Device::Cpu(buf), TenOp::CmpLt) => todo!(),
        (Device::Cpu(buf), TenOp::CmpNe) => todo!(),
        (Device::Cpu(buf), TenOp::CmpEq) => todo!(),
        (Device::Cpu(buf), TenOp::Xor) => todo!(),
        (Device::Cpu(buf), TenOp::Or) => todo!(),
        (Device::Cpu(buf), TenOp::And) => todo!(),
        (Device::Cpu(buf), TenOp::ThreeFry) => todo!(),
        (Device::Cpu(buf), TenOp::Sub) => todo!(),
        (Device::Cpu(buf), TenOp::FDiv) => todo!(),
        (Device::Cpu(buf), TenOp::Pow) => todo!(),
        (Device::Cpu(buf), TenOp::Where) => todo!(),
        (Device::Cpu(buf), TenOp::MulAcc) => todo!(),
        (Device::Cpu(buf), TenOp::ReduceAxis) => todo!(),
        (Device::Cpu(buf), TenOp::Reduce) => todo!(),
        (Device::Cpu(buf), TenOp::AllReduce) => todo!(),
        (Device::Gpu(buf), TenOp::Exp2) => todo!(),
        (Device::Gpu(buf), TenOp::Log2) => todo!(),
        (Device::Gpu(buf), TenOp::Sin) => todo!(),
        (Device::Gpu(buf), TenOp::Sqrt) => todo!(),
        (Device::Gpu(buf), TenOp::Recip) => todo!(),
        (Device::Gpu(buf), TenOp::Neg) => todo!(),
        (Device::Gpu(buf), TenOp::Trunc) => todo!(),
        (Device::Gpu(buf), TenOp::Add) => todo!(),
        (Device::Gpu(buf), TenOp::Mul) => todo!(),
        (Device::Gpu(buf), TenOp::Shl) => todo!(),
        (Device::Gpu(buf), TenOp::Shr) => todo!(),
        (Device::Gpu(buf), TenOp::IDiv) => todo!(),
        (Device::Gpu(buf), TenOp::Max) => todo!(),
        (Device::Gpu(buf), TenOp::Mod) => todo!(),
        (Device::Gpu(buf), TenOp::CmpLt) => todo!(),
        (Device::Gpu(buf), TenOp::CmpNe) => todo!(),
        (Device::Gpu(buf), TenOp::CmpEq) => todo!(),
        (Device::Gpu(buf), TenOp::Xor) => todo!(),
        (Device::Gpu(buf), TenOp::Or) => todo!(),
        (Device::Gpu(buf), TenOp::And) => todo!(),
        (Device::Gpu(buf), TenOp::ThreeFry) => todo!(),
        (Device::Gpu(buf), TenOp::Sub) => todo!(),
        (Device::Gpu(buf), TenOp::FDiv) => todo!(),
        (Device::Gpu(buf), TenOp::Pow) => todo!(),
        (Device::Gpu(buf), TenOp::Where) => todo!(),
        (Device::Gpu(buf), TenOp::MulAcc) => todo!(),
        (Device::Gpu(buf), TenOp::ReduceAxis) => todo!(),
        (Device::Gpu(buf), TenOp::Reduce) => todo!(),
        (Device::Gpu(buf), TenOp::AllReduce) => todo!(),
    }
}