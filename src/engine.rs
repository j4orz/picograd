// use std::rc::Rc;

// use crate::{rsten::Tensor, runtime::Device};

// pub struct Session {
//     pub device: Device,
//     pub engine: Engine,
// }

// pub enum Engine { Eager(View), Lazy(usize) }

// pub struct View {
//     shape: Vec<usize>,
//     stride: Vec<usize>,
//     storage: Rc<Storage>,
// }
// pub struct Storage { pub data: Vec<f32>, pub grad: Option<Tensor> }

// // The variant order preserves the intended toposort priority.
// pub enum UOpKind {
//     NoOp, Sink, Unique, Device, Kernel, Precast, RewriteError, // uops that aren't rendered
//     Child, Children, // track children
//     Copy, Bufferize, Buffer, BufferView, MSelect, MStack, // buffer ops 
//     Contiguous, ContiguousBackward, Detach, Fuse, Realize, // ops that adjust the behavior of the scheduler
//     Block, BlockStart, BlockEnd, BlockFinal, // blocks in linearizer (only used there) 
//     Reshape, Permute, Expand, Pad, Shrink, Flip, // movement ops! these only exist in the tensor graph 
//     Multi, // MULTI is really a movement op
//     View, // view is what all movement ops become
//     Valid, // TODO: remove VALID with the VIEW(CONST(DEVICE)) refactor
//     DefineGlobal, DefineLocal, DefineReg, // TODO: unify these ops into the levels of the memory hierarchy. depends on ASSIGN is STORE
//     DefineVar, Bind, // this is for symbolic shapes
//     Special, // this is a RANGE for GPU dimensions, similar to symbolic shapes but not exactly
//     ReduceAxis, Reduce, AllReduce, // reduce
//     Unroll, Contract, Gep, Vectorize, Cat, PtrCat, // optimization helper ops
//     Cast, BitCast, Exp2, Log2, Sin, Sqrt, Recip, Neg, Trunc, // UnaryOps 
//     Load, Store, // load/store before math
//     Assign, // TODO: ASSIGN is STORE, remove ASSIGN
//     Wmma, // tensor core math op, not elementwise
//     Index, // INDEX is a BinaryOp similar to ADD, but it operates on pointers
//     Add, Mul, Shl, Shr, IDiv, Max, Mod, CmpLt, CmpNe, CmpEq, Xor, Or, And, ThreeFry, Sub, FDiv, Pow, // binops
//     Where, MulAcc, // ternops
//     Barrier, Range, If, EndRange, EndIf, // controlflowops
//     VConst, Const, // consts. VCONST is a vectorized const
//     Custom, CustomI, // CUSTOM/CUSTOMI are used to output strings into codegen. the I makes the string inline
// }

// pub struct UOp {
//     kind:  UOpKind
// }


// // use crate::uop::UOp;

// // enum Engine { Eager, Lazy }

// // pub fn forward(op: UOp) -> () {
// //     let e = Engine::Eager;
// //     match e {
// //     Engine::Eager => eagerly_eval(op),
// //     Engine::Lazy => lazily_compile(op),
// //     }
// // }

// // pub fn eagerly_eval(op: UOp) -> () {
// //     match op {
// //     UOp::View => todo!(),
// //     UOp::Add => todo!(),
// //     UOp::Reduce => todo!(),
// //     UOp::Where => todo!(),
// //     UOp::Contract => todo!(),
// //     _ => unimplemented!() // compiler-specific ops
// //     }
// // }

// // pub fn lazily_compile(op: UOp) -> () {
// //     match op {
// //     UOp::NoOp | UOp::Sink | UOp::Unique | UOp::Device | UOp::Kernel | UOp::Precast | UOp::RewriteError => todo!(), // uops that aren't rendered
// //     UOp::Child | UOp::Children => todo!(), // track children
// //     UOp::Copy | UOp::Bufferize | UOp::Buffer | UOp::BufferView | UOp::MSelect | UOp::MStack => todo!(), // buffer ops
// //     UOp::Contiguous | UOp::ContiguousBackward | UOp::Detach | UOp::Fuse | UOp::Realize => todo!(), // ops that adjust the behavior of the scheduler
// //     UOp::Block | UOp::BlockStart | UOp::BlockEnd | UOp::BlockFinal => todo!(), // blocks in linearizer (only used there)
// //     UOp::Reshape | UOp::Permute | UOp::Expand | UOp::Pad | UOp::Shrink | UOp::Flip => todo!(), // movement ops! these only exist in the tensor graph
// //     UOp::Multi => todo!(), // MULTI is really a movement op
// //     UOp::View => todo!(), // view is what all movement ops become
// //     UOp::Valid => todo!(), // TODO: remove VALID with the VIEW(CONST(DEVICE)) refactor
// //     UOp::DefineGlobal | UOp::DefineLocal | UOp::DefineReg => todo!(), // TODO: unify these ops into the levels of the memory hierarchy. depends on ASSIGN is STORE
// //     UOp::DefineVar | UOp::Bind => todo!(), // this is for symbolic shapes
// //     UOp::Special => todo!(), // this is a RANGE for GPU dimensions, similar to symbolic shapes but not exactly
// //     UOp::ReduceAxis | UOp::Reduce | UOp::AllReduce => todo!(), // reduce
// //     UOp::Unroll | UOp::Contract | UOp::Gep | UOp::Vectorize | UOp::Cat | UOp::PtrCat => todo!(), // optimization helper ops
// //     UOp::Cast | UOp::BitCast | UOp::Exp2 | UOp::Log2 | UOp::Sin | UOp::Sqrt | UOp::Recip | UOp::Neg | UOp::Trunc => todo!(), // UnaryOps
// //     UOp::Load | UOp::Store => todo!(), // load/store before math
// //     UOp::Assign => todo!(), // TODO: ASSIGN is STORE, remove ASSIGN
// //     UOp::Wmma => todo!(), // tensor core math op, not elementwise
// //     UOp::Index => todo!(), // INDEX is a BinaryOp similar to ADD, but it operates on pointers
// //     UOp::Add | UOp::Mul | UOp::Shl | UOp::Shr | UOp::IDiv | UOp::Max | UOp::Mod | UOp::CmpLt | UOp::CmpNe | UOp::CmpEq | UOp::Xor | UOp::Or | UOp::And | UOp::ThreeFry | UOp::Sub | UOp::FDiv | UOp::Pow => todo!(), // binops
// //     UOp::Where | UOp::MulAcc => todo!(), // ternops
// //     UOp::Barrier | UOp::Range | UOp::If | UOp::EndRange | UOp::EndIf => todo!(), // controlflowops
// //     UOp::VConst | UOp::Const => todo!(), // consts. VCONST is a vectorized const
// //     UOp::Custom | UOp::CustomI => todo!(), // CUSTOM/CUSTOMI are used to output strings into codegen. the I makes the string inline
// //     }
// // }