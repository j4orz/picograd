// The variant order preserves the intended toposort priority.
pub enum UOp {
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