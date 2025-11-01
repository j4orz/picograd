from __future__ import annotations
from . import _pgrs

class StridedTensor():
  """
  picograd follows the classical architecture of numerical array programming
  see: (Dodson, Lewis 1985) https://dl.acm.org/doi/pdf/10.1145/1057935.1057937

  - Level  2:  (BCKWD: compile, ad/gd)provides the high level mathematical primitives of
                                      compilation, automatic differentiation, gradient descent
  - Level  0/1:(FWD: cuBLAS/cuDNN)    provides performance primitives that require the knowledge of
                                      microarchitecture to obtain peak theoretical throughput (FLOP/S)
  - Level -1:  (DATA: ndarray/tensor) provides the foundational multi-dimensional array data structure
  """
  # ***** Tensor Compile, AD, GD (Level 2) *****
  def gradient(self, *targets:StridedTensor, gradient:StridedTensor|None=None, materialize_grads=False) -> list[StridedTensor]:
    """
    Computes the gradient of the targets with respect to self.

    ```python exec="true" source="above" session="tensor" result="python"
    x = Tensor.eye(3)
    y = Tensor([[2.0,0,-2.0]])
    z = y.matmul(x).sum()
    dx, dy = z.gradient(x, y)

    print(dx.tolist())  # dz/dx
    print(dy.tolist())  # dz/dy
    ```
    """
    assert gradient is not None or self.shape == tuple(), "when no gradient is provided, backward must be called on a scalar tensor"
    if not (self.is_floating_point() and all(t.is_floating_point() for t in targets)): raise RuntimeError("only float Tensors have gradient")
    if gradient is None: gradient = StridedTensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False)
    target_uops = [x.uop for x in targets]
    grads = compute_gradient(self.uop, gradient.uop, set(target_uops))
    ret = []
    for x in target_uops:
      if (y:=grads.get(x)) is None:
        if materialize_grads: y = x.const_like(0)
        else: raise RuntimeError(f"{x}\n\nnot found in\n\n{self.uop}")
      ret.append(y)
    # create returned Tensors
    return [StridedTensor(u, device=t.device) for t,u in zip(targets, ret)]

  def backward(self, gradient:StridedTensor|None=None) -> StridedTensor:
    """
    Propagates the gradient of a tensor backwards through the computation graph.
    If the 'gradient' argument is not provided, the tensor must be a scalar, and the gradient is implicitly set to 1.0.
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    t.sum().backward()
    print(t.grad.numpy())
    ```
    """
    all_uops = self.uop.toposort()
    tensors_need_grad: list[StridedTensor] = [t for tref in all_tensors if (t:=tref()) is not None and \
                                       t.uop in all_uops and t.requires_grad]
    # clear contexts
    for t,g in zip(tensors_need_grad, self.gradient(*tensors_need_grad, gradient=gradient, materialize_grads=True)):
      assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
      t.grad = g if t.grad is None else (t.grad + g)
    return self

  # ***** Tensor DNN, BLAS Operations (Level 1/0) *****
  # fa/mm

  # element ops
  # --math
  # --unary
  def exp2(self) -> StridedTensor: return self.cast(least_upper_float(self.dtype))._apply_uop(UOp.exp2)
  def sigmoid(self) -> StridedTensor: return (1 + (self * (-1/math.log(2))).exp2()).reciprocal()
  # --activation
  def tanh(self) -> StridedTensor: return 2.0 * ((2.0 * self).sigmoid()) - 1.0
  # --broacsted
  # --processing

  # reduce ops

  # movement ops
  # --high level
  def gather(self:StridedTensor, dim:int, index: StridedTensor) -> StridedTensor:
    """
    Gathers values along an axis specified by `dim`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.gather(1, Tensor([[0, 0], [1, 0]])).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  def cat(self:StridedTensor, *args:StridedTensor, dim:int=0) -> StridedTensor:
    """
    Concatenates self with other `Tensor` in `args` along an axis specified by `dim`.
    All tensors must have the same shape except in the concatenating dimension.

    ```python exec="true" source="above" session="tensor" result="python"
    t0, t1, t2 = Tensor([[1, 2]]), Tensor([[3, 4]]), Tensor([[5, 6]])
    print(t0.cat(t1, t2, dim=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t0.cat(t1, t2, dim=1).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  def stack(self:StridedTensor, *args:StridedTensor, dim:int=0) -> StridedTensor:
    """
    Concatenates self with other `Tensor` in `args` along a new dimension specified by `dim`.

    ```python exec="true" source="above" session="tensor" result="python"
    t0, t1, t2 = Tensor([1, 2]), Tensor([3, 4]), Tensor([5, 6])
    print(t0.stack(t1, t2, dim=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t0.stack(t1, t2, dim=1).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  def repeat(self, repeats, *args) -> StridedTensor:
    """
    Repeats tensor number of times along each dimension specified by `repeats`.
    `repeats` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.repeat(4, 2).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.repeat(4, 2, 1).shape)
    ```
    """
    raise NotImplementedError("TODO")

  def split(self, sizes:int|Sequence[int], dim:int=0) -> tuple[StridedTensor, ...]:
    """
    Splits the tensor into chunks along the dimension specified by `dim`.
    If `sizes` is an integer, it splits into equally sized chunks if possible, otherwise the last chunk will be smaller.
    If `sizes` is a list, it splits into `len(sizes)` chunks with size in `dim` according to `size`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(10).reshape(5, 2)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    split = t.split(2)
    print("\\n".join([repr(x.numpy()) for x in split]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    split = t.split([1, 4])
    print("\\n".join([repr(x.numpy()) for x in split]))
    ```
    """
    raise NotImplementedError("TODO")

  def chunk(self, chunks:int, dim:int=0) -> list[StridedTensor]:
    """
    Splits the tensor into `chunks` number of chunks along the dimension `dim`.
    If the tensor size along `dim` is not divisible by `chunks`, all returned chunks will be the same size except the last one.
    The function may return fewer than the specified number of chunks.

    ```python exec="true" source="above" session="tensor" result="python"
    chunked = Tensor.arange(11).chunk(6)
    print("\\n".join([repr(x.numpy()) for x in chunked]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    chunked = Tensor.arange(12).chunk(6)
    print("\\n".join([repr(x.numpy()) for x in chunked]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    chunked = Tensor.arange(13).chunk(6)
    print("\\n".join([repr(x.numpy()) for x in chunked]))
    ```
    """
    raise NotImplementedError("TODO")

  def unfold(self, dim:int, size:sint, step:int) -> StridedTensor:
    """
    Unfolds the tensor along dimension `dim` into overlapping windows.

    Each window has length `size` and begins every `step` elements of `self`.
    Returns the input tensor with dimension `dim` replaced by dims `(n_windows, size)`
    where `n_windows = (self.shape[dim] - size) // step + 1`.

    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(8).unfold(0,2,2)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(27).reshape(3,3,3).unfold(-1,2,3)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```
    """
    raise NotImplementedError("TODO")

  def squeeze(self, dim:int|None=None) -> StridedTensor:
    """
    Returns a tensor with specified dimensions of input of size 1 removed.
    If `dim` is not specified, all dimensions with size 1 are removed.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.zeros(2, 1, 2, 1, 2)
    print(t.squeeze().shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.squeeze(0).shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.squeeze(1).shape)
    ```
    """
    raise NotImplementedError("TODO")

  def unsqueeze(self, dim:int) -> StridedTensor:
    """
    Returns a tensor with a new dimension of size 1 inserted at the specified `dim`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(t.unsqueeze(0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.unsqueeze(1).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  # --low level
  def view(self, shape:tuple[sint, ...], *args) -> StridedTensor:
    """`.view` is an alias for `.reshape`."""
    raise NotImplementedError("TODO")

  def reshape(self, shape, *args) -> StridedTensor:
    """
    Returns a tensor with the same data as the original tensor but with a different shape.
    `shape` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6)
    print(t.reshape(2, 3).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  # .expand() .permute() .flip() .shrink() .pad()

  # ***** Tensor Constructors (Level -1) *****
  # --high level: .randn_like, randint, normal, uniform, scaled_uniform, kaiming_normal, randperm, multinomial

  # --low level:
  @staticmethod
  def empty(*shape, device:str|tuple[str, ...]|None=None, dtype:DTypeLike|None=None, **kwargs) -> StridedTensor:
    """
    Creates an empty tensor with the given shape.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3)
    print(t.shape)
    ```
    """
    raise NotImplementedError("TODO")
  def empty_like(self, **kwargs) -> StridedTensor:
    """
    Creates an empty tensor with the same shape as `self`.
    If `dtype` is not specified, the dtype of `self` is used.
    """
    return StridedTensor.empty(self.shape, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)

  @staticmethod
  def rand(*shape, device:str|None=None, dtype:DTypeLike|None=None, contiguous:bool=True, **kwargs) -> StridedTensor:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[0, 1)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.rand(2, 3)
    print(t.numpy())
    ```
    """
    raise NotImplementedError("TODO")

  @staticmethod
  def full(shape:tuple[sint, ...], fill_value:ConstType, **kwargs) -> StridedTensor:
    """
    Creates a tensor with the given shape, filled with the given value.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), 42).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), False).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  @staticmethod
  def zeros(*shape, **kwargs) -> StridedTensor:
    """
    Creates a tensor with the given shape, filled with zeros.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.zeros(2, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.zeros(2, 3, dtype=dtypes.int32).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  @staticmethod
  def ones(*shape, **kwargs) -> StridedTensor:
    """
    Creates a tensor with the given shape, filled with ones.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(2, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(2, 3, dtype=dtypes.int32).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  @staticmethod
  def arange(start, stop=None, step=1, **kwargs) -> StridedTensor:
    """
    Returns a 1-D tensor of size `ceil((stop - start) / step)` with values from `[start, stop)`, with spacing between values given by `step`.

    If `stop` is not specified, values are generated from `[0, start)` with the given `step`.

    If `stop` is specified, values are generated from `[start, stop)` with the given `step`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5, 10).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5, 10, 2).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5.5, 10, 2).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  @staticmethod
  def linspace(start:int|float, stop:int|float, steps:int, **kwargs) -> StridedTensor:
    """
    Returns a 1-D tensor of `steps` evenly spaced values from `start` to `stop`, inclusive.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.linspace(0, 10, 5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.linspace(-1, 1, 5).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  @staticmethod
  def eye(n:int, m:int|None=None, **kwargs) -> StridedTensor:
    """
    Returns a 2-D tensor with `n` rows and `m` columns, with ones on the diagonal and zeros elsewhere.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.eye(3).numpy())
    ```

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.eye(2, 4).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  # ***** Tensor Data (Level -1) *****  
  @property
  def ndim(self) -> int:
    """
    Returns the number of dimensions in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    print(t.ndim)
    ```
    """
    raise NotImplementedError("TODO")

  def numel(self) -> sint:
    """
    Returns the total number of elements in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(t.numel())
    ```
    """
    raise NotImplementedError("TODO")

  def element_size(self) -> int:
    """
    Returns the size in bytes of an individual element in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([5], dtype=dtypes.int16)
    print(t.element_size())
    ```
    """
    raise NotImplementedError("TODO")

  def nbytes(self) -> int:
    """
    Returns the total number of bytes of all elements in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([8, 9], dtype=dtypes.float)
    print(t.nbytes())
    ```
    """
    raise NotImplementedError("TODO")

  def is_floating_point(self) -> bool:
    """
    Returns `True` if the tensor contains floating point types, i.e. is one of `dtypes.float64`, `dtypes.float32`,
    `dtypes.float16`, `dtypes.bfloat16`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([8, 9], dtype=dtypes.float32)
    print(t.is_floating_point())
    ```
    """
    raise NotImplementedError("TODO")

  def size(self, dim:int|None=None) -> sint|tuple[sint, ...]:
    """
    Returns the size of the tensor. If `dim` is specified, return the length along dimension `dim`. Otherwise return the shape of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[4, 5, 6], [7, 8, 9]])
    print(t.size())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.size(dim=1))
    ```
    """
    raise NotImplementedError("TODO")

  def __init__(self):
    self.shape, self.stride = [], []

class ScheduledTensor():
  # Pipeline: Tensor -> LazyData -> ScheduleItem -> ExecItem -> Codegen -> Source -> Binary -> Runtime  
  # Middleend (Optimization)
  # -kernelize: graph rewrites
  # -schedule_with_vars: feeds graph to scheduler and memory planner
  # -realize: hands schedule to run_schedule
  def __init__(self):
    self.todo = True

__all__ = ["StridedTensor", "ScheduledTensor"]
