from __future__ import annotations
from . import _pgrs

class Tensor():
  def __init__(self, x: int, y: int) -> None:
    self.x = x
    self.y = y

  def foo(self) -> str:
    return _pgrs.sum_as_string(self.x, self.y)
  
  # ***** Tensor Properties *****
  
  @property
  def ndim(self) -> int:
    """
    Returns the number of dimensions in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    print(t.ndim)
    ```
    """
    return 0

  def numel(self) -> sint:
    """
    Returns the total number of elements in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(t.numel())
    ```
    """
    return 0

  def element_size(self) -> int:
    """
    Returns the size in bytes of an individual element in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([5], dtype=dtypes.int16)
    print(t.element_size())
    ```
    """
    return 0

  def nbytes(self) -> int:
    """
    Returns the total number of bytes of all elements in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([8, 9], dtype=dtypes.float)
    print(t.nbytes())
    ```
    """
    return 0

  def is_floating_point(self) -> bool:
    """
    Returns `True` if the tensor contains floating point types, i.e. is one of `dtypes.float64`, `dtypes.float32`,
    `dtypes.float16`, `dtypes.bfloat16`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([8, 9], dtype=dtypes.float32)
    print(t.is_floating_point())
    ```
    """
    return 0

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
    return 0
  
  def matmul(self, other: Tensor) -> Tensor:
    """
    Performs matrix multiplication between two tensors.

    You can pass in the `reverse` keyword argument to control the order of the matrix multiplication.
    You can pass in the optional `dtype` keyword argument to control the data type of the accumulation.

    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    print(a.matmul(b).numpy())
    ```
    """
    return other.dot(self, dtype=dtype) if reverse else self.dot(other, dtype=dtype)

  # math ops
  # unary ops
  # reduce ops
  # broadcasted ops
  # activation ops
  # processing ops
  # nn ops

  # movement high level
  def gather(self:Tensor, dim:int, index: Tensor) -> Tensor:
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
    return 0

  def cat(self:Tensor, *args:Tensor, dim:int=0) -> Tensor:
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
    return 0

  def stack(self:Tensor, *args:Tensor, dim:int=0) -> Tensor:
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
    return 0

  def repeat(self, repeats, *args) -> Tensor:
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
    return 0

  def split(self, sizes:int|Sequence[int], dim:int=0) -> tuple[Tensor, ...]:
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
    return 0

  def chunk(self, chunks:int, dim:int=0) -> list[Tensor]:
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
    return 0

  def unfold(self, dim:int, size:sint, step:int) -> Tensor:
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
    return 0

  def squeeze(self, dim:int|None=None) -> Tensor:
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
    return 0

  def unsqueeze(self, dim:int) -> Tensor:
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
    return 0

  # movement low level
  # ***** movement low level ops *****

  def view(self, shape:tuple[sint, ...], *args) -> Tensor:
    """`.view` is an alias for `.reshape`."""
    return 0

  def reshape(self, shape, *args) -> Tensor:
    """
    Returns a tensor with the same data as the original tensor but with a different shape.
    `shape` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6)
    print(t.reshape(2, 3).numpy())
    ```
    """
    return 0

  def expand(self, shape, *args) -> Tensor:
    """
    Returns a tensor that is expanded to the shape that is specified.
    Expand can also increase the number of dimensions that a tensor has.

    Passing a `-1` or `None` to a dimension means that its size will not be changed.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.expand(4, -1).numpy())
    ```
    """
    return 0

  def permute(self, order, *args) -> Tensor:
    """
    Returns a tensor that is a permutation of the original tensor.
    The new tensor has the same data as the original tensor but with the dimensions permuted according to the order specified.
    `order` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3, 5)
    print(t.shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.permute(2, 0, 1).shape)
    ```
    """
    return 0

  def flip(self, axis, *args) -> Tensor:
    """
    Returns a tensor that reverses the order of the original tensor along given `axis`.
    `axis` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.flip(0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.flip((0, 1)).numpy())
    ```
    """
    return 0

  def shrink(self, arg:tuple[tuple[sint, sint]|None, ...]) -> Tensor:
    """
    Returns a tensor that shrinks the each axis based on input arg.
    `arg` must have the same length as `self.ndim`.
    For each axis, it can be `None`, which means no shrink, or a tuple `(start, end)` that works the same as Python slice.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(3, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.shrink(((None, (1, 3)))).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.shrink((((0, 2), (0, 2)))).numpy())
    ```
    """
    return 0

  def pad(self, padding:Sequence[sint]|Sequence[tuple[sint, sint]|None], mode:str="constant", value:float=0.0) -> Tensor:
    """
    Returns a tensor with padding applied based on the input `padding`.

    `padding` supports two padding structures:

    1. Flat padding: `(padding_left, padding_right, padding_top, padding_bottom, ...)`
        - This structure matches PyTorch's pad.
        - `padding` length must be even.

    2. Group padding: `(..., (padding_top, padding_bottom), (padding_left, padding_right))`
        - This structure matches pad for JAX, NumPy, TensorFlow, and others.
        - For each axis, padding can be `None`, meaning no padding, or a tuple `(start, end)`.
        - `padding` must have the same length as `self.ndim`.

    Padding values can be negative, resulting in dimension shrinks that work similarly to Python negative slices.
    Padding modes is selected with `mode` which supports `constant`, `reflect` and `replicate`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.pad((1, 2, 0, -1)).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.pad(((None, None, (0, -1), (1, 2)))).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.pad((1, 2, 0, -1), value=-float('inf')).numpy())
    ```
    """
    return 0

  # rng high level
  def randn_like(self, dtype:DTypeLike|None=None, requires_grad:bool|None=None, **kwargs) -> Tensor:
    """
    Creates a tensor with the same shape and sharding as `self`, filled with random values from a normal distribution with mean 0 and variance 1.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.randn_like(t).numpy())
    ```
    """
    return 0

  @staticmethod
  def randn(*shape, dtype:DTypeLike|None=None, requires_grad:bool|None=None, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a normal distribution with mean `0` and standard deviation `1`.
    If `dtype` is not specified, the default type is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.randn(2, 3).numpy())
    ```
    """
    return 0

  @staticmethod
  def randint(*shape, low=0, high=10, dtype=dtypes.int32, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random integer values generated uniformly from the interval `[low, high)`.
    If `dtype` is not specified, the default type is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.randint(2, 3, low=5, high=10).numpy())
    ```
    """
    return 0

  @staticmethod
  def normal(*shape, mean=0.0, std=1.0, requires_grad:bool|None=None, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a normal distribution with the given `mean` and standard deviation `std`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.normal(2, 3, mean=10, std=2).numpy())
    ```
    """
    return 0

  @staticmethod
  def uniform(*shape, low=0.0, high=1.0, dtype:DTypeLike|None=None, requires_grad:bool|None=None, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[low, high)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.uniform(2, 3, low=2, high=10).numpy())
    ```
    """
    return 0

  @staticmethod
  def scaled_uniform(*shape, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution
    over the interval `[-prod(shape)**-0.5, prod(shape)**-0.5)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.scaled_uniform(2, 3).numpy())
    ```
    """
    return 0

  # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
  @staticmethod
  def glorot_uniform(*shape, **kwargs) -> Tensor:
    """
    <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.glorot_uniform(2, 3).numpy())
    ```
    """
    return 0

  # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
  @staticmethod
  def kaiming_uniform(*shape, a:float = 0.01, **kwargs) -> Tensor:
    """
    <https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.kaiming_uniform(2, 3).numpy())
    ```
    """
    return 0

  # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_
  @staticmethod
  def kaiming_normal(*shape, a:float = 0.01, **kwargs) -> Tensor:
    """
    <https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.kaiming_normal(2, 3).numpy())
    ```
    """
    return 0

  @staticmethod
  def randperm(n:int, device=None, dtype=dtypes.int32, **kwargs) -> Tensor:
    """
    Returns a tensor with a random permutation of integers from `0` to `n-1`.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.randperm(6).numpy())
    ```
    """
    return 0

  def multinomial(self:Tensor, num_samples:int = 1, replacement:bool = False) -> Tensor:
    return 0

  # constructors
  @staticmethod
  def empty(*shape, device:str|tuple[str, ...]|None=None, dtype:DTypeLike|None=None, **kwargs) -> Tensor:
    """
    Creates an empty tensor with the given shape.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3)
    print(t.shape)
    ```
    """
    return 0
  def empty_like(self, **kwargs) -> Tensor:
    """
    Creates an empty tensor with the same shape as `self`.
    If `dtype` is not specified, the dtype of `self` is used.
    """
    return Tensor.empty(self.shape, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)

  @staticmethod
  def from_blob(ptr:int, shape:tuple[int, ...], **kwargs) -> Tensor:
    """
    Exposes the pointer as a Tensor without taking ownership of the original data.
    The pointer must remain valid for the entire lifetime of the created Tensor.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.
    """
    return 0

  @staticmethod
  def from_url(url:str, gunzip:bool=False, **kwargs) -> Tensor:
    """
    Creates a Tensor from a URL.

    This is the preferred way to access Internet resources.
    It currently returns a DISK Tensor, but in the future it may return an HTTP Tensor.
    This also will soon become lazy (when possible) and not print progress without DEBUG.

    The `gunzip` flag will gzip extract the resource and return an extracted Tensor.
    """
    return 0

  _seed: int = int(time.time())
  _device_seeds: dict[str, Tensor] = {}
  _device_rng_counters: dict[str, Tensor] = {}
  @staticmethod
  def manual_seed(seed=0) -> None:
    """
    Sets the seed for random operations.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.rand(5).numpy())
    print(Tensor.rand(5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)  # reset to the same seed
    print(Tensor.rand(5).numpy())
    print(Tensor.rand(5).numpy())
    ```
    """
    return 0

  @staticmethod
  def _threefry_random_bits(key:Tensor, counts0:Tensor, counts1:Tensor) -> Tensor:
    return 0

  @staticmethod
  def rand(*shape, device:str|None=None, dtype:DTypeLike|None=None, contiguous:bool=True, **kwargs) -> Tensor:
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
    return 0

  @staticmethod
  def full(shape:tuple[sint, ...], fill_value:ConstType, **kwargs) -> Tensor:
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
    return 0

  @staticmethod
  def zeros(*shape, **kwargs) -> Tensor:
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
    return 0

  @staticmethod
  def ones(*shape, **kwargs) -> Tensor:
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
    return 0

  @staticmethod
  def arange(start, stop=None, step=1, **kwargs) -> Tensor:
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
    return 0

  @staticmethod
  def linspace(start:int|float, stop:int|float, steps:int, **kwargs) -> Tensor:
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
    return 0

  @staticmethod
  def eye(n:int, m:int|None=None, **kwargs) -> Tensor:
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
    return 0

  def full_like(self, fill_value:ConstType, **kwargs) -> Tensor:
    """
    Creates a tensor with the same shape as `self`, filled with the given value.
    If `dtype` is not specified, the dtype of `self` is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.full_like(t, 42).numpy())
    ```
    """
    return 0

  def zeros_like(self, **kwargs) -> Tensor:
    """
    Creates a tensor with the same shape as `self`, filled with zeros.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.zeros_like(t).numpy())
    ```
    """
    return 0

  def ones_like(self, **kwargs) -> Tensor:
    """
    Creates a tensor with the same shape as `self`, filled with ones.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.zeros(2, 3)
    print(Tensor.ones_like(t).numpy())
    ```
    """
    return 0

  def rand_like(self, **kwargs) -> Tensor:
    """
    Creates a tensor with the same shape and sharding as `self`, filled with random values from a uniform distribution over the interval `[0, 1)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.rand_like(t).numpy())
    ```
    """
    return 0

__all__ = ["Tensor"]
