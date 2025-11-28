from dataclasses import dataclass
from picograd.dtype import DType
from picograd.engine.op import Op, OpCode, PatternMatcher

# -kernelize: graph rewrites
# -schedule_with_vars: feeds graph to scheduler and memory planner
# -realize: hands schedule to run_schedule

class PatternMatcher:
  """
  ...
  """
  def __init__(): raise NotImplementedError

class Pattern:
  """
  ...
  """
  def __init__(): raise NotImplementedError

@dataclass(frozen=True)
class TensorCore: # D = A * B + C, A is (M x K), B is (K x N), C and D are (M x N)
  dims: tuple[int,int,int] # N, M, K
  threads: int # number of threads that construct the warp
  elements_per_thread: tuple[int, int, int] # elements per-thread to load/store from A/B/C
  dtype_in: DType # dtype for A and B
  dtype_out: DType # dtype for C and D
  opts: tuple[str, ...] # ordered tuple of "ux" or "lx" specifying kernel opts to perform. "ux" upcasts dim x and "lx" localizes dim x
  # (local_swizzle, upcast_swizzle, reduce_swizzle)
  # l<num> is the num axis of the locals, similar for u<num> and upcasts, r<num> and reduces
  swizzle: tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]]

class Renderer:
  """
  picograd follows tinygrad (and torch/xla and swift for tensorflow) with lazy graph capture, see (Suhan et al. https://arxiv.org/abs/2102.13267)
  and modifying the semantics of the programming model where users must explicitly materialize data with .realize(),
  as opposed to pt2 which maintains the eager programming model surface via graph capture at the host-language level (python bytecode interception)
  see (Ansel et al. https://docs.pytorch.org/assets/pytorch2-2.pdf)
  """
  device: str = ""
  suffix: str = ""
  # TODO: make this generic with a list of supported types
  supports_float4: bool = True
  has_local: bool = True
  has_threads: bool = False
  has_shared: bool = True
  # NOTE: these two should be in (x,y,z) order to match the max_sizes argument in get_grouped_dims
  global_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  local_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  shared_max: int = 32768
  tensor_cores: list[TensorCore] = []
  pre_matcher: PatternMatcher|None = None
  extra_matcher: PatternMatcher|None = None
  code_for_op: dict[OpCode, Callable] = {}

  def __reduce__(self): return self.__class__, ()
  def render(self, uops:list[Op]) -> str: raise NotImplementedError("needs a renderer")