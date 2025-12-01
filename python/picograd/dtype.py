from __future__ import annotations
import ctypes
import math
import struct
from typing import Callable, Final, Literal
from dataclasses import dataclass
from enum import Enum, auto

# from picograd.engine.opnode import OpNode

sint = int
# Variable = OpNode
ConstType = float|int|bool
FmtStr = Literal['?', 'b', 'B', 'h', 'H', 'i', 'I', 'q', 'Q', 'e', 'f', 'd']

class InvalidTypeMetaClass(type):
  instance:None|InvalidType = None
  def __call__(cls):
    if (ret:=InvalidTypeMetaClass.instance) is not None: return ret
    InvalidTypeMetaClass.instance = ret = super().__call__()
    return ret

class InvalidType(metaclass=InvalidTypeMetaClass):
  def __eq__(self, other): return self is other
  def __lt__(self, other): return self is not other
  def __gt__(self, other): return self is not other
  def __hash__(self): return id(self)
  def __repr__(self): return "Invalid"
  def __reduce__(self): return (InvalidType, ())  # Return the global Invalid instance


# ************ DTypes ************
# all DTypes should only be created once
class DTypeMetaClass(type):
  dcache: dict[tuple, DType] = {}
  def __call__(cls, *args, **kwargs):
    if (ret:=DTypeMetaClass.dcache.get(args, None)) is not None: return ret
    DTypeMetaClass.dcache[args] = ret = super().__call__(*args)
    return ret

@dataclass(frozen=True, eq=False)
class DType(metaclass=DTypeMetaClass):
  priority: int  # this determines when things get upcasted
  itemsize: int
  name: str
  fmt: FmtStr|None
  count: int
  _scalar: DType|None
  @staticmethod
  def new(priority:int, itemsize:int, name:str, fmt:FmtStr|None): return DType(priority, itemsize, name, fmt, 1, None)
ConstLike = ConstType|InvalidType|tuple[ConstType|InvalidType, ...] # Variable
DTypeLike = str|DType

class dtypes:
  void: Final[DType] = DType.new(-1, 0, "void", None)
  index: Final[DType] = DType.new(-1,100, "index", None)
  bool: Final[DType] = DType.new(0, 1, "bool", '?')
  int8: Final[DType] = DType.new(1, 1, "signed char", 'b')
  uint8: Final[DType] = DType.new(2, 1, "unsigned char", 'B')
  int16: Final[DType] = DType.new(3, 2, "short", 'h')
  uint16: Final[DType] = DType.new(4, 2, "unsigned short", 'H')
  int32: Final[DType] = DType.new(5, 4, "int", 'i')
  uint32: Final[DType] = DType.new(6, 4, "unsigned int", 'I')
  int64: Final[DType] = DType.new(7, 8, "long", 'q')
  uint64: Final[DType] = DType.new(8, 8, "unsigned long", 'Q')
  fp8e4m3: Final[DType] = DType.new(9, 1, "float8_e4m3", None)
  fp8e5m2: Final[DType] = DType.new(10, 1, "float8_e5m2", None)
  float16: Final[DType] = DType.new(11, 2, "half", 'e')
  # bfloat16 has higher priority than float16, so least_upper_dtype(dtypes.int64, dtypes.uint64) = dtypes.float16
  bfloat16: Final[DType] = DType.new(12, 2, "__bf16", None)
  float32: Final[DType] = DType.new(13, 4, "float", 'f')
  float64: Final[DType] = DType.new(14, 8, "double", 'd')
  fp8s = (fp8e4m3, fp8e5m2)
  floats = fp8s + (float16, bfloat16, float32, float64)
  uints = (uint8, uint16, uint32, uint64)
  sints = (int8, int16, int32, int64)
  ints = uints + sints
  all = floats + ints + (bool, index) # noqa: A003

def float_to_fp16(x):
  try: return struct.unpack('e', struct.pack('e', float(x)))[0]
  except OverflowError: return math.copysign(math.inf, x)

def float_to_bf16(x):
  if not math.isfinite(x): return x
  u = struct.unpack('I', struct.pack('f', x))[0]
  u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
  return struct.unpack('f', struct.pack('I', u))[0]


# fp8-float conversions based on https://gitlab.com/nvidia/headers/cuda-individual/cudart/-/blob/main/cuda_fp8.hpp
def float_to_fp8(x: float, dtype: DType) -> int:
  assert dtype in dtypes.fp8s, "Only for fp8s"
  # e4m3 don't support inf, return 0x7f(+NaN) and 0xff(-NaN) to match jax
  # NaN is unordered, can't compare with zero, use math.copysign to get sign
  if dtype == dtypes.fp8e4m3 and not math.isfinite(x): return 0x7f if math.copysign(1, x) > 0 else 0xff
  if dtype == dtypes.fp8e5m2 and math.isinf(x): return 0x7c if math.copysign(1, x) > 0 else 0xfc
  config = {
      dtypes.fp8e4m3: {"EXP_BIAS": 7, "SIGNIFICAND_BITS": 4, "MANTISSA_MASK": 0x7, "MINDENORM_O2": 0x3F50000000000000,
              "OVERFLOW_THRESHOLD": 0x407D000000000000, "MAXNORM": 0x7E, "MINNORM": 0x3F90000000000000, "INF_VALUE": 0x7F},
      dtypes.fp8e5m2: {"EXP_BIAS": 15, "SIGNIFICAND_BITS": 3, "MANTISSA_MASK": 0x3, "MINDENORM_O2": 0x3EE0000000000000,
              "OVERFLOW_THRESHOLD": 0x40EE000000000000 - 1, "MAXNORM": 0x7B, "MINNORM": 0x3F10000000000000, "INF_VALUE": 0x7E}
  }[dtype]
  xbits, = struct.unpack('Q', struct.pack('d', x))
  FP8_DP_HALF_ULP = 1 << (53 - config["SIGNIFICAND_BITS"] - 1)
  sign = ((xbits >> 63) & 1) << 7
  exp = (((xbits >> 52) & 0x7FF) - 1023 + config["EXP_BIAS"])
  mantissa = (xbits >> (53 - config["SIGNIFICAND_BITS"])) & config["MANTISSA_MASK"]
  absx = xbits & 0x7FFFFFFFFFFFFFFF

  if absx <= config["MINDENORM_O2"]: res = 0
  elif absx > 0x7FF0000000000000: res = 0x7F if dtype == dtypes.fp8e4m3 else 0x7E | mantissa
  elif absx > config["OVERFLOW_THRESHOLD"]: res = config["MAXNORM"]
  elif absx >= config["MINNORM"]:
    res = ((exp << (config["SIGNIFICAND_BITS"] - 1)) | mantissa)
    round_bits = xbits & ((FP8_DP_HALF_ULP << 1) - 1)
    if (round_bits > FP8_DP_HALF_ULP) or (round_bits == FP8_DP_HALF_ULP and (mantissa & 1)): res = res + 1
  else:
    shift = 1 - exp
    mantissa |= 1 << (config["SIGNIFICAND_BITS"] - 1)
    res = (mantissa >> shift)
    round_bits = (xbits | (1 << (53 - 1))) & ((FP8_DP_HALF_ULP << (shift + 1)) - 1)
    if (round_bits > (FP8_DP_HALF_ULP << shift)) or (round_bits == (FP8_DP_HALF_ULP << shift) and (res & 1)):
      res = res + 1

  res |= sign
  return int(res)

def fp8_to_float(x: int, dtype: DType) -> float:
  assert dtype in dtypes.fp8s, "Only for fp8s"
  ur = x << 8

  if dtype == dtypes.fp8e5m2 and (ur & 0x7FFF) > 0x7C00: ur = 0x7FFF
  elif dtype == dtypes.fp8e4m3:
    sign = ur & 0x8000
    exponent = ((ur & 0x7800) >> 1) + 0x2000
    mantissa = (ur & 0x0700) >> 1
    absx = x & 0x7F
    if absx == 0x7F: ur = 0x7FFF
    elif exponent == 0x2000:
      if mantissa != 0:
        mantissa <<= 1
        while (mantissa & 0x0400) == 0:
          mantissa <<= 1
          exponent -= 0x0400
        mantissa &= 0x03FF
      else:
        exponent = 0
      ur = (sign | exponent) | mantissa
    else:
      ur = (sign | exponent) | mantissa

  half_bytes = struct.pack('<H', ur)
  float32_val = struct.unpack('e', half_bytes)[0]
  return float(float32_val)

truncate: dict[DType, Callable] = {
  dtypes.bool: bool,
  dtypes.float16: float_to_fp16, dtypes.bfloat16: lambda x: float_to_bf16(float(x)),
  **{fp8: (lambda x, dtype=fp8: fp8_to_float(float_to_fp8(x, dtype), dtype)) for fp8 in dtypes.fp8s},
  dtypes.float32: lambda x: ctypes.c_float(x).value, dtypes.float64: lambda x: ctypes.c_double(x).value,
  dtypes.uint8: lambda x: ctypes.c_uint8(x).value, dtypes.uint16: lambda x: ctypes.c_uint16(x).value,
  dtypes.uint32: lambda x: ctypes.c_uint32(x).value, dtypes.uint64: lambda x: ctypes.c_uint64(x).value,
  dtypes.int8: lambda x: ctypes.c_int8(x).value, dtypes.int16: lambda x: ctypes.c_int16(x).value, dtypes.int32: lambda x: ctypes.c_int32(x).value,
  dtypes.int64: lambda x: ctypes.c_int64(x).value
}

# ************ PtrDTypes ************
class AddrSpace(Enum):
  def __repr__(self): return str(self)
  GLOBAL = auto(); LOCAL = auto(); REG = auto()  # noqa: E702
  
@dataclass(frozen=True, eq=False)
class PtrDType(DType):
  _base: DType
  addrspace: AddrSpace
  v: int
  size: int = -1  # -1 is unlimited size