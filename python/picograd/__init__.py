print("""
      ⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️
      importing teenygrad: the bridge from micrograd to tinygrad
      ⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️
      """)

"""
                                                                            ,,  
  mm         Rest in Pure Land Dr. Thomas Zhang ND., R.TCMP, R.Ac.        `7MM  
  MM                                                                        MM  
mmMMmm .gP"Ya   .gP"Ya `7MMpMMMb.`7M'   `MF'.P"Ybmmm `7Mb,od8 ,6"Yb.   ,M""bMM  
  MM  ,M'   Yb ,M'   Yb  MM    MM  VA   ,V :MI  I8     MM' "'8)   MM ,AP    MM  
  MM  8M"""""" 8M""""""  MM    MM   VA ,V   WmmmP"     MM     ,pm9MM 8MI    MM  
  MM  YM.    , YM.    ,  MM    MM    VVV   8M          MM    8M   MM `Mb    MM  
  `Mbmo`Mbmmd'  `Mbmmd'.JMML  JMML.  ,V     YMMMMMb  .JMML.  `Moo9^Yo.`Wbmd"MML.
                                    ,V     6'     dP                            
                                 OOb"      Ybmmmd'

teenygrad is a teaching deep learning framework that is the bridge from micrograd to tinygrad
see: https://github.com/karpathy/micrograd and https://github.com/tinygrad/tinygrad

teenygrad comes with a free opensource textbook
"The Struture and Interpretation of Tensor Programs"
see: : https://j4orz.ai/sitp/

in *CHAPTER 1* of the SITP book: https://j4orz.ai/sitp/1
you will develop all the required machinery for the `ndarray`/`Tensor` abstraction including:
  1. domain specific language
    0. intermediate representation with `OpNode` graph vertices and their `OpCode` function types
    1. frontend sugar with `Tensor` ndarray
    2. middleend engine with an `.evaluate`or that implements BLAS-like CPU kernels
  2. device runtimes: `Buffer` `Allocator` memory management and `Kernel` `Compiler` compute management

in *CHAPTER 2* of the SITP book: https://j4orz.ai/sitp/2
you will develop the two primary pytorch1 abstractions for training deep `torch.nn`s in the "age of research" including:
  1. optimization and differentiation with `optim.sgd` and `Tensor.backward()`
  2. parallel acceleration with cuBLAS-like GPU kernels

in *CHAPTER 3* of the SITP book: https://j4orz.ai/sitp/3
you will modify the engine in order to train deep `torch.nn`s in the "age of scaling by updating
  1. middleend engine's eager interpreter to just-in-time/lazy compiler
    A. optimizer: todo..
    B. generator: todo..
"""
from .sugar import nn, optim
from .sugar.tensor import Tensor
__all__ = ["Tensor"]

from importlib import import_module as _import_module
_pgrs = _import_module("picograd._pgrs")