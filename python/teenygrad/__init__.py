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

teenygrad is a teaching deep learning framework that is the bridge from micrograd to tinygrad capable of training nanogpt
teenygrad comes with free course notes. https://j4orz.ai/sitp/
  - in part 1 you implement a multidimensional `Tensor` and accelerated `BLAS` kernels.
  - in part 2 you implement `.backward()` and accelerated `cuBLAS` kernels for the age of research.
  - in part 3 you implement a fusion compiler with `OpNode` graph IR for the age of scaling.
"""
from .frontend import nn, optim
from .frontend.tensor import Tensor
__all__ = ["Tensor"]

from importlib import import_module as _import_module
teenygradrs = _import_module("teenygrad._rs")