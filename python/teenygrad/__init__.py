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

As a compiler writer for domain specific cloud languages, I became frustrated with the non-constructiveness and disjointedness
of my learning experience in the discipline of machine learning systems, particularly with domain specific tensor languages.
These course notes for The Structure and Interpretation of Tensor Programs is my personal answer to these frustrations.  It's:
- the spiritual successor to LLM101n. You will develop your own deep learning framework teenygrad (sharing 90% of its abstractions with tinygrad)
  in order to run training and inference for the exact same nanogpt[0] and nanochat[1] models you developed in Eureka's LLM101n[2] course.
- inspired by the whirlwind tour form of SICP/HTDP/PAPL1 applied to substance of training neural networks, hacking deep learning frameworks, and programming massively parallel processors.
- opensource book creating an omakase learning experience with the teenygrad codebase and the best high level intuitive visualizers from across the world

[0]: https://github.com/karpathy/nanogpt
[1]: https://github.com/karpathy/nanochat
[2]: https://eurekalabs.ai/

**Contents**
in part 1 you implement a multidimensional `Tensor` and accelerated `BLAS` kernels.
in part 2 you implement `.backward()` and accelerated `cuBLAS` kernels for the age of research.
in part 3 you implement a fusion compiler with `OpNode` graph IR for the age of scaling.
If you empathize with some of my frustrations, you may benefit from the course too.
I've intentionally designed it so that the only pre-requisite required is the ability to program with highschool-level math.

Good luck on your journey.
Are you ready to begin?
"""
from .frontend import nn, optim
from .frontend.tensor import Tensor
__all__ = ["Tensor"]

from importlib import import_module as _import_module
eagkers = _import_module("teenygrad._rs")
