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
teenygrad is a teaching deep learning framework that bridges micograd to tinygrad which consists of a
1. sugar: domain specific ndarray language (python `Tensor`) that provides automatic differentiation, optimizers, and neural network layers
2. engine: an interpreter and compiler pipeline for an abstract compute language (`OpNode`/\OpCode`) that the tensor frontend desugars/lowers to.
    a. interpreter pipeline "pt1's age of researcH":
      when evaluating the model, a graph of those decomposed ops will be traced dynamically at runtime (as opposed to just-in-time source to source transform like autograd/jax),
      which then serves as the data structure for automatic differentiation to apply backpropagation and route gradients throughout the graph
    b. compiler pipeline "pt2's age of scaling":
      which modifies the language implementation strategy from eager interpretation to just-in-time/lazy compilation
      in order to obtain a global view of the computational graph, and to apply optimizations; the primary one being fusion.
    
    gpu accelerated kernels (cuda c/hip c)
3. runtime: memory (`Buffer` `Allocator`) and compute (`Kernel` `Compiler`)
"""
from .sugar import nn, optim
from .sugar.tensor import Tensor
__all__ = ["Tensor"]

from importlib import import_module as _import_module
_pgrs = _import_module("picograd._pgrs")