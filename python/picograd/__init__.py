"""
picograd is a teaching deep learning framework that bridges micograd to tinygrad
which consists of a domain specific ndarray language and heterogenous runtime
1. a Device runtime that provides memory buffer allocation and compute kernel compilation
2. a Op intermediate representation that provides an abstract compute language at a higher semantic level than HIP/CUDA C
3. a Tensor domain specific ndarray programming language that provides automatic differentiation, optimizers, and neural network layers
for more details on the usage of each layer of abstraction, see examples/abstractions.py

picograd follows pt1 (chainer/autograd) for the "age of research" which takes the numpy ndarray and adds
  - gpu acceleration to forward kernels with .forward()
  - automatic differentiation with backward kernels with .backward()
  picograd's forward passes for the tensor object decompose (desugar) methods into a more primitive set of ops.
  the exact decomposition and intermediate representation is taken directly from tinygrad's RISC-y opset of element ops, reduce ops, and movement ops.
  when evaluating the model, a graph of those decomposed ops will be traced dynamically at runtime (as opposed to just-in-time source to source transform like autograd/jax),
  which then serves as the data structure for automatic differentiation to apply backpropagation and route gradients throughout the graph

picograd also follows pt2 for the "age of scaling" which modifies the language implementation strategy from eager interpretation to just-in-time/lazy compilation
  in order to obtain a global view of the computational graph, and to apply optimizations; the primary one being fusion.
  specifically, picograd follows tinygrad (and torch/xla and swift for tensorflow) with lazy graph capture, see (Suhan et al. https://arxiv.org/abs/2102.13267)
  and modifying the semantics of the programming model where users must explicitly materialize data with .realize(),
  as opposed to pt2 which maintains the eager programming model surface via graph capture at the host-language level (python bytecode interception)
  see (Ansel et al. https://docs.pytorch.org/assets/pytorch2-2.pdf)
"""
print("picograd: a bridge from micrograd to tinygrad")

from . import optim
from . import nn
from .tensor import Tensor
from .device import Runtime

__all__ = ["Runtime", "Tensor", "nn", "optim"]

from importlib import import_module as _import_module
_pgrs = _import_module("picograd._pgrs")