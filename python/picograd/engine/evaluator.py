from __future__ import annotations
import os
from typing import TYPE_CHECKING
# from picograd.device import Allocator
from picograd.engine.op import OpCode
if TYPE_CHECKING: from picograd.frontend.tensor import Tensor
from math import exp, sin