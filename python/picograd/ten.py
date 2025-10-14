"""High-level tensor helpers."""

from . import _pgrs

class Tensor():
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def foo(self) -> str:
        """Return the sum of two numbers using the Rust extension."""
        return _pgrs.sum_as_string(self.x, self.y)


    def bar(self) -> str:
        """Return a friendly banner from the Python layer."""
        return "hello from picograd"


__all__ = ["Tensor"]
