from __future__ import annotations

class Device:
  """
  picograd's device runtimes follow the tinygrad runtime architecture where a
  VendorRuntime manages compute with a Program and memory with an Allocator
  """
  def __init__(self): raise NotImplementedError("todo")

class VendorRuntime:
  def __init__(self, device:str, allocator:Allocator): raise NotImplementedError("todo")

# class TinyHCQRuntime:
#   def __init__(self, device:str, allocator:Allocator): raise NotImplementedError("todo")

class Buffer:
  def __init__(): raise NotImplementedError("")

class Allocator:
  def __init__(self): raise NotImplementedError("todo")
  def alloc(self, size:int, options:BufferSpec|None=None):
    assert size > 0, f"alloc size must be positive, getting {size}"
    return self._alloc(size, options if options is not None else self.default_buffer_spec)
  def free(self, opaque, size:int, options:BufferSpec|None=None):
    self._free(opaque, options if options is not None else self.default_buffer_spec)

class VendorCompiler:
  def __init__(self, cachekey:str|None=None): self.cachekey = None if DISABLE_COMPILER_CACHE else cachekey
  def compile(self, src:str) -> bytes: return src.encode()   # NOTE: empty compiler is the default
  def compile_cached(self, src:str) -> bytes:
    if self.cachekey is None or (lib := diskcache_get(self.cachekey, src)) is None:
      assert not getenv("ASSERT_COMPILE"), f"tried to compile with ASSERT_COMPILE set\n{src}"
      lib = self.compile(src)
      if self.cachekey is not None: diskcache_put(self.cachekey, src, lib)
    return lib
  def disassemble(self, lib:bytes): pass
  def __init__(self): raise NotImplementedError("todo")


ALL_DEVICES = ["CPU", "HIP"]