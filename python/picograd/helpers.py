import platform
import sys
import time, ctypes, subprocess

# DEBUG = ContextVar("DEBUG", 0)
OSX, WIN = platform.system() == "Darwin", sys.platform == "win32"

def init_c_struct_t(fields: tuple[tuple[str, type[ctypes._SimpleCData]], ...]):
  class CStruct(ctypes.Structure):
    _pack_, _fields_ = 1, fields
  return CStruct
def init_c_var(ctypes_var, creat_cb): return (creat_cb(ctypes_var), ctypes_var)[1]
def mv_address(mv): return ctypes.addressof(ctypes.c_char.from_buffer(mv))

def system(cmd:str, **kwargs) -> str:
  st = time.perf_counter()
  ret = subprocess.check_output(cmd.split(), **kwargs).decode().strip()
  # if DEBUG >= 1: print(f"system: '{cmd}' returned {len(ret)} bytes in {(time.perf_counter() - st)*1e3:.2f} ms")
  print(f"system: '{cmd}' returned {len(ret)} bytes in {(time.perf_counter() - st)*1e3:.2f} ms")
  return ret