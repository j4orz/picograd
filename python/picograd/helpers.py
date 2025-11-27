import functools, platform, sys, os, time, ctypes, subprocess
from typing import overload

DEBUG = 0 # ContextVar("DEBUG", 0)
OSX, WIN = platform.system() == "Darwin", sys.platform == "win32"
LRU = 1 # ContextVar("LRU", 1)

@overload
def getenv(key:str) -> int: ...
@overload
def getenv(key:str, default:T) -> T: ...
@functools.cache
def getenv(key:str, default:Any=0): return type(default)(os.getenv(key, default))

def unwrap_class_type(cls_t): return cls_t.func if isinstance(cls_t, functools.partial) else cls_t

def suppress_finalizing(func):
  def wrapper(*args, **kwargs):
    try: return func(*args, **kwargs)
    except (RuntimeError, AttributeError, TypeError, ImportError):
      if not getattr(sys, 'is_finalizing', lambda: True)(): raise # re-raise if not finalizing
  return wrapper

def colored(st, color:str|None, background=False): # replace the termcolor library
  colors = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
  return f"\u001b[{10*background+60*(color.upper() == color)+30+colors.index(color.lower())}m{st}\u001b[0m" if color is not None else st

def system(cmd:str, **kwargs) -> str:
  st = time.perf_counter()
  ret = subprocess.check_output(cmd.split(), **kwargs).decode().strip()
  # if DEBUG >= 1: print(f"system: '{cmd}' returned {len(ret)} bytes in {(time.perf_counter() - st)*1e3:.2f} ms")
  print(f"system: '{cmd}' returned {len(ret)} bytes in {(time.perf_counter() - st)*1e3:.2f} ms")
  return ret