import teenygrad
import torch
import unittest

def helper_test_generic_square(name, N, f1, f2, onearg=False):
  torch.manual_seed(0)
  torch_a = (torch.rand(N, N, dtype=torch_dt) - 0.5).to(torch_device)
  torch_b = (torch.rand(N, N, dtype=torch_dt) - 0.5).to(torch_device) if not onearg else None

  tiny_a = Tensor(torch_a.cpu().numpy())
  tiny_b = Tensor(torch_b.cpu().numpy()) if not onearg else None

  helper_test_generic(f"{name:30s} {N:5d}x{N:5d}", f1, (torch_a, torch_b), TinyJit(f2), (tiny_a, tiny_b))

def helper_test_matvec(name, N, M):
  torch.manual_seed(0)
  torch_a = (torch.rand(N, dtype=torch_dt) - 0.5).to(torch_device)
  torch_b = (torch.rand(N, M, dtype=torch_dt) - 0.5).to(torch_device)

  tiny_a = Tensor(torch_a.cpu().numpy())
  tiny_b = Tensor(torch_b.cpu().numpy())

  helper_test_generic(f"{name:30s} {N:5d}x{M:5d}", lambda a,b: a@b, (torch_a, torch_b), TinyJit(lambda a,b:a@b), (tiny_a, tiny_b))

prefix = None
def helper_test_generic(name, f1, f1_args, f2, f2_args):
  global prefix
  with torch.no_grad():
    val_torch, et_torch = helper_test_speed(f1, *f1_args)
  val_tinygrad, et_tinygrad = helper_test_speed(f2, *f2_args)

  desc = "faster" if et_torch > et_tinygrad else "slower"
  flops = save_ops*1e-6
  mem = save_mem*1e-6
  print(("\r" if not CI else "")+f"{name:42s} {et_torch:7.2f} ms ({flops/et_torch:9.2f} GFLOPS {mem/et_torch:7.2f} GB/s) in torch, {et_tinygrad:7.2f} ms ({flops/et_tinygrad:9.2f} GFLOPS {mem/et_tinygrad:7.2f} GB/s) in tinygrad, {colorize_float(et_tinygrad/et_torch)} {desc} {flops:10.2f} MOPS {mem:8.2f} MB")  # noqa: E501
  atol, rtol = (1e-2, 1e-2) if torch_dt == torch.float16 else (1e-3, 1e-3)
  np.testing.assert_allclose(val_tinygrad, val_torch, atol=atol, rtol=rtol)

save_ops, save_mem = 0, 0
CNT = getenv("CNT", 8)
def helper_test_speed(f1, *args):
  global save_ops, save_mem
  ets = []
  ret = None
  cache_defeat = np.zeros((2048,2048))
  for i in range(CNT):
    del ret

    # operation cache defeats
    args = [(x+1).realize() if isinstance(x, Tensor) else (None if x is None else (x+1)) for x in args]
    args = [(x-1).realize() if isinstance(x, Tensor) else (None if x is None else (x-1)) for x in args]

    # force syncing
    [x.numpy() if isinstance(x, Tensor) or str(torch_device) == "cpu" else x.cpu().numpy() for x in args if x is not None]

    # clear 32MB global memory cache (CPU and global memory only)
    cache_defeat += 1

    # manual pre sync
    if isinstance(args[0], Tensor):
      local_device = Device[args[0].device]
      local_device.synchronize()
    else: sync()

    GlobalCounters.global_ops = 0
    GlobalCounters.global_mem = 0
    st = time.perf_counter()
    ret = f1(*args)
    if isinstance(ret, Tensor): local_device.synchronize()
    else: sync()
    et = (time.perf_counter() - st) * 1000
    if i >= 1: ets.append(et)
    if GlobalCounters.global_ops:
      save_ops, save_mem = GlobalCounters.global_ops, GlobalCounters.global_mem
  return ret.numpy() if isinstance(ret, Tensor) else ret.cpu().numpy(), np.min(ets)

@unittest.skipIf(getenv("BIG") == 0, "no big tests")
@unittest.skipIf(getenv("MOCKGPU"), "no MOCKGPUs")
class TestBigSpeed(unittest.TestCase):
  def test_add(self):
    def f(a, b): return a+b
    helper_test_generic_square('add', 8192, f, f)
  def test_exp(self):
    def f(a, b): return a.exp()
    helper_test_generic_square('exp', 8192, f, f, onearg=True)
  def test_gemm_2048(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 2048, f, f)
  def test_gemm_4096(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 4096, f, f)
  def test_matvec_4096_16384(self): helper_test_matvec('matvec_4096_16384', 4096, 16384)
  def test_matvec_16384_4096(self): helper_test_matvec('matvec_16384_4096', 16384, 4096)

class TestSpeed(unittest.TestCase):
  def test_sub(self):
    def f(a, b): return a-b
    helper_test_generic_square('sub', 4096, f, f)

  def test_neg(self):
    def f(a, b): return -a
    helper_test_generic_square('neg', 4096, f, f, onearg=True)

  def test_exp(self):
    def f(a, b): return a.exp()
    helper_test_generic_square('exp', 2048, f, f, onearg=True)

  def test_sqrt(self):
    def f(a, b): return a.sqrt()
    helper_test_generic_square('sqrt', 2048, f, f, onearg=True)

  def test_relu(self):
    def f(a, b): return a.relu()
    helper_test_generic_square('relu', 4096, f, f, onearg=True)

  def test_max(self):
    def f(a, b): return a.max()
    helper_test_generic_square('max', 4096, f, f, onearg=True)

  def test_gemm(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 1024, f, f)

  def test_gemm_small(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 256, f, f)

  def test_matvec_1024_1024(self): helper_test_matvec('matvec_1024_1024', 1024, 1024)
  def test_matvec_1024_4096(self): helper_test_matvec('matvec_1024_4096', 1024, 4096)
  def test_matvec_4096_1024(self): helper_test_matvec('matvec_4096_1024', 4096, 1024)
  def test_matvec_4096_4096(self): helper_test_matvec('matvec_4096_4096', 4096, 4096)

if __name__ == '__main__':
    unittest.main()