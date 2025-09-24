from picograd import numpy as np, eager, lazy #, PROF, reset_prof
# from picograd.functional import grad, jit, vmap_tensor

def demo_eager_vs_lazy():
    x = np.rand(1024, 1024)
    w = np.rand(1024, 1024)
    b = np.rand(1024)

    # Eager: 3 kernels (matmul, add, relu)
    # reset_prof()
    with eager():
        y1 = (x @ w + b).relu()
        _ = y1.numpy()  # materialize
    # print("[eager] kernels:", PROF.kernels, "bytes R/W:", PROF.bytes_read, PROF.bytes_written)

    # Lazy: 1 kernel after simple fusion (matmul+add+relu)
    # reset_prof()
    with lazy():
        y2 = (x @ w + b).relu().realize()
        _ = y2.numpy()
    # print("[lazy ] kernels:", PROF.kernels, "bytes R/W:", PROF.bytes_read, PROF.bytes_written)

# def demo_grad_and_jit():
#     def f(x, w, b):
#         return (x @ w + b).relu().sum()

#     g_w = grad(f, argnums=(1,))
#     x = Tensor.rand(8, 16); w = Tensor.rand(16, 32); b = Tensor.rand(32)

#     dw = g_w(x, w, b)
#     print("dw shape:", dw.shape)

#     f_fast = jit(f)
#     _ = f_fast(x, w, b)

# def demo_vmap():
#     def f(x):  # (N, D)->(), sum of relu
#         return x.relu().sum()
#     vf = vmap_tensor(f, in_axes=0, out_axes=0)
#     x = Tensor.rand(4, 16, 16)
#     y = vf(x)
#     print("vmap out shape:", y.numpy().shape)

# if __name__ == "__main__":
#     demo_eager_vs_lazy()
#     demo_grad_and_jit()
#     demo_vmap()