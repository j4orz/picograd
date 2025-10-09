from math import sin, exp
import numpy as np
import matplotlib.pyplot as plot

# AD

# symbolic:
# numeric: finite differences (slow and inaccurate)
# automatic: "as exact as symbolic differentiation but with no expression swell"


# computational graph -> linearized computational graph. -> bauer paths (sum over paths)
# example from paul d. hovland

# forward mode AD: topological vertex elimination (dynamic programming?)
# reverse mode AD "backpropagation", "errors moving back through time":
# reverse topological vertex elimination
# forward vs reverse is jsut choosing a better order for chain rule
# cubic vs quadratic. order is flexible bc matmul is associative

# it makes sense to go forward if you have f: R -> R^n (i.e parameterizing high dimensional vector field)

# forward and reverse mode are just two extreme of the dynamic programming schedules
# finding the best schedule is NP-hard (what reduction? coloring?)


def f(x1, x2):
    a = exp(x1)
    b = sin(x2)
    c = b*x2
    d = a*c
    return a*d
    

def tanh(x):
    y = np.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)

def main():
    x = np.random.randn(3, 4)
    A = np.random.randn(3, 4)
    t = np.random.randn(3, 4, 5)
    print("hello world")
    print("random vector x\n", x)
    print("==============================")
    print("random matrix x\n", A)
    print("==============================")
    print("random tensor x\n", t)

    grad_tanh = grad(tanh)
    print(grad_tanh(1.0))
    print(tanh(1.0001)-tanh(0.9999)) / 0.0002

    foo = np.linspace(-7, 7, 200)
    # first, second, third, fourth, fifth, sixth derivatives??

def ffn(X, W):
    # H1
    # H2
    # out
    return 5

if __name__ == "__main__":
    main()