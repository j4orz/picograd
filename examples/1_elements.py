import picograd
import matplotlib.pyplot as plt

def tensor_on_het():
    print("tensors on heterogeneous runtimes: picograd.Tensor")

def vecs_as_probs():
    print("vectors as probabilities: picograd.distributions")

def mat_as_data_and_fn():
    print("matrices as data and functions: picograd.linalg")

def matcalc_for_opt():
    print("matrix calculus for optimization: picograd.optim")

def learn_gaussian_with_regression():
    print("learning gaussisans with regressin: price prediction")

def main():
    print("sitp ch1: elements of learning") # read ch1.0: https://j4orz.ai/sitp/1.0
    tensor_on_het()                      # read ch1.1: https://j4orz.ai/sitp/1.1
    vecs_as_probs()                      # read ch1.2: https://j4orz.ai/sitp/1.2
    mat_as_data_and_fn()                 # read ch1.3: https://j4orz.ai/sitp/1.3
    matcalc_for_opt()                    # read ch1.4: https://j4orz.ai/sitp/1.4
    learn_gaussian_with_regression()     # read ch1.5: https://j4orz.ai/sitp/1.5

if __name__ == "__main__":
    main()