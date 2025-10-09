import picograd
import matplotlib.pyplot as plt

def ad():
    print("automatic differentiation")

def sgd():
    print("stochastic gradient descent")

def adam():
    print("adam")

def muon():
    print("muon")

def main():
    print("sitp ch3: anatomy of the autograd") # read ch3.0: https://j4orz.ai/sitp/3.0
    ad()                                 # read ch3.1: https://j4orz.ai/sitp/3.1
    sgd()                                # read ch3.2: https://j4orz.ai/sitp/3.2
    adam()                               # read ch3.3: https://j4orz.ai/sitp/3.3
    muon()                               # read ch3.4: https://j4orz.ai/sitp/3.4

if __name__ == "__main__":
    main()