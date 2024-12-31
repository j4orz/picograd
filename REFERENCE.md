# Software 2.0 Reference

**Contents**
1. Hypotheses
    - [Linear models](#linear-models)
    - Non-linear non-parametric models
    - Non-linear parametric models
        - FFN
        - RNN
        - CNN
        - GPT
2. Optimization
    - SGD
    - Adam
    - Shampoo

## Linear Models

**Classification**: recovers discriminative classifier $p(Y=c|\mathbf{X}=\mathbf{x}; \boldsymbol{\theta})$ where $Y \overset{\text{iid}}{\sim} Ber(p)$. First we start off with binary classification so $\mathcal{Y}=\{0,1\}$. The logistic regression (better named as sigmoidal classification) assumption is to take a linear hypothesis $h_{\boldsymbol{\theta}}(\mathbf{x})$ defined as

$$
\begin{align*}
h_{\boldsymbol{\theta}}&: \mathbb{R}^{d} \to [0,1] \\
h(\mathbf{x};\mathbf{\boldsymbol{\theta}})&:= \sigma(\boldsymbol{\theta}^{\top}\mathbf{x}) = \frac{1}{1+\mathrm{e}^{-\boldsymbol{\theta}^{\top}\mathbf{x}}}
\end{align*}
$$

which typechecks since $\boldsymbol{\theta}^{\top} \in \mathbb{R}^{1,d}$ (so $\boldsymbol{\theta}^{\top}: \mathbb{R}^{d} \to \mathbb{R}$) and $\sigma: \mathbb{R} \to [0,1]$, and interpret it as $p$. That is,

$$
\begin{align*}
p(Y=1|X=x;\boldsymbol{\theta}) &:= \sigma(\boldsymbol{\theta}^{\top}\boldsymbol{x}) \tag{logistic regression assumption} \\
\implies p(Y=0|X=x;\boldsymbol{\theta}) &= 1- \sigma(\theta^{\top}\mathbf{x}) \tag{total law of prob} \\
\end{align*}
$$

which we can formulate as a continuous and differentiable probability density function $p(Y=y|X=x) = [\sigma(\theta^{\top}\mathbf{x})]^y[1-\sigma(\theta^{\top}\mathbf{x})]^{1-y}$. Two things to note:

1. Do not confuse the 0/1 in $\mathcal{Y}=\{0,1\}$ with the 0/1 in $h_{\theta}: \mathbb{R}^{d} \to [0,1]$. The former is the sample space mapped by random variable $Y:\Omega \to \mathbb{N}$, whereas the latter is the probability measure assigned to that event conditioned on $X$. We could change the label encodings to be $\mathcal{Y}=\{-1,1\}$. Also, multiclass classification has $\mathcal{Y}=\{0,1,...,k\}$ with the same type $h_{\theta}: \mathbb{R}^{d} \to [0,1]$.
2. Do not confuse parameter $p$ with parameters $\boldsymbol{\theta}$. It's helpful to conceptualize the former decomposing into the latter. $;$ is used to denote the notion of "parameterized by", which is not a 1-1 substitution.


Then, taking the maximum likelihood estimate (MLE) of the negative log-likelihood yields:

$$
\begin{align*}

\hat{\mathbf{\theta}} &\in \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\mathop{\text{argmin}}} -\log \mathcal{L}(\boldsymbol{\theta}) \\
 &\in \underset{\boldsymbol{\theta}}{\mathop{\text{argmin}}} -\log \prod_{i=1}^n p(y^{(i)}|\mathbf{x}^{(i)};\boldsymbol{\theta}) \tag{conditional prob after Y iid} \\
 &\in \underset{\boldsymbol{\theta}}{\mathop{\text{argmin}}} -\log \prod_{i=1}^n [\sigma(\boldsymbol{\theta}^{\top}\mathbf{x}^{(i)})]^{y^{(i)}}[1-\sigma(\boldsymbol{\theta}^{\top}\mathbf{x}^{(i)})]^{1-y^{(i)}} \tag{logistic regression assumption}\\
 &\in \underset{\boldsymbol{\theta}}{\mathop{\text{argmin}}} -\sum_{i=1}^n y^{(i)}\log \sigma(\boldsymbol{\theta}^{\top}\mathbf{x}^{(i)}) + (1-y^{(i)})\log (1-\sigma(\boldsymbol{\theta}^{\top}\mathbf{x}^{(i)})) \tag{log laws} \\
 &\in \underset{\boldsymbol{\theta}}{\mathop{\text{argmin}}} -\sum_{i=1}^n y^{(i)}\log \hat{y}^{(i)} + (1-y^{(i)})\log (1-\hat{y}^{(i)}) \tag{$\hat{y}^{(i)} = \sigma(\boldsymbol{\theta}^{\top}\mathbf{x}^{(i)})$}\\
&\in \underset{\boldsymbol{\theta}}{\mathop{\text{argmin}}} \sum_{i=1}^n \mathbb{H}_{ce}(y^{(i)}, \hat{y}^{(i)}) \tag{binary cross entropy}\\

\end{align*}
$$

where
1. $\hat{p} := \hat{\boldsymbol{\theta}}$ (the estimators are 1-1)
2. $\mathcal{L}$ is used to denote likelihood, whereas the negative log likelihood $-log\mathcal{L}$ is the entire loss function.
3. $\in$ is used rather than $=$ to denote the potential existence of multiple optima, even though in this case, the solution is unique since the log-likelihood function is convex. Finding *the* optimum was more important back when the field was dominated by statistical learning methods, but, this has became less important over time with the empirical success of non-linear models such as deep neural networks.

Finally, while $\mathop{\text{argmin}}$ is usually implemented algorithmically (as opposed to numerically or symbolically) via automatic differentiation, we will save autograd for deep neural networks, and for now, derive the gradient manually.

$$
\begin{align*}
\nabla_{\boldsymbol{\theta}}\text{NLL}(\boldsymbol{\theta}) &= \nabla_{\boldsymbol{\theta}}[-\sum_{i=1}^n y^{(i)}\log \hat{y}^{(i)} + (1-y^{(i)})\log (1-\hat{y}^{(i)})] \tag{$\hat{y}^{(i)} = \sigma(\boldsymbol{\theta}^{\top}\mathbf{x}^{(i)})$} \\
&= -\sum_{i=1}^n \nabla_{\boldsymbol{\theta}}[y^{(i)}\log \hat{y}^{(i)} + (1-y^{(i)})\log (1-\hat{y}^{(i)})] \tag{$\nabla$ is linear} \\
&= -\sum_{i=1}^n \nabla_{\boldsymbol{\theta}}[y^{(i)}\log \hat{y}^{(i)} + (1-y^{(i)})\log (1-\hat{y}^{(i)})] \tag{} \\

\end{align*}
$$

so $\theta^{t+1} := \theta^{t} - \eta \nabla_{\theta} \text{NLL}(\theta)$.

**Regression**: recovers discriminative model $p(Y=y|\mathbf{X}=\mathbf{x}; \theta)$ where $Y \overset{\text{iid}}{\sim} Nor(\mu, \sigma^2).$ The linear regression assumption is to take a linear hypothesis $h_{\theta}(\mathbf{x})$ defined as

$$
\begin{align*}
h_{\theta}&: \mathbb{R}^{d} \to \mathbb{R} \\
h_{\theta}(\mathbf{x})&:= \theta^{\top}\mathbf{x}
\end{align*}
$$

**Generalized linear models**

## Non-linear Parametric Models

**Neural Networks (NN)**: To motivate the functions that neural networks implement, the biological inspiration is ignored in favor of the mathematical specification. The previous family of functions that we considered as hypotheses were linear models, so the obvious idea is to come up with a non-linear function class $h(\mathbf{x}; \mathbf{w}, \mathbf{b}) := W\phi(\mathbf{x}) + \mathbf{b}$. We can recursively compose these non-linear functions with the aim of "lifting" the *representation* of the data into more complex "motifs". A loose analogy is to conceptualize a neural network with multiple compositions (layers) as a function which parses the representation of the data [(Olah 2015)](https://colah.github.io/posts/2015-09-NN-Types-FP/):

$$
\begin{align*}
h&: \mathbb{R}^{d_0} \to \mathbb{R}^{d_L} \\
h(\mathbf{x}; \mathbf{w}) &:= W_l \circ (\phi \circ W_{l-1}) \circ \cdots \circ (\phi \circ W_1) \circ \mathbf{x}
\end{align*}
$$

Network design started out with manual feature engineering of $\phi: \mathbb{R}^{d_i} \to [0,1]^n$, and evolved into deep learning, a suitecase term that refers to end-to-end representation learning of networks with multiple compositions.

While the optimization and generalization of other non-linear models such as kernel methods and gaussian processes are formally well-understood (with functional analysis and bayesian probability, respectively), the primary method of inquiry for neural networks have been empiricism. There are very interesting open questions for the theoretician, but for now, we proceed in this document with the agreement that the state of deep learning is more similar to alchemy than it is to chemistry.


With that said, the domain of modelling that we are interested in is language. The feed forward neural network (fnn) will be covered first, and each major advance in network architecture thereafter.

**Feedforward Neural Networks (FNN)**

**Recurrent Neural Networks (RNN)**

**Convolutional Neural Networks (CNN)**

**Generative Pretrained Transformers (GPT)**

**ChatGPT (GPT + RLHF)**

## Optimizers

## SGD
The general definition of the derivative...important for understanding different optimizers that use momentum and preconditioning techniques. They're just using gradients with different notions of distance (inner product). Speedrun gpt2 training runs.

## Adam

## Shampoo

## Non-linear Non-Parametric Models
**Kernel Methods**

**Gaussian Processes**