# Software 2.0 Reference

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

1. Do not confuse the 0/1 in $\mathcal{Y}=\{0,1\}$ with the 0/1 in $h_{\theta}: \mathbb{R}^{d} \to [0,1]$. The former is the sample space mapped by random variable $Y:\Omega \to \mathbb{N}$, whereas the latter is the probability measure assigned to that event conditioned on $X$. Multiclass classification has $\mathcal{Y}=\{0,1,...,k\}$ with the same $h_{\theta}: \mathbb{R}^{d} \to [0,1]$.
2. Do not confuse parameter $p$ with parameters $\boldsymbol{\theta}$. It's helpful to conceptualize the former decomposing into the latter. $;$ is used to denote the notion of "parameterized by", which is not a 1-1 substitution.


Then, taking the maximum likelihood estimate (MLE) of the negative log-likelihood yields:

$$
\begin{align*}

\hat{\mathbf{\theta}} &\in \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\mathop{\text{argmin}}} -\log \mathcal{L}(\boldsymbol{\theta}) \\
 &\in \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\mathop{\text{argmin}}} -\log \prod_{i=1}^n p(y^{(i)}|\mathbf{x}^{(i)};\boldsymbol{\theta}) \tag{conditional prob after Y iid} \\
 &\in \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\mathop{\text{argmin}}} -\log \prod_{i=1}^n [\sigma(\boldsymbol{\theta}^{\top}\mathbf{x}^{(i)})]^{y^{(i)}}[1-\sigma(\boldsymbol{\theta}^{\top}\mathbf{x}^{(i)})]^{1-y^{(i)}} \tag{logistic regression assumption}\\
 &\in \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\mathop{\text{argmin}}} -\sum_{i=1}^n y^{(i)}\log \sigma(\boldsymbol{\theta}^{\top}\mathbf{x}^{(i)}) + (1-y^{(i)})\log (1-\sigma(\boldsymbol{\theta}^{\top}\mathbf{x}^{(i)})) \tag{log laws} \\
 &\in \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\mathop{\text{argmin}}} -\sum_{i=1}^n y^{(i)}\log \hat{y}^{(i)} + (1-y^{(i)})\log (1-\hat{y}^{(i)}) \tag{$\hat{y}^{(i)} = \sigma(\boldsymbol{\theta}^{\top}\mathbf{x}^{(i)})$}\\

\end{align*}
$$

where
1. $\hat{p} := \hat{\boldsymbol{\theta}}$ (the estimators are 1-1)
2. $\mathcal{L}$ is used to denote likelihood, whereas the negative log likelihood $-log\mathcal{L}$ is the entire loss function.
3. $\in$ is used rather than $=$ to denote the potential existence of multiple optima, even though in this case, the solution is unique since the log-likelihood function is convex. Finding *the* optimum was more important back when the field was dominated by statistical learning methods, but, this has became less important over time with the empirical success of non-linear models such as deep neural networks.

Finally, $\mathop{\text{argmin}}$ is implemented algorithmically (as opposed to numerically or symbolically) with gradient descent as $\theta^{t+1} := \theta^{t} - \alpha \nabla_{\boldsymbol{\theta}} NLL(\theta)$.

where

$$
\begin{align*}
\nabla_{\boldsymbol{\theta}}NLL(\boldsymbol{\theta}) &= \nabla[-\sum_{i=1}^n y^{(i)}\log \hat{y}^{(i)} + (1-y^{(i)})\log (1-\hat{y}^{(i)})] \tag{$\hat{y}^{(i)} = \sigma(\boldsymbol{\theta}^{\top}\mathbf{x}^{(i)})$} \\
&= -\sum_{i=1}^n \nabla_{\boldsymbol{\theta}}[y^{(i)}\log \hat{y}^{(i)} + (1-y^{(i)})\log (1-\hat{y}^{(i)})] \tag{$\nabla$ linear} \\
&= -\sum_{i=1}^n \nabla_{\boldsymbol{\theta}}[y^{(i)}\log \hat{y}^{(i)} + (1-y^{(i)})\log (1-\hat{y}^{(i)})] \tag{} \\

\end{align*}
$$

so $\theta^{t+1} := \theta^{t} - \alpha \nabla_{\theta} NLL(\theta)$.

**Regression**: recovers discriminative model $p(Y=y|\mathbf{X}=\mathbf{x}; \theta)$ where $Y \overset{\text{iid}}{\sim} Nor(\mu, \sigma^2).$ The linear regression assumption is to take a linear hypothesis $h_{\theta}(\mathbf{x})$ defined as

$$
\begin{align*}
h_{\theta}&: \mathbb{R}^{d} \to \mathbb{R} \\
h_{\theta}(\mathbf{x})&:= \theta^{\top}\mathbf{x}
\end{align*}
$$

**Generalized linear models**

## Non-linear Non-Parametric Models

**Kernel Methods**

## Non-linear Parametric Models
**Neural Networks (NN)**

*Prediction*
TODO: generalize classifiction (N) and regression (R) into exponential family.

*Generation*
just sampling from a classification model?

from here on out, we're going to focus on generative sequence modelling,
building our way up to large language models (1B, 10B, 70B??).

_Double Descent_
_Grokking_
_Lottery Tickets_

**Recurrent Neural Networks (RNN)**

**Generative Pretrained Transformers (GPT)**

**ChatGPT (GPT + RLHF)**