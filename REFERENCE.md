# Software 2.0 Math

## Linear Models

**Classification**: recovers discriminative model $p(Y=c|\mathbf{X}=\mathbf{x}; \theta)$ where $Y \overset{\text{iid}}{\sim} Ber(p)$. The logistic regression (better named as sigmoidal classification) assumption is to take a linear hypothesis $h_{\theta}(\mathbf{x})$ defined as

$$
\begin{align*}
h_{\theta}&: \mathbb{R}^{d} \to [0,1] \\
h_{\theta}(\mathbf{x})&:= \sigma(\theta^{\top}\mathbf{x}) = \frac{1}{1+\mathrm{e}^{-\theta^{\top}\mathbf{x}}}
\end{align*}
$$

which typechecks since $\theta^{\top}: \mathbb{R}^{d} \to \mathbb{R}$ and $\sigma: \mathbb{R} \to [0,1]$, and interpret it as $p$. That is,

$$
\begin{align*}
p(Y=1|X=x) &:= \sigma(\theta^{\top}\mathbf{x}) \tag{logistic regression assumption} \\
\implies p(Y=0|X=x) &= 1- \sigma(\theta^{\top}\mathbf{x}) \tag{total law of prob} \\
\end{align*}
$$

which we can formulate as a continuous and differentiable probability density function $p(Y=y|X=x) = [\sigma(\theta^{\top}\mathbf{x})]^y[1-\sigma(\theta^{\top}\mathbf{x})]^{1-y}$.

Then, taking the maximum likelihood estimate (MLE) of the negative log-likelihood yields:

$$
\begin{align*}

\hat{\mathbf{\theta}} &\in \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\mathop{\text{argmin}}} -\log \mathcal{L}(\boldsymbol{\theta}) \\
 &\in \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\mathop{\text{argmin}}} -\log \prod_{i=1}^n p(y^{(i)}|\mathbf{x}^{(i)};\boldsymbol{\theta}) \tag{conditional prob after Y iid} \\
 &\in \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\mathop{\text{argmin}}} -\log \prod_{i=1}^n [\sigma(\theta^{\top}\mathbf{x})]^y[1-\sigma(\theta^{\top}\mathbf{x})]^{1-y} \tag{logistic regression assumption}\\
 &\in \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\mathop{\text{argmin}}} \sum_{i=1}^n y^{(i)}\log \sigma(\theta^{\top}\mathbf{x}^{(i)}) - (1-y^{(i)})\log (1-\sigma(\theta^{\top}\mathbf{x}^{(i)})) \tag{log laws} \\
 &\in \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\mathop{\text{argmin}}} \sum_{i=1}^n y^{(i)}\log \hat{y}^{(i)} - (1-y^{(i)})\log (1-\hat{y}^{(i)}) \tag{$\hat{y}^{(i)} = \sigma(\theta^{\top}\mathbf{x}^{(i)})$}\\

\end{align*}
$$

where $\in$ is used rather than $=$ to indicate the existence of multiple optima. In this specific case, the solution is unique since the log-likelihood function is convex, which was important when the field was dominated by statistical learning methods. However, with the empirical success of non-linear models such as deep neural networks, finding *the* optimum is not as important.

Finally, $\mathop{\text{argmin}}$ is implemented algorithmically with gradient descent as $\theta^{t+1} := \theta^{t} - \alpha \nabla_{\theta} NLL(\theta)$.

where

$$
\begin{align*}

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

**Linear language models**: models language with $(\Omega, \mathcal{F}, \mathbb{P})$ probability space..

## Non-linear Models

**Kernel Methods**

**Neural Networks**

**Large Language Models**

