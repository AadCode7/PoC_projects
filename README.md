# PoC_projects

# 1] Linear Regression

### 1.1 — What and why

Linear regression is a type of supervised machine-learning algorithm that learns from the labelled datasets and maps the data points with most optimized linear functions which can be used for prediction on new datasets.~ It's the simplest supervised learning algorithm for regression: transparent, fast, interpretable, and a foundation for more advanced methods. Use it for baseline modeling, understanding linear relationships, and for problems where interpretability matters. 


### 1.2 — Linear regression — intuition and math

We assume a dataset ((x^{(i)}, y^{(i)})*{i=1}^m). For a single example (x \in \mathbb{R}^n) (with bias term), the model predicts
[
\hat{y} = h*\theta(x) = \theta^T x = \sum_{j=0}^{n} \theta_j x_j,
]
where (x_0 = 1) (bias). The learning task is to find parameter vector (\theta) that gives predictions close to (y).

Define the mean squared error (MSE) cost:
[
J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2.
]
Minimizing (J(\theta)) yields the best-fit linear model under squared-error. ([cs229.stanford.edu][2])


### 1.3 — Cost function and normal equation

Because (J(\theta)) is a convex quadratic in (\theta), the global minimum can be solved analytically. In matrix form, with (X\in\mathbb{R}^{m\times (n+1)}) and (y\in\mathbb{R}^m),
[
\theta^* = (X^T X)^{-1} X^T y
]
provided (X^T X) is invertible. This is the normal equation. For small feature counts or when you want exact closed-form solution, this is convenient; for large-scale problems, iterative optimizers like gradient descent are often preferred. See a concise derivation here. ([eli.thegreenplace.net][3])


# 4] Gradient descent 

Gradient Descent is an iterative optimization algorithm used to minimize a cost function by adjusting model parameters in the direction of the steepest descent of the function’s gradient. Gradient descent iteratively updates parameters to reduce the cost. The gradient of (J) w.r.t. (\theta_j) is
[
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}.
]

Update rule (batch gradient descent):
[
\theta := \theta - \alpha \nabla_\theta J(\theta)
]
which expands to
[
\theta_j := \theta_j - \alpha \cdot \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
]
for each (j). (\alpha) is the learning rate.

Variants:

* Batch gradient descent: compute gradient on full dataset per update. Stable but can be slow per step. ([DataCamp][4])
* Stochastic gradient descent (SGD): update using each individual example. Faster updates, more noisy, good for online/streaming settings. ([IBM][5])
* Mini-batch: compromise — use small batches (e.g., 32, 64). Works well in practice and is the de-facto choice for many ML problems. ([ruder.io][6])

Because linear regression's cost is convex, gradient descent will converge to the same global minimum (no local minima issues). ([cs229.stanford.edu][2])


### 4.1 — Practical items: learning rate, scaling, regularization, convergence

**Learning rate (\alpha):**

* Too large: divergence or oscillation.
* Too small: painfully slow convergence.
  A typical tactic: start with a moderate (\alpha) (e.g., 0.01 — problem dependent), monitor loss, and use learning-rate schedules or adaptive optimizers for automation. ([ruder.io][6])

**Feature scaling / normalization:**
Gradient descent benefits a lot from scaling features to similar ranges (standardization or min-max). If features vary by orders of magnitude, gradients move in imbalanced directions and convergence slows.

### 4.2 **Regularization (Ridge / L2, Lasso / L1):**
To reduce overfitting and stabilize solutions when features are correlated, add penalty terms:

* Ridge (L2): (J(\theta) + \frac{\lambda}{2m}|\theta|_2^2)
* Lasso (L1): (J(\theta) + \frac{\lambda}{m}|\theta|_1)
  Regularization also helps when (X^T X) is nearly singular. Scikit-learn has efficient solvers for these. ([scikit-learn.org][1])

**Convergence checks:**

* Monitor training loss vs iterations.
* Stop when the absolute or relative change in loss falls below a tolerance or after a max number of epochs.
* Use learning curves and validation set to detect under/overfitting.

### 4.3 — Common pitfalls and tips

* Not scaling features: slows convergence and makes choosing a good learning rate hard.
* Choosing learning rate by guesswork: use log-scale sweep and pick the largest stable (\alpha).
* Not monitoring validation error: you might overfit without noticing.
* Forgetting the bias term: include a column of ones or use intercept handling in libraries.
* Using normal equation on very large feature sets: if (n) large, inversion of (X^T X) is expensive and numerically unstable; use iterative solvers or regularization. ([eli.thegreenplace.net][3])
