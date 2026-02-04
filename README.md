# PoC_projects

## 1] Linear Regression

### 1.1 — What and why

Linear regression is a supervised machine learning algorithm that learns from labelled data and models the relationship between input features and a target variable using a linear function. The goal is to find the most optimized straight line or hyperplane that best fits the data and can be used to make predictions on unseen data.

It is one of the simplest regression algorithms and is widely used because it is fast, interpretable, and mathematically well understood. Linear regression is often used as a baseline model and as a foundation for understanding more complex machine learning techniques.


### 1.2 — Linear regression — intuition and math

Assume a dataset with `m` training examples:

```
(x(1), y(1)), (x(2), y(2)), ..., (x(m), y(m))
```

Each input example `x` has `n` features. To handle the intercept term, we add a bias feature:

```
x0 = 1
```

The hypothesis (prediction function) of linear regression is:

```
y_hat = h_theta(x) = theta^T * x
```

Expanded form:

```
y_hat = theta0 * x0 + theta1 * x1 + theta2 * x2 + ... + thetan * xn
```

Here:

* `theta` is the parameter vector
* `theta0` is the bias (intercept)
* `thetai` represents the weight for feature `xi`

The objective is to find the values of `theta` such that the predicted values `y_hat` are as close as possible to the true values `y`.

---

### Mean Squared Error (Cost Function)

To measure how well the model performs, we use the Mean Squared Error (MSE) cost function:

```
J(theta) = (1 / (2m)) * Σ (h_theta(x(i)) - y(i))^2
```

Where:

* `m` is the number of training examples
* `h_theta(x(i))` is the predicted value
* `y(i)` is the actual value

The factor `1 / (2m)` is used for mathematical convenience when taking derivatives.

Minimizing this cost function results in the best-fit linear model under the squared-error assumption.

Reference: Stanford CS229 Lecture Notes [2]


### 1.3 — Cost function and normal equation

The cost function `J(theta)` for linear regression is **convex**, meaning it has only one global minimum and no local minima.

Because of this property, the optimal parameters can be solved analytically using the **Normal Equation**.

In matrix form:

```
theta* = (X^T X)^(-1) X^T y
```

Where:

* `X` is the design matrix of shape `(m, n+1)`
* `y` is the target vector of shape `(m, 1)`
* `X^T` is the transpose of `X`

This method directly computes the optimal parameters without iteration.

Pros:

* No learning rate required
* Exact solution

Cons:

* Computationally expensive for large feature sets
* Matrix inversion can be numerically unstable

For large datasets or high-dimensional problems, iterative methods like gradient descent are preferred.

Reference: Eli Bendersky [3]


## 2] Gradient Descent

Gradient Descent is an iterative optimization algorithm used to minimize a cost function by updating model parameters in the direction of the steepest decrease of the function.

Instead of solving for parameters analytically, gradient descent gradually moves toward the minimum by taking small steps proportional to the negative gradient.


### Gradient of the cost function

The partial derivative of the cost function with respect to a parameter `theta_j` is:

```
∂J(theta) / ∂theta_j = (1 / m) * Σ (h_theta(x(i)) - y(i)) * x_j(i)
```

This gradient tells us how much the cost function changes when `theta_j` is adjusted.


### Update rule (Batch Gradient Descent)

The general update rule is:

```
theta = theta - alpha * gradient
```

Expanded per parameter:

```
theta_j = theta_j - alpha * (1 / m) * Σ (h_theta(x(i)) - y(i)) * x_j(i)
```

Where:

* `alpha` is the learning rate
* All parameters are updated simultaneously


### Variants of Gradient Descent

* **Batch Gradient Descent**
  Uses the entire dataset to compute gradients at each step. Stable but slow for large datasets.
  Reference: DataCamp [4]

* **Stochastic Gradient Descent (SGD)**
  Updates parameters using one training example at a time. Faster but noisy updates.
  Reference: IBM [5]

* **Mini-batch Gradient Descent**
  Uses small batches (e.g. 32 or 64 samples). Balances speed and stability.
  This is the most commonly used variant in practice.
  Reference: Ruder [6]

Since the linear regression cost function is convex, gradient descent is guaranteed to converge to the global minimum.

Reference: Stanford CS229 [2]


### 2.1 — Practical items: learning rate, scaling, regularization, convergence

#### Learning rate (alpha)

* Too large: cost function may diverge or oscillate
* Too small: convergence becomes very slow

A common strategy is to start with a moderate value such as `0.01`, monitor the loss, and adjust as needed.


#### Feature scaling / normalization

Gradient descent converges much faster when features are on similar scales.

Common techniques:

* Standardization (zero mean, unit variance)
* Min-max scaling

Without scaling, parameters associated with large-valued features dominate updates and slow convergence.

### 2.2 — Regularization (Ridge / L2, Lasso / L1)

Regularization helps prevent overfitting and improves numerical stability.

**Ridge Regression (L2):**

```
J(theta) + (lambda / (2m)) * ||theta||^2
```

**Lasso Regression (L1):**

```
J(theta) + (lambda / m) * ||theta||_1
```

Where:

* `lambda` controls the strength of regularization

Regularization is especially useful when features are highly correlated or when `X^T X` is close to singular.

### Convergence checks

* Plot cost vs iterations
* Stop training when loss improvement falls below a threshold
* Use validation curves to detect overfitting or underfitting

### 2.3 — Common pitfalls and tips

* Not scaling features leads to slow convergence
* Poor learning rate selection causes divergence or slow training
* Ignoring validation error can hide overfitting
* Forgetting the bias term leads to incorrect models
* Using the normal equation on very large feature sets is computationally expensive

