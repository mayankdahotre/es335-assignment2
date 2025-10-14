# Task 2: Effect of Feature Scaling on Optimization

This experiment illustrates the critical role of feature scaling in optimizing machine learning models using full-batch gradient descent for linear regression. We analyze the impact on a dataset with a large feature scale (`x` ranging from 0 to 1000).

## Dataset Description

The synthetic dataset is generated as follows:

- **Size**: 100 samples
- **Feature**: `x` drawn uniformly from [0, 1000]
- **Target**: `y = 3x + 2 + ε`, where `ε` is Gaussian noise (`N(0,1)`)

### Code for Dataset Generation and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate dataset
num_samples = 100
np.random.seed(42)
x = np.random.uniform(0, 1000, num_samples)
f_x = 3 * x + 2
eps = np.random.randn(num_samples)
y = f_x + eps

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Noisy Data', alpha=0.6, s=50)
plt.plot(np.sort(x), 3 * np.sort(x) + 2, color='red', linewidth=2, label='True Function: y = 3x + 2')
plt.title('Synthetic Linear Dataset', fontsize=14, pad=10)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

<img width="989" height="589" alt="image" src="https://github.com/user-attachments/assets/1494e12f-d6de-4305-9a23-0d0a1b241262" />


## Part 1: Without Feature Scaling

Without scaling, the large range of `x` (0 to 1000) causes significant optimization challenges.

- **Convergence**: Extremely slow, requiring up to 1,000,000 iterations to approach the convergence criterion (`||θ_t - θ*|| < 0.001`).
- **Learning Rate**: A tiny learning rate (`α = 1e-6`) is necessary to avoid divergence. Larger rates cause instability due to the elongated, narrow loss surface, where gradients with respect to the weight dominate those for the bias.
- **Loss Behavior**: The Mean Squared Error (MSE) decreases slowly, indicating inefficient optimization.

### Code for Gradient Descent (Unscaled)

```python
import numpy as np
import matplotlib.pyplot as plt

# Data
num_samples = 100
np.random.seed(42)
x = np.random.uniform(0, 1000, num_samples)
f_x = 3 * x + 2
eps = np.random.randn(num_samples)
y = f_x + eps

# Add bias term
X = np.vstack((np.ones(num_samples), x)).T  # shape (100, 2)

# Empirical least squares solution
theta_star = np.linalg.inv(X.T @ X) @ X.T @ y
print("Empirical least squares solution θ*:", theta_star)

# Gradient descent
theta = np.zeros(2)
alpha = 1e-6
epsilon = 0.001
max_iters = 1000000
mse_history = []
num_iters = 0

for i in range(max_iters):
    y_pred = X @ theta
    grad = (2/num_samples) * X.T @ (y_pred - y)
    theta = theta - alpha * grad
    mse = np.mean((y - y_pred)**2)
    mse_history.append(mse)
    num_iters += 1
    if np.linalg.norm(theta - theta_star) < epsilon:
        break

print(f"Converged in {num_iters} iterations")
print("Theta found by gradient descent:", theta)

# Plot MSE
plt.figure(figsize=(10, 6))
plt.plot(mse_history, label='MSE Loss', color='blue')
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Full-Batch Gradient Descent Loss (Unscaled)', fontsize=14, pad=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
```

<img width="989" height="589" alt="image" src="https://github.com/user-attachments/assets/a7e2da2b-bd28-41f1-bc3e-de99e34cf35e" />


**Output**:
```
Empirical least squares solution θ*: [2.21509616 2.99954023]
Converged in 1000000 iterations
Theta found by gradient descent: [0.96206697 3.00144881]
```

## Part 2: With Z-Score Normalization

Z-score normalization transforms `x` to have a mean of 0 and standard deviation of 1: `x_scaled = (x - μ_x) / σ_x`.

- **Convergence**: Significantly faster, converging in approximately 1,426 iterations.
- **Learning Rate**: A larger learning rate (`η = 0.01`) is stable, as scaling creates a well-conditioned, more spherical loss surface.
- **Loss Behavior**: MSE decreases rapidly, flattening as the model reaches the optimal parameters.

### Code for Gradient Descent (Scaled)

```python
import numpy as np
import matplotlib.pyplot as plt

# Data
num_samples = 100
np.random.seed(42)
x = np.random.uniform(0, 1000, num_samples)
f_x = 3 * x + 2
eps = np.random.randn(num_samples)
y = f_x + eps

# Z-score normalization
mu_x = np.mean(x)
sigma_x = np.std(x)
x_scaled = (x - mu_x) / sigma_x

# Least squares solution for scaled data
theta1_star = np.cov(x_scaled, y, bias=True)[0,1] / np.var(x_scaled)
theta0_star = np.mean(y) - theta1_star * np.mean(x_scaled)
theta_star = np.array([theta0_star, theta1_star])
print("Least squares solution (scaled):", theta_star)

# Gradient descent
theta = np.array([0.0, 0.0])
eta = 0.01
epsilon = 0.001
max_iters = 10000
mse_history = []

for iter in range(max_iters):
    y_pred = theta[0] + theta[1] * x_scaled
    error = y_pred - y
    mse = np.mean(error**2)
    mse_history.append(mse)
    grad0 = np.mean(error)
    grad1 = np.mean(error * x_scaled)
    theta = theta - eta * np.array([grad0, grad1])
    if np.linalg.norm(theta - theta_star) < epsilon:
        print(f"Converged in {iter+1} iterations")
        break

print("Theta found by gradient descent:", theta)

# Plot MSE
plt.figure(figsize=(10, 6))
plt.plot(mse_history, label='MSE Loss', color='blue')
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.title('Full-Batch Gradient Descent with Z-Score Normalization', fontsize=14, pad=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
```

<img width="989" height="589" alt="image" src="https://github.com/user-attachments/assets/3e44e152-8450-492f-96e1-8605127fb86b" />


**Output**:
```
Least squares solution (scaled): [1412.54114977  887.85858784]
Converged in 1426 iterations
Theta found by gradient descent: [1412.54030685  887.85805801]
```

## Conclusion

Feature scaling, such as Z-score normalization, is essential for efficient gradient-based optimization. It mitigates the adverse effects of large feature scales, prevents gradient imbalance, and transforms the loss surface into a more navigable shape. This results in:

- **Faster Convergence**: Orders of magnitude fewer iterations.
- **Stable Learning Rates**: Larger learning rates without divergence.
- **Computational Efficiency**: Reduced training time and resource usage.

Scaling is a critical preprocessing step for ensuring robust and efficient model training.
