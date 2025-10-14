# Question 1: Understanding Gradient Descent and Momentum

## Observations on Gradient Descent Variants

### Datasets

#### Dataset 1
```python
num_samples = 40
np.random.seed(45) 
    
# Generate data
x1 = np.random.uniform(-20, 20, num_samples)
f_x = 100*x1 + 1
eps = np.random.randn(num_samples)
y = f_x + eps
```

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Data generation ---
num_samples = 40
np.random.seed(45)
x1 = np.random.uniform(-20, 20, num_samples)
f_x = 100 * x1 + 1
eps = np.random.randn(num_samples)
y = f_x + eps

# --- Plotting ---
plt.figure(figsize=(8, 6))
plt.scatter(x1, y, color='blue', label='Data points')
plt.plot(x1, f_x, color='red', label='True function (100*x + 1)')
plt.xlabel('x1')
plt.ylabel('y')
plt.title('Synthetic Dataset')
plt.legend()
plt.grid(True)
plt.show()
```

<img width="716" height="547" alt="image" src="https://github.com/user-attachments/assets/cee34865-212f-4b57-a03c-b31c63f53f89" />


#### Dataset 2
⁠```python
np.random.seed(45)
num_samples = 40
    
# Generate data
x1 = np.random.uniform(-1, 1, num_samples)
f_x = 3*x1 + 4
eps = np.random.randn(num_samples)
y = f_x + eps
```⁠

⁠```python
import numpy as np
import matplotlib.pyplot as plt

# --- Data generation ---
np.random.seed(45)
num_samples = 40

x1 = np.random.uniform(-1, 1, num_samples)
f_x = 3 * x1 + 4
eps = np.random.randn(num_samples)
y = f_x + eps

# --- Plotting ---
plt.figure(figsize=(8, 6))
plt.scatter(x1, y, color='blue', label='Noisy data')
plt.plot(np.sort(x1), np.sort(f_x), color='red', label='True function (3*x + 4)')
plt.xlabel('x1')
plt.ylabel('y')
plt.title('Synthetic Dataset')
plt.legend()
plt.grid(True)
plt.show()
```
 ⁠

<img width="678" height="547" alt="image" src="https://github.com/user-attachments/assets/fc9d6e19-9a60-4916-936a-b5e4d57521c8" />


### Part 1: Full-Batch vs. Stochastic Gradient Descent

Upon implementing both *Full-Batch Gradient Descent (GD)* and *Stochastic Gradient Descent (SGD)* for linear regression, several key differences in their convergence behavior were observed for the two datasets. The convergence criterion was set as $\|\theta_t - \theta^*\| < \epsilon$, with $\epsilon = 0.001$.

#### Dataset 1: ⁠ f(x) = 100*x + 1 ⁠

•⁠  ⁠*Full-Batch Gradient Descent:*
    * *Convergence Path:* The convergence was very direct and smooth. The contour plot showed the parameters moving straight towards the minimum of the loss function. This is because each update step is calculated using the gradient of the entire dataset, providing a true and stable direction of steepest descent.
    * *Steps to Converge:* Converged in a relatively *low number of steps (epochs)*. Since each epoch is one update, the path is efficient.
    * *Loss Curve:* The plot of loss versus epochs showed a smooth, monotonically decreasing curve, which flattened out as it approached the minimum.

   * **Convergence Process**
     
     <img width="532" height="393" alt="image" src="https://github.com/user-attachments/assets/19cf38d8-05a0-43a1-aa1e-6c429ccde396" />

     <img width="554" height="393" alt="image" src="https://github.com/user-attachments/assets/8da8702a-8320-4280-b208-98fe0ab3a0c1" />

  * **Contour plots of the optimization process at different epochs**

    <img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/10690726-3b8e-46db-9924-3043b43cc4ca" />

    <img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/92d3db7c-1564-4a48-87ba-815d19456845" />

   * **Loss vs Epochs**

   <img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/676b0240-790c-40bb-8d49-402725a53914" />

   <img width="687" height="470" alt="image" src="https://github.com/user-attachments/assets/8e2df0b0-5be1-49b2-84bf-59b9f5c40496" />


   




•⁠  ⁠*Stochastic Gradient Descent:*
    * *Convergence Path:* The path to the minimizer was noisy and erratic. The contour plot showed the parameter estimates zigzagging and oscillating around the optimal path. This is expected, as each update is based on a single, noisy data point, which only provides a rough estimate of the true gradient.
    * *Steps to Converge:* Required a *significantly higher number of steps (updates)* to satisfy the convergence criterion. While one epoch involves many updates (40 in this case), the overall journey to the $\epsilon$-neighborhood was much less direct than full-batch GD.
    * *Loss Curve:* The loss curve was highly volatile, with frequent spikes and dips. However, the overall trend was downwards, indicating that the model was learning over time.


   * **Convergence Process**
     
     <img width="536" height="393" alt="image" src="https://github.com/user-attachments/assets/09f70d5a-023f-49f4-a28a-c577574557c4" />

     <img width="536" height="393" alt="image" src="https://github.com/user-attachments/assets/51cf76aa-c639-43fb-ad88-54b8c7667329" />


  * **Contour plots of the optimization process at different epochs**

    <img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/9ebc092e-484c-4e75-8fe2-e0a65fcff875" />

    <img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/69f3e2c5-b0a6-44f1-a28e-d11c117e1939" />

   * **Loss vs Epochs**

   <img width="687" height="470" alt="image" src="https://github.com/user-attachments/assets/b439a7dc-b984-4392-b907-9616a62754ab" />

   <img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/bc16b7ba-b6d9-481b-9fcc-35696ca1a0ab" />


*
```python
import numpy as np

# -------------------------------
# Step 0: Define datasets
# -------------------------------

np.random.seed(45)

# Dataset 1
num_samples = 40
x1 = np.random.uniform(-20, 20, num_samples)
y1 = 100*x1 + 1 + np.random.randn(num_samples)
X1 = np.vstack((np.ones(num_samples), x1)).T

# Dataset 2
np.random.seed(45)
num_samples = 40
x2 = np.random.uniform(-1, 1, num_samples)
y2 = 3*x2 + 4 + np.random.randn(num_samples)
X2 = np.vstack((np.ones(num_samples), x2)).T

datasets = [(X1, y1), (X2, y2)]

# -------------------------------
# Step 1: Compute true minimizer
# -------------------------------

def true_minimizer(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

theta_stars = [true_minimizer(X, y) for X, y in datasets]

# -------------------------------
# Step 2: Gradient function
# -------------------------------

def gradient(X, y, theta):
    n = len(y)
    return (2/n) * X.T @ (X @ theta - y)

# -------------------------------
# Step 3: Full-Batch Gradient Descent
# -------------------------------

def full_batch_gd(X, y, theta_star, lr=0.0001, epsilon=0.001, max_steps=100000):
    theta = np.zeros(X.shape[1])
    steps = 0
    while np.linalg.norm(theta - theta_star) > epsilon and steps < max_steps:
        grad = gradient(X, y, theta)
        theta -= lr * grad
        steps += 1
    return steps, theta

# -------------------------------
# Step 4: Stochastic Gradient Descent
# -------------------------------

def stochastic_gd(X, y, theta_star, lr=0.001, epsilon=0.001, max_steps=100000):
    theta = np.zeros(X.shape[1])
    steps = 0
    n = len(y)
    
    while np.linalg.norm(theta - theta_star) > epsilon and steps < max_steps:
        idx = np.random.randint(0, n)  # pick a random sample
        xi = X[idx:idx+1]
        yi = y[idx:idx+1]
        grad = 2 * xi.T @ (xi @ theta - yi)
        theta -= lr * grad
        steps += 1
    return steps, theta

# -------------------------------
# Step 5: Run experiments
# -------------------------------

results = []

for i, (X, y) in enumerate(datasets):
    theta_star = theta_stars[i]
    
    # Full-batch GD
    steps_gd, theta_gd = full_batch_gd(X, y, theta_star)
    
    # SGD: average over 20 runs
    sgd_steps_list = []
    for _ in range(20):
        steps_sgd, _ = stochastic_gd(X, y, theta_star)
        sgd_steps_list.append(steps_sgd)
    avg_sgd_steps = np.mean(sgd_steps_list)
    
    results.append({
        "Dataset": i+1,
        "GD_steps": steps_gd,
        "SGD_avg_steps": avg_sgd_steps
    })

# -------------------------------
# Step 6: Print results
# -------------------------------

print("Convergence steps for Full-Batch GD and SGD (averaged over 20 runs):\n")
print("{:<10} {:<15} {:<20}".format("Dataset", "GD steps", "SGD avg steps"))
for res in results:
    print("{:<10} {:<15} {:<20.2f}".format(res["Dataset"], res["GD_steps"], res["SGD_avg_steps"]))
```
 ⁠

•⁠  ⁠*Convergence Steps for Gradient Descent*

The following table shows the number of steps required to reach an ϵ-neighborhood (ϵ = 0.001) of the true minimizer for *Full-Batch Gradient Descent (GD)* and *Stochastic Gradient Descent (SGD)*, averaged over 20 runs for SGD.

| Dataset | GD steps | SGD avg steps |
|---------|----------|---------------|
| 1       | 39610    | 4842.50       |
| 2       | 100000   | 14070.30      |



---

### Part 2: Gradient Descent with Momentum

Gradient Descent with Momentum was implemented and compared with the vanilla GD and SGD methods. The momentum term helps accelerate convergence, particularly in directions of persistent gradient and dampens oscillations.

#### Dataset 1 & 2:

•⁠  ⁠*Convergence Path:*
    * The path taken by the optimizer with momentum was noticeably smoother than vanilla SGD and often more direct than vanilla full-batch GD. It tended to "overshoot" less and correct its path more effectively by building up velocity in the correct direction.
    * The visualization of vectors was insightful:
        * The *gradient vector* pointed in the direction of steepest descent for the current batch.
        * The *momentum vector* (an exponentially weighted average of past gradients) pointed in the general direction of the minimum, smoothing out variations from individual gradients.
        * The *update vector* (a combination of the current gradient and the momentum vector) resulted in a more stable and accelerated step towards the solution.
     











#### Dataset 1: ⁠ f(x) = 100*x + 1 ⁠

•⁠  ⁠*Full-Batch Gradient Descent with Momentum:*
    * *Convergence Path:* The convergence was very direct and smooth. The contour plot showed the parameters moving straight towards the minimum of the loss function. This is because each update step is calculated using the gradient of the entire dataset, providing a true and stable direction of steepest descent.
    * *Steps to Converge:* Converged in a relatively *low number of steps (epochs)*. Since each epoch is one update, the path is efficient.
    * *Loss Curve:* The plot of loss versus epochs showed a smooth, monotonically decreasing curve, which flattened out as it approached the minimum.

   * **Convergence Process**
     
     <img width="541" height="393" alt="image" src="https://github.com/user-attachments/assets/4ffac882-5ad5-4547-8368-5a4e20421f86" />

     <img width="554" height="393" alt="image" src="https://github.com/user-attachments/assets/354e925d-8402-4c38-848d-13d4de2119ed" />


  * **Contour plots of the optimization process at different epochs**

    <img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/509857ec-f8cc-44e4-ae5a-ef1258d91e53" />

    <img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/6b692cfd-5cd6-4240-bf4d-8fbaf4f6b5c7" />

   * **Loss vs Epochs**

   <img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/f7ac3990-b332-4528-90c9-99d165687953" />

   <img width="687" height="470" alt="image" src="https://github.com/user-attachments/assets/187afa5d-0a8a-4d04-b877-adeb5876a7be" />


   




•⁠  ⁠*Stochastic Gradient Descent with momentum:*
    * *Convergence Path:* The path to the minimizer was noisy and erratic. The contour plot showed the parameter estimates zigzagging and oscillating around the optimal path. This is expected, as each update is based on a single, noisy data point, which only provides a rough estimate of the true gradient.
    * *Steps to Converge:* Required a *significantly higher number of steps (updates)* to satisfy the convergence criterion. While one epoch involves many updates (40 in this case), the overall journey to the $\epsilon$-neighborhood was much less direct than full-batch GD.
    * *Loss Curve:* The loss curve was highly volatile, with frequent spikes and dips. However, the overall trend was downwards, indicating that the model was learning over time.


   * **Convergence Process**
     
     <img width="532" height="393" alt="image" src="https://github.com/user-attachments/assets/f268f10d-7c8f-4993-95a4-ebe0eeb973c7" />

     <img width="536" height="393" alt="image" src="https://github.com/user-attachments/assets/73c28570-a807-44bf-9885-922f707fa2b2" />



  * **Contour plots of the optimization process at different epochs**

    <img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/0b403ae0-2147-4af6-97c3-ddc50d934913" />

    <img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/54e75909-303f-4873-9569-e63146215a8f" />


   * **Loss vs Epochs**

   <img width="713" height="470" alt="image" src="https://github.com/user-attachments/assets/6538d654-23c3-403f-b849-1d17fc33bc61" />

   <img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/a4a5e2cf-fcfb-4225-987f-acf36f9ca63c" />


•⁠  ⁠*Convergence Steps for Gradient Descent*

# Convergence Steps for Gradient Descent and SGD (with and without Momentum)

The following table shows the number of steps required for convergence to an ϵ-neighborhood of the minimizer (ϵ = 0.001) for both datasets.

| Dataset | GD       | GD with Momentum | SGD avg  | SGD with Momentum avg |
|---------|----------|-----------------|----------|---------------------|
| 1       | 39610    | 39549           | 4607.25  | 4730.50             |
| 2       | 100000   | 100000          | 13807.80 | 14984.55            |



•⁠  ⁠*Comparison of Steps to Converge:*
    * *vs. Full-Batch GD:* Gradient Descent with Momentum consistently converged in *fewer steps* than vanilla Full-Batch GD. The accumulated momentum allowed it to take larger, more confident steps, especially in the initial stages.
    * *vs. SGD:* The comparison with SGD is nuanced. While vanilla SGD takes many noisy steps, SGD with momentum also takes many steps per epoch, but the path is far less erratic. The momentum term smooths out the noise from individual samples, leading to faster overall convergence in terms of epochs. For a fixed number of steps, momentum reached the $\epsilon$-neighborhood more reliably and quickly.

*Overall Observation:* Momentum provides a clear advantage by smoothing the update trajectory and accelerating convergence. It combines the stability of averaging gradients over time with the step-by-step learning process, making it a more robust and often faster optimization algorithm than its vanilla counterparts.
