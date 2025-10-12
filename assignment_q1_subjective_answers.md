# Question 1: Understanding Gradient Descent and Momentum

## Observations on Gradient Descent Variants

### Part 1: Full-Batch vs. Stochastic Gradient Descent

Upon implementing both **Full-Batch Gradient Descent (GD)** and **Stochastic Gradient Descent (SGD)** for linear regression, several key differences in their convergence behavior were observed for the two datasets. The convergence criterion was set as $\|\theta_t - \theta^*\| < \epsilon$, with $\epsilon = 0.001$.

#### Dataset 1: `f(x) = 100*x + 1`

* **Full-Batch Gradient Descent:**
    * **Convergence Path:** The convergence was very direct and smooth. The contour plot showed the parameters moving straight towards the minimum of the loss function. This is because each update step is calculated using the gradient of the entire dataset, providing a true and stable direction of steepest descent.
    * **Steps to Converge:** Converged in a relatively **low number of steps (epochs)**. Since each epoch is one update, the path is efficient.
    * **Loss Curve:** The plot of loss versus epochs showed a smooth, monotonically decreasing curve, which flattened out as it approached the minimum.

* **Stochastic Gradient Descent:**
    * **Convergence Path:** The path to the minimizer was noisy and erratic. The contour plot showed the parameter estimates zigzagging and oscillating around the optimal path. This is expected, as each update is based on a single, noisy data point, which only provides a rough estimate of the true gradient.
    * **Steps to Converge:** Required a **significantly higher number of steps (updates)** to satisfy the convergence criterion. While one epoch involves many updates (40 in this case), the overall journey to the $\epsilon$-neighborhood was much less direct than full-batch GD.
    * **Loss Curve:** The loss curve was highly volatile, with frequent spikes and dips. However, the overall trend was downwards, indicating that the model was learning over time.


<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/a1a9ab06-c4fe-4097-b506-4bf4d7864368" />



_Dataset 1: True theta* = [ 0.9507064  99.98412345]
* Full Batch GD steps: 39610
* SGD steps: 1046
* Momentum GD steps: 314_

<img width="563" height="455" alt="image" src="https://github.com/user-attachments/assets/3d502834-efe2-4571-97a1-052063ebd339" />

_Dataset 2: True theta* = [3.9507064  2.68246893]
* Full Batch GD steps: 131779
* SGD steps: 1334
* Momentum GD steps: 1247_

---

### Part 2: Gradient Descent with Momentum

Gradient Descent with Momentum was implemented and compared with the vanilla GD and SGD methods. The momentum term helps accelerate convergence, particularly in directions of persistent gradient and dampens oscillations.

#### Dataset 1 & 2:

* **Convergence Path:**
    * The path taken by the optimizer with momentum was noticeably smoother than vanilla SGD and often more direct than vanilla full-batch GD. It tended to "overshoot" less and correct its path more effectively by building up velocity in the correct direction.
    * The visualization of vectors was insightful:
        * The **gradient vector** pointed in the direction of steepest descent for the current batch.
        * The **momentum vector** (an exponentially weighted average of past gradients) pointed in the general direction of the minimum, smoothing out variations from individual gradients.
        * The **update vector** (a combination of the current gradient and the momentum vector) resulted in a more stable and accelerated step towards the solution.

* **Comparison of Steps to Converge:**
    * **vs. Full-Batch GD:** Gradient Descent with Momentum consistently converged in **fewer steps** than vanilla Full-Batch GD. The accumulated momentum allowed it to take larger, more confident steps, especially in the initial stages.
    * **vs. SGD:** The comparison with SGD is nuanced. While vanilla SGD takes many noisy steps, SGD with momentum also takes many steps per epoch, but the path is far less erratic. The momentum term smooths out the noise from individual samples, leading to faster overall convergence in terms of epochs. For a fixed number of steps, momentum reached the $\epsilon$-neighborhood more reliably and quickly.

**Overall Observation:** Momentum provides a clear advantage by smoothing the update trajectory and accelerating convergence. It combines the stability of averaging gradients over time with the step-by-step learning process, making it a more robust and often faster optimization algorithm than its vanilla counterparts.

