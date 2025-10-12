# Question 2: Effect of Feature Scaling on Optimisation

## Observations on Feature Scaling

This experiment clearly demonstrates the critical importance of feature scaling for the efficient optimization of machine learning models. We used full-batch gradient descent for linear regression on a dataset with a large feature scale (`x` ranging from 0 to 1000).

### Part 1: Without Feature Scaling

* **Convergence:** The optimizer struggled immensely to converge. It required an **extremely large number of iterations** to meet the convergence criterion ($\|\theta_t - \theta^*\| < 0.001$).
* **Learning Rate:** A very small learning rate had to be used. Any reasonably large learning rate caused the loss to diverge immediately. This is because the loss surface was shaped like a very narrow, elongated ellipse. The gradients with respect to the weight were much larger than the gradients with respect to the bias, causing the updates to oscillate wildly across the narrow valley of the loss function instead of moving along it towards the minimum.
* **Loss Plot:** The MSE loss vs. iterations plot showed a very slow decrease. Even after many iterations, the loss decreased in tiny increments, highlighting the inefficiency of the optimization process.



---

### Part 2: With Z-Score Normalization

After applying Z-score normalization ($x_{scaled} = \frac{x - \mu_x}{\sigma_x}$), the feature `x` was transformed to have a mean of 0 and a standard deviation of 1.

* **Convergence:** The optimizer converged **dramatically faster**. The number of iterations required to reach the same $\epsilon$-neighborhood of the minimizer was reduced by several orders of magnitude.
* **Learning Rate:** A much larger, more effective learning rate could be used without the risk of divergence. The feature scaling reshaped the loss surface to be more spherical or well-conditioned. This allowed the gradients to point more directly toward the minimum.
* **Loss Plot:** The MSE loss vs. iterations plot showed a steep and rapid decline, quickly flattening out as the model converged to the optimal parameters.

**Conclusion:** Feature scaling is not just a "nice-to-have" but a crucial preprocessing step for gradient-based optimization algorithms. It prevents certain features from dominating the gradient calculation, stabilizes the learning process, and allows the optimizer to find the minimum much more efficiently, saving significant computational time.
