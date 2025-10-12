# Question 5: Logistic Regression in PyTorch

## Observations on PyTorch Logistic Regression Implementation

A custom `LogisticTorch` class was implemented from scratch using PyTorch and compared against `sklearn.linear_model.LogisticRegression` on the `make_moons` dataset.

### Performance Comparison

* **Accuracy:** Both the custom `LogisticTorch` model and the `sklearn` model achieved very similar accuracy scores, typically around **85-88%**. Neither model could achieve perfect accuracy.

* **Reason for Imperfect Accuracy:** The `make_moons` dataset is inherently **non-linearly separable**. The data is shaped like two interlocking crescent moons. Logistic Regression, by its nature, is a linear classifier. It can only create a single straight line (or hyperplane in higher dimensions) to separate the classes. No matter where this line is placed, it will inevitably misclassify some points from both moon shapes. This fundamental limitation of the model, not the implementation, is the reason for the imperfect accuracy.

### Decision Boundary

* The decision boundary plots for both models were, as expected, **linear**. They both produced a straight line that attempted to best separate the two classes. The orientation and position of this line were nearly identical for both the PyTorch and scikit-learn implementations, confirming that our custom model was behaving correctly and converging to the same optimal linear solution as the established library.



### Loss Curve

* The loss curve for the `LogisticTorch` model, plotting the Binary Cross-Entropy loss against the number of training epochs, showed a **smooth, downward trend**. The loss decreased rapidly in the initial epochs and then gradually flattened out as the model's parameters converged to their optimal values. This curve is a clear indicator that the gradient descent optimization process was working correctly and the model was successfully learning from the data.

**Conclusion:** The from-scratch PyTorch implementation successfully replicated the behavior of the standard `sklearn` logistic regression model. Both models demonstrated the inherent limitations of a linear classifier when applied to non-linear data. The exercise confirmed a solid understanding of the underlying mechanics of logistic regression, including the forward pass (linear layer + sigmoid), loss calculation (BCE Loss), and backpropagation (gradient descent).
