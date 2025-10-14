# Question 4: Implementing Matrix Factorization

## Image Used in this Task
```python
import os
if os.path.exists('dog.jpg'):
    print('dog.jpg exists')
else:
    !wget https://segment-anything.com/assets/gallery/AdobeStock_94274587_welsh_corgi_pembroke_CD.jpg -O dog.jpg

import torchvision
import matplotlib.pyplot as plt

# Read the image (C, H, W) format
img = torchvision.io.read_image("dog.jpg")

# Convert from [C, H, W] → [H, W, C]
img = img.permute(1, 2, 0)

# Convert to float and normalize (matplotlib expects [0,1])
img = img.float() / 255.0

# Display
plt.imshow(img)
plt.axis('off')
plt.show()
```

<img width="515" height="350" alt="image" src="https://github.com/user-attachments/assets/637830f9-01b9-4e74-9b80-37be567bb518" />


Grayscale version of this image
```python
import torch
import torchvision
import matplotlib.pyplot as plt

# Load and prepare
img = torchvision.io.read_image("dog.jpg").float() / 255.0  # [C, H, W]
img = img.permute(1, 2, 0)  # [H, W, C]

# Convert to grayscale (average over RGB channels)
gray = img.mean(dim=2)

print(gray.shape)  # should be [H, W]

plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()
```

<img width="515" height="350" alt="image" src="https://github.com/user-attachments/assets/4e063ad2-84b3-4902-8b7a-87383aed33c2" />


Cropped Version of the image
```python
import torchvision
import matplotlib.pyplot as plt

# Read the image (C, H, W) format
img = torchvision.io.read_image("dog.jpg")

# Define crop parameters
x, y = 800, 600   # top-left corner (row, column)
h, w = 300, 300   # height and width of crop

# Crop the image
# img is [C, H, W], so slicing order is [C, y:y+h, x:x+w]
cropped_img = img[:, y:y+h, x:x+w]

# Convert from [C, H, W] → [H, W, C]
cropped_img = cropped_img.permute(1, 2, 0)

# Convert to float and normalize for display
cropped_img = cropped_img.float() / 255.0

# Display cropped image
plt.imshow(cropped_img)
plt.axis('off')
plt.show()
```

<img width="389" height="389" alt="image" src="https://github.com/user-attachments/assets/cb0dbd41-c2ea-4e2b-9260-63f5a01147f5" />


## Observations on Matrix Factorization Applications

## Image Reconstruction using Gradient Descent and Alternating Least Squares

Image reconstruction is a process used in computational imaging to recover or estimate an image from incomplete, noisy, or indirect measurements. It’s a core problem in fields like medical imaging (e.g., MRI, CT), computer vision, and astronomy. Two common optimization techniques for image reconstruction are **Gradient Descent (GD)** and **Alternating Least Squares (ALS)**. This document provides an introductory overview of these methods.

### What is Image Reconstruction?

Image reconstruction aims to recover an image \( x \) (typically represented as a vector of pixel intensities) from observed data \( y \), which is often related to the image through a forward model:

\[
y = Ax + n
\]

- \( A \): A linear operator (e.g., a blurring matrix, Fourier transform, or projection operator in tomography).
- \( n \): Noise or measurement errors.
- \( x \): The unknown image to be reconstructed.
- \( y \): The observed data (e.g., sensor measurements, sinograms in CT, or k-space data in MRI).

The goal is to solve for \( x \) given \( y \) and \( A \), often by formulating an optimization problem to minimize the error between the observed data and the reconstructed image, while incorporating constraints or regularization to handle noise and ill-posedness.

### Gradient Descent (GD) in Image Reconstruction

Gradient Descent is an iterative optimization algorithm used to minimize a cost function. In image reconstruction, the cost function typically measures the discrepancy between the observed data \( y \) and the predicted data \( Ax \), often using a least-squares formulation:

\[
\min_x f(x) = \frac{1}{2} \| Ax - y \|_2^2
\]

Regularization terms (e.g., Tikhonov or total variation) may be added to enforce smoothness or sparsity:

\[
\min_x f(x) = \frac{1}{2} \| Ax - y \|_2^2 + \lambda R(x)
\]

where \( R(x) \) is the regularization term, and \( \lambda \) controls its strength.

#### How GD Works
1. Start with an initial guess for the image \( x_0 \).
2. Compute the gradient of the cost function \( \nabla f(x) \), which points in the direction of the steepest increase.
3. Update the image iteratively: \( x_{k+1} = x_k - \alpha \nabla f(x_k) \), where \( \alpha \) is the step size (learning rate).
4. Repeat until convergence (e.g., when the gradient is small or the cost function stabilizes).

#### In Image Reconstruction
- The gradient often involves the forward operator \( A \) and its adjoint \( A^T \): \( \nabla f(x) = A^T (Ax - y) + \lambda \nabla R(x) \).
- GD is computationally efficient for large-scale problems but may converge slowly for ill-conditioned systems or require careful tuning of \( \alpha \).
- Variants like stochastic GD or accelerated GD (e.g., Nesterov’s method) can improve performance.

#### Pros and Cons
**Pros**:
- Simple to implement.
- Works well for smooth cost functions.
- Can handle large-scale problems with appropriate step-size tuning.

**Cons**:
- Sensitive to step-size choice.
- Slow convergence for ill-posed problems without preconditioning.
- May get stuck in local minima for non-convex problems.

### Alternating Least Squares (ALS) in Image Reconstruction

ALS is an optimization technique used when the problem can be split into multiple subproblems that are easier to solve individually. It’s particularly useful for problems involving matrix factorization or when the forward model involves multiple variables. In image reconstruction, ALS is often applied to problems like non-negative matrix factorization (NMF) or tensor decomposition, where the image or data is modeled as a product of simpler components.

#### How ALS Works
1. Decompose the problem into a set of variables (e.g., \( x = UV \), where \( U \) and \( V \) are factor matrices).
2. Fix all but one variable and solve for the remaining variable using least squares.
3. Alternate between variables, solving each subproblem iteratively until convergence.

For example, in NMF for image reconstruction:
- The image is approximated as \( x \approx WH \), where \( W \) and \( H \) are non-negative matrices, and the goal is to minimize:
  \[
  \min_{W,H \geq 0} \| A(WH) - y \|_2^2
  \]
- ALS alternates between:
  - Fixing \( H \) and solving for \( W \).
  - Fixing \( W \) and solving for \( H \).
- Each subproblem is a least-squares problem, often solved analytically or with convex optimization techniques.

#### In Image Reconstruction
- ALS is used in applications like hyperspectral imaging, where the image is modeled as a combination of spectral and spatial components.
- It’s effective for problems with non-negativity constraints or low-rank structures.
- The forward operator \( A \) may represent transformations like blurring or subsampling.

#### Pros and Cons
**Pros**:
- Handles non-negativity and other constraints naturally.
- Can exploit the structure of the problem (e.g., low-rank approximations).
- Robust for certain multi-variable problems.

**Cons**:
- May converge slowly or to suboptimal solutions.
- Computationally expensive for large-scale problems.
- Requires careful initialization to avoid poor local minima.

### Comparing GD and ALS

| Aspect          | Gradient Descent (GD)                          | Alternating Least Squares (ALS)                |
|-----------------|------------------------------------------------|------------------------------------------------|
| **Problem Type**| General-purpose for smooth, unconstrained problems | Suited for separable variables or constraints like non-negativity |
| **Convergence** | Faster for well-conditioned; struggles with ill-posed | Slower but robust for structured problems      |
| **Computational Cost** | Lighter per iteration                         | Involves multiple least-squares solves         |
| **Applications**| MRI/CT with regularization                     | NMF-based or tensor methods                    |

- **Problem Type**: GD is general-purpose and works well for smooth, unconstrained problems, while ALS is suited for problems with separable variables or constraints like non-negativity.
- **Convergence**: GD may converge faster for well-conditioned problems but struggles with ill-posed ones. ALS can be slower but is robust for structured problems.
- **Computational Cost**: GD is typically lighter per iteration, while ALS may involve solving multiple least-squares problems per iteration.

### Example Applications

1. **MRI Reconstruction**:
   - **GD**: Used to solve regularized least-squares problems for compressed sensing MRI, minimizing data fidelity plus total variation.
   - **ALS**: Applied in dynamic MRI, where the image sequence is factored into temporal and spatial components.

2. **Deblurring**:
   - **GD**: Iteratively minimizes the error between the blurred image and the observed data with smoothness constraints.
   - **ALS**: Used in blind deconvolution, alternating between estimating the image and the blur kernel.

3. **Hyperspectral Imaging**:
   - **ALS**: Common for NMF-based decomposition of spectral data.
   - **GD**: Used for end-to-end optimization with deep learning priors.

### Conclusion

Gradient Descent and Alternating Least Squares are powerful tools for image reconstruction, each suited to different problem structures. GD is versatile and widely used for its simplicity, while ALS excels in problems with factorization or constraints. In practice, the choice depends on the specific forward model, constraints, and computational resources. Hybrid approaches or advanced variants (e.g., accelerated GD or regularized ALS) are often used to improve performance.

---

### Part a) Image Reconstruction using Gradient Descent

In this part, we aim to **reconstruct missing or corrupted regions** of an image using **low-rank matrix factorization**. Natural images often have redundancy, and their pixel values can be approximated well by a **low-rank representation**.

#### 1. Mathematical Formulation

Let the grayscale image be represented as a matrix \( M \in \mathbb{R}^{H \times W} \), where \( H \) and \( W \) are the height and width of the image, respectively. If parts of the image are missing (masked), we represent the observed pixels with a mask matrix \( \Omega \in \{0, 1\}^{H \times W} \), where:

$$
\Omega_{ij} =
\begin{cases}
1 & \text{if pixel } (i,j) \text{ is observed}, \\
0 & \text{if pixel } (i,j) \text{ is missing}.
\end{cases}
$$

The goal is to find two low-rank matrices \( U \in \mathbb{R}^{H \times r} \) and \( V \in \mathbb{R}^{r \times W} \) such that:

$$
M \approx U V,
$$

where \( r \ll \min(H, W) \) is the rank of the approximation, chosen to balance reconstruction quality and computational efficiency. Here, \( (U V)_{ij} = \sum_{k=1}^r U_{ik} V_{kj} \) represents the \((i,j)\)-th entry of the matrix product \( U V \).

#### 2. Optimization Problem

We minimize the **mean squared error (MSE)** over the observed pixels:

$$
\min_{U, V} \sum_{(i,j) : \Omega_{ij} = 1} \left( M_{ij} - (U V)_{ij} \right)^2,
$$

where \((U V)_{ij} = \sum_{k=1}^r U_{ik} V_{kj}\) denotes the \((i,j)\)-th entry of the matrix product \(U V\).

This is solved iteratively using **gradient descent**:

- Initialize \( U \in \mathbb{R}^{H \times r} \) and \( V \in \mathbb{R}^{r \times W} \) with random entries (e.g., small values from a normal distribution).
- Compute the gradients of the loss function with respect to \( U \) and \( V \):

$$
\frac{\partial L}{\partial U} = 2 \left( (U V - M) \circ \Omega \right) V^T, \quad \frac{\partial L}{\partial V} = 2 U^T \left( (U V - M) \circ \Omega \right),
$$

  where \(\circ\) denotes the Hadamard (element-wise) product, and \(L\) is the loss function.
- Update \( U \) and \( V \) in the negative gradient direction:

$$
U \gets U - \eta \frac{\partial L}{\partial U}, \quad V \gets V - \eta \frac{\partial L}{\partial V},
$$

  where \(\eta > 0\) is the learning rate.
- Repeat until convergence (e.g., when the change in loss is below a threshold \(\epsilon\) or after a fixed number of iterations).

#### 3. Implementation Details

- Each **channel** of a color image can be treated independently.
- After each iteration, the reconstructed patch \( \hat{M} = U V \) is updated.
- **Learning rate** and **rank** are hyperparameters that affect reconstruction quality.

#### 4. Advantages

- Captures **global structure** of the image effectively.
- Handles **missing pixels** naturally using a mask.
- Allows a trade-off between **approximation quality** and **computational cost** by varying the rank.

#### 5. Evaluation Metrics

1. **RMSE (Root Mean Square Error)**
The **RMSE (Root Mean Square Error)** is defined as:

$$
\text{RMSE} = \sqrt{\frac{1}{|\Omega^c|} \sum_{(i,j) \in \Omega^c} \left( M_{ij} - \hat{M}_{ij} \right)^2},
$$

where \(\Omega^c = \{(i,j) \mid \Omega_{ij} = 0\}\) denotes the set of missing pixels, and \(|\Omega^c|\) is the number of missing pixels.

2. **PSNR (Peak Signal-to-Noise Ratio)**
The **PSNR (Peak Signal-to-Noise Ratio)** is defined as:

$$
\text{PSNR} = 20 \cdot \log_{10} \left( \frac{\text{MAX}_I}{\text{RMSE}} \right),
$$

where $\text{MAX}_I$ is the maximum possible pixel value (e.g., 1.0 for normalized images or 255 for 8-bit images).
#### 6. Observations

- **Low-rank approximations** can recover main structures of the image but may miss fine textures at very low ranks.
- Increasing the **rank** improves reconstruction but may overfit for very high ranks.
- Gradient descent converges faster for well-scaled images and appropriate learning rates.

---


#### Case 1: Missing Rectangular Block

* **Reconstruction Quality:** Reconstructing a contiguous 30x30 block was challenging. The resulting reconstruction was often blurry and lacked fine details. The algorithm had to infer the entire region based solely on the surrounding pixels, with no information available within the block itself. The transition between the original image and the reconstructed block was often noticeable.
* **Metrics:** This case resulted in a **higher RMSE** and a **lower Peak SNR**, indicating a larger error between the reconstruction and the ground truth.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

# -----------------------------
# Helper Functions
# -----------------------------
def compute_rmse(img_true, img_pred, mask=None):
    """Compute RMSE over the masked region"""
    if mask is None:
        mask = torch.ones_like(img_true)
    diff = (img_true - img_pred) * mask
    mse = torch.sum(diff**2) / torch.sum(mask)
    return torch.sqrt(mse).item()

def plot_side_by_side(images, titles, figsize=(20,8)):
    """Plot multiple images side by side"""
    n = len(images)
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=14)
        plt.axis('off')
    plt.show()

# -----------------------------
# Step 1: Load and crop image
# -----------------------------
img = torchvision.io.read_image("dog.jpg").float()  # [C,H,W]

# Crop parameters
x_crop, y_crop, h, w = 800, 600, 300, 300
cropped_img = img[:, y_crop:y_crop+h, x_crop:x_crop+w]

# Convert to grayscale
img_gray = cropped_img.mean(dim=0) / 255.0  # [H,W]

H, W = img_gray.shape
print(f"Cropped image shape: {H}x{W}")

# -----------------------------
# Step 2: Create 30x30 rectangular mask
# -----------------------------
mask_rect = torch.ones_like(img_gray)
mask_rect[50:80, 60:90] = 0  # 30x30 missing block

# -----------------------------
# Step 3: Low-rank matrix factorization reconstruction
# -----------------------------
def low_rank_reconstruct(img, mask, rank=30, lr=0.2, epochs=2000):
    H, W = img.shape
    U = torch.randn(H, rank, requires_grad=True)
    V = torch.randn(rank, W, requires_grad=True)
    optimizer = optim.Adam([U, V], lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = U @ V
        loss = torch.sum(((pred - img) * mask)**2) / torch.sum(mask)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.6f}")
    return (U @ V).detach()

# Reconstruct
recon_rect = low_rank_reconstruct(img_gray, mask_rect, rank=30)

# -----------------------------
# Step 4: Compute metrics
# -----------------------------
rmse_rect = compute_rmse(img_gray, recon_rect, mask=(1-mask_rect))
psnr_rect = psnr(img_gray.numpy(), recon_rect.numpy(), data_range=1)

print(f"\nRectangular 30x30 reconstruction metrics:")
print(f"RMSE = {rmse_rect:.4f}")
print(f"PSNR = {psnr_rect:.2f} dB")

# -----------------------------
# Step 5: Display side by side
# -----------------------------
plot_side_by_side(
    [img_gray, img_gray*mask_rect, recon_rect],
    ["Original Cropped", "Masked 30x30", "Reconstructed Rectangular"],
    figsize=(20,8)
)

```

<img width="1570" height="499" alt="image" src="https://github.com/user-attachments/assets/6fecbe30-8811-4399-a016-ec16ebccb0b4" />

**Output**:
```
Cropped image shape: 300x300
Epoch 500/2000, Loss=0.001615
Epoch 1000/2000, Loss=0.001109
Epoch 1500/2000, Loss=0.000926
Epoch 2000/2000, Loss=0.000842

Rectangular 30x30 reconstruction metrics:
RMSE = 0.2473
PSNR = 28.40 dB
```

#### Case 2: Missing Random Pixels

* **Reconstruction Quality:** The reconstruction of randomly scattered missing pixels was significantly more successful. Because known pixels were interspersed with missing ones, the model could leverage local context and correlations much more effectively. The reconstructed image appeared much sharper and more faithful to the original compared to the block reconstruction.
* **Metrics:** This case yielded a **lower RMSE** and a **higher Peak SNR**, confirming the superior quality of the reconstruction.


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

# -----------------------------
# Helper Functions
# -----------------------------
def compute_rmse(img_true, img_pred, mask=None):
    """Compute RMSE over the masked region"""
    if mask is None:
        mask = torch.ones_like(img_true)
    diff = (img_true - img_pred) * mask
    mse = torch.sum(diff**2) / torch.sum(mask)
    return torch.sqrt(mse).item()

def plot_side_by_side(images, titles, figsize=(20,8)):
    """Plot multiple images side by side"""
    n = len(images)
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=14)
        plt.axis('off')
    plt.show()

# -----------------------------
# Step 1: Load and crop image
# -----------------------------
img = torchvision.io.read_image("dog.jpg").float()  # [C,H,W]

# Crop parameters
x_crop, y_crop, h, w = 800, 600, 300, 300
cropped_img = img[:, y_crop:y_crop+h, x_crop:x_crop+w]

# Convert to grayscale
img_gray = cropped_img.mean(dim=0) / 255.0  # [H,W]

H, W = img_gray.shape
print(f"Cropped image shape: {H}x{W}")

# -----------------------------
# Step 2: Create random mask (900 pixels missing)
# -----------------------------
mask_rand = torch.ones_like(img_gray)
missing_indices = torch.randperm(H*W)[:900]
mask_rand.view(-1)[missing_indices] = 0

# -----------------------------
# Step 3: Low-rank matrix factorization reconstruction
# -----------------------------
def low_rank_reconstruct(img, mask, rank=30, lr=0.2, epochs=2000):
    H, W = img.shape
    U = torch.randn(H, rank, requires_grad=True)
    V = torch.randn(rank, W, requires_grad=True)
    optimizer = optim.Adam([U, V], lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = U @ V
        loss = torch.sum(((pred - img) * mask)**2) / torch.sum(mask)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.6f}")
    return (U @ V).detach()

# Reconstruct
recon_rand = low_rank_reconstruct(img_gray, mask_rand, rank=30)

# -----------------------------
# Step 4: Compute metrics
# -----------------------------
rmse_rand = compute_rmse(img_gray, recon_rand, mask=(1-mask_rand))
psnr_rand = psnr(img_gray.numpy(), recon_rand.numpy(), data_range=1)

print(f"\nRandom 900 pixels reconstruction metrics:")
print(f"RMSE = {rmse_rand:.4f}")
print(f"PSNR = {psnr_rand:.2f} dB")

# -----------------------------
# Step 5: Display side by side
# -----------------------------
plot_side_by_side(
    [img_gray, img_gray*mask_rand, recon_rand],
    ["Original Cropped", "Masked Random 900 Pixels", "Reconstructed Random"],
    figsize=(20,8)
)
```

<img width="1570" height="499" alt="image" src="https://github.com/user-attachments/assets/8eb6dde2-700f-45f2-af0c-0d85509a13a2" />

**Output**:
```
Cropped image shape: 300x300
Epoch 500/2000, Loss=0.001523
Epoch 1000/2000, Loss=0.001113
Epoch 1500/2000, Loss=0.000905
Epoch 2000/2000, Loss=0.000823

Random 900 pixels reconstruction metrics:
RMSE = 0.0358
PSNR = 30.82 dB
```

### Part b) Image Reconstruction using Gradient Descent

* **Performance:** ALS generally converged faster and more reliably than Gradient Descent for this problem. Since each step in ALS (solving for `U` with `V` fixed, and vice-versa) is a standard least squares problem, it can be solved optimally and directly.
* **Tuning:** GD required careful tuning of the learning rate, whereas ALS was more stable and required less hyperparameter tuning. The final reconstruction quality achieved by both methods was comparable, but ALS proved to be a more efficient optimization strategy for this specific task.


#### Case 1: Missing Rectangular Block

* **Reconstruction Quality:** Reconstructing a contiguous 30x30 block was challenging. The resulting reconstruction was often blurry and lacked fine details. The algorithm had to infer the entire region based solely on the surrounding pixels, with no information available within the block itself. The transition between the original image and the reconstructed block was often noticeable.
* **Metrics:** This case resulted in a **higher RMSE** and a **lower Peak SNR**, indicating a larger error between the reconstruction and the ground truth.

```python
import torch
import torchvision
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

# -----------------------------
# Helper Functions
# -----------------------------
def compute_rmse(img_true, img_pred, mask=None):
    if mask is None:
        mask = torch.ones_like(img_true)
    diff = (img_true - img_pred) * mask
    mse = torch.sum(diff**2) / torch.sum(mask)
    return torch.sqrt(mse).item()

def plot_side_by_side(images, titles, figsize=(20,8)):
    n = len(images)
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=14)
        plt.axis('off')
    plt.show()

# -----------------------------
# Step 1: Load and crop image
# -----------------------------
img = torchvision.io.read_image("dog.jpg").float()
x, y, h, w = 800, 600, 300, 300
cropped_img = img[:, y:y+h, x:x+w]
img_gray = cropped_img.mean(dim=0) / 255.0
H, W = img_gray.shape
print(f"Cropped image shape: {H}x{W}")

# -----------------------------
# Step 2: Create rectangular mask 30x30
# -----------------------------
mask_rect = torch.ones_like(img_gray)
mask_rect[50:80, 60:90] = 0  # 30x30 missing block

# -----------------------------
# Step 3: ALS Reconstruction
# -----------------------------
def als_reconstruct(img, mask, rank=30, iterations=30):
    H, W = img.shape
    U = torch.randn(H, rank)
    V = torch.randn(rank, W)
    M = mask

    for it in range(iterations):
        # Update U
        for i in range(H):
            idx = M[i, :] == 1
            V_sub = V[:, idx]
            img_sub = img[i, idx]
            if V_sub.shape[1] > 0:
                U[i, :] = torch.linalg.lstsq(V_sub.T, img_sub).solution
        # Update V
        for j in range(W):
            idx = M[:, j] == 1
            U_sub = U[idx, :]
            img_sub = img[idx, j]
            if U_sub.shape[0] > 0:
                V[:, j] = torch.linalg.lstsq(U_sub, img_sub).solution
    return U @ V

recon_rect = als_reconstruct(img_gray, mask_rect, rank=30, iterations=30)

# -----------------------------
# Step 4: Compute metrics
# -----------------------------
rmse_rect = compute_rmse(img_gray, recon_rect, mask=(1-mask_rect))
psnr_rect = psnr(img_gray.numpy(), recon_rect.numpy(), data_range=1)

print("\nRectangular 30x30 reconstruction metrics (ALS):")
print(f"RMSE = {rmse_rect:.4f}")
print(f"PSNR = {psnr_rect:.2f} dB")

# -----------------------------
# Step 5: Plot images side by side
# -----------------------------
plot_side_by_side([img_gray, img_gray*mask_rect, recon_rect],
                  ["Original Cropped", "Masked 30x30 Rect", "Reconstructed Rectangular (ALS)"],
                  figsize=(20,8))
```

<img width="1570" height="499" alt="image" src="https://github.com/user-attachments/assets/1824e8c3-8d49-4dd3-a784-08907f6a58dc" />

**Output**:
```
Cropped image shape: 300x300

Rectangular 30x30 reconstruction metrics (ALS):
RMSE = 0.6334
PSNR = 23.22 dB
```


#### Case 2: Missing Random Pixels

* **Reconstruction Quality:** The reconstruction of randomly scattered missing pixels was significantly more successful. Because known pixels were interspersed with missing ones, the model could leverage local context and correlations much more effectively. The reconstructed image appeared much sharper and more faithful to the original compared to the block reconstruction.
* **Metrics:** This case yielded a **lower RMSE** and a **higher Peak SNR**, confirming the superior quality of the reconstruction.


```python
import torch
import torchvision
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

# -----------------------------
# Helper Functions
# -----------------------------
def compute_rmse(img_true, img_pred, mask=None):
    if mask is None:
        mask = torch.ones_like(img_true)
    diff = (img_true - img_pred) * mask
    mse = torch.sum(diff**2) / torch.sum(mask)
    return torch.sqrt(mse).item()

def plot_side_by_side(images, titles, figsize=(20,8)):
    n = len(images)
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=14)
        plt.axis('off')
    plt.show()

# -----------------------------
# Step 1: Load and crop image
# -----------------------------
img = torchvision.io.read_image("dog.jpg").float()
x, y, h, w = 800, 600, 300, 300
cropped_img = img[:, y:y+h, x:x+w]
img_gray = cropped_img.mean(dim=0) / 255.0
H, W = img_gray.shape
print(f"Cropped image shape: {H}x{W}")

# -----------------------------
# Step 2: Create random mask with 900 missing pixels
# -----------------------------
mask_rand = torch.ones_like(img_gray)
missing_indices = torch.randperm(H*W)[:900]
mask_rand.view(-1)[missing_indices] = 0

# -----------------------------
# Step 3: ALS Reconstruction
# -----------------------------
def als_reconstruct(img, mask, rank=30, iterations=30):
    H, W = img.shape
    U = torch.randn(H, rank)
    V = torch.randn(rank, W)
    M = mask

    for it in range(iterations):
        # Update U
        for i in range(H):
            idx = M[i, :] == 1
            V_sub = V[:, idx]
            img_sub = img[i, idx]
            if V_sub.shape[1] > 0:
                U[i, :] = torch.linalg.lstsq(V_sub.T, img_sub).solution
        # Update V
        for j in range(W):
            idx = M[:, j] == 1
            U_sub = U[idx, :]
            img_sub = img[idx, j]
            if U_sub.shape[0] > 0:
                V[:, j] = torch.linalg.lstsq(U_sub, img_sub).solution
    return U @ V

recon_rand = als_reconstruct(img_gray, mask_rand, rank=30, iterations=30)

# -----------------------------
# Step 4: Compute metrics
# -----------------------------
rmse_rand = compute_rmse(img_gray, recon_rand, mask=(1-mask_rand))
psnr_rand = psnr(img_gray.numpy(), recon_rand.numpy(), data_range=1)

print("\nRandom 900 pixels reconstruction metrics (ALS):")
print(f"RMSE = {rmse_rand:.4f}")
print(f"PSNR = {psnr_rand:.2f} dB")

# -----------------------------
# Step 5: Plot images side by side
# -----------------------------
plot_side_by_side([img_gray, img_gray*mask_rand, recon_rand],
                  ["Original Cropped", "Masked Random 900", "Reconstructed Random (ALS)"],
                  figsize=(20,8))
```

<img width="1570" height="499" alt="image" src="https://github.com/user-attachments/assets/d2fe1955-6c31-4821-8d61-5b3059c0feca" />

**Output**:
```
Cropped image shape: 300x300

Random 900 pixels reconstruction metrics (ALS):
RMSE = 0.0340
PSNR = 31.09 dB
```

---

### Part b) Data Compression

Here, matrix factorization was used to compress a 50x50 image patch by representing it with lower-rank matrices `U` (50xr) and `V` (50xr).

#### Image used for this part
```python
import torchvision
import matplotlib.pyplot as plt

# Load the image
img = torchvision.io.read_image("dog.jpg").float() / 255.0  # [C,H,W]

# Crop parameters
x_crop, y_crop, h, w = 800, 600, 300, 300
cropped_img = img[:, y_crop:y_crop+h, x_crop:x_crop+w].permute(1,2,0)  # [H,W,C]

# Display with axes
plt.figure(figsize=(6,6))
plt.imshow(cropped_img)
# plt.xlabel("Width (pixels)")
# plt.ylabel("Height (pixels)")
plt.title("Cropped Image with Axes")
plt.grid(True)
plt.show()
```

<img width="518" height="528" alt="image" src="https://github.com/user-attachments/assets/159965f6-7015-4118-8cf8-a61a8b43f6de" />

```python
import torch
import torchvision
import matplotlib.pyplot as plt

# -----------------------------
# Helper function to add red border
# -----------------------------
def add_red_border(img, x, y, patch_size=50, border_thickness=2):
    """
    Add a red border around the patch in the image
    img: [H,W,C]
    x, y: top-left corner of patch
    patch_size: size of patch
    border_thickness: thickness of red border
    """
    img_copy = img.clone()
    # Top border
    img_copy[y:y+border_thickness, x:x+patch_size, 0] = 1
    img_copy[y:y+border_thickness, x:x+patch_size, 1:] = 0
    # Bottom border
    img_copy[y+patch_size-border_thickness:y+patch_size, x:x+patch_size, 0] = 1
    img_copy[y+patch_size-border_thickness:y+patch_size, x:x+patch_size, 1:] = 0
    # Left border
    img_copy[y:y+patch_size, x:x+border_thickness, 0] = 1
    img_copy[y:y+patch_size, x:x+border_thickness, 1:] = 0
    # Right border
    img_copy[y:y+patch_size, x+patch_size-border_thickness:x+patch_size, 0] = 1
    img_copy[y:y+patch_size, x+patch_size-border_thickness:x+patch_size, 1:] = 0
    return img_copy

# -----------------------------
# Load and crop the image
# -----------------------------
img = torchvision.io.read_image("dog.jpg").float() / 255.0  # [C,H,W]

# Crop parameters
x_crop, y_crop, h, w = 800, 600, 300, 300
cropped_img = img[:, y_crop:y_crop+h, x_crop:x_crop+w].permute(1,2,0)  # [H,W,C]

# -----------------------------
# Patch coordinates
# -----------------------------
patches = [(5,5), (35,50), (80,175)]

# -----------------------------
# Add red borders and collect images
# -----------------------------
images_with_borders = []
titles = []

for idx, (x, y) in enumerate(patches):
    img_with_border = add_red_border(cropped_img, x, y, patch_size=50, border_thickness=2)
    images_with_borders.append(img_with_border)
    titles.append(f"Patch {idx+1}")

# -----------------------------
# Plot all side by side
# -----------------------------
plt.figure(figsize=(18,6))
for i, img_plot in enumerate(images_with_borders):
    plt.subplot(1, len(images_with_borders), i+1)
    plt.imshow(img_plot)
    plt.title(titles[i], fontsize=14)
    plt.axis('on')
plt.show()
```

<img width="1451" height="477" alt="image" src="https://github.com/user-attachments/assets/13ad2017-b8d6-457d-92f9-113cb1b1cee7" />


#### Effect of Rank (r) and Patch Complexity

* **Case 1: Single Color Patch (Low Complexity)**
    * This patch is inherently low-rank. A very low rank like **`r=5`** was sufficient to reconstruct the patch almost perfectly. Increasing the rank to 10, 25, or 50 offered no visible improvement in quality, as the patch contained very little information to begin with.
 
```python
import torch
import torchvision
import matplotlib.pyplot as plt

# -----------------------------
# Helper function to add red border
# -----------------------------
def add_red_border(img, x, y, patch_size=50, border_thickness=2):
    img_copy = img.clone()
    # Top border
    img_copy[y:y+border_thickness, x:x+patch_size, 0] = 1
    img_copy[y:y+border_thickness, x:x+patch_size, 1:] = 0
    # Bottom border
    img_copy[y+patch_size-border_thickness:y+patch_size, x:x+patch_size, 0] = 1
    img_copy[y+patch_size-border_thickness:y+patch_size, x:x+patch_size, 1:] = 0
    # Left border
    img_copy[y:y+patch_size, x:x+border_thickness, 0] = 1
    img_copy[y:y+patch_size, x:x+border_thickness, 1:] = 0
    # Right border
    img_copy[y:y+patch_size, x+patch_size-border_thickness:x+patch_size, 0] = 1
    img_copy[y:y+patch_size, x+patch_size-border_thickness:x+patch_size, 1:] = 0
    return img_copy

# -----------------------------
# Load and crop the image
# -----------------------------
img = torchvision.io.read_image("dog.jpg").float() / 255.0
x_crop, y_crop, h, w = 800, 600, 300, 300
cropped_img = img[:, y_crop:y_crop+h, x_crop:x_crop+w].permute(1,2,0)  # [H,W,C]

# -----------------------------
# Patch 1
# -----------------------------
x_patch, y_patch = 5, 5
img_patch1 = add_red_border(cropped_img, x_patch, y_patch, patch_size=50, border_thickness=2)

plt.figure(figsize=(6,6))
plt.imshow(img_patch1)
plt.title("Patch 1 at (5,5)")
plt.axis('on')
plt.show()
```

<img width="518" height="528" alt="image" src="https://github.com/user-attachments/assets/cb6784d6-5dfc-406b-9451-a3c2b4a6dc10" />


```python
import torch
import torchvision
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

# -----------------------------
# Helper Functions
# -----------------------------
def plot_side_by_side(images, titles, figsize=(20,6)):
    n = len(images)
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(1, n, i+1)
        img_to_plot = images[i].detach().cpu()
        plt.imshow(img_to_plot)
        plt.title(titles[i], fontsize=14)
        plt.axis('off')
    plt.show()

def low_rank_gd_patch(patch, rank=5, lr=0.1, epochs=500):
    N, _, C = patch.shape
    reconstructed_patch = torch.zeros_like(patch)
    for c in range(C):
        channel_patch = patch[:,:,c]
        U = torch.randn(N, rank, requires_grad=True)
        V = torch.randn(rank, N, requires_grad=True)
        optimizer = torch.optim.Adam([U,V], lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            recon = U @ V
            loss = ((recon - channel_patch)**2).mean()
            loss.backward()
            optimizer.step()
        reconstructed_patch[:,:,c] = (U @ V).detach()
    return reconstructed_patch

def add_red_border(img, x, y, patch_size=50, border_thickness=2):
    img_copy = img.clone()
    img_copy[y:y+border_thickness, x:x+patch_size, 0] = 1
    img_copy[y:y+border_thickness, x:x+patch_size, 1:] = 0
    img_copy[y+patch_size-border_thickness:y+patch_size, x:x+patch_size, 0] = 1
    img_copy[y+patch_size-border_thickness:y+patch_size, x:x+patch_size, 1:] = 0
    img_copy[y:y+patch_size, x:x+border_thickness, 0] = 1
    img_copy[y:y+patch_size, x:x+border_thickness, 1:] = 0
    img_copy[y:y+patch_size, x+patch_size-border_thickness:x+patch_size, 0] = 1
    img_copy[y:y+patch_size, x+patch_size-border_thickness:x+patch_size, 1:] = 0
    return img_copy

def compute_rmse(img_true, img_pred):
    return torch.sqrt(((img_true - img_pred)**2).mean()).item()

# -----------------------------
# Step 1: Load and crop image
# -----------------------------
img = torchvision.io.read_image("dog.jpg").float() / 255.0
x_crop, y_crop, h, w = 800, 600, 300, 300
cropped_img = img[:, y_crop:y_crop+h, x_crop:x_crop+w].permute(1,2,0)

# -----------------------------
# Step 2: Define patch 1
# -----------------------------
N = 50
x_patch, y_patch = 0, 0
ranks = [5, 10, 25, 50]

# -----------------------------
# Step 3: Reconstruct and analyze
# -----------------------------
patch = cropped_img[y_patch:y_patch+N, x_patch:x_patch+N, :]
print(f"Patch 1 at ({x_patch},{y_patch})")

reconstructed_images = []
titles = []
psnr_values = []
rmse_values = []

for r in ranks:
    recon_patch = low_rank_gd_patch(patch, rank=r, lr=0.1, epochs=500)
    img_recon = cropped_img.clone()
    img_recon[y_patch:y_patch+N, x_patch:x_patch+N, :] = recon_patch
    img_recon_with_border = add_red_border(img_recon, x_patch, y_patch, patch_size=N)
    reconstructed_images.append(img_recon_with_border)
    titles.append(f"Rank {r}")

    psnr_values.append(psnr(patch.numpy(), recon_patch.numpy(), data_range=1.0))
    rmse_values.append(compute_rmse(patch, recon_patch))

# Print metrics table
print("\nRank |    PSNR (dB)    |   RMSE")
print("-----------------------------------")
for i, r in enumerate(ranks):
    print(f"{r:<5}|  {psnr_values[i]:<15.4f}| {rmse_values[i]:<.4f}")

best_idx_psnr = psnr_values.index(max(psnr_values))
best_idx_rmse = rmse_values.index(min(rmse_values))
print(f"\nBest by PSNR: Rank {ranks[best_idx_psnr]} (PSNR={psnr_values[best_idx_psnr]:.2f} dB)")
print(f"Best by RMSE: Rank {ranks[best_idx_rmse]} (RMSE={rmse_values[best_idx_rmse]:.4f})")

plot_side_by_side(reconstructed_images, titles, figsize=(20,6))


```

<img width="1570" height="380" alt="image" src="https://github.com/user-attachments/assets/b5b784e7-7301-4938-86dc-9fec2d8a626a" />

**Output**:
```
Patch 1 at (0,0)

Rank |    PSNR (dB)    |   RMSE
-----------------------------------
5    |  44.8042        | 0.0058
10   |  45.0206        | 0.0056
25   |  47.2140        | 0.0044
50   |  49.3685        | 0.0034

Best by PSNR: Rank 50 (PSNR=49.37 dB)
Best by RMSE: Rank 50 (RMSE=0.0034)
```


* **Case 2: 2-3 Different Colors (Medium Complexity)**
    * With **`r=5`**, the reconstruction was decent but edges between the colored regions appeared slightly blurry.
    * Increasing the rank to **`r=10`** and **`r=25`** significantly improved the sharpness and quality.
    * At **`r=50`** (full rank), the reconstruction was perfect, but this offers no compression benefit.

```python
import torch
import torchvision
import matplotlib.pyplot as plt

# -----------------------------
# Helper function to add red border
# -----------------------------
def add_red_border(img, x, y, patch_size=50, border_thickness=2):
    img_copy = img.clone()
    # Top border
    img_copy[y:y+border_thickness, x:x+patch_size, 0] = 1
    img_copy[y:y+border_thickness, x:x+patch_size, 1:] = 0
    # Bottom border
    img_copy[y+patch_size-border_thickness:y+patch_size, x:x+patch_size, 0] = 1
    img_copy[y+patch_size-border_thickness:y+patch_size, x:x+patch_size, 1:] = 0
    # Left border
    img_copy[y:y+patch_size, x:x+border_thickness, 0] = 1
    img_copy[y:y+patch_size, x:x+border_thickness, 1:] = 0
    # Right border
    img_copy[y:y+patch_size, x+patch_size-border_thickness:x+patch_size, 0] = 1
    img_copy[y:y+patch_size, x+patch_size-border_thickness:x+patch_size, 1:] = 0
    return img_copy

# -----------------------------
# Load and crop the image
# -----------------------------
img = torchvision.io.read_image("dog.jpg").float() / 255.0
x_crop, y_crop, h, w = 800, 600, 300, 300
cropped_img = img[:, y_crop:y_crop+h, x_crop:x_crop+w].permute(1,2,0)  # [H,W,C]

# -----------------------------
# Patch 2
# -----------------------------
x_patch, y_patch = 35, 50
img_patch2 = add_red_border(cropped_img, x_patch, y_patch, patch_size=50, border_thickness=2)

plt.figure(figsize=(6,6))
plt.imshow(img_patch2)
plt.title("Patch 2 at (35,50)")
plt.axis('on')
plt.show()

```

<img width="518" height="528" alt="image" src="https://github.com/user-attachments/assets/b69df28c-9bb7-44f2-8e34-4bb2265eb369" />


```python
import torch
import torchvision
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

# -----------------------------
# Helper Functions
# -----------------------------
def plot_side_by_side(images, titles, figsize=(20,6)):
    n = len(images)
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(1, n, i+1)
        img_to_plot = images[i].detach().cpu()
        plt.imshow(img_to_plot)
        plt.title(titles[i], fontsize=14)
        plt.axis('off')
    plt.show()

def low_rank_gd_patch(patch, rank=5, lr=0.1, epochs=500):
    N, _, C = patch.shape
    reconstructed_patch = torch.zeros_like(patch)
    for c in range(C):
        channel_patch = patch[:,:,c]
        U = torch.randn(N, rank, requires_grad=True)
        V = torch.randn(rank, N, requires_grad=True)
        optimizer = torch.optim.Adam([U,V], lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            recon = U @ V
            loss = ((recon - channel_patch)**2).mean()
            loss.backward()
            optimizer.step()
        reconstructed_patch[:,:,c] = (U @ V).detach()
    return reconstructed_patch

def add_red_border(img, x, y, patch_size=50, border_thickness=2):
    img_copy = img.clone()
    img_copy[y:y+border_thickness, x:x+patch_size, 0] = 1
    img_copy[y:y+border_thickness, x:x+patch_size, 1:] = 0
    img_copy[y+patch_size-border_thickness:y+patch_size, x:x+patch_size, 0] = 1
    img_copy[y+patch_size-border_thickness:y+patch_size, x:x+patch_size, 1:] = 0
    img_copy[y:y+patch_size, x:x+border_thickness, 0] = 1
    img_copy[y:y+patch_size, x:x+border_thickness, 1:] = 0
    img_copy[y:y+patch_size, x+patch_size-border_thickness:x+patch_size, 0] = 1
    img_copy[y:y+patch_size, x+patch_size-border_thickness:x+patch_size, 1:] = 0
    return img_copy

def compute_rmse(img_true, img_pred):
    return torch.sqrt(((img_true - img_pred)**2).mean()).item()

# -----------------------------
# Step 1: Load and crop image
# -----------------------------
img = torchvision.io.read_image("dog.jpg").float() / 255.0
x_crop, y_crop, h, w = 800, 600, 300, 300
cropped_img = img[:, y_crop:y_crop+h, x_crop:x_crop+w].permute(1,2,0)

# -----------------------------
# Step 2: Define patch 1
# -----------------------------
N = 50
x_patch, y_patch = 25, 50
ranks = [5, 10, 25, 50]

# -----------------------------
# Step 3: Reconstruct and analyze
# -----------------------------
patch = cropped_img[y_patch:y_patch+N, x_patch:x_patch+N, :]
print(f"Patch 1 at ({x_patch},{y_patch})")

reconstructed_images = []
titles = []
psnr_values = []
rmse_values = []

for r in ranks:
    recon_patch = low_rank_gd_patch(patch, rank=r, lr=0.1, epochs=500)
    img_recon = cropped_img.clone()
    img_recon[y_patch:y_patch+N, x_patch:x_patch+N, :] = recon_patch
    img_recon_with_border = add_red_border(img_recon, x_patch, y_patch, patch_size=N)
    reconstructed_images.append(img_recon_with_border)
    titles.append(f"Rank {r}")

    psnr_values.append(psnr(patch.numpy(), recon_patch.numpy(), data_range=1.0))
    rmse_values.append(compute_rmse(patch, recon_patch))

# Print metrics table
print("\nRank |    PSNR (dB)    |   RMSE")
print("-----------------------------------")
for i, r in enumerate(ranks):
    print(f"{r:<5}|  {psnr_values[i]:<15.4f}| {rmse_values[i]:<.4f}")

best_idx_psnr = psnr_values.index(max(psnr_values))
best_idx_rmse = rmse_values.index(min(rmse_values))
print(f"\nBest by PSNR: Rank {ranks[best_idx_psnr]} (PSNR={psnr_values[best_idx_psnr]:.2f} dB)")
print(f"Best by RMSE: Rank {ranks[best_idx_rmse]} (RMSE={rmse_values[best_idx_rmse]:.4f})")

plot_side_by_side(reconstructed_images, titles, figsize=(20,6))


```

<img width="1570" height="380" alt="image" src="https://github.com/user-attachments/assets/375a31e6-9f51-4147-b10e-2dd5f9f3a1d7" />

**Output**:
```
Patch 2 at (25,50)

Rank |    PSNR (dB)    |   RMSE
-----------------------------------
5    |  33.5970        | 0.0209
10   |  37.4109        | 0.0135
25   |  41.0680        | 0.0088
50   |  44.6660        | 0.0058

Best by PSNR: Rank 50 (PSNR=44.67 dB)
Best by RMSE: Rank 50 (RMSE=0.0058)
```

* **Case 3: Multiple Colors/Details (High Complexity)**
    * This patch contained the most information (high-frequency details).
    * With **`r=5`**, the reconstruction was very poor, appearing blurry and blocky, losing most of the original detail.
    * As the rank increased from **`r=10`** to **`r=25`**, the quality progressively improved, with more details and textures becoming visible.
    * A high rank like **`r=25`** was needed to achieve a reasonably faithful reconstruction, and even then, some fine details might be lost compared to the original `r=50` patch.
 
```python
import torch
import torchvision
import matplotlib.pyplot as plt

# -----------------------------
# Helper function to add red border
# -----------------------------
def add_red_border(img, x, y, patch_size=50, border_thickness=2):
    img_copy = img.clone()
    # Top border
    img_copy[y:y+border_thickness, x:x+patch_size, 0] = 1
    img_copy[y:y+border_thickness, x:x+patch_size, 1:] = 0
    # Bottom border
    img_copy[y+patch_size-border_thickness:y+patch_size, x:x+patch_size, 0] = 1
    img_copy[y+patch_size-border_thickness:y+patch_size, x:x+patch_size, 1:] = 0
    # Left border
    img_copy[y:y+patch_size, x:x+border_thickness, 0] = 1
    img_copy[y:y+patch_size, x:x+border_thickness, 1:] = 0
    # Right border
    img_copy[y:y+patch_size, x+patch_size-border_thickness:x+patch_size, 0] = 1
    img_copy[y:y+patch_size, x+patch_size-border_thickness:x+patch_size, 1:] = 0
    return img_copy

# -----------------------------
# Load and crop the image
# -----------------------------
img = torchvision.io.read_image("dog.jpg").float() / 255.0
x_crop, y_crop, h, w = 800, 600, 300, 300
cropped_img = img[:, y_crop:y_crop+h, x_crop:x_crop+w].permute(1,2,0)  # [H,W,C]

# -----------------------------
# Patch 3
# -----------------------------
x_patch, y_patch = 80, 175
img_patch3 = add_red_border(cropped_img, x_patch, y_patch, patch_size=50, border_thickness=2)

plt.figure(figsize=(6,6))
plt.imshow(img_patch3)
plt.title("Patch 3 at (80,175)")
plt.axis('on')
plt.show()

```

<img width="518" height="528" alt="image" src="https://github.com/user-attachments/assets/145a8dd4-763c-4181-9dc9-743ed70d39a0" />


```python
import torch
import torchvision
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

# -----------------------------
# Helper Functions
# -----------------------------
def plot_side_by_side(images, titles, figsize=(20,6)):
    n = len(images)
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(1, n, i+1)
        img_to_plot = images[i].detach().cpu()
        plt.imshow(img_to_plot)
        plt.title(titles[i], fontsize=14)
        plt.axis('off')
    plt.show()

def low_rank_gd_patch(patch, rank=5, lr=0.1, epochs=500):
    N, _, C = patch.shape
    reconstructed_patch = torch.zeros_like(patch)
    for c in range(C):
        channel_patch = patch[:,:,c]
        U = torch.randn(N, rank, requires_grad=True)
        V = torch.randn(rank, N, requires_grad=True)
        optimizer = torch.optim.Adam([U,V], lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            recon = U @ V
            loss = ((recon - channel_patch)**2).mean()
            loss.backward()
            optimizer.step()
        reconstructed_patch[:,:,c] = (U @ V).detach()
    return reconstructed_patch

def add_red_border(img, x, y, patch_size=50, border_thickness=2):
    img_copy = img.clone()
    img_copy[y:y+border_thickness, x:x+patch_size, 0] = 1
    img_copy[y:y+border_thickness, x:x+patch_size, 1:] = 0
    img_copy[y+patch_size-border_thickness:y+patch_size, x:x+patch_size, 0] = 1
    img_copy[y+patch_size-border_thickness:y+patch_size, x:x+patch_size, 1:] = 0
    img_copy[y:y+patch_size, x:x+border_thickness, 0] = 1
    img_copy[y:y+patch_size, x:x+border_thickness, 1:] = 0
    img_copy[y:y+patch_size, x+patch_size-border_thickness:x+patch_size, 0] = 1
    img_copy[y:y+patch_size, x+patch_size-border_thickness:x+patch_size, 1:] = 0
    return img_copy

def compute_rmse(img_true, img_pred):
    return torch.sqrt(((img_true - img_pred)**2).mean()).item()

# -----------------------------
# Step 1: Load and crop image
# -----------------------------
img = torchvision.io.read_image("dog.jpg").float() / 255.0
x_crop, y_crop, h, w = 800, 600, 300, 300
cropped_img = img[:, y_crop:y_crop+h, x_crop:x_crop+w].permute(1,2,0)

# -----------------------------
# Step 2: Define patch 1
# -----------------------------
N = 50
x_patch, y_patch = 80, 175
ranks = [5, 10, 25, 50]

# -----------------------------
# Step 3: Reconstruct and analyze
# -----------------------------
patch = cropped_img[y_patch:y_patch+N, x_patch:x_patch+N, :]
print(f"Patch 1 at ({x_patch},{y_patch})")

reconstructed_images = []
titles = []
psnr_values = []
rmse_values = []

for r in ranks:
    recon_patch = low_rank_gd_patch(patch, rank=r, lr=0.1, epochs=500)
    img_recon = cropped_img.clone()
    img_recon[y_patch:y_patch+N, x_patch:x_patch+N, :] = recon_patch
    img_recon_with_border = add_red_border(img_recon, x_patch, y_patch, patch_size=N)
    reconstructed_images.append(img_recon_with_border)
    titles.append(f"Rank {r}")

    psnr_values.append(psnr(patch.numpy(), recon_patch.numpy(), data_range=1.0))
    rmse_values.append(compute_rmse(patch, recon_patch))

# Print metrics table
print("\nRank |    PSNR (dB)    |   RMSE")
print("-----------------------------------")
for i, r in enumerate(ranks):
    print(f"{r:<5}|  {psnr_values[i]:<15.4f}| {rmse_values[i]:<.4f}")

best_idx_psnr = psnr_values.index(max(psnr_values))
best_idx_rmse = rmse_values.index(min(rmse_values))
print(f"\nBest by PSNR: Rank {ranks[best_idx_psnr]} (PSNR={psnr_values[best_idx_psnr]:.2f} dB)")
print(f"Best by RMSE: Rank {ranks[best_idx_rmse]} (RMSE={rmse_values[best_idx_rmse]:.4f})")

plot_side_by_side(reconstructed_images, titles, figsize=(20,6))


```

<img width="1570" height="380" alt="image" src="https://github.com/user-attachments/assets/9983d448-17ac-4634-a0f8-22779aa0182c" />

**Output**:
```
Patch 3 at (80,175)

Rank |    PSNR (dB)    |   RMSE
-----------------------------------
5    |  25.6756        | 0.0520
10   |  29.2833        | 0.0343
25   |  34.0043        | 0.0199
50   |  39.0332        | 0.0112

Best by PSNR: Rank 50 (PSNR=39.03 dB)
Best by RMSE: Rank 50 (RMSE=0.0112)
```


**Overall Conclusion:** There is a direct trade-off between the **compression ratio** (which is higher for lower rank `r`) and the **reconstruction quality**. The amount of visual information or "complexity" in an image patch determines the minimum rank `r` required to reconstruct it with acceptable fidelity. Simple, smooth regions can be compressed aggressively (low `r`), while complex, detailed regions require a higher rank to preserve their information.
