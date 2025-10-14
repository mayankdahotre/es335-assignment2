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

## Observations on Matrix Factorization Applications

### Part a) Image Reconstruction

Matrix factorization was used to reconstruct missing regions in an image by approximating the image matrix `M` as a product of two lower-rank matrices, `U` and `V^T`.

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


#### Alternating Least Squares (ALS) vs. Gradient Descent (GD)

* **Performance:** ALS generally converged faster and more reliably than Gradient Descent for this problem. Since each step in ALS (solving for `U` with `V` fixed, and vice-versa) is a standard least squares problem, it can be solved optimally and directly.
* **Tuning:** GD required careful tuning of the learning rate, whereas ALS was more stable and required less hyperparameter tuning. The final reconstruction quality achieved by both methods was comparable, but ALS proved to be a more efficient optimization strategy for this specific task.

---

### Part b) Data Compression

Here, matrix factorization was used to compress a 50x50 image patch by representing it with lower-rank matrices `U` (50xr) and `V` (50xr).

#### Effect of Rank (r) and Patch Complexity

* **Case 1: Single Color Patch (Low Complexity)**
    * This patch is inherently low-rank. A very low rank like **`r=5`** was sufficient to reconstruct the patch almost perfectly. Increasing the rank to 10, 25, or 50 offered no visible improvement in quality, as the patch contained very little information to begin with.

* **Case 2: 2-3 Different Colors (Medium Complexity)**
    * With **`r=5`**, the reconstruction was decent but edges between the colored regions appeared slightly blurry.
    * Increasing the rank to **`r=10`** and **`r=25`** significantly improved the sharpness and quality.
    * At **`r=50`** (full rank), the reconstruction was perfect, but this offers no compression benefit.

* **Case 3: Multiple Colors/Details (High Complexity)**
    * This patch contained the most information (high-frequency details).
    * With **`r=5`**, the reconstruction was very poor, appearing blurry and blocky, losing most of the original detail.
    * As the rank increased from **`r=10`** to **`r=25`**, the quality progressively improved, with more details and textures becoming visible.
    * A high rank like **`r=25`** was needed to achieve a reasonably faithful reconstruction, and even then, some fine details might be lost compared to the original `r=50` patch.

**Overall Conclusion:** There is a direct trade-off between the **compression ratio** (which is higher for lower rank `r`) and the **reconstruction quality**. The amount of visual information or "complexity" in an image patch determines the minimum rank `r` required to reconstruct it with acceptable fidelity. Simple, smooth regions can be compressed aggressively (low `r`), while complex, detailed regions require a higher rank to preserve their information.
