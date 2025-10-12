# Question 4: Implementing Matrix Factorization

## Observations on Matrix Factorization Applications

### Part a) Image Reconstruction

Matrix factorization was used to reconstruct missing regions in an image by approximating the image matrix `M` as a product of two lower-rank matrices, `U` and `V^T`.

#### Case 1: Missing Rectangular Block

* **Reconstruction Quality:** Reconstructing a contiguous 30x30 block was challenging. The resulting reconstruction was often blurry and lacked fine details. The algorithm had to infer the entire region based solely on the surrounding pixels, with no information available within the block itself. The transition between the original image and the reconstructed block was often noticeable.
* **Metrics:** This case resulted in a **higher RMSE** and a **lower Peak SNR**, indicating a larger error between the reconstruction and the ground truth.

#### Case 2: Missing Random Pixels

* **Reconstruction Quality:** The reconstruction of randomly scattered missing pixels was significantly more successful. Because known pixels were interspersed with missing ones, the model could leverage local context and correlations much more effectively. The reconstructed image appeared much sharper and more faithful to the original compared to the block reconstruction.
* **Metrics:** This case yielded a **lower RMSE** and a **higher Peak SNR**, confirming the superior quality of the reconstruction.

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
