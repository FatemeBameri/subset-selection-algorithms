import numpy as np
import matplotlib.pyplot as plt
from dppy.finite_dpps import FiniteDPP

# 1. تولید داده‌ی دوبعدی
np.random.seed(0)
X = np.vstack([
    np.random.multivariate_normal([0, 0], 0.1*np.eye(2), 50),
    np.random.multivariate_normal([1, 1], 0.1*np.eye(2), 50)
])

# 2. تعریف ماتریس L با کرنل گاوسی
def rbf_kernel(X, sigma=0.5):
    pairwise_sq_dists = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)
    return np.exp(-pairwise_sq_dists / (2 * sigma**2))

L = rbf_kernel(X, sigma=0.3)
L += 1e-8 * np.eye(L.shape[0])   # (پیشنهادی) پایدارسازی عددی

# 3. تعریف DPP با L-ensemble
dpp = FiniteDPP('likelihood', L=L)

# 4. نمونه‌گیری k-DPP (مثلا انتخاب 10 نقطه متنوع)
dpp.sample_exact_k_dpp(size=10)
subset_idx = dpp.list_of_samples[0]

# 5. نمایش
plt.scatter(X[:, 0], X[:, 1], c='lightgray', label="All points")
plt.scatter(X[subset_idx, 0], X[subset_idx, 1], c='red', marker='o', s=80, label="DPP subset")
plt.legend()
plt.title("k-DPP sampling (L-ensemble): Selection of 10 points")
plt.show()
