
#######################################
#-------------MRC----------------------
#######################################
import torch
import numpy as np
import tqdm
from typing import Union, Optional

class MRCSamplerGPU:
    """
    Greedy MRC coreset selection (GPU-friendly)
    Selects subset minimizing distance-correlation (R) and entropy.
    """

    def __init__(
        self,
        percentage: float,
        restarts: int = 5,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        if not 0 < percentage < 1:
            raise ValueError("percentage must be in (0,1)")
        self.percentage = percentage
        self.restarts = restarts
        self.device = (
            torch.device(device)
            if device is not None
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.seed = seed

    # -------------------
    # Helper 1: distance covariance
    # -------------------
    def _distance_covariance(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Computes V²(X, Y): distance covariance squared."""
        X = X.to(self.device)
        Y = Y.to(self.device)
        n = X.shape[0]

        # distance matrices
        a = torch.cdist(X, X, p=2)
        b = torch.cdist(Y, Y, p=2)

        # double centering
        A = a - a.mean(0, keepdim=True) - a.mean(1, keepdim=True) + a.mean()
        B = b - b.mean(0, keepdim=True) - b.mean(1, keepdim=True) + b.mean()

        V2_xy = torch.mean(A * B)
        return V2_xy

    # -------------------
    # Helper 2: distance correlation R(X,Y)
    # -------------------
    def _distance_correlation(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        V2_xy = self._distance_covariance(X, Y)
        V2_xx = self._distance_covariance(X, X)
        V2_yy = self._distance_covariance(Y, Y)

        if V2_xx <= 0 or V2_yy <= 0:
            return torch.tensor(0.0, device=self.device)
        R = V2_xy / torch.sqrt(V2_xx * V2_yy)
        return R

    # -------------------
    # Helper 3: entropy measure
    # -------------------
    def _entropy(self, X: torch.Tensor) -> float:
        """Estimate information entropy of subset (Shannon style)."""
        X = X.to(self.device)
        cov = torch.cov(X.T) + 1e-6 * torch.eye(X.shape[1], device=self.device)
        det = torch.det(cov)
        det = torch.clamp(det, min=1e-10)
        entropy = 0.5 * torch.log(det)
        return entropy.item()

    # -------------------
    # Main MRC selection
    # -------------------
    def _mrc_once(self, X: torch.Tensor, k: int) -> list:
        N, D = X.shape
        indices = list(range(N))
        selected = []

        # انتخاب اولیه تصادفی
        first = int(torch.randint(0, N, (1,), device=self.device).item())
        selected.append(first)

        # حلقه انتخاب
        for _ in tqdm.trange(1, k, desc="MRC-Greedy", leave=False):
            min_corr = float("inf")
            next_idx = None

            for j in range(N):
                if j in selected:
                    continue

                # محاسبه‌ی بیشترین وابستگی با اعضای انتخاب‌شده
                maxR = 0.0
                for s in selected:
                    R = self._distance_correlation(X[j:j+1], X[s:s+1])
                    maxR = max(maxR, R.item())

                # انتخاب کمترین همبستگی
                if maxR < min_corr:
                    min_corr = maxR
                    next_idx = j

            selected.append(next_idx)

        return selected

    # -------------------
    # run(): اجرای چند باره و انتخاب زیرمجموعه با کمترین آنتروپی
    # -------------------
    def run(self, features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, device=self.device, dtype=torch.float32)
        else:
            features = features.to(self.device).float()

        N, D = features.shape
        k = max(1, int(self.percentage * N))

        best_subset = None
        best_entropy = float("inf")

        for r in range(self.restarts):
            selected = self._mrc_once(features, k)
            subset = features[selected]
            ent = self._entropy(subset)
            if ent < best_entropy:
                best_entropy = ent
                best_subset = subset

        print(f"Final MRC subset selected with entropy = {best_entropy:.4f}")

        if isinstance(best_subset, torch.Tensor):
            return best_subset.detach().cpu().numpy()
        else:
            return np.asarray(best_subset, dtype=np.float32)
        

###############################################
#---------------GreedyMRC----------------------
###############################################
import torch
import numpy as np
import tqdm
from typing import Union, Optional


class GreedyMRCSamplerGPU:
    """
    Memory-efficient Streaming Greedy MRC (no full NxN allocation)
    --------------------------------------------------------------
    Approximation:
      - Uses RBF kernel similarity as proxy for correlation
      - Avoids building full N×N matrix (stream computation)
    """

    def __init__(
        self,
        percentage: float,
        sigma: float = 2.0,
        restarts: int = 3,
        block_size: int = 1024,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        if not 0 < percentage < 1:
            raise ValueError("percentage must be in (0,1)")
        self.percentage = percentage
        self.sigma = sigma
        self.restarts = restarts
        self.block_size = block_size
        self.device = (
            torch.device(device)
            if device is not None
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    # RBF similarity between a vector and matrix
    def _rbf_sim(self, x, X):
        dists = torch.cdist(x[None, :], X, p=2)
        sims = torch.exp(-dists ** 2 / (2 * self.sigma ** 2))
        return sims.squeeze(0)

    def _entropy(self, X: torch.Tensor) -> float:
        cov = torch.cov(X.T) + 1e-6 * torch.eye(X.shape[1], device=self.device)
        det = torch.det(cov)
        det = torch.clamp(det, min=1e-10)
        return 0.5 * torch.log(det).item()

    def _mrc_once(self, X: torch.Tensor, k: int) -> list:
        N, D = X.shape
        selected = []
        first = int(torch.randint(0, N, (1,), device=self.device).item())
        selected.append(first)

        max_sim = torch.zeros(N, device=self.device)

        for _ in tqdm.trange(1, k, desc="MRC-Stream", leave=False):
            # آخرین نمونه انتخاب‌شده
            x_sel = X[selected[-1]]
            # محاسبهٔ شباهت با همه نقاط (در بلاک‌های کوچک)
            for start in range(0, N, self.block_size):
                end = min(start + self.block_size, N)
                sims = self._rbf_sim(x_sel, X[start:end])
                max_sim[start:end] = torch.maximum(max_sim[start:end], sims)

            # حذف اندیس‌های انتخاب‌شده
            max_sim[selected] = float("inf")

            # انتخاب کمترین شباهت
            next_idx = torch.argmin(max_sim).item()
            selected.append(next_idx)

        return selected

    def run(self, features: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, device=self.device, dtype=torch.float32)
        else:
            features = features.to(self.device).float()

        N, D = features.shape
        k = max(1, int(self.percentage * N))

        print(f"🚀 Running streaming MRC: selecting {k}/{N} samples...")

        best_subset = None
        best_entropy = float("inf")

        for r in range(self.restarts):
            print(f"Restart {r + 1}/{self.restarts}")
            selected = self._mrc_once(features, k)
            subset = features[selected]
            ent = self._entropy(subset)

            if ent < best_entropy:
                best_entropy = ent
                best_subset = subset

        print(f"Final streaming MRC subset with entropy = {best_entropy:.4f}")

        if isinstance(best_subset, torch.Tensor):
            return best_subset.detach().cpu().numpy()
        else:
            return np.asarray(best_subset, dtype=np.float32)

#############
'''
######## In the main function:
        elif name == "greedy_mrc":
            return patchcore.sampler.GreedyMRCSamplerGPU(
                                 percentage=percentage,
                                restarts=3,
                                sigma=2.0,
                                block_size=512,
                                device=device
                                )
'''
