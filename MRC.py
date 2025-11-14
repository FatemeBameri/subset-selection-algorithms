#######################################
#-------------MRC----------------------
#######################################
import torch
import numpy as np
from tqdm import trange
from typing import Union, Optional

class MRCSampler(BaseSampler):
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
        super().__init__(percentage) 
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
        R =  torch.sqrt(V2_xy / torch.sqrt(V2_xx * V2_yy))
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

        first = int(torch.randint(0, N, (1,), device=self.device).item())
        selected.append(first)

        for _ in trange(1, k, desc="MRC", leave=False):
            min_corr = float("inf")
            next_idx = None

            for j in range(N):
                if j in selected:
                    continue

                maxR = 0.0
                for s in selected:
                    R = self._distance_correlation(X[j:j+1], X[s:s+1])
                    maxR = max(maxR, R.item())

                if maxR < min_corr:
                    min_corr = maxR
                    next_idx = j

            selected.append(next_idx)

        return selected

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

        print(f"✅ Final MRC subset selected with entropy = {best_entropy:.4f}")
        return best_subset


class FastApproxMRCSampler(BaseSampler):
    """
    Multi-run sampler with:
       - greedy k-center selection (approximate)
       - approximate entropy metric: mean(d) * std(d)
    """

    def __init__(self, percentage=0.1, n_runs=5, proj_dim=128, device="cuda"):
        super().__init__(percentage)
        self.n_runs = n_runs
        self.proj_dim = proj_dim
        self.device = torch.device(device)
        self.proj = None

    def _convert_input(self, X):
        """Convert numpy → torch and move to device."""
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.tensor(X, dtype=torch.float32, device=self.device)

    def _project(self, X):
        """Fast random projection."""
        if self.proj is None:
            D = X.shape[1]
            self.proj = torch.randn(
                D, self.proj_dim, device=self.device
            ) / (self.proj_dim ** 0.5)
        return X @ self.proj

    def _approx_entropy(self, d):
        """Very fast entropy approximation."""
        return d.mean() * d.std()

    @torch.no_grad()
    def run(self, X):
        # Save original features (needed to return the subset later)
        original_X = X

        # Convert numpy → torch
        X = self._convert_input(X)
        N = X.shape[0]
        k = int(max(1, N * self.percentage))

        best_score = float("inf")
        best_selected = None

        for _ in range(self.n_runs):

            # ---- Projection ----
            Xp = self._project(X)

            # ---- Greedy k-center ----
            selected = []
            first = torch.randint(0, N, (1,), device=self.device).item()
            selected.append(first)

            d = torch.norm(Xp - Xp[first], dim=1)

            for _ in range(k - 1):
                nxt = torch.argmax(d).item()
                selected.append(nxt)

                center = Xp[nxt]
                d = torch.minimum(d, torch.norm(Xp - center, dim=1))

            selected_tensor = torch.tensor(selected, device=self.device)

            # ---- Approx entropy ----
            score = self._approx_entropy(d)

            if score < best_score:
                best_score = score
                best_selected = selected_tensor.clone()

        # Sort indices
        indices = torch.sort(best_selected)[0].cpu().numpy()

        # ---- Return subset of original features ----
        if isinstance(original_X, torch.Tensor):
            subset = original_X[indices].cpu().numpy()
        else:
            subset = original_X[indices]

        return subset
