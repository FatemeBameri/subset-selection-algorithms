from tqdm import trange
from typing import Union, Optional
import torch


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
