import torch
import numpy as np
import tqdm
from typing import Union, Optional


class GreedyKDppSamplerGPU:
    def __init__(
        self,
        percentage: float,
        sigma: float = 5.0,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        """
        GPU-friendly greedy KDPP sampler (fixed).
        Args:
            percentage: fraction in (0,1) of points to select.
            sigma: RBF bandwidth.
            device: 'cuda' or 'cpu' or torch.device. If None autodetects.
            seed: optional random seed for reproducibility.
        """
        if not 0 < percentage < 1:
            raise ValueError("percentage must be in (0,1)")
        self.percentage = percentage
        self.sigma = float(sigma)
        self.device = (
            torch.device(device)
            if device is not None
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )
        self.seed = seed
        if seed is not None:
            # set seeds (numpy + torch)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _ensure_2d_tensor(X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 1:
            return X.unsqueeze(0)
        return X

    def _rbf_kernel_vector(self, x: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF kernel between vector x [D] and matrix X [M,D] on device.
        Returns shape [M], dtype float32.
        """
        # ensure shapes
        x = x.to(self.device).float()
        X = X.to(self.device).float()
        # X: [M,D], x: [D] or [1,D]
        if x.ndim == 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        # compute squared distances
        # (X - x)**2 sum along dim=1
        dists = torch.sum((X - x) ** 2, dim=1)
        return torch.exp(-dists / (2.0 * (self.sigma ** 2)))

    def run(self, features: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Input: features [N, D] (np.ndarray or torch.Tensor)
        Output: subset features [k, D] (same type as input)
        """
        if isinstance(features, np.ndarray):
            if features.ndim != 2:
                raise ValueError("features numpy array must be 2D [N,D]")
            is_numpy = True
            X = torch.tensor(features, device=self.device, dtype=torch.float32)
        elif isinstance(features, torch.Tensor):
            if features.ndim != 2:
                raise ValueError("features tensor must be 2D [N,D]")
            is_numpy = False
            X = features.to(self.device).float()
        else:
            raise TypeError("features must be np.ndarray or torch.Tensor")

        N, D = X.shape
        if N == 0:
            return np.zeros((0, D), dtype=np.float32) if is_numpy else torch.zeros((0, D), device=features.device)

        k = max(1, int(self.percentage * N))
        if k > N:
            k = N

        # selected indices (on CPU list)
        selected = []
        # mask of remaining (on device)
        mask = torch.ones(N, dtype=torch.bool, device=self.device)
        # max_sim vector for all N (float)
        max_sim = torch.zeros(N, dtype=torch.float32, device=self.device)

        # first selection: random
        first_idx = int(torch.randint(0, N, (1,), device=self.device).item())
        selected.append(first_idx)
        mask[first_idx] = False

        # compute initial max_sim for remaining
        remaining_idx = torch.where(mask)[0]  # always 1D
        if remaining_idx.numel() > 0:
            sims = self._rbf_kernel_vector(X[first_idx], X[remaining_idx])  # shape [M]
            max_sim[remaining_idx] = sims

        # greedy loop
        for _ in tqdm.tqdm(range(1, k), desc="Greedy KDPP GPU"):
            # find index among remaining with minimal max_sim
            # min over masked entries
            masked_vals = max_sim[mask]  # 1D tensor
            # safe-guard if masked_vals empty
            if masked_vals.numel() == 0:
                break
            local_argmin = int(torch.argmin(masked_vals).item())
            remaining_idx = torch.where(mask)[0]  # update remaining (1D)
            next_idx = int(remaining_idx[local_argmin].item())
            selected.append(next_idx)
            # mark removed
            mask[next_idx] = False

            # recompute remaining and update max_sim on remaining
            remaining_idx = torch.where(mask)[0]
            if remaining_idx.numel() == 0:
                break
            sim = self._rbf_kernel_vector(X[next_idx], X[remaining_idx])  # [M]
            # update only remaining positions
            max_sim[remaining_idx] = torch.maximum(max_sim[remaining_idx], sim)

        selected = np.array(selected, dtype=np.int64)

        # return same type as input
        if is_numpy:
            subset = features[selected]  # numpy indexing
            # ensure float32 contiguous for FAISS
            return np.ascontiguousarray(subset.astype(np.float32, copy=False))
        else:
            # original features was torch.Tensor -> return same device as original
            subset = X[selected]  # currently on self.device
            # move back to original device if needed
            if features.device != self.device:
                subset = subset.to(features.device)
            return subset

'''
import numpy as np
import torch

N, D = 2000, 64
features_np = np.random.randn(N, D).astype(np.float32)
sampler = GreedyKDppSamplerGPU(percentage=0.05, sigma=5.0, seed=42)
subset_np = sampler.run(features_np)
print("subset_np.shape:", subset_np.shape)  # should be (k, D)

# torch input
features_t = torch.randn(N, D)
sampler_t = GreedyKDppSamplerGPU(percentage=0.05, sigma=5.0, seed=42)
subset_t = sampler_t.run(features_t)
print("subset_t.shape:", subset_t.shape)
'''

