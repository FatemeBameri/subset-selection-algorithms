from dppy.finite_dpps import FiniteDPP
import numpy as np
import torch

class DPPSampler(BaseSampler):
    def __init__(self, percentage: float, device: str = "cpu"):
        super().__init__(percentage)
        self.device = torch.device(device)

    def run(self, features):
        self._store_type(features)
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy()
        else:
            features_np = features

        # نرمال‌سازی
        features_np = features_np / np.linalg.norm(features_np, axis=1, keepdims=True)

        n = len(features_np)
        k = max(1, int(n * self.percentage))

        # Kernel ماتریس
        L = features_np @ features_np.T

        # اجرای k-DPP
        dpp = FiniteDPP('likelihood', **{'L': L})
        subset_indices = dpp.sample_k_dpp(k=k)

        subset = features_np[subset_indices]
        subset = torch.tensor(subset, device=self.device) if not self.features_is_numpy else subset

        return subset
