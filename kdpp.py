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

import numpy as np
import torch
from dppy.finite_dpps import FiniteDPP
from typing import Union

class ApproximateDPPSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: str = "cpu",
        subset_size: int = 10000,
        random_state: int = 0,
        mcmc_iters: int = 1000,
    ):
        """
        Args:
            percentage: fraction of total items to sample (e.g. 0.1)
            device: "cpu" or "cuda" (output type restored to input type)
            subset_size: size of random subset to run exact/approx DPP on
            random_state: RNG seed for reproducibility
            mcmc_iters: number of MCMC iterations if falling back to sample_mcmc_k
        """
        super().__init__(percentage)
        self.device = torch.device(device)
        self.subset_size = int(subset_size)
        self.rng = np.random.RandomState(random_state)
        self.mcmc_iters = int(mcmc_iters)

    def _ensure_psd(self, L: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Ensure L is symmetric PSD by eigen-clipping and symmetrizing."""
        # symmetrize first (numerical safety)
        L = (L + L.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(L)
        # clip tiny negative eigenvalues to zero
        eigvals_clipped = np.clip(eigvals, 0.0, None)
        # rebuild L
        L_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        # final symmetrize
        return (L_psd + L_psd.T) / 2.0

    def run(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        self._store_type(features)

        # convert to numpy for dppy
        if isinstance(features, torch.Tensor):
            X = features.detach().cpu().numpy()
        else:
            X = np.asarray(features)

        N = X.shape[0]
        k = max(1, int(N * self.percentage))

        # choose random subset indices
        subset_n = min(self.subset_size, N)
        subset_idx = self.rng.choice(N, subset_n, replace=False)
        X_sub = X[subset_idx]

        # normalize rows (avoid zero vectors)
        norms = np.linalg.norm(X_sub, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_sub = X_sub / norms

        # build small kernel L_sub = X_sub X_sub^T
        L_sub = X_sub @ X_sub.T

        # ensure PSD (important for dppy.sample_exact_k_dpp)
        L_sub = self._ensure_psd(L_sub)

        # create FiniteDPP with L_sub
        dpp = FiniteDPP('likelihood', L=L_sub)

        # desired k on the subset
        k_sub = min(k, X_sub.shape[0])

        # Try available sampling methods in a robust order:
        selected_sub_idx = None

        # 1) try exact k-DPP method if exists (older names differ across versions)
        if hasattr(dpp, "sample_exact_k_dpp"):
            try:
                dpp.sample_exact_k_dpp(size=int(k_sub))
                selected_sub_idx = np.array(dpp.list_of_samples[-1])
            except Exception as e:
                # fallthrough to next option
                selected_sub_idx = None

        # 2) try MCMC-based k-DPP if available (sample_mcmc_k)
        if selected_sub_idx is None and hasattr(dpp, "sample_mcmc_k"):
            try:
                # sample_mcmc_k signature may accept size and n_iter (varies by version)
                # we'll try a couple of common calling patterns
                try:
                    dpp.sample_mcmc_k(size=int(k_sub), n_iter=self.mcmc_iters)
                except TypeError:
                    # some versions expect (k, n_iter)
                    dpp.sample_mcmc_k(int(k_sub), self.mcmc_iters)
                selected_sub_idx = np.array(dpp.list_of_samples[-1])
            except Exception:
                selected_sub_idx = None

        # 3) try sample_exact (may or may not support size argument)
        if selected_sub_idx is None and hasattr(dpp, "sample_exact"):
            try:
                # try size keyword
                try:
                    dpp.sample_exact(size=int(k_sub))
                except TypeError:
                    # fallback: some versions accept k as first arg
                    dpp.sample_exact(int(k_sub))
                selected_sub_idx = np.array(dpp.list_of_samples[-1])
            except Exception:
                selected_sub_idx = None

        # if still nothing, raise explicit error with explanation
        if selected_sub_idx is None:
            raise RuntimeError(
                "DPP sampling failed: none of sample_exact_k_dpp / sample_mcmc_k / sample_exact succeeded. "
                "Check dppy version and available methods. You can also try increasing subset_size or using a different fallback."
            )

        # map back to original indices and return subset of features
        selected_idx = subset_idx[selected_sub_idx]
        sel_feats = X[selected_idx]

        # restore type
        if not self.features_is_numpy:
            sel_feats = torch.tensor(sel_feats, device=self.features_device, dtype=torch.float32)
        return sel_feats


# Approximation version of KDPP

class ApproximateDPPSampler(BaseSampler):
    def __init__(self, percentage: float, device: str = "cpu", subset_size: int = 10000, mcmc_iters: int = 500):
        super().__init__(percentage)
        self.device = torch.device(device)
        self.subset_size = subset_size
        self.mcmc_iters = mcmc_iters

    def run(self, features):
        self._store_type(features)
        if isinstance(features, torch.Tensor):
            X = features.cpu().numpy()
        else:
            X = features

        N = len(X)
        k = max(1, int(N * self.percentage))

       
        subset_idx = np.random.choice(N, min(self.subset_size, N), replace=False)
        X_sub = X[subset_idx]
        X_sub = X_sub / (np.linalg.norm(X_sub, axis=1, keepdims=True) + 1e-10)
        L_sub = X_sub @ X_sub.T

       
        eigvals, eigvecs = np.linalg.eigh(L_sub)
        eigvals[eigvals < 0] = 0 
        L_sub = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L_sub += np.eye(L_sub.shape[0]) * 1e-6  # regularization

       
        rank_L = np.sum(eigvals > 1e-8)
        k_sub = min(k, rank_L)

      
        dpp = FiniteDPP('likelihood', **{'L': L_sub})
        selected_sub_idx = None

        try:
           
            dpp.sample_exact_k_dpp(size=k_sub)
            selected_sub_idx = np.array(dpp.list_of_samples[-1])
        except Exception as e1:
            print(f"sample_exact_k_dpp failed: {e1}")
            try:
                dpp.sample_mcmc_k_dpp(size=k_sub, n_iter=self.mcmc_iters)
                selected_sub_idx = np.array(dpp.list_of_samples[-1])
            except Exception as e2:
                print(f"sample_mcmc_k_dpp failed: {e2}")
                try:
                    dpp.sample_exact()
                    selected_sub_idx = np.array(dpp.list_of_samples[-1])
                except Exception as e3:
                    print(f"sample_exact failed: {e3}")
                    dpp.sample_mcmc(n_iter=self.mcmc_iters)
                    selected_sub_idx = np.array(dpp.list_of_samples[-1])


        if selected_sub_idx is None or len(selected_sub_idx) == 0:
            print("DPP sampling failed, falling back to random subset.")
            selected_sub_idx = np.random.choice(L_sub.shape[0], k_sub, replace=False)


        selected_idx = subset_idx[selected_sub_idx]
        subset = X[selected_idx]

        if not self.features_is_numpy:
            subset = torch.tensor(subset, device=self.features_device)

        return subset
