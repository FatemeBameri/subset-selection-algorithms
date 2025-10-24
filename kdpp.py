from dppy.finite_dpps import FiniteDPP
import numpy as np
import inspect
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

        
        features_np = features_np / np.linalg.norm(features_np, axis=1, keepdims=True)

        n = len(features_np)
        k = max(1, int(n * self.percentage))

        # Kernel 
        L = features_np @ features_np.T

        # k-DPP
        dpp = FiniteDPP('likelihood', **{'L': L})
        subset_indices = dpp.sample_k_dpp(k=k)

        subset = features_np[subset_indices]
        subset = torch.tensor(subset, device=self.device) if not self.features_is_numpy else subset

        return subset


# Approximation version of KDPP

class ApproximateDPPSampler(BaseSampler):
    def __init__(self, percentage: float, device: str = "cpu", subset_size: int = 10000, mcmc_iters: int = 5000):
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

       
        X_sub = X_sub / np.linalg.norm(X_sub, axis=1, keepdims=True)

       
        L_sub = X_sub @ X_sub.T
        L_sub = 0.5 * (L_sub + L_sub.T) 
        L_sub += np.eye(L_sub.shape[0]) * 1e-6  # regularization

       
        dpp = FiniteDPP('likelihood', **{'L': L_sub})
        k_sub = min(k, len(X_sub))
        selected_sub_idx = None

     
        # 1) exact k-DPP
        if hasattr(dpp, "sample_exact_k_dpp"):
            try:
                dpp.sample_exact_k_dpp(size=int(k_sub))
                selected_sub_idx = np.array(dpp.list_of_samples[-1])
                print("Used sample_exact_k_dpp")
            except Exception as e:
                print(f"sample_exact_k_dpp failed: {e}")

        # 2) MCMC k-DPP
        if selected_sub_idx is None and hasattr(dpp, "sample_mcmc_k_dpp"):
            try:
                dpp.sample_mcmc_k_dpp(size=int(k_sub), n_iter=self.mcmc_iters)
                selected_sub_idx = np.array(dpp.list_of_samples[-1])
                print("Used sample_mcmc_k_dpp")
            except Exception as e:
                print(f"sample_mcmc_k_dpp failed: {e}")

       
        if selected_sub_idx is None and hasattr(dpp, "sample_exact"):
            try:
                try:
                    dpp.sample_exact(size=int(k_sub))
                except TypeError:
                    dpp.sample_exact(int(k_sub))
                selected_sub_idx = np.array(dpp.list_of_samples[-1])
                print("Used sample_exact")
            except Exception as e:
                print(f"sample_exact failed: {e}")

        
        if selected_sub_idx is None and hasattr(dpp, "sample_mcmc"):
            try:
                sig = inspect.signature(dpp.sample_mcmc)
                if "mode" in sig.parameters:
                    for mode_try in ("k_dpp", "k-dpp", "k", "fixed"):
                        try:
                            dpp.sample_mcmc(mode_try, n_iter=self.mcmc_iters)
                            selected_sub_idx = np.array(dpp.list_of_samples[-1])
                            print(f"Used sample_mcmc with mode='{mode_try}'")
                            break
                        except Exception:
                            continue
                else:
                    dpp.sample_mcmc(n_iter=self.mcmc_iters)
                    selected_sub_idx = np.array(dpp.list_of_samples[-1])
                    print("Used sample_mcmc (no mode)")
            except Exception as e:
                print(f"sample_mcmc failed: {e}")

     
        if selected_sub_idx is None or len(selected_sub_idx) == 0:
            print("DPP sampling failed. Using random fallback.")
            selected_sub_idx = np.random.choice(L_sub.shape[0], int(k_sub), replace=False)

     
        selected_idx = subset_idx[selected_sub_idx]
        subset = X[selected_idx]

        if not self.features_is_numpy:
            subset = torch.tensor(subset, device=self.features_device)

        return subset

