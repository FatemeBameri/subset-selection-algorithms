import torch
import numpy as np
from typing import Union, Optional

class GPUKMedoidsSampler:
    """
    Fast K-Medoids sampler for coresets using PyTorch (GPU if available).
    - Handles feature tuples or numpy arrays safely
    - Computes distances in batches to avoid CUDA OOM
    """

    def __init__(self,
                 k: Optional[int] = None,
                 percentage: Optional[float] = None,
                 max_iter: int = 20,
                 sample_limit_for_medoid: int = 2000,
                 batch_size: int = 1000,
                 device: Optional[str] = None,
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        assert (k is not None) ^ (percentage is not None), "Provide exactly one of k or percentage"
        self.k = k
        self.percentage = percentage
        self.max_iter = max_iter
        self.sample_limit_for_medoid = sample_limit_for_medoid
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.verbose = verbose

    # --- reduce مخصوص KMedoids ---
    def _reduce_for_kmedoids(self, feature):
        if isinstance(feature, tuple):
            feature = feature[0]  # فقط بخش اصلی
        if isinstance(feature, np.ndarray):
            feature = torch.tensor(feature, dtype=torch.float32, device=self.device)
        else:
            feature = feature.to(self.device, dtype=torch.float32)
        return feature.reshape(len(feature), -1)

    def _choose_k(self, n_samples: int) -> int:
        if self.k is not None:
            return max(1, min(self.k, n_samples))
        else:
            k = max(1, int(n_samples * self.percentage))
            return min(k, n_samples)

    def _kmedoids_pp_init(self, data: torch.Tensor, k: int):
        rng = np.random.default_rng(self.random_state)
        first = int(rng.integers(0, data.shape[0]))
        medoid_indices = [first]

        with torch.no_grad():
            medoids = data[medoid_indices]
            dists = torch.cdist(data, medoids).squeeze(1)
            for _ in range(1, k):
                probs = (dists.cpu().numpy() ** 2)
                total = probs.sum()
                if total <= 0:
                    next_idx = int(rng.integers(0, data.shape[0]))
                else:
                    probs = probs / total
                    next_idx = int(rng.choice(data.shape[0], p=probs))
                medoid_indices.append(next_idx)
                medoids = data[medoid_indices]
                dists = torch.min(torch.cdist(data, medoids).min(dim=1).values, dists)
        return torch.tensor(medoid_indices, dtype=torch.long, device=self.device)

    def _recompute_medoid_for_cluster(self, data: torch.Tensor, cluster_idx: torch.Tensor):
        m = cluster_idx.shape[0]
        if m == 0:
            return None
        if m > self.sample_limit_for_medoid:
            rng = np.random.default_rng(self.random_state)
            choose = rng.choice(m, size=self.sample_limit_for_medoid, replace=False)
            subs_idx = cluster_idx[choose]
            subpoints = data[subs_idx]
            pair_d = torch.cdist(subpoints, subpoints)
            costs = pair_d.sum(dim=1)
            best_local = costs.argmin().item()
            return int(subs_idx[best_local].item())
        else:
            points = data[cluster_idx]
            pair_d = torch.cdist(points, points)
            costs = pair_d.sum(dim=1)
            best_local = costs.argmin().item()
            return int(cluster_idx[best_local].item())

    # --- batch-wise cdist ---
    def _batch_cdist_labels(self, data: torch.Tensor, medoids: torch.Tensor):
        batch_size = self.batch_size
        all_labels = []
        all_min_dists = []
        for start in range(0, data.shape[0], batch_size):
            batch = data[start:start+batch_size]
            dists = torch.cdist(batch, medoids)
            labels = torch.argmin(dists, dim=1)
            min_dists = dists.min(dim=1).values
            all_labels.append(labels)
            all_min_dists.append(min_dists)
        return torch.cat(all_labels), torch.cat(all_min_dists)

    def run(self, features: Union[np.ndarray, torch.Tensor]):
        # --- reduce و تبدیل مطمئن به tensor ---
        processed_features = [self._reduce_for_kmedoids(f) for f in features]
        data = torch.cat(processed_features, dim=0)  # همه tensor روی device هستند

        n_samples = data.shape[0]
        k = self._choose_k(n_samples)

        if self.verbose:
            print(f"[KMedoids] n={n_samples}, k={k}, device={self.device}")

        # init medoids
        medoid_indices = self._kmedoids_pp_init(data, k)
        prev_inertia = None

        for it in range(self.max_iter):
            medoids_tensor = data[medoid_indices]
            labels, min_dists = self._batch_cdist_labels(data, medoids_tensor)
            inertia = min_dists.sum().item()

            if self.verbose:
                print(f" it={it}, inertia={inertia:.6f}")

            if prev_inertia is not None and abs(prev_inertia - inertia) <= 1e-4:
                if self.verbose:
                    print(" Converged by tol.")
                break
            prev_inertia = inertia

            # recompute medoids
            new_medoid_indices = medoid_indices.clone()
            for cluster_id in range(k):
                cluster_mask = (labels == cluster_id)
                if not cluster_mask.any():
                    available = torch.tensor([i for i in range(n_samples)
                                              if int(i) not in new_medoid_indices.cpu().tolist()],
                                             device=self.device, dtype=torch.long)
                    if available.numel() > 0:
                        rand_idx = int(torch.randint(0, available.numel(), (1,), device=self.device).item())
                        new_medoid_indices[cluster_id] = available[rand_idx]
                    continue
                cluster_idx = torch.nonzero(cluster_mask, as_tuple=False).squeeze(1)
                best = self._recompute_medoid_for_cluster(data, cluster_idx)
                if best is not None:
                    new_medoid_indices[cluster_id] = best

            if torch.equal(new_medoid_indices, medoid_indices):
                if self.verbose:
                    print(" Medoids unchanged -> converged.")
                break
            medoid_indices = new_medoid_indices

        medoid_indices = torch.unique(medoid_indices)
        return data[medoid_indices], medoid_indices

'''
sampler = GPUKMedoidsSampler(
    k=50,
    max_iter=20,
    sample_limit_for_medoid=1500,
    batch_size=1000,
    random_state=42,
    verbose=True
)

medoids, medoid_idxs = sampler.run(features)
'''
