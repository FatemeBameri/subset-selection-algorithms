import numpy as np
import torch
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances
from typing import Union
import abc

# ---- نسخه CLARA با device ----
class CLARASampler(BaseSampler):
    """
    CLARA (Clustering LARge Applications) Sampler.
    Efficient subset selection using repeated K-Medoids clustering on samples.
    """

    def __init__(
        self,
        percentage: float,
        n_clusters: int = 10,
        n_iter: int = 5,
        random_state: int = 42,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(percentage)
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.random_state = np.random.RandomState(random_state)
        self.device = torch.device(device)

    def run(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Run CLARA sampling.
        Args:
            features: Tensor [N, D] or ndarray [N, D]
        Returns:
            A subset of representative samples (≈ percentage * N)
        """
        self._store_type(features)

        # تبدیل به NumPy برای KMedoids
        if isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = features

        N = len(features_np)
        n_samples = max(1, int(N * self.percentage))

        best_medoids = None
        best_cost = np.inf

        for i in range(self.n_iter):
            subset_size = min(1000, N)
            subset_idx = self.random_state.choice(N, subset_size, replace=False)
            subset = features_np[subset_idx]

            # اجرای KMedoids روی subset
            kmedoids = KMedoids(
                n_clusters=min(self.n_clusters, subset.shape[0]),
                metric="euclidean",
                random_state=self.random_state
            )
            kmedoids.fit(subset)
            medoids = kmedoids.cluster_centers_

            # هزینه روی کل داده‌ها
            distances = pairwise_distances(features_np, medoids, metric="euclidean")
            cost = np.sum(np.min(distances, axis=1))

            #print(f"[CLARA Iter {i+1}/{self.n_iter}] Cost = {cost:.2f}")

            if cost < best_cost:
                best_cost = cost
                best_medoids = medoids

        # انتخاب نزدیک‌ترین داده‌ها به medoids
        distances = pairwise_distances(features_np, best_medoids, metric="euclidean")
        closest_indices = np.argmin(distances, axis=0)
        selected_indices = np.unique(closest_indices)[:n_samples]

        subset = features_np[selected_indices]

        # بازگرداندن نوع اصلی و انتقال به device اصلی
        subset_tensor = torch.tensor(subset, device=self.device)
        if self.features_is_numpy:
            return subset_tensor.cpu().numpy()
        else:
            return subset_tensor.to(self.features_device)
