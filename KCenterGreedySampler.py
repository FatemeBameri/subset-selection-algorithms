import abc
import torch
import numpy as np
from tqdm import tqdm
from typing import Union
from sklearn.metrics import pairwise_distances


class KCenterGreedySampler(BaseSampler):
    def __init__(self, percentage: float, device: torch.device):
        super().__init__(percentage)
        self.device = device

    def run(
            self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Perform greedy k-center sampling on input features.

        Args:
            features: Tensor [N, D] or ndarray [N, D] of features

        Returns:
            Subset of features with size determined by self.percentage
        """
        self._store_type(features)

        if not isinstance(features, np.ndarray):
            features_np = features.cpu().numpy()
        else:
            features_np = features

        n_samples = max(1, int(len(features_np) * self.percentage))
        selected_indices = self._kcenter_greedy(features_np, n_samples)
        subset = features_np[selected_indices]

        if not self.features_is_numpy:
            subset = torch.tensor(subset, device=self.features_device)
        return subset

    @staticmethod
    def _kcenter_greedy(X: np.ndarray, n_samples: int) -> np.ndarray:
        N = X.shape[0]
        if n_samples >= N:
            return np.arange(N)

        selected = [np.random.randint(N)]
        distances = pairwise_distances(X, X[selected], metric="euclidean").flatten()

        for _ in range(1, n_samples):
            idx = np.argmax(distances)
            selected.append(idx)

            new_dist = pairwise_distances(X, X[[idx]], metric="euclidean").flatten()
            distances = np.minimum(distances, new_dist)

        return np.array(selected)
        
        
#*****************************************************
#*****************************************************


class ApproximateKCenterGreedySampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        batch_size: int = 5000,
    ):
        super().__init__(percentage)
        self.device = device
        self.number_of_starting_points = number_of_starting_points
        self.batch_size = batch_size
        self.use_faiss = faiss_available  

    def run(self, features):
        self._store_type(features)
        if not isinstance(features, np.ndarray):
            features_np = features.cpu().numpy().astype(np.float32)
        else:
            features_np = features.astype(np.float32)

        n_samples = max(1, int(len(features_np) * self.percentage))
        selected_indices = self._approximate_kcenter_greedy(features_np, n_samples)

        subset = features_np[selected_indices]
        if not self.features_is_numpy:
            subset = torch.tensor(subset, device=self.features_device)
        return subset

    def _approximate_kcenter_greedy(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        N = X.shape[0]
        if n_samples >= N:
            return np.arange(N)

        start_points = np.random.choice(N, min(self.number_of_starting_points, N), replace=False)
        selected = list(start_points)

        min_distances = self._compute_min_distances(X, X[selected])

        for _ in tqdm(range(len(selected), n_samples), desc="Approx K-Center (fast)"):
            idx = np.argmax(min_distances)
            selected.append(idx)
            new_dists = self._compute_min_distances(X, X[[idx]])
            min_distances = np.minimum(min_distances, new_dists)

        return np.array(selected)

    def _compute_min_distances(self, X, centers):
        if self.use_faiss:
            X32 = X.astype(np.float32)
            centers32 = centers.astype(np.float32)
            index = faiss.IndexFlatL2(X32.shape[1])
            index.add(centers32)
            D, _ = index.search(X32, 1)
            return np.sqrt(D).flatten()
        else:
            N = X.shape[0]
            batch_size = self.batch_size
            min_dists = np.full(N, np.inf, dtype=np.float32)

            for i in range(0, N, batch_size):
                X_batch = X[i:i+batch_size]
                dists = np.linalg.norm(X_batch[:, None, :] - centers[None, :, :], axis=2)
                batch_min = np.min(dists, axis=1)
                min_dists[i:i+batch_size] = np.minimum(min_dists[i:i+batch_size], batch_min)
            return min_dists
