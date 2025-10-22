from typing import Union
import numpy as np
import torch
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

        # تبدیل به NumPy برای محاسبه فاصله‌ها
        if not isinstance(features, np.ndarray):
            features_np = features.cpu().numpy()
        else:
            features_np = features

        n_samples = max(1, int(len(features_np) * self.percentage))
        selected_indices = self._kcenter_greedy(features_np, n_samples)
        subset = features_np[selected_indices]

        # بازگرداندن نوع اصلی
        if not self.features_is_numpy:
            subset = torch.tensor(subset, device=self.features_device)
        return subset

    @staticmethod
    def _kcenter_greedy(X: np.ndarray, n_samples: int) -> np.ndarray:
        N = X.shape[0]
        if n_samples >= N:
            return np.arange(N)

        # انتخاب اولین نمونه تصادفی
        selected = [np.random.randint(N)]
        distances = pairwise_distances(X, X[selected], metric="euclidean").flatten()

        for _ in range(1, n_samples):
            idx = np.argmax(distances)
            selected.append(idx)
            # به‌روزرسانی فاصله‌ها با نزدیک‌ترین نمونه انتخاب شده
            new_dist = pairwise_distances(X, X[[idx]], metric="euclidean").flatten()
            distances = np.minimum(distances, new_dist)

        return np.array(selected)
#*****************************************************
#*****************************************************
#*****************************************************
class ApproximateKCenterGreedySampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
    ):
        """
        Approximate K-Center Greedy Sampler:
        Similar to KCenterGreedy but uses a subset of starting points
        to reduce computational cost.
        """
        super().__init__(percentage)
        self.device = device
        self.number_of_starting_points = number_of_starting_points

    def run(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        self._store_type(features)

        # تبدیل به NumPy برای محاسبه فاصله‌ها
        if not isinstance(features, np.ndarray):
            features_np = features.cpu().numpy()
        else:
            features_np = features

        n_samples = max(1, int(len(features_np) * self.percentage))
        selected_indices = self._approximate_kcenter_greedy(features_np, n_samples)
        subset = features_np[selected_indices]

        # بازگرداندن نوع اصلی
        if not self.features_is_numpy:
            subset = torch.tensor(subset, device=self.features_device)
        return subset

    def _approximate_kcenter_greedy(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Approximate version of K-Center Greedy using subset-based initialization.
        """
        N = X.shape[0]
        if n_samples >= N:
            return np.arange(N)

        # انتخاب چند نقطه تصادفی برای تقریب اولیه
        start_points = np.random.choice(N, min(self.number_of_starting_points, N), replace=False)
        selected = list(start_points)

        # محاسبه فاصله میان تمام نقاط و نقاط اولیه (به جای کل ماتریس N×N)
        distances = pairwise_distances(X, X[start_points], metric="euclidean")
        min_distances = np.mean(distances, axis=1)

        for _ in tqdm.tqdm(range(len(selected), n_samples), desc="Approx K-Center"):
            # انتخاب نقطه با بیشترین فاصله از نقاط انتخاب‌شده
            idx = np.argmax(min_distances)
            selected.append(idx)

            # محاسبه فاصله جدید فقط با همان نقطه انتخاب‌شده
            new_dist = pairwise_distances(X, X[[idx]], metric="euclidean").flatten()
            min_distances = np.minimum(min_distances, new_dist)

        return np.array(selected)
