from typing import Union
import torch
import numpy as np
from sklearn_extra.cluster import KMedoids


class KMedoidsSampler(BaseSampler):
    def __init__(self, percentage: float, random_state: int = 42):
        """
        Subset selection using k-Medoids (PAM) algorithm.

        Args:
            percentage: float in (0,1), fraction of data to select
            random_state: random seed for reproducibility
        """
        super().__init__(percentage)
        self.random_state = random_state

    def run(
            self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        # ذخیره نوع داده اصلی
        self._store_type(features)

        # تبدیل به numpy برای KMedoids
        if not isinstance(features, np.ndarray):
            features_np = features.cpu().numpy()
        else:
            features_np = features

        # تعیین تعداد نمونه‌های انتخابی
        n_samples = max(1, int(len(features_np) * self.percentage))

        # اجرای k-Medoids
        kmedoids = KMedoids(n_clusters=n_samples, metric="euclidean", random_state=0)
        kmedoids.fit(features_np)

        # اندیس نمونه‌های انتخاب شده (medoids)
        subset_indices = kmedoids.medoid_indices_
        subset = features_np[subset_indices]

        # بازگرداندن نوع داده اصلی
        return self._restore_type(torch.tensor(subset) if not self.features_is_numpy else subset)
