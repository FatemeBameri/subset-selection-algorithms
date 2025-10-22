from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import torch

class KMeansPlusSampler(BaseSampler):
    def __init__(self, percentage: float, random_state: int = 0):
        super().__init__(percentage)
        self.random_state = random_state

    def run(self, features):
        self._store_type(features)

        # تبدیل به numpy
        if not isinstance(features, np.ndarray):
            features_np = features.cpu().numpy()
        else:
            features_np = features

        n_clusters = max(1, int(len(features_np) * self.percentage))

        # اجرای k-means
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=self.random_state)
        kmeans.fit(features_np)

        # انتخاب نزدیک‌ترین داده‌ها به مراکز
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features_np)
        subset = features_np[closest_indices]

        # برگرداندن نوع اصلی
        if not self.features_is_numpy:
            subset = torch.tensor(subset, device=self.features_device)

        return subset
