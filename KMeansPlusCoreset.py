import numpy as np
from sklearn.cluster import MiniBatchKMeans

class KMeansPlusSampler(BaseSampler):
    def __init__(self, percentage: float, random_state: int = 0, batch_size: int = 1024, sample_size: int = 50000):
        super().__init__(percentage)
        self.random_state = random_state
        self.batch_size = batch_size
        self.sample_size = sample_size 

    def run(self, features):
        self._store_type(features)
        features_np = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else features

        if len(features_np) > self.sample_size:
            idxs = np.random.choice(len(features_np), self.sample_size, replace=False)
            features_sample = features_np[idxs]
        else:
            features_sample = features_np

        n_clusters = max(1, int(len(features_np) * self.percentage))

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            init="k-means++",
            batch_size=self.batch_size,
            random_state=self.random_state,
            n_init=1,
            max_iter=100
        )
        kmeans.fit(features_sample)

        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32, device='cpu') 

        cluster_centers = torch.tensor(kmeans.cluster_centers_, device=features.device, dtype=features.dtype)
        distances = torch.cdist(cluster_centers, features)
        closest_indices = torch.argmin(distances, dim=1)
        subset = features[closest_indices]

        return subset
