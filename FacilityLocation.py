from apricot import FacilityLocationSelection
from typing import Union
import numpy as np
import torch


class FacilityLocationSampler(BaseSampler):
    """
    Submodular Facility Location coreset sampler using apricot.
    Efficient, diverse subset selection suitable for large feature sets.
    """

    def __init__(self, percentage: float, device: str = "cpu"):
        super().__init__(percentage)
        self.device = torch.device(device)

    def run(self, features):
      
        self._store_type(features)

        # numpy
        if isinstance(features, torch.Tensor):
            X = features.detach().cpu().numpy()
        else:
            X = np.array(features)

        n = len(X)
        k = max(1, int(n * self.percentage))

        # Facility Location selection
        selector = FacilityLocationSelection(
            n_samples=k,
            metric='cosine',     # می‌توان تغییر داد به 'euclidean' یا 'precomputed'
            random_state=0
        )

        subset = selector.fit_transform(X)

        if not self.features_is_numpy:
            subset = torch.tensor(subset, device=self.device)

        return subset

#############################
############################


class ApproximateFacilitySampler(BaseSampler):
    def __init__(
            self,
            percentage: float,
            device: str = "cpu",
            subset_size: int = 20000,
            metric: str = "cosine",
            random_state: int = 0
    ):
        """
        Approximate Submodular Facility Location Sampler using a random subset.

        Args:
            percentage: fraction of total features to select
            device: "cpu" or "cuda"
            subset_size: size of random subset to run selection on
            metric: similarity metric for Facility Location ("cosine", "euclidean", etc.)
            random_state: seed for reproducibility
        """
        super().__init__(percentage)
        self.device = torch.device(device)
        self.subset_size = subset_size
        self.metric = metric
        self.random_state = random_state

    def run(self, features: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        self._store_type(features)

        # Convert to numpy if input is torch.Tensor
        if isinstance(features, torch.Tensor):
            X = features.detach().cpu().numpy()
        else:
            X = features

        N = X.shape[0]
        k = max(1, int(N * self.percentage))

        # Step 1: Select a random subset
        subset_idx = np.random.choice(N, min(self.subset_size, N), replace=False)
        X_sub = X[subset_idx]

        # Step 2: Apply Facility Location Selection on subset
        fl = FacilityLocationSelection(
            n_samples=min(k, len(X_sub)),
            metric=self.metric,
            random_state=self.random_state
        )
        fl.fit(X_sub)
        selected_sub_idx = fl.ranking  # indices within the subset

        # Step 3: Map back to original dataset
        selected_idx = subset_idx[selected_sub_idx]
        subset = X[selected_idx]

        # Restore type
        if not self.features_is_numpy:
            subset = torch.tensor(subset, device=self.features_device)

        return subset
