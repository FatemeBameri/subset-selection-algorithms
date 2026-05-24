import numpy as np
import torch
from apricot import FacilityLocationSelection
from typing import Union

class ApproximateFacilitySampler(BaseSampler):
    def __init__(
            self,
            percentage: float,
            device: str = "cuda",
            subset_size: int = 30000,
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
        self.my_metric = metric


    def run(self, features: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        self._store_type(features)

        # Convert to numpy if input is torch.Tensor
        if isinstance(features, torch.Tensor):
            X = features.detach().cpu().numpy()
        else:
            X = features

        N = X.shape[0]
        k = max(1, int(N * self.percentage))
        #self.subset_size = k

        # Step 1: Select a random subset
        subset_idx = np.random.choice(N, min(self.subset_size, N), replace=False)
        X_sub = X[subset_idx]


        # Step 2: Apply Facility Location Selection on subset
        fl = FacilityLocationSelection(
            n_samples=min(k, len(X_sub)),
            metric=self.my_metric,
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

