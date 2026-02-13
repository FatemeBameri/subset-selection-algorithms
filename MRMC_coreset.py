
class HybridSquaredLossMRMC(BaseSampler):
    """
    Hybrid sampler: first top-k squared-loss (variance), then greedy k-center selection.
    Compatible with PatchCore memory bank.
    """
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        squared_loss_fraction: float = 0.5,
        dimension_to_project_features_to: int = 128,
    ):
        """
        Args:
            percentage (float): final fraction of features to keep
            squared_loss_fraction (float): fraction to keep after squared-loss selection before k-center
            device (torch.device)
        """
        super().__init__(percentage)
        self.device = device
        self.squared_loss_fraction = squared_loss_fraction
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def run(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:

        if self.percentage >= 1.0:
            return features

        self._store_type(features)

        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)

        features = features.to(self.device)
        reduced_features = self._reduce_features(features)

        num_initial = max(1, int(len(features) * self.squared_loss_fraction))
        mu = torch.mean(reduced_features, dim=0, keepdim=True)
        squared_loss = torch.sum((reduced_features - mu)**2, dim=1)
        _, topk_indices = torch.topk(squared_loss, k=num_initial, largest=True)
        features_topk = reduced_features[topk_indices]

        num_final = max(1, int(len(features) * self.percentage))
        selected_indices = self._greedy_k_center(features_topk, num_final)

        final_indices = topk_indices[selected_indices]
        features = features[final_indices]


        return self._restore_type(features)

    @staticmethod
    def _compute_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Batchwise Euclidean distance."""
        a2 = (a**2).sum(dim=1, keepdim=True)
        b2 = (b**2).sum(dim=1, keepdim=True).T
        ab = a @ b.T
        return torch.sqrt((a2 - 2*ab + b2).clamp(min=0))

    def _greedy_k_center(self, features: torch.Tensor, k: int) -> np.ndarray:
        """Greedy k-center selection (deterministic)."""
        N = features.shape[0]
        if k >= N:
            return np.arange(N)

        # Initialize distances
        distances = torch.full((N,), float('inf'), device=features.device)
        selected = []

        # pick first point arbitrarily
        first_idx = 0
        selected.append(first_idx)
        dist_new = self._compute_distances(features, features[first_idx:first_idx+1]).squeeze()
        distances = torch.min(distances, dist_new)

        for _ in range(1, k):
            next_idx = torch.argmax(distances).item()
            selected.append(next_idx)
            dist_new = self._compute_distances(features, features[next_idx:next_idx+1]).squeeze()
            distances = torch.min(distances, dist_new)

        return np.array(selected)

