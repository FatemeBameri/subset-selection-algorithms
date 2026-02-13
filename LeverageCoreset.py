from sklearn.utils.extmath import randomized_svd

class LeverageScoreSampler(BaseSampler):
    def __init__(self, percentage: float, device: str = "cpu", rank: int = 64):
        """
        Leverage Score Sampling for coreset selection.

        Args:
            percentage (float)
            device (str)
            rank (int)
        """
        super().__init__(percentage)
        self.device = torch.device(device)
        self.rank = rank

    def run(self, features):
        """
        Leverage Score Sampling.
        Args:
            features (torch.Tensor or np.ndarray): (N, D)
        Returns:
        """
        self._store_type(features)

        if isinstance(features, torch.Tensor):
            X = features.detach().cpu().numpy()
        else:
            X = features

        N, D = X.shape
        k = max(1, int(N * self.percentage))

        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

        try:
            U, S, Vt = randomized_svd(
                X,
                n_components=min(self.rank, D),
                random_state=0
            )
        except Exception as e:
            raise RuntimeError(f"Randomized SVD failed: {e}")

        leverage_scores = np.sum(U ** 2, axis=1)
        leverage_scores = np.clip(leverage_scores, 0, None)
        leverage_scores /= np.sum(leverage_scores)

        selected_idx = np.random.choice(N, size=k, replace=False, p=leverage_scores)
        subset = X[selected_idx]

        if not self.features_is_numpy:
            subset = torch.tensor(subset, device=self.features_device, dtype=torch.float32)

        return subset

