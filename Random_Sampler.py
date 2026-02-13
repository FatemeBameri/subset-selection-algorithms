
class RandomSampler(BaseSampler):
    def __init__(self, percentage: float):
        super().__init__(percentage)

    def run(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        num_random_samples = int(len(features) * self.percentage)
        subset_indices = np.random.choice(len(features), num_random_samples, replace=False)
        subset_indices = np.array(subset_indices)
        return features[subset_indices]
