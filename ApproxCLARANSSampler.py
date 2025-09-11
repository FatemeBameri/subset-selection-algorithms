import torch
import numpy as np
from typing import Union
from pyclustering.cluster.clarans import clarans

class AutoDeviceApproxCLARANSSampler:
    """
    Approximate CLARANS استاندارد با Auto CPU/GPU
    - مدویدهای واقعی
    - Approximation اصولی از طریق subset تصادفی
    - فاصله‌ها روی GPU در صورت موجود بودن
    - جلوگیری از خطای تعداد medoid بیشتر از subset
    """

    def __init__(self, percentage: float, numlocal: int = 3, maxneighbor: int = 20,
                 random_state: int = None):
        self.percentage = percentage
        self.numlocal = numlocal
        self.maxneighbor = maxneighbor
        self.random_state = random_state
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run(self, features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        # تبدیل داده‌ها به torch.Tensor روی device
        if not isinstance(features, torch.Tensor):
            data_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        else:
            data_tensor = features.to(self.device, dtype=torch.float32)

        n_samples = data_tensor.shape[0]
        n_medoids = max(1, int(n_samples * self.percentage))

        # تنظیم خودکار subset و batch
        sample_fraction = min(0.1, 5000 / n_samples)  # حداکثر 5000 نمونه
        sample_size = max(200, int(n_samples * sample_fraction))
        # جلوگیری از خطا: تعداد medoid نباید از subset بیشتر شود
        n_medoids = min(n_medoids, sample_size)
        batch_size = max(1000, int(n_samples * 0.05))

        rng = np.random.default_rng(self.random_state)
        sample_indices = rng.choice(n_samples, size=sample_size, replace=False)
        sample_data = data_tensor[sample_indices].cpu().numpy()  # pyclustering نیاز به numpy

        # اجرای CLARANS واقعی روی subset
        clarans_instance = clarans(
            data=sample_data.tolist(),
            number_clusters=n_medoids,
            numlocal=self.numlocal,
            maxneighbor=self.maxneighbor
        )
        clarans_instance.process()
        sample_medoid_indices = clarans_instance.get_medoids()
        sample_medoids = sample_data[sample_medoid_indices]

        # نگاشت مدویدها به نزدیک‌ترین نمونه واقعی در کل دیتاست
        medoid_indices = []
        for medoid in sample_medoids:
            min_dist = float('inf')
            closest_idx = None
            for start in range(0, n_samples, batch_size):
                batch = data_tensor[start:start+batch_size]
                dists = torch.norm(batch - torch.tensor(medoid, device=self.device), dim=1)
                idx_in_batch = torch.argmin(dists).item()
                if dists[idx_in_batch] < min_dist:
                    min_dist = dists[idx_in_batch].item()
                    closest_idx = start + idx_in_batch
            medoid_indices.append(closest_idx)

        medoid_indices = torch.tensor(medoid_indices, device=self.device)
        return data_tensor[medoid_indices]

'''
sampler = AutoDeviceApproxCLARANSSampler(
    percentage=0.1,
    numlocal=5,
    maxneighbor=50,
    random_state=42
)

coreset = sampler.run(features)  # features: numpy array یا torch.Tensor
'''

