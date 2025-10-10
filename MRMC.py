import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


#----------------------------MRMC----------------------------
class MRMC_CoresetSampler(BaseSampler):
    def __init__(self, percentage: float, rho: float = 1/3, r_param: float = 2.0, alpha_random: float = 0.15):
        super().__init__(percentage)
        self.rho = rho
        self.r_param = r_param
        self.alpha_random = alpha_random

    def run(self, phi: Union[torch.Tensor, np.ndarray], L_reg: Union[torch.Tensor, np.ndarray], labels: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        phi: importance scores (phi)
        L_reg: regularization scores (L_reg)
        labels: class labels (numpy or tensor)
        returns: indices of selected core-set
        """
        # تبدیل همه به numpy برای ساده‌تر شدن پردازش
        if isinstance(phi, torch.Tensor):
            phi = phi.cpu().numpy()
        if isinstance(L_reg, torch.Tensor):
            L_reg = L_reg.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        N = len(phi)
        core_size = int(self.percentage * N)
        num_random = int(self.alpha_random * core_size)
        num_top = core_size - num_random
        U_all = phi - self.r_param * L_reg

        core_indices = []

        # مرحله 1: top phi + random
        for c in np.unique(labels):
            idx_c = np.where(labels == c)[0]
            k_top = num_top // len(np.unique(labels))
            topk = idx_c[np.argsort(-phi[idx_c])[:k_top]]
            k_rand = max(num_random // len(np.unique(labels)), 1)
            remaining = np.setdiff1d(idx_c, topk, assume_unique=True)
            randk = remaining if len(remaining) < k_rand else np.random.choice(remaining, k_rand, replace=False)
            core_indices.extend(np.concatenate([topk, randk]))

        core_indices = np.unique(np.array(core_indices))
        if len(core_indices) < core_size:
            needed = core_size - len(core_indices)
            remaining_all = np.setdiff1d(np.arange(N), core_indices, assume_unique=True)
            add_idx = remaining_all[np.argsort(-phi[remaining_all])[:needed]]
            core_indices = np.concatenate([core_indices, add_idx])
        elif len(core_indices) > core_size:
            core_indices = core_indices[:core_size]

        # مرحله 2: C' و top-U_all
        C_prime_size = int(np.floor(self.rho * len(core_indices)))
        C_prime_indices = []
        for c in np.unique(labels):
            idx_c = np.intersect1d(np.where(labels==c)[0], core_indices, assume_unique=True)
            if len(idx_c) == 0: continue
            topk = idx_c[np.argsort(-phi[idx_c])[:max(1, C_prime_size // len(np.unique(labels)))]]
            C_prime_indices.extend(topk)

        C_prime_indices = np.unique(np.array(C_prime_indices))
        remaining = np.setdiff1d(np.arange(N), C_prime_indices, assume_unique=True)
        add_count = len(core_indices) - len(C_prime_indices)
        add_indices = []
        for c in np.unique(labels):
            rem_cls = np.intersect1d(remaining, np.where(labels==c)[0], assume_unique=True)
            if len(rem_cls) == 0: continue
            top_add = rem_cls[np.argsort(-U_all[rem_cls])[:max(1, add_count // len(np.unique(labels)))]]
            add_indices.extend(top_add)
        add_indices = np.unique(np.array(add_indices))
        if len(add_indices) < add_count:
            need = add_count - len(add_indices)
            rem2 = np.setdiff1d(remaining, add_indices, assume_unique=True)
            add2 = rem2[np.argsort(-U_all[rem2])[:need]]
            add_indices = np.concatenate([add_indices, add2])

        core_final_indices = np.unique(np.concatenate([C_prime_indices, add_indices]))
        if len(core_final_indices) < core_size:
            need = core_size - len(core_final_indices)
            rem = np.setdiff1d(np.arange(N), core_final_indices, assume_unique=True)
            add_more = rem[np.argsort(-U_all[rem])[:need]]
            core_final_indices = np.concatenate([core_final_indices, add_more])
        elif len(core_final_indices) > core_size:
            core_final_indices = core_final_indices[:core_size]

        return core_final_indices

class MRMC_CoresetSampler(BaseSampler):
    """
    Fast Approximate MRMC Coreset Sampler
    - PatchCore-compatible (run(features))
    - Uses approximate diversity for speed
    - Handles torch.Tensor or np.ndarray
    """

    def __init__(self, percentage: float, rho: float = 1/3, r_param: float = 2.0,
                 alpha_random: float = 0.15, device: str = "cuda", 
                 diversity_fraction: float = 0.1, batch_size: int = 256):
        """
        Args:
            percentage: fraction of features to select (0-1)
            rho: fraction of core that forms C'
            r_param: regularization weight
            alpha_random: fraction of core that is random
            device: "cuda" or "cpu"
            diversity_fraction: fraction of features used to approximate diversity
            batch_size: batch size for computing distances
        """
        super().__init__(percentage)
        self.rho = rho
        self.r_param = r_param
        self.alpha_random = alpha_random
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.diversity_fraction = diversity_fraction
        self.batch_size = batch_size

    @torch.no_grad()
    def run(self, features):
        is_numpy = isinstance(features, np.ndarray)
        if is_numpy:
            features = torch.from_numpy(features).float()
        else:
            features = features.detach().float()

        features = features.to(self.device)
        N, D = features.shape
        if N == 0:
            return features

        # ---- Compute φ and L_reg ----
        phi = features.norm(dim=1)
        L_reg = torch.var(features, dim=1)

        # ---- Approximate diversity ----
        subset_size = max(1, int(N * self.diversity_fraction))
        subset_idx = torch.randperm(N)[:subset_size]
        subset_feats = features[subset_idx]

        diversity_score = []
        for i in range(0, N, self.batch_size):
            batch = features[i:i+self.batch_size]
            dist = torch.cdist(batch, subset_feats)
            diversity_score.append(dist.mean(dim=1))
        diversity_score = torch.cat(diversity_score)

        # ---- Combined score ----
        U_all = phi - self.r_param * L_reg + diversity_score  # diversity implicit weight = 1

        # ---- Select top samples ----
        core_size = int(self.percentage * N)
        num_random = int(self.alpha_random * core_size)
        num_top = core_size - num_random

        top_indices = torch.argsort(U_all, descending=True)[:num_top].cpu().numpy()
        remaining = np.setdiff1d(np.arange(N), top_indices, assume_unique=True)
        rand_indices = np.random.choice(remaining, size=num_random, replace=False)
        core_indices = np.unique(np.concatenate([top_indices, rand_indices]))

        # ---- Return in original type ----
        if is_numpy:
            return features.cpu().numpy()[core_indices]
        else:
            return features[core_indices].to(features.device)

class MRMC_CoresetSampler2(BaseSampler):
    """
    Fast Cluster-based MRMC Sampler
    PatchCore-compatible (run(features))
    Combines representativeness + informativeness using clustering
    """

    def __init__(
        self, 
        percentage: float, 
        rho: float = 1/3, 
        r_param: float = 2.0, 
        alpha_random: float = 0.15,
        n_clusters: int = 50,
        device: str = "cuda"
    ):
        super().__init__(percentage)
        self.rho = rho
        self.r_param = r_param
        self.alpha_random = alpha_random
        self.n_clusters = n_clusters
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def run(self, features):
        is_numpy = isinstance(features, np.ndarray)
        if is_numpy:
            feats = torch.from_numpy(features).float()
        else:
            feats = features.detach().cpu().float()

        N, D = feats.shape
        if N == 0:
            return features

        # ---- Compute basic metrics ----
        phi = feats.norm(dim=1).numpy()        # informativeness
        L_reg = torch.var(feats, dim=1).numpy()  # regularization term
        U_all = phi - self.r_param * L_reg

        # ---- Fast clustering to ensure diversity ----
        kmeans = MiniBatchKMeans(
            n_clusters=min(self.n_clusters, N // 5 + 1),
            batch_size=512,
            n_init="auto"
        ).fit(feats.numpy())

        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        labels = kmeans.labels_

        # Representative sample per cluster (max U_all in cluster)
        selected = []
        for c in range(len(centers)):
            idx_c = np.where(labels == c)[0]
            if len(idx_c) == 0:
                continue
            best_idx = idx_c[np.argmax(U_all[idx_c])]
            selected.append(best_idx)

        # ---- Add more samples to reach target ----
        selected = np.array(selected)
        core_size = int(self.percentage * N)
        remaining = np.setdiff1d(np.arange(N), selected, assume_unique=True)
        extra_needed = max(0, core_size - len(selected))

        # Mix of top-U and random
        top_indices = np.argsort(-U_all[remaining])[:int(extra_needed * (1 - self.alpha_random))]
        rand_indices = np.random.choice(remaining, size=int(extra_needed * self.alpha_random), replace=False)

        core_indices = np.unique(np.concatenate([selected, remaining[top_indices], rand_indices]))
        if len(core_indices) > core_size:
            core_indices = core_indices[:core_size]

        # ---- Return result ----
        if is_numpy:
            return features[core_indices]
        else:
            return features[core_indices.to(features.device) if torch.is_tensor(core_indices)
                            else torch.tensor(core_indices, device=features.device)]

'''
    elif name == "mrmc_coreset":
        return patchcore.sampler.MRMCoresetSampler(percentage=0.1, device=torch.device("cpu"))

    elif name == "mrmc_coreset":
        return patchcore.sampler.MRMC_CoresetSampler(
                 percentage=0.3,
                 rho=1/3,
                 r_param=2.0,
                 alpha_random=0.1,
                 n_clusters=50
               )
               
    elif name == "mrmc_coreset":
       return patchcore.sampler.MRMC_CoresetSampler(
             percentage=0.3,
             rho=1/3,
             r_param=2.0,
            alpha_random=0.15
     )
'''


