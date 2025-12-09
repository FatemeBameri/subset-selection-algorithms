# ---------------------------
# ELFSCoresetSampler (ELFS-inspired, PatchCore-friendly)
# ---------------------------
class ELFSCoresetSampler(BaseSampler):
    """
    ELFS-inspired coreset sampler for patch-features (PatchCore).
    Modes:
      - mode="feature" : instability = noise-proxy in feature space (no labels needed)
      - mode="pseudo"   : instability = proxy training-dynamics computed if pseudo_labels or image_embeddings provided
    Key ideas preserved from ELFS:
      - training-dynamics proxy (optional)
      - representativeness (distance to cluster center or global mean)
      - rarity (mean distance to anchors ≈ inverse of density)
      - combine signals into composite score, choose top-L, then k-center greedy for diversity
    Designed to be memory-friendly with chunking/subsampling.
    """

    def __init__(
        self,
        percentage: float,
        device: torch.device,
        mode: str = "feature",  # "feature" or "pseudo"
        reduce_dim: Optional[int] = 128,  # reduce features if high-dim (None to skip)
        density_anchor_count: int = 500,  # number of anchors to approximate density/rarity
        k_density: int = 5,
        noise_T: int = 2,
        noise_std: float = 1e-2,
        training_proxy_epochs: int = 3,  # for mode="pseudo" (kept small)
        training_proxy_lr: float = 1e-2,
        batch_size_instability: int = 1024,
        topk_pool_multiplier: float = 5.0,
        w_repr: float = 1.0,
        w_density: float = 1.0,
        w_unc: float = 1.0,
        candidate_seed: Optional[int] = None,
    ):
        super().__init__(percentage)
        assert mode in ("feature", "pseudo"), "mode must be 'feature' or 'pseudo'"
        self.device = device
        self.mode = mode
        self.reduce_dim = reduce_dim
        self.density_anchor_count = max(1, int(density_anchor_count))
        self.k_density = max(1, int(k_density))
        self.noise_T = max(0, int(noise_T))
        self.noise_std = float(noise_std)
        self.training_proxy_epochs = max(0, int(training_proxy_epochs))
        self.training_proxy_lr = float(training_proxy_lr)
        self.batch_size_instability = max(1, int(batch_size_instability))
        self.topk_pool_multiplier = max(1.0, float(topk_pool_multiplier))
        self.w_repr = float(w_repr)
        self.w_density = float(w_density)
        self.w_unc = float(w_unc)
        self.candidate_seed = candidate_seed

        # optional projection layer (created lazily when dims known)
        self._proj = None
        self._proj_in_dim = None

    # -------------------------
    # helpers: reduce dim lazily
    # -------------------------
    def _maybe_project(self, feats: torch.Tensor) -> torch.Tensor:
        if self.reduce_dim is None:
            return feats
        if self._proj is None or self._proj_in_dim != feats.shape[1]:
            # create linear projection on the fly (no gradient required)
            self._proj_in_dim = feats.shape[1]
            proj = torch.nn.Linear(self._proj_in_dim, self.reduce_dim, bias=False)
            # small random init is fine; keep on device
            proj.to(self.device)
            proj.eval()
            with torch.no_grad():
                self._proj = proj
        with torch.no_grad():
            feats = self._proj(feats)
        return feats

    # -------------------------
    # representativeness: distance to centroid or cluster centroid
    # if cluster_centroids provided (torch.Tensor [K,D]) use distance to assigned centroid mean.
    # else use global mean distance (default)
    # -------------------------
    def _compute_representativeness(self, feats: torch.Tensor, cluster_centroids: Optional[torch.Tensor] = None, cluster_assignments: Optional[torch.Tensor] = None) -> torch.Tensor:
        # feats: [N,D]
        if cluster_centroids is not None and cluster_assignments is not None:
            # distance to assigned cluster centroid (makes representativeness local to cluster)
            centroids = cluster_centroids[cluster_assignments]  # [N,D]
            return torch.norm(feats - centroids, dim=1)
        else:
            mean_vec = feats.mean(dim=0, keepdim=True)
            return torch.norm(feats - mean_vec, dim=1)

    # -------------------------
    # rarity (inverse density) approx:
    # compute mean distance to a set of anchors (sampled) -> larger means rarer
    # memory-friendly via chunking
    # -------------------------
    def _compute_rarity(self, feats: torch.Tensor, anchors: Optional[torch.Tensor] = None, anchor_count: Optional[int] = None) -> torch.Tensor:
        N, D = feats.shape
        if anchor_count is None:
            anchor_count = self.density_anchor_count
        anchor_count = min(anchor_count, N)
        if anchors is None:
            # sample anchors randomly
            if self.candidate_seed is not None:
                torch.manual_seed(self.candidate_seed)
            perm = torch.randperm(N, device=feats.device)[:anchor_count]
            anchors = feats[perm]  # [A, D]
        else:
            anchors = anchors.to(feats.device)

        # compute mean Euclidean distance from each feat to the anchors (approximate rarity)
        # chunking to save memory
        mean_dist = torch.zeros(N, device=feats.device)
        chunk = 1024
        with torch.no_grad():
            for i in range(0, N, chunk):
                q = feats[i:i+chunk]  # [b, D]
                # use torch.cdist which is optimized and can be chunked
                d = torch.cdist(q, anchors, p=2.0)  # [b, A]
                mean_dist[i:i+chunk] = d.mean(dim=1)
        # mean_dist large => rarer
        return mean_dist

    # -------------------------
    # instability proxies
    # A) noise-proxy: add small Gaussian noise in feature space and measure average displacement
    # B) training-dynamics-proxy: if pseudo_labels and image embeddings available, train a small linear probe
    #    for a few epochs and record per-sample losses (AUM or mean loss used as score).
    # -------------------------
    def _compute_instability_noise(self, feats: torch.Tensor) -> torch.Tensor:
        N, D = feats.shape
        if self.noise_T <= 0:
            return torch.zeros(N, device=feats.device)
        inst = torch.zeros(N, device=feats.device)
        with torch.no_grad():
            for start in range(0, N, self.batch_size_instability):
                end = min(start + self.batch_size_instability, N)
                batch = feats[start:end]  # [b, D]
                sims = torch.zeros(end - start, self.noise_T, device=feats.device)
                for t in range(self.noise_T):
                    noise = torch.randn_like(batch) * self.noise_std
                    pert = batch + noise
                    sims[:, t] = torch.norm(batch - pert, dim=1)
                inst[start:end] = sims.mean(dim=1)
        # larger => more unstable
        return inst

    def _compute_instability_training_proxy(
        self,
        image_embeddings: torch.Tensor,
        pseudo_labels: torch.Tensor,
        linear_hidden: int = 0,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """
        Lightweight proxy training dynamics:
         - image_embeddings: [M, Dimg] image-level embeddings (one per image)
         - pseudo_labels: [M] long tensor with cluster assignments or pseudo labels
        Returns:
         - per-image score (mean loss across training steps or AUM-like score).
        NOTE: keep epochs small (1..5) and do training on CPU if GPU memory is constrained.
        """
        device = self.device
        epochs = self.training_proxy_epochs if epochs is None else epochs
        lr = self.training_proxy_lr if lr is None else lr

        M, Dimg = image_embeddings.shape
        # tiny linear probe
        # map Dimg -> num_classes (using simple linear)
        num_classes = int(pseudo_labels.max().item()) + 1
        probe = torch.nn.Linear(Dimg, num_classes).to(device)
        opt = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9)

        # prepare dataset (convert to tensors)
        X = image_embeddings.to(device)
        y = pseudo_labels.to(device).long()

        # record per-sample loss over epochs (for a simple AUM-like score)
        losses = torch.zeros(M, epochs, device=device)
        ce = torch.nn.CrossEntropyLoss(reduction="none")

        probe.train()
        for e in range(epochs):
            # small random permutation
            perm = torch.randperm(M, device=device)
            for i in range(0, M, batch_size):
                idx = perm[i:i+batch_size]
                xb = X[idx]
                yb = y[idx]
                logits = probe(xb)
                loss_vec = ce(logits, yb)  # [b]
                loss = loss_vec.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                # store loss per sample
                losses[idx, e] = loss_vec.detach()
        probe.eval()
        # AUM-like score: mean loss across epochs (higher => harder / more uncertain)
        score = losses.mean(dim=1).detach().cpu()
        # return as tensor on sampler device (we will map image->patch later)
        return score.to(device)


    # ---------- min-max normalization utility ----------
    def _minmax_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor x to [0,1] along its values.
        Safe: handles constant tensors and nan/inf by returning zeros in degenerate case.
        Input: 1-D or any-shaped tensor -> returns same shape tensor.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)

        # move to device to be safe
        x = x.to(self.device)

        # flatten then compute min/max safely
        mn = torch.min(x)
        mx = torch.max(x)

        # if all equal or invalid range, return zeros of same shape
        if not torch.isfinite(mn) or not torch.isfinite(mx) or mx <= mn:
            return torch.zeros_like(x)

        # normal case
        return (x - mn) / (mx - mn)
    # -------------------------
    # small k-center greedy (operates on candidate pool)
    # optimized for memory (works with torch.norm and in-place updates)
    # -------------------------
    def _kcenter_greedy(self, feats: torch.Tensor, m: int, seed: Optional[int] = None) -> np.ndarray:
        """
        feats: [L, D] (candidate pool)
        returns indices (numpy) of length m
        """
        L = feats.shape[0]
        if m >= L:
            return np.arange(L, dtype=np.int64)
        if seed is None:
            seed = self.candidate_seed
        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
            first = torch.randint(0, L, (1,), device=feats.device).item()
            selected = [first]
            # distances to selected set (initialize)
            dists = torch.norm(feats - feats[first:first+1], dim=1)
            for _ in range(1, m):
                idx = torch.argmax(dists).item()
                selected.append(idx)
                newd = torch.norm(feats - feats[idx:idx+1], dim=1)
                dists = torch.minimum(dists, newd)
            return np.array(selected, dtype=np.int64)

    # -------------------------
    # main run:
    # features: [N, D] patch features (torch.Tensor or np.ndarray)
    # optional kwargs (mode='pseudo'):
    #   image_embeddings: torch.Tensor [M, Dimg]  (one embedding per image)
    #   image_to_patch_map: list or tensor mapping patch_idx -> image_idx (length N)
    #   pseudo_labels: optional precomputed pseudo_labels [M]
    # -------------------------
    def run(self, features: Union[torch.Tensor, np.ndarray], **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """
        Run sampler.

        kwargs (optional):
          - image_embeddings: torch.Tensor [M, Dimg] (required only if mode='pseudo' and you want training dynamics)
          - image_to_patch_map: torch.LongTensor [N] mapping patch_idx -> image_idx (required to map image scores to patches)
          - pseudo_labels: torch.LongTensor [M] if already computed (optional)
          - anchors: torch.Tensor [A, D] optional anchors for rarity computation
        """
        if self.percentage == 1:
            return features

        self._store_type(features)
        if isinstance(features, np.ndarray):
            feats = torch.from_numpy(features).float().to(self.device)
        else:
            feats = features.float().to(self.device)

        N, D = feats.shape

        # optional projection
        feats_proj = self._maybe_project(feats) if (self.reduce_dim is not None and self.reduce_dim > 0) else feats

        # ---------- representativeness ----------
        # use cluster-based representativeness if pseudo provided (prefer local)
        cluster_centroids = None
        cluster_assignments = None
        if self.mode == "pseudo" and "pseudo_labels" in kwargs and "image_embeddings" in kwargs:
            # if image-level clusters exist, map them to patches via image_to_patch_map and compute centroid per cluster
            pseudo_labels = kwargs["pseudo_labels"]
            image_embeddings = kwargs["image_embeddings"].to(self.device)
            image_to_patch_map = kwargs.get("image_to_patch_map", None)
            # compute cluster centroids in image-embedding space and then map to patch-feature space by averaging patch-features per image cluster
            # fallback: use global mean
            try:
                # cluster -> list of image indices
                K = int(pseudo_labels.max().item()) + 1
                # compute patch-level centroid per cluster by averaging patches of images in cluster
                if image_to_patch_map is not None:
                    # compute per-image mean of corresponding patches in feats_proj
                    # image_embeddings must align with pseudo_labels
                    M = image_embeddings.shape[0]
                    img_patch_means = torch.zeros((M, feats_proj.shape[1]), device=feats_proj.device)
                    counts = torch.zeros(M, device=feats_proj.device)
                    # compute per-image mean patch-features (chunked)
                    for p_idx in range(N):
                        img_idx = int(image_to_patch_map[p_idx])
                        img_patch_means[img_idx] += feats_proj[p_idx]
                        counts[img_idx] += 1
                    nonzero = counts > 0
                    img_patch_means[nonzero] /= counts[nonzero].unsqueeze(1)
                    # now cluster centroid in patch-feature space (avg of images' patch_means per cluster)
                    centroids = torch.zeros((K, feats_proj.shape[1]), device=feats_proj.device)
                    cluster_counts = torch.zeros(K, device=feats_proj.device)
                    for img_idx in range(M):
                        c = int(pseudo_labels[img_idx].item())
                        centroids[c] += img_patch_means[img_idx]
                        cluster_counts[c] += 1
                    nonzero = cluster_counts > 0
                    centroids[nonzero] /= cluster_counts[nonzero].unsqueeze(1)
                    # cluster_assignments for patches: map each patch to its image's cluster
                    cluster_assignments = image_to_patch_map.clone().to(self.device)
                    cluster_assignments = cluster_assignments.apply_(lambda x: int(pseudo_labels[int(x)].item()))
                    cluster_centroids = centroids
                else:
                    # fallback: use global mean
                    cluster_centroids = None
                    cluster_assignments = None
            except Exception:
                cluster_centroids = None
                cluster_assignments = None

        repr_score = self._compute_representativeness(feats_proj, cluster_centroids, cluster_assignments)

        # ---------- rarity (approx density) ----------
        anchors = kwargs.get("anchors", None)
        rarity_score = self._compute_rarity(feats_proj, anchors=anchors, anchor_count=self.density_anchor_count)

        # ---------- instability ----------
        if self.mode == "feature":
            inst_score = self._compute_instability_noise(feats_proj)
        else:
            # pseudo mode -> try training dynamics proxy if required args present
            if "image_embeddings" in kwargs and ("pseudo_labels" in kwargs):
                # compute image-level training-dynamics proxy
                image_embeddings = kwargs["image_embeddings"]
                pseudo_labels = kwargs["pseudo_labels"]
                train_scores = self._compute_instability_training_proxy(image_embeddings, pseudo_labels, epochs=self.training_proxy_epochs, lr=self.training_proxy_lr)
                # map image scores to patch scores via image_to_patch_map
                image_to_patch_map = kwargs.get("image_to_patch_map", None)
                if image_to_patch_map is not None:
                    inst_score = torch.zeros(N, device=feats_proj.device)
                    for p_idx in range(N):
                        img_idx = int(image_to_patch_map[p_idx])
                        inst_score[p_idx] = train_scores[img_idx]
                else:
                    # fallback: use noise proxy
                    inst_score = self._compute_instability_noise(feats_proj)
            else:
                # fallback to noise proxy
                inst_score = self._compute_instability_noise(feats_proj)

        # ---------- normalize each signal -----------
        repr_n = self._minmax_normalize(repr_score)
        rarity_n = self._minmax_normalize(rarity_score)  # larger => rarer
        inst_n = self._minmax_normalize(inst_score)

        # composite (note: choose sign of weights depending on desired behavior)
        # default: repr (higher => more representative), rarity (higher => rarer), inst (higher => more informative/harder)
        composite = (self.w_repr * repr_n) + (self.w_density * rarity_n) + (self.w_unc * inst_n)

        # ---------- candidate pool selection ----------
        m = max(1, int(N * self.percentage))
        L = min(N, max(m, int(m * self.topk_pool_multiplier)))
        topk_vals, top_inds = torch.topk(composite, k=L, largest=True)
        top_inds = top_inds.cpu().numpy().astype(np.int64)

        # ---------- run k-center greedy on candidate pool ----------
        candidate_feats = feats[top_inds].to(self.device)
        selected_local = self._kcenter_greedy(candidate_feats, m, seed=self.candidate_seed)
        selected_indices = top_inds[selected_local]

        # return reduced features (same type as input)
        res = feats[selected_indices]
        return self._restore_type(res)
