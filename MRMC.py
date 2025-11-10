import torch
import torch.nn as nn
from tqdm import tqdm
import abc
from typing import Union


class MRMCCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        model: nn.Module,
        dataloader,
        device: torch.device,
        R: int = 10,
        rho: float = 0.3,
        gamma: float = 2,
        lr: float = 0.1,
        proxy_epochs: int = 200
    ):
        super().__init__(percentage)
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.R = R
        self.rho = rho
        self.gamma = gamma
        self.lr = lr
        self.proxy_epochs = proxy_epochs

    def run(self, features=None):
        #self._store_type(features)
        self.model.to(self.device)
        self.model.train()

        num_samples = len(self.dataloader.dataset)
        coreset_size = int(self.percentage * num_samples)
        all_losses = torch.zeros((self.R, num_samples), device=self.device)

        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9,weight_decay=5e-4)

        # ------------------------ Phase 1: R rounds mini-batch SGD ------------------------
        for r in range(self.R):
            print(f"[Round {r+1}/{self.R}] Training model and collecting losses...")
            batch_losses = torch.zeros(num_samples, device=self.device)
            for batch_idx, (indices, (inputs, targets)) in enumerate(tqdm(self.dataloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss_mean = loss.mean()
                loss_mean.backward()
                optimizer.step()
                batch_losses[indices] = loss.detach()
            all_losses[r] = batch_losses

        # ------------------------ Phase 2: compute φ_MRMC ------------------------
        print("Computing φ_MRMC (vectorized exponential fitting)...")

        R = self.R
        r_values = torch.arange(1, R + 1, device=self.device, dtype=torch.float32)  # (R,)
        num_samples = all_losses.shape[1]

        all_losses = torch.clamp(all_losses, min=1e-8)

        y = torch.log(all_losses)  # (R, N)
        mean_r = torch.mean(r_values)
        mean_y = torch.mean(y, dim=0)  # (N,)

        cov_ry = torch.sum((r_values.unsqueeze(1) - mean_r) * (y - mean_y.unsqueeze(0)), dim=0)
        var_r = torch.sum((r_values - mean_r) ** 2)

        w = -cov_ry / (var_r + 1e-12)  # (N,)
        log_q = mean_y + w * mean_r  # (N,)
        q = torch.exp(log_q)  # (N,)

        phi_mrmc = q * (1 - torch.exp(-w * R))

        print(f"φ_MRMC computed: shape={phi_mrmc.shape}, min={phi_mrmc.min():.4f}, max={phi_mrmc.max():.4f}")

        # ------------------------ Phase 3: coreset selection ------------------------
        if self.rho == 1:
            _, topk_idx = torch.topk(phi_mrmc, coreset_size)
            return self._restore_type(topk_idx)

        # ------------------------ Phase 4: top ρ|C| for proxy ------------------------
        num_top = int(self.rho * coreset_size)
        _, top_indices = torch.topk(phi_mrmc, num_top)
        top_indices = top_indices.to(self.device)

        # ------------------------ Phase 5: train lightweight proxy model ------------------------
        feature_dim = self.model(
            torch.zeros(1, *self.dataloader.dataset[0][1][0].shape).to(self.device)
        ).flatten(1).shape[1]

        proxy = nn.Linear(feature_dim, 1).to(self.device)
        proxy_opt = torch.optim.Adam(proxy.parameters(), lr=self.lr)

        print(f"Training proxy model on top {num_top} samples...")
        for epoch in range(self.proxy_epochs):
            proxy_loss_sum = 0.0
            for batch_idx, (indices, (inputs, _)) in enumerate(self.dataloader):
                inputs = inputs.to(self.device)
                indices = indices.to(self.device)
                mask = (indices.unsqueeze(1) == top_indices).any(dim=1)
                if mask.sum() == 0:
                    continue
                inputs_subset = inputs[mask]
                idx_subset = indices[mask]

                with torch.no_grad():
                    feats = self.model(inputs_subset).flatten(1)

                preds = proxy(feats).squeeze()
                true_vals = phi_mrmc[idx_subset]
                loss = torch.mean((preds - true_vals) ** 2)

                proxy_opt.zero_grad()
                loss.backward()
                proxy_opt.step()
                proxy_loss_sum += loss.item()
            print(f"Proxy Epoch {epoch + 1}: Loss={proxy_loss_sum:.4f}")

        # ------------------------ Phase 6: compute φ_reg for remaining samples ------------------------
        all_indices = torch.arange(len(phi_mrmc), device=self.device)
        remaining_mask = ~torch.isin(all_indices, top_indices)
        remaining_indices = all_indices[remaining_mask]

        phi_reg = torch.zeros_like(phi_mrmc)
        with torch.no_grad():
            for batch_idx, (indices, (inputs, _)) in enumerate(self.dataloader):
                inputs = inputs.to(self.device)
                indices = indices.to(self.device)
                mask = torch.isin(indices, remaining_indices)
                if mask.sum() == 0:
                    continue
                inputs_subset = inputs[mask]
                idx_subset = indices[mask]

                feats = self.model(inputs_subset).flatten(1)
                preds = proxy(feats).squeeze()
                loss_vals = torch.abs(preds - phi_mrmc[idx_subset])
                phi_reg[idx_subset] = torch.exp(-loss_vals)

        # ------------------------ Phase 7: final selection ------------------------
        num_remaining = coreset_size - num_top
        final_scores = phi_mrmc[remaining_indices] * (phi_reg[remaining_indices] ** self.gamma)
        _, remaining_top_idx = torch.topk(final_scores, num_remaining)
        final_indices = torch.cat([top_indices, remaining_indices[remaining_top_idx]])

        if features is not None:
            features_np = features.detach().cpu().numpy() if torch.is_tensor(features) else np.asarray(features)
            selected_features = features_np[final_indices.cpu().numpy()]
            print(f"[DEBUG] Returning selected features with shape {selected_features.shape}")
            return selected_features
            
        return final_indices


'''
       elif name == "MRMC2":
       
            import torch.nn as nn
            import torchvision.models as models
            model = models.wide_resnet50_2(weights='IMAGENET1K_V1')
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 15) 
            model = model.to(device)
            
            return patchcore.sampler.MRMCCoresetSampler(
            percentage=0.1,
            model=model,
            dataloader=train_loader2,
            device=torch.device("cuda:0"),
            R=10,
            rho=1 / 3,
            gamma=2
        )
        
'''
