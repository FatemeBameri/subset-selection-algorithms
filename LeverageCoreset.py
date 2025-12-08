import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd

class LeverageScoreSampler(BaseSampler):
    def __init__(self, percentage: float, device: str = "cpu", rank: int = 64):
        """
        Leverage Score Sampling for coreset selection.
        انتخاب زیرمجموعه از داده‌ها بر اساس اهمیت‌شان در زیرفضای اصلی (low-rank subspace).

        Args:
            percentage (float): درصد داده‌ها برای انتخاب (مثلاً 0.1 برای 10%)
            device (str): دستگاه اجرا ("cpu" یا "cuda")
            rank (int): تعداد مؤلفه‌های SVD برای تقریب (مثلاً 64)
        """
        super().__init__(percentage)
        self.device = torch.device(device)
        self.rank = rank

    def run(self, features):
        """
        اجرای الگوریتم Leverage Score Sampling.
        Args:
            features (torch.Tensor یا np.ndarray): ویژگی‌های داده با ابعاد (N, D)
        Returns:
            زیرمجموعه‌ی انتخاب‌شده از همان نوع ورودی
        """
        self._store_type(features)

        # تبدیل داده‌ها به numpy در صورت نیاز
        if isinstance(features, torch.Tensor):
            X = features.detach().cpu().numpy()
        else:
            X = features

        N, D = X.shape
        k = max(1, int(N * self.percentage))

        # --- 1. نرمال‌سازی ردیف‌ها ---
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

        # --- 2. محاسبه‌ی SVD تصادفی (تقریبی و سریع) ---
        try:
            U, S, Vt = randomized_svd(
                X,
                n_components=min(self.rank, D),
                random_state=0
            )
        except Exception as e:
            raise RuntimeError(f"Randomized SVD failed: {e}")

        # --- 3. محاسبه‌ی امتیازهای اهرمی ---
        leverage_scores = np.sum(U ** 2, axis=1)
        leverage_scores = np.clip(leverage_scores, 0, None)
        leverage_scores /= np.sum(leverage_scores)

        # --- 4. نمونه‌برداری بر اساس امتیازها ---
        selected_idx = np.random.choice(N, size=k, replace=False, p=leverage_scores)
        subset = X[selected_idx]

        # --- 5. بازگرداندن نوع اصلی (torch یا numpy) ---
        if not self.features_is_numpy:
            subset = torch.tensor(subset, device=self.features_device, dtype=torch.float32)

        return subset

