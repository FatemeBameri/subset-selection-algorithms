import torch
import numpy as np
from typing import Union

class CoverageCentricCoreSampler:
    """
    نسخه کامل CoverageCentricCoreSampler سازگار با PatchCore جدید
    - حالت‌ها: random، monotonic، stratified
    - حفظ mislabel masking و الگوریتم اصلی
    """

    def __init__(self, percentage: float, mode: str = "random", key: str = "score",
                 descending: bool = True, class_balanced: bool = False,
                 mis_ratio: float = 0.1, stratas: int = 50):
        if not 0 < percentage <= 1:
            raise ValueError("Percentage must be in (0,1].")
        self.percentage = percentage
        self.mode = mode
        self.key = key
        self.descending = descending
        self.class_balanced = class_balanced
        self.mis_ratio = mis_ratio
        self.stratas = stratas

    def run(self, data_score: Union[dict, torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        اگر dict داده شود، از keys اصلی الگوریتم استفاده می‌کنیم.
        اگر tensor یا numpy داده شود، از norm feature به عنوان score استفاده می‌کنیم.
        """
        is_numpy_input = False
        if isinstance(data_score, dict):
            # dict اصلی
            if self.key in data_score:
                total_num = len(data_score[self.key])
            else:
                total_num = len(data_score["targets"])
            coreset_num = int(self.percentage * total_num)
            features_score = data_score.get(self.key, None)
            if features_score is not None and isinstance(features_score, np.ndarray):
                features_score = torch.from_numpy(features_score)
        else:
            # tensor یا numpy
            if isinstance(data_score, np.ndarray):
                data_score = torch.from_numpy(data_score)
                is_numpy_input = True
            total_num = data_score.shape[0]
            coreset_num = int(self.percentage * total_num)
            features_score = torch.norm(data_score, dim=1)  # استفاده از norm به جای score

        # حالت‌های انتخاب coreset
        if self.mode == "random":
            selected_idx = self._random_selection(total_num, coreset_num)

        elif self.mode == "monotonic":
            selected_idx = self._score_monotonic_selection(
                data_score, features_score, coreset_num
            )

        elif self.mode == "stratified":
            # اعمال mislabel masking
            mis_num = int(self.mis_ratio * total_num)
            data_score, score_index = self._mislabel_mask(
                data_score, mis_num, coreset_num, features_score
            )
            selected_idx, _ = self._stratified_sampling(
                data_score, coreset_num, features_score
            )
            if score_index is not None:
                selected_idx = score_index[selected_idx]

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # بازگرداندن نوع ورودی
        if isinstance(data_score, dict):
            return selected_idx
        else:
            if is_numpy_input:
                return data_score[selected_idx].numpy()
            else:
                return data_score[selected_idx]

    # -------------------- متدهای کمکی --------------------
    @staticmethod
    def _random_selection(total_num, coreset_num):
        print('Random selection.')
        return torch.randperm(total_num)[:coreset_num]

    def _score_monotonic_selection(self, data_score, features_score, coreset_num):
        print('Monotonic selection.')
        if features_score is None and isinstance(data_score, dict):
            features_score = data_score[self.key]
        score_sorted_index = torch.argsort(features_score, descending=self.descending)

        if self.class_balanced and isinstance(data_score, dict) and "targets" in data_score:
            all_index = torch.arange(len(data_score["targets"]))
            targets_list = data_score["targets"][score_sorted_index]
            targets_unique = torch.unique(targets_list)
            selected_index = []
            for target in targets_unique:
                mask = (targets_list == target)
                target_index = all_index[mask]
                num_select = int(mask.sum() * self.percentage)
                selected_index += list(target_index[:num_select])
            selected_index = torch.tensor(selected_index)
            return score_sorted_index[selected_index]
        else:
            return score_sorted_index[:coreset_num]

    @staticmethod
    def _mislabel_mask(data_score, mis_num, coreset_num, features_score):
        if isinstance(data_score, dict) and "accumulated_margin" in data_score:
            mis_score = data_score["accumulated_margin"]
            mis_sorted_index = torch.argsort(mis_score, descending=False)
            hard_index = mis_sorted_index[:mis_num]
            print(f'Mislabel masking: prune {len(hard_index)} hard samples.')
            easy_index = mis_sorted_index[mis_num:]
            if features_score is not None:
                data_score["score"] = data_score["score"][easy_index]
            return data_score, easy_index
        else:
            return data_score, None

    def _stratified_sampling(self, data_score, coreset_num, features_score):
        print('Stratified sampling...')
        if features_score is None and isinstance(data_score, dict):
            features_score = data_score[self.key]
        score = features_score
        total_num = coreset_num
        min_score = torch.min(score)
        max_score = torch.max(score) * 1.0001
        step = (max_score - min_score) / self.stratas

        def bin_range(k):
            return min_score + k*step, min_score + (k+1)*step

        strata_num = []
        for i in range(self.stratas):
            start, end = bin_range(i)
            num = torch.logical_and(score >= start, score < end).sum()
            strata_num.append(num)
        strata_num = torch.tensor(strata_num)

        # تخصیص بودجه
        sorted_index = torch.argsort(strata_num)
        sort_bins = strata_num[sorted_index]
        num_bin = len(strata_num)
        rest_exp_num = total_num
        budgets = []
        for i in range(num_bin):
            rest_bins = num_bin - i
            avg = rest_exp_num // rest_bins
            cur_num = min(sort_bins[i].item(), avg)
            budgets.append(cur_num)
            rest_exp_num -= cur_num
        rst = torch.zeros(num_bin, dtype=torch.int)
        rst[sorted_index] = torch.tensor(budgets, dtype=torch.int)
        budgets = rst

        selected_index = []
        sample_index = torch.arange(len(score))
        for i in range(self.stratas):
            start, end = bin_range(i)
            mask = torch.logical_and(score >= start, score < end)
            pool = sample_index[mask]
            if len(pool) > 0:
                rand_index = torch.randperm(pool.shape[0])
                selected_index += [pool[idx].item() for idx in rand_index[:budgets[i]]]

        return torch.tensor(selected_index), None

'''
sampler = CoverageCentricCoreSampler(
    percentage=0.1,    # 10٪ نمونه‌ها
    mode="stratified", # random / monotonic / stratified
    key="score",
    descending=True,
    class_balanced=True,
    mis_ratio=0.1
)

coreset_indices = sampler.run(data_score)
'''
