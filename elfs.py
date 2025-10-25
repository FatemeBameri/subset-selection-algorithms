import torch
import torch.nn.functional as F


class ELFS_CoresetSampler(BaseSampler):
    def __init__(self, percentage, device, metric_key="forgetting", descending=True):
        super().__init__(percentage)  # ^ درصد به BaseSampler پاس داده شد
        self.device = device
        self.metric_key = metric_key  # ^ پارامتر جدید
        self.descending = descending  # ^ پارامتر جدید

    def run(self, *args, **kwargs):
        # حالت جدید با td_log و dataset
        if 'td_log' in kwargs and 'dataset' in kwargs:
            td_log = kwargs['td_log']
            dataset = kwargs['dataset']
            return self.score_monotonic_selection(
                td_log, dataset,
                percentage=self.percentage,
                metric_key=self.metric_key,
                descending=self.descending
            )

        # حالت قدیمی
        elif len(args) == 1:
            features = args[0]
            return features

        else:
            raise ValueError("TD_CoresetSampler requires either td_log & dataset, or features as single argument")

    def score_monotonic_selection(self, td_log, dataset, percentage=0.1, descending=True, metric_key="forgetting"):
        data_importance = self.compute_training_dynamics(td_log, dataset)
        score = data_importance[metric_key]
        score_sorted_index = score.argsort(descending=descending)
        total_num = int(percentage * score.shape[0])

        #print(f"Selecting top {percentage*100:.1f}% based on '{metric_key}' metric.")
        #print(f"Top {metric_key} values: {score[score_sorted_index[:10]]}")
        #print(f"Bottom {metric_key} values: {score[score_sorted_index[-10:]]}")

        return score_sorted_index[:total_num]

    @staticmethod
    def compute_training_dynamics(td_log, dataset):
        """Compute training dynamics for TD_CoresetSampler using 'is_anomaly' labels."""
        data_size = len(dataset)
        # استخراج targets از فیلد is_anomaly
        targets = torch.tensor([dataset[i]['is_anomaly'] for i in range(data_size)], dtype=torch.int32)

        data_importance = {
            'targets': targets,
            'correctness': torch.zeros(data_size, dtype=torch.int32),
            'forgetting': torch.zeros(data_size, dtype=torch.int32),
            'last_correctness': torch.zeros(data_size, dtype=torch.int32),
            'accumulated_margin': torch.zeros(data_size, dtype=torch.float32)
        }

        for i, item in enumerate(td_log):
            if i % 1000 == 0:
                print(f"Processing batch {i}/{len(td_log)}")

            output = torch.tensor(item['output'], dtype=torch.float32)
            output = F.softmax(output, dim=-1)
            predicted = output.argmax(dim=1)
            index = item['idx'].type(torch.long)
            label = targets[index]

            correctness = (predicted == label).type(torch.int)
            data_importance['forgetting'][index] += torch.logical_and(
                data_importance['last_correctness'][index] == 1, correctness == 0)
            data_importance['last_correctness'][index] = correctness
            data_importance['correctness'][index] += data_importance['last_correctness'][index]

            batch_idx = range(output.shape[0])
            target_prob = output[batch_idx, label]
            output[batch_idx, label] = 0
            other_highest_prob = torch.max(output, dim=1)[0]
            margin = target_prob - other_highest_prob
            data_importance['accumulated_margin'][index] += margin

        return data_importance


