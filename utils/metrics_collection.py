import pickle
import os


class metric_collection:
    def __init__(self, base_path):
        self.metrics = {
                    'hits_10_overall': 0,
                    'hits_3_overall': 0,
                    'hits_1_overall': 0,
                    'mrr_overall': 0,
                    'hits_10': {},
                    'hits_3': {},
                    'hits_1': {},
                    'mrr': {},
                    "epoch_time_avg": {},
                    "epoch_time_sum": {},
                    'num_epoch': {}
                }
        self.base_path = base_path

    def update_eval_metrics(self, time, mrr, hit_1, hit_3, hit_10):
        self.metrics['hits_10'][time] = hit_10
        self.metrics['hits_3'][time] = hit_3
        self.metrics['hits_1'][time] = hit_1
        self.metrics['mrr'][time] = mrr

    def update_eval_accumulated_metrics(self, accumulative_val_result):
        self.metrics['hits_10_overall'] = accumulative_val_result['hit_10']
        self.metrics['hits_3_overall'] = accumulative_val_result['hit_3']
        self.metrics['hits_1_overall'] = accumulative_val_result['hit_1']
        self.metrics['mrr_overall'] = accumulative_val_result['mrr']

    def update_time(self, time, epoch_time_gauge):
        self.metrics['epoch_time_avg'][time] = epoch_time_gauge.get_mean()
        self.metrics['epoch_time_sum'][time] = epoch_time_gauge.get_sum()
        self.metrics['num_epoch'][time] = epoch_time_gauge.get_count()

    def save(self):
        filename = os.path.join(self.base_path, "metrics-per-snapshot.pt")
        with open(filename, 'wb') as handle:
            pickle.dump(self.metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)


class counter_gauge:
    def __init__(self):
        self.val_total = 0
        self.count = 0

    def add(self, val):
        self.val_total += val
        self.count += 1

    def get_mean(self):
        return self.val_total / self.count

    def get_sum(self):
        return self.val_total

    def get_count(self):
        return self.count

    def reset(self):
        self.val_total = 0
        self.count = 0