import pickle
import os
import numpy as np
from collections import defaultdict

class metric_collection:
    def __init__(self, args, base_path):
        length = args.end_time_step - args.start_time_step
        self.metrics = {
                    'hits_10_overall': 0,
                    'hits_3_overall': 0,
                    'hits_1_overall': 0,
                    'mrr_overall': 0,
                    'raw_hits_10': {},
                    'raw_hits_3': {},
                    'raw_hits_1': {},
                    'raw_mrr': {},
                    'first_positive_hits_10': {},
                    'first_positive_hits_3': {},
                    'first_positive_hits_1': {},
                    'first_positive_mrr': {},
                    'deleted_facts_hits_10': {},
                    'deleted_facts_hits_3': {},
                    'deleted_facts_hits_1': {},
                    'deleted_facts_mrr': {},
                    'both_positive_hits_10': {},
                    'both_positive_hits_3': {},
                    'both_positive_hits_1': {},
                    'both_positive_mrr': {},
                    'mean_relative_ranks': {},
                    'hits_10': {},
                    'hits_3': {},
                    'hits_1': {},
                    'mrr': {},
                    "epoch_time_avg": {},
                    "epoch_time_sum": {},
                    'num_epoch': {},
                    'training_data_size':{}
                }
        if args.plot_gradient:
            self.metrics['gradient_cosine'] = defaultdict(list)
        self.base_path = base_path
        self.start_time = args.start_time_step
        self.diff_time_eval_results = np.zeros((length, length, 4))

    def update_gradient_similarity(self, time, similarity):
        self.metrics['gradient_cosine'][time].append(similarity)

    def update_deleted_facts_ranks(self, time, mrr, hit_1, hit_3, hit_10):
        self.metrics['deleted_facts_hits_10'][time] = hit_10
        self.metrics['deleted_facts_hits_3'][time] = hit_3
        self.metrics['deleted_facts_hits_1'][time] = hit_1
        self.metrics['deleted_facts_mrr'][time] = mrr

    def update_raw_ranks(self, time, mrr, hit_1, hit_3, hit_10):
        self.metrics['raw_hits_10'][time] = hit_10
        self.metrics['raw_hits_3'][time] = hit_3
        self.metrics['raw_hits_1'][time] = hit_1
        self.metrics['raw_mrr'][time] = mrr

    def update_first_positive_ranks(self, time, mrr, hit_1, hit_3, hit_10):
        self.metrics['first_positive_hits_10'][time] = hit_10
        self.metrics['first_positive_hits_3'][time] = hit_3
        self.metrics['first_positive_hits_1'][time] = hit_1
        self.metrics['first_positive_mrr'][time] = mrr

    def update_both_positive_ranks(self, time, mrr, hit_1, hit_3, hit_10):
        self.metrics['both_positive_hits_10'][time] = hit_10
        self.metrics['both_positive_hits_3'][time] = hit_3
        self.metrics['both_positive_hits_1'][time] = hit_1
        self.metrics['both_positive_mrr'][time] = mrr

    def update_mean_relative_ranks(self, time, mean_relative_rank):
        self.metrics['mean_relative_ranks'][time] = mean_relative_rank

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

    def update_training_data_size(self, time, size):
        self.metrics['training_data_size'][time] = size

    def update_diff_time_eval_results(self, cur_time, target_time, mrr, hit_1, hit_3, hit_10):
        self.diff_time_eval_results[cur_time - self.start_time][target_time - self.start_time][:] = mrr, hit_1, hit_3, hit_10

    def save(self):
        filename = os.path.join(self.base_path, "metrics-per-snapshot.pt")
        with open(filename, 'wb') as handle:
            pickle.dump(self.metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

        diff_time_filename = os.path.join(self.base_path, "diff-time-eval-results.pt")
        np.save(diff_time_filename, self.diff_time_eval_results)


class eval_metric_collection(metric_collection):
    def __init__(self, start_time_step, end_time_step, base_path, test_set):
        self.metrics = {
                    'hits_10_overall': 0,
                    'hits_3_overall': 0,
                    'hits_1_overall': 0,
                    'mrr_overall': 0,
                    'raw_hits_10': {},
                    'raw_hits_3': {},
                    'raw_hits_1': {},
                    'raw_mrr': {},
                    'first_positive_hits_10': {},
                    'first_positive_hits_3': {},
                    'first_positive_hits_1': {},
                    'first_positive_mrr': {},
                    'deleted_facts_hits_10': {},
                    'deleted_facts_hits_3': {},
                    'deleted_facts_hits_1': {},
                    'deleted_facts_mrr': {},
                    'both_positive_hits_10': {},
                    'both_positive_hits_3': {},
                    'both_positive_hits_1': {},
                    'both_positive_mrr': {},
                    'mean_relative_ranks': {},
                    'hits_10': {},
                    'hits_3': {},
                    'hits_1': {},
                    'mrr': {}
                }
        self.test_set = test_set
        self.base_path = base_path
        self.start_time = start_time_step
        length = end_time_step - start_time_step
        self.diff_time_eval_results = np.zeros((length, length, 4))

    def copy(self):
        filename = os.path.join(self.base_path, "metrics-per-snapshot.pt")
        with open(filename, 'rb') as fp:
            res = pickle.load(fp)

        self.metrics["epoch_time_avg"] = res["epoch_time_avg"]
        self.metrics["epoch_time_sum"] = res["epoch_time_sum"]
        self.metrics['num_epoch'] = res['num_epoch']
        self.metrics['training_data_size'] = res['training_data_size']

    def save(self):
        filename = os.path.join(self.base_path, "{}-metrics-per-snapshot.pt".format("val" if not self.test_set else "test"))

        with open(filename, 'wb') as handle:
            pickle.dump(self.metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

        diff_time_filename = os.path.join(self.base_path, "{}-diff-time-eval-results.pt".format("val" if not self.test_set else "test"))
        np.save(diff_time_filename, self.diff_time_eval_results)


class counter_gauge:
    def __init__(self):
        self.val_total = 0
        self.count = 0

    def add(self, val):
        self.val_total += val
        self.count += 1

    def get_mean(self):
        if self.count == 0:
            return 0
        return self.val_total / self.count

    def get_sum(self):
        return self.val_total

    def get_count(self):
        return self.count

    def reset(self):
        self.val_total = 0
        self.count = 0