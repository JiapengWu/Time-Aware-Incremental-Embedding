import torch
import numpy as np
from collections import defaultdict
import pdb
# import sys
# import os
# sys.path.append(os.getcwd())
from utils.dataset import fill_latest_currence_time_step, get_prev_triples
from utils.dataset import load_quadruples
from utils.args import process_args
from utils.dataset import load_quadruples_tensor, id2entrel, get_total_number
from utils.util_functions import plot_frequency_stats, count_frequency_value_lst, analyze_top_samples, print_per_step_top_patterns
import matplotlib.pyplot as plt
from collections import Counter


def calc_aggregated_statistics(stats_per_time_agg, items, stats_per_time, target_time, cur_time):
    for item in items:
        if item in stats_per_time[cur_time].keys():
            stats_per_time_agg[target_time][item] += stats_per_time[cur_time][item]


def temp_func():
    return defaultdict(int)


def count_freq_per_time(train_data):
    # train_graph_dict, _, _ = build_interpolation_graphs(args)
    triple_freq_per_time_step = defaultdict(temp_func)
    ent_pair_freq_per_time_step = defaultdict(temp_func)
    sub_freq_per_time_step = defaultdict(temp_func)
    obj_freq_per_time_step = defaultdict(temp_func)
    rel_freq_per_time_step = defaultdict(temp_func)
    sub_rel_freq_per_time_step = defaultdict(temp_func)
    rel_obj_freq_per_time_step = defaultdict(temp_func)

    for quad in train_data:
        sub, rel, obj, tim = tuple(quad)
        triple_freq_per_time_step[tim][(sub, rel, obj)] += 1
        ent_pair_freq_per_time_step[tim][(sub, obj)] += 1
        sub_freq_per_time_step[tim][sub] += 1
        obj_freq_per_time_step[tim][obj] += 1
        rel_freq_per_time_step[tim][rel] += 1
        sub_rel_freq_per_time_step[tim][(sub, rel)] += 1
        rel_obj_freq_per_time_step[tim][(rel, obj)] += 1

    return triple_freq_per_time_step, ent_pair_freq_per_time_step, sub_freq_per_time_step, obj_freq_per_time_step, rel_freq_per_time_step, sub_rel_freq_per_time_step, rel_obj_freq_per_time_step


class ReservoirSampler:
    def __init__(self, args, time2quads_train):
        self.args = args
        self.time2quads_train = time2quads_train

        self.start_time_step = args.start_time_step
        self.end_time_step = args.end_time_step

        self.lambda_triple = args.lambda_triple
        self.lambda_ent_pair = args.lambda_ent_pair
        self.lambda_ent_rel = args.lambda_ent_rel
        self.lambda_ent = args.lambda_ent
        self.sigma = args.sigma

        self.train_seq_len = args.train_seq_len
        self.train_data, self.train_times = load_quadruples(args.dataset, 'train.txt')

        self.frequency_sampling = args.frequency_sampling
        self.inverse_frequency_sampling = args.inverse_frequency_sampling
        self.positive_sample_size = args.num_samples_each_time_step * args.train_seq_len
        self.discounted_multiplier = args.cur_frequency_discount_factor * self.train_seq_len

        _, num_rels, _ = get_total_number(args.dataset, 'stat.txt')
        self.id2ent, self.id2rel = id2entrel(args.dataset, num_rels)
        if self.frequency_sampling or self.inverse_frequency_sampling:
            self.count_frequency()
            self.pre_calc_sample_rate()

    def sample(self, time):
        all_hist_quads = torch.cat([self.time2quads_train[cur_time] for cur_time in
                range(max(0, time - self.train_seq_len), time)])

        sample_size = min(len(all_hist_quads), self.positive_sample_size)
        if self.frequency_sampling or self.inverse_frequency_sampling:
            sample_rate_array = np.array(self.sample_rate_cache[time])
            sampled_idx = np.random.choice(len(all_hist_quads), sample_size, p=sample_rate_array / np.sum(sample_rate_array))
        else:
            perm = torch.randperm(all_hist_quads.size(0))
            sampled_idx = perm[:sample_size]

        return all_hist_quads[sampled_idx]

    def analyze_top_samples(self, time):
        all_hist_quads = torch.cat([self.time2quads_train[cur_time] for cur_time in
                                    range(max(0, time - self.train_seq_len), time)])
        analyze_top_samples(self, time, all_hist_quads, 1000, self.id2ent, self.id2rel)

    def pre_calc_sample_rate(self, print_details=False, plot_details=False):
        self.sample_rate_cache = defaultdict(list)

        for target_time in range(self.start_time_step, self.end_time_step):
            hist_target_triple_freq = self.triple_freq_per_time_step_agg[target_time]
            hist_target_ent_pair_freq = self.ent_pair_freq_per_time_step_agg[target_time]
            hist_target_sub_rel_freq = self.sub_rel_freq_per_time_step_agg[target_time]
            hist_target_rel_obj_freq = self.rel_obj_freq_per_time_step_agg[target_time]
            hist_target_sub_freq = self.sub_freq_per_time_step_agg[target_time]
            hist_target_obj_freq = self.obj_freq_per_time_step_agg[target_time]

            # self.triple_freq_per_time_step, self.ent_pair_freq_per_time_step, self.sub_freq_per_time_step, self.obj_freq_per_time_step, \
            # self.rel_freq_per_time_step, self.sub_rel_freq_per_time_step, self.rel_obj_freq_per_time_step = count_freq_per_time(self.train_data)

            cur_target_triple_freq = self.triple_freq_per_time_step[target_time]
            cur_target_ent_pair_freq = self.ent_pair_freq_per_time_step[target_time]
            cur_target_sub_rel_freq = self.sub_rel_freq_per_time_step[target_time]
            cur_target_rel_obj_freq = self.rel_obj_freq_per_time_step[target_time]
            cur_target_sub_freq = self.sub_freq_per_time_step[target_time]
            cur_target_obj_freq = self.obj_freq_per_time_step[target_time]

            drop_rate_lst = self.sample_rate_cache[target_time]

            if plot_details:

                print(target_time, self.train_seq_len)

                plot_frequency_stats(cur_target_triple_freq.values(), cur_target_ent_pair_freq.values(), cur_target_sub_rel_freq.values(),
                                     cur_target_rel_obj_freq.values(), cur_target_sub_freq.values(), cur_target_obj_freq.values(), historical=False)
                plot_frequency_stats(hist_target_triple_freq.values(), hist_target_ent_pair_freq.values(), hist_target_sub_rel_freq.values(),
                                     hist_target_rel_obj_freq.values(), hist_target_sub_freq.values(), hist_target_obj_freq.values())

                all_time_quads = torch.cat([self.time2quads_train[cur_time] for cur_time in
                                        range(max(0, target_time - self.train_seq_len), target_time)])

                hist_target_triple_freq_lst = []
                hist_target_ent_pair_freq_lst = []
                hist_target_sub_rel_freq_lst = []
                hist_target_rel_obj_freq_lst = []
                hist_target_sub_freq_lst = []
                hist_target_obj_freq_lst = []

                cur_target_triple_freq_lst = []
                cur_target_ent_pair_freq_lst = []
                cur_target_sub_rel_freq_lst = []
                cur_target_rel_obj_freq_lst = []
                cur_target_sub_freq_lst = []
                cur_target_obj_freq_lst = []

                for s, r, o, t in all_time_quads:
                    s, r, o, t = s.item(), r.item(), o.item(), t.item()

                    hist_target_triple_freq_lst.append(hist_target_triple_freq[(s, r, o)])
                    hist_target_ent_pair_freq_lst.append(hist_target_ent_pair_freq[(s, o)])
                    hist_target_sub_rel_freq_lst.append(hist_target_sub_rel_freq[(s, r)])
                    hist_target_rel_obj_freq_lst.append(hist_target_rel_obj_freq[(r, o)])
                    hist_target_sub_freq_lst.append(hist_target_sub_freq[s])
                    hist_target_obj_freq_lst.append(hist_target_obj_freq[o])

                    cur_target_triple_freq_lst.append(cur_target_triple_freq[(s, r, o)])
                    cur_target_ent_pair_freq_lst.append(cur_target_ent_pair_freq[(s, o)])
                    cur_target_sub_rel_freq_lst.append(cur_target_sub_rel_freq[(s, r)])
                    cur_target_rel_obj_freq_lst.append(cur_target_rel_obj_freq[(r, o)])
                    cur_target_sub_freq_lst.append(cur_target_sub_freq[s])
                    cur_target_obj_freq_lst.append(cur_target_obj_freq[o])

                plot_frequency_stats(cur_target_triple_freq_lst, cur_target_ent_pair_freq_lst, cur_target_sub_rel_freq_lst,
                         cur_target_rel_obj_freq_lst, cur_target_sub_freq_lst, cur_target_obj_freq_lst, all_time=True, historical=False)

                plot_frequency_stats(hist_target_triple_freq_lst, hist_target_ent_pair_freq_lst, hist_target_sub_rel_freq_lst,
                         hist_target_rel_obj_freq_lst, hist_target_sub_freq_lst, hist_target_obj_freq_lst, all_time=True)

                # pdb.set_trace()

            if print_details:
                for s, r, o, t in self.time2quads_train[target_time]:
                    s, r, o, t = s.item(), r.item(), o.item(), t.item()
                    print("{}\t{}\t{}\t{}".format(self.id2ent[s], self.id2rel[r], self.id2ent[o], t))
                    print("{:5} {:5} {:5} {:5} {:5} {:5}".format(hist_target_triple_freq[(s, r, o)],
                                                                 hist_target_ent_pair_freq[(s, o)],
                                                                 hist_target_sub_rel_freq[(s, r)],
                                                                 hist_target_rel_obj_freq[(r, o)],
                                                                 hist_target_sub_freq[s],
                                                                 hist_target_obj_freq[o]))
                pdb.set_trace()

            for cur_time in range(max(0, target_time - self.train_seq_len), target_time):
                cur_quads = self.time2quads_train[cur_time]
                if print_details:
                    for s, r, o, t in cur_quads:
                        s, r, o, t = s.item(), r.item(), o.item(), t.item()
                        print("{}\t{}\t{}\t{}".format(self.id2ent[s], self.id2rel[r], self.id2ent[o], t))
                        print("{:5} {:5} {:5} {:5} {:5} {:5}".format(hist_target_triple_freq[(s, r, o)],
                                                                           hist_target_ent_pair_freq[(s, o)],
                                                                           hist_target_sub_rel_freq[(s, r)],
                                                                           hist_target_rel_obj_freq[(r, o)],
                                                                           hist_target_sub_freq[s],
                                                                           hist_target_obj_freq[o]))
                    pdb.set_trace()

                for s, r, o, t in cur_quads:
                    s, r, o = s.item(), r.item(), o.item()

                    rate = 1 + self.lambda_triple * np.log(1 + hist_target_triple_freq[(s, r, o)]) \
                             + self.lambda_triple * self.discounted_multiplier * np.log(1 + cur_target_triple_freq[(s, r, o)]) \
                             + self.lambda_ent_pair * np.log(1 + hist_target_ent_pair_freq[(s, o)]) \
                             + self.lambda_ent_pair * self.discounted_multiplier * np.log(1 + cur_target_ent_pair_freq[(s, o)]) \
                             + self.lambda_ent_rel * (np.log(1 + hist_target_sub_rel_freq[(s, r)]) + np.log(1 + hist_target_rel_obj_freq[(r, o)])) \
                             + self.lambda_ent_rel * self.discounted_multiplier * (np.log(1 + cur_target_sub_rel_freq[(s, r)]) + np.log(1 + cur_target_rel_obj_freq[(r, o)])) \
                             + self.lambda_ent * (np.log(1 + hist_target_sub_freq[s]) + np.log(1 + hist_target_obj_freq[o])) + \
                             + self.lambda_ent * self.discounted_multiplier * (np.log(1 + cur_target_sub_freq[s]) + np.log(1 + cur_target_obj_freq[o]))

                    if self.inverse_frequency_sampling:
                        rate = 1 / rate
                    rate *= np.exp((cur_time - target_time) / self.sigma)
                    drop_rate_lst.append(rate)

    def count_frequency(self):
        self.triple_freq_per_time_step, self.ent_pair_freq_per_time_step, self.sub_freq_per_time_step, self.obj_freq_per_time_step, \
            self.rel_freq_per_time_step, self.sub_rel_freq_per_time_step, self.rel_obj_freq_per_time_step = count_freq_per_time(self.train_data)

        self.triple_freq_per_time_step_agg = defaultdict(temp_func)
        self.ent_pair_freq_per_time_step_agg = defaultdict(temp_func)
        self.sub_rel_freq_per_time_step_agg = defaultdict(temp_func)
        self.rel_obj_freq_per_time_step_agg = defaultdict(temp_func)
        self.sub_freq_per_time_step_agg = defaultdict(temp_func)
        self.obj_freq_per_time_step_agg = defaultdict(temp_func)
        self.rel_freq_per_time_step_agg = defaultdict(temp_func)

        for target_time in range(self.start_time_step, self.end_time_step):
            for cur_time in range(max(0, target_time - self.train_seq_len), target_time):
                for s, r, o, t in self.time2quads_train[cur_time]:
                    s, r, o = s.item(), r.item(), o.item()
                    self.triple_freq_per_time_step_agg[target_time][(s, r, o)] += 1
                    self.ent_pair_freq_per_time_step_agg[target_time][(s, o)] += 1
                    self.sub_rel_freq_per_time_step_agg[target_time][(s, r)] += 1
                    self.rel_obj_freq_per_time_step_agg[target_time][(r, o)] += 1
                    self.sub_freq_per_time_step_agg[target_time][s] += 1
                    self.obj_freq_per_time_step_agg[target_time][o] += 1
                    self.rel_freq_per_time_step_agg[target_time][r] += 1

# We want to sample from the deleted edge reservoir such that both for training and evaluation. We want to make those facts
# that were previously true but currently untrue to be "hard negatives", and hope this would also help reduce the rank of the
# deleted edges in the validation set. During the training at time step t, we sample from the reservoir of previously deleted
# "triples". For each negative triple (s, r, o), we add a corresponding positive triple (s, r, o, t') to the training set.
# During evaluation, we evaluate the model on all of the previously deleted facts

class DeletedEdgeReservoir:
    def __init__(self, args, time2quads_train):
        self.time2quads_train = time2quads_train
        self.deleted_edge_sample_size = args.deleted_edge_sample_size
        self.train_seq_len = args.train_seq_len
        # (s, r, o) -> newest t
        self.training_reservoir = {}
        self.construct_init_deleted_edge_reservoir(args.start_time_step)

    def construct_init_deleted_edge_reservoir(self, target_time):
        for t in range(target_time):
            fill_latest_currence_time_step(self.time2quads_train, self.training_reservoir, t)
            # fill_latest_currence_time_step(self.time2quads_val, self.val_reservoir, t)

    def sample_deleted_edges_train(self, time):
        fill_latest_currence_time_step(self.time2quads_train, self.training_reservoir, time)
        train_reservoir_quads = get_prev_triples(self.training_reservoir, time, self.train_seq_len)
        num_reservoir_quads = len(train_reservoir_quads)
        idx = np.random.choice(num_reservoir_quads, min(self.deleted_edge_sample_size, num_reservoir_quads), replace=False)
        return torch.from_numpy(train_reservoir_quads[idx])

    # def get_deleted_edges_val(self, time):
    #     fill_latest_currence_time_step(self.time2quads_val, self.val_reservoir, time)
    #     val_reservoir_quads = get_prev_triples(self.val_reservoir, time, self.train_seq_len, train=False)
    #     # pdb.set_trace()
    #     return torch.from_numpy(val_reservoir_quads)


if __name__ == '__main__':
    args = process_args()

    time2quads_train, time2quads_val, time2quads_test = \
        load_quadruples_tensor(args.dataset, 'train.txt', 'valid.txt', 'test.txt')

    reservoir_sampler = ReservoirSampler(args, time2quads_train)
    reservoir_sampler.count_frequency()
    reservoir_sampler.pre_calc_sample_rate(print_details=True)
