import numpy as np
import torch
from utils.utils import cuda, get_true_head_and_tail_per_graph


class CorruptTriples:
    def __init__(self, model):
        self.model = model
        self.args = model.args
        self.negative_rate = self.args.negative_rate
        self.use_cuda = self.args.use_cuda
        self.graph_dict_train = model.graph_dict_train
        self.get_true_hear_and_tail()

    def get_true_hear_and_tail(self):
        self.true_heads_train = dict()
        self.true_tails_train = dict()
        for t, g in self.graph_dict_train.items():
            triples = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
            true_head, true_tail = get_true_head_and_tail_per_graph(triples)
            self.true_heads_train[t] = true_head
            self.true_tails_train[t] = true_tail

    def negative_sampling(self, quadruples):
        size_of_batch = quadruples.shape[0]
        negative_rate = min(self.negative_rate, len(self.model.known_entities))
        neg_tail_samples = np.zeros((size_of_batch, 1 + negative_rate), dtype=int)
        neg_head_samples = np.zeros((size_of_batch, 1 + negative_rate), dtype=int)
        neg_tail_samples[:, 0] = quadruples[:, 2]
        neg_head_samples[:, 0] = quadruples[:, 0]
        labels = torch.zeros(size_of_batch)
        for i in range(size_of_batch):
            s, r, o, t = quadruples[i]
            s, r, o, t = s.item(), r.item(), o.item(), t.item()
            idx_dict = self.model.graph_dict_train[t].ids
            tail_samples = self.corrupt_triple(s, r, o, negative_rate, self.true_tails_train[t], idx_dict, corrupt_tail=True)
            head_samples = self.corrupt_triple(s, r, o, negative_rate, self.true_heads_train[t], idx_dict, corrupt_tail=False)
            neg_tail_samples[i][0] = idx_dict[o]
            neg_head_samples[i][0] = idx_dict[s]
            neg_tail_samples[i, 1:] = tail_samples
            neg_head_samples[i, 1:] = head_samples

        neg_tail_samples, neg_head_samples = torch.from_numpy(neg_tail_samples), torch.from_numpy(neg_head_samples)
        if self.use_cuda:
            neg_tail_samples, neg_head_samples, labels = \
                cuda(neg_tail_samples, self.args.n_gpu), cuda(neg_head_samples, self.args.n_gpu), cuda(labels,
                                                                                                     self.args.n_gpu)
        return neg_tail_samples.long(), neg_head_samples.long(), labels

    def corrupt_triple(self, s, r, o, negative_rate, other_true_entities, idx_dict, corrupt_tail=True):
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < negative_rate:
            # import pdb; pdb.set_trace()
            if self.args.negative_sample_all_entities:
                negative_sample = np.random.randint(self.model.num_ents, size=negative_rate)
            else:
                negative_sample = np.random.choice(self.model.known_entities, size=negative_rate)
            true_entities = other_true_entities[(s, r)] if corrupt_tail else other_true_entities[(r, o)]
            mask = np.in1d(
                negative_sample,
                [idx_dict[i.item()] for i in true_entities],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        return np.concatenate(negative_sample_list)[:negative_rate]
