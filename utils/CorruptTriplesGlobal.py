import numpy as np
import torch
from utils.utils import cuda, get_true_subject_and_object_per_graph
np.random.seed(0)


class CorruptTriplesGlobal:
    def __init__(self, model):
        self.model = model
        self.args = model.args
        self.negative_rate = self.args.negative_rate
        self.use_cuda = self.args.use_cuda
        self.get_true_subject_object_global()

    def set_known_entities(self):
        self.all_known_entities = self.model.all_known_entities
        self.known_entities = self.model.all_known_entities[self.args.end_time_step]\
            if self.args.train_base_model else self.model.known_entities

    def get_true_subject_object_global(self):
        self.true_subjects_train_global_dict = dict()
        self.true_objects_train_global_dict = dict()
        for t, quads in self.model.time2quads_train.items():
            true_subjects_dict, true_objects_dict = get_true_subject_and_object_per_graph(quads[:, :3])
            self.true_subjects_train_global_dict[t] = true_subjects_dict
            self.true_objects_train_global_dict[t] = true_objects_dict

    def negative_sampling(self, quadruples, negative_rate, use_fixed_known_entities=True):
        size_of_batch = quadruples.shape[0]

        if use_fixed_known_entities:
            negative_rate = min(negative_rate, len(self.known_entities))

        neg_object_samples = np.zeros((size_of_batch, 1 + negative_rate), dtype=int)
        neg_subject_samples = np.zeros((size_of_batch, 1 + negative_rate), dtype=int)
        neg_object_samples[:, 0] = quadruples[:, 2]
        neg_subject_samples[:, 0] = quadruples[:, 0]
        labels = torch.zeros(size_of_batch)
        for i in range(size_of_batch):
            # import pdb; pdb.set_trace()
            s, r, o, t = quadruples[i]
            s, r, o, t = s.item(), r.item(), o.item(), t.item()

            known_entities = self.known_entities if use_fixed_known_entities else self.all_known_entities[t]

            tail_samples = self.corrupt_triple(s, r, o, negative_rate, self.true_objects_train_global_dict[t], known_entities, corrupt_object=True)
            head_samples = self.corrupt_triple(s, r, o, negative_rate, self.true_subjects_train_global_dict[t], known_entities, corrupt_object=False)
            neg_object_samples[i][0] = o
            neg_subject_samples[i][0] = s
            neg_object_samples[i, 1:] = tail_samples
            neg_subject_samples[i, 1:] = head_samples

        neg_object_samples, neg_subject_samples = torch.from_numpy(neg_object_samples), torch.from_numpy(neg_subject_samples)
        if self.use_cuda:
            neg_object_samples, neg_subject_samples, labels = \
                cuda(neg_object_samples, self.args.n_gpu), cuda(neg_subject_samples, self.args.n_gpu), cuda(labels, self.args.n_gpu)
        return neg_object_samples.long(), neg_subject_samples.long(), labels

    def corrupt_triple(self, s, r, o, negative_rate, other_true_entities_dict, known_entities, corrupt_object=True):
        negative_sample_list = []
        negative_sample_size = 0
        true_entities = other_true_entities_dict[(s, r)] if corrupt_object else other_true_entities_dict[(r, o)]
        while negative_sample_size < negative_rate:
            # import pdb; pdb.set_trace()
            negative_sample = np.random.choice(known_entities, size=negative_rate)
            mask = np.in1d(
                negative_sample,
                true_entities,
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        return np.concatenate(negative_sample_list)[:negative_rate]