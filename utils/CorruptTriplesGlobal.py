import numpy as np
import torch
from utils.util_functions import cuda, get_true_subject_and_object_per_graph
import os
import pickle
import shelve
from utils.util_functions import write_to_shelve, write_to_default_dict
from collections import defaultdict
import pdb


class CorruptTriplesGlobal:
    def __init__(self, model):
        self.model = model
        self.args = model.args
        np.random.seed(self.args.np_seed)
        self.negative_rate = self.args.negative_rate
        self.use_cuda = self.args.use_cuda
        # print("Constructing train filter")
        self.get_true_subject_object_global()

    def set_known_entities(self):
        self.all_known_entities = self.model.all_known_entities
        self.known_entities = self.model.all_known_entities[self.args.end_time_step - 1] \
            if self.args.train_base_model else self.model.known_entities

    def get_true_subject_object_global(self):
        true_subject_path = os.path.join(self.args.dataset, "true_subjects_train.db")
        true_object_path = os.path.join(self.args.dataset, "true_objects_train.db")

        if os.path.exists(os.path.join(self.args.dataset, "true_subjects_train.db.dat")) and \
                os.path.exists(os.path.join(self.args.dataset, "true_objects_train.db.dat")):
            print("loading the training shelve")
            self.true_subjects_train_global_dict = shelve.open(true_subject_path)
            self.true_objects_train_global_dict = shelve.open(true_object_path)
        else:
            print("computing the training shelve")
            # true_subjects_train_global_defaultdict = defaultdict(dict)
            # true_objects_train_global_defaultdict = defaultdict(dict)

            self.true_subjects_train_global_dict = shelve.open(true_subject_path)
            self.true_objects_train_global_dict = shelve.open(true_object_path)

            for t, quads in self.model.time2quads_train.items():
                true_subjects_dict, true_objects_dict = get_true_subject_and_object_per_graph(quads[:, :3])
                write_to_shelve(self.true_subjects_train_global_dict, true_subjects_dict, t)
                write_to_shelve(self.true_objects_train_global_dict, true_objects_dict, t)

    '''
    def get_true_subject_object_global(self):
        true_subject_path = os.path.join(self.args.dataset, "true_subjects_train.pt")
        true_object_path = os.path.join(self.args.dataset, "true_objects_train.pt")
        if os.path.exists(true_subject_path) and os.path.exists(true_object_path):
            with open(true_subject_path, "rb") as f:
                self.true_subjects_train_global_dict = pickle.load(f)
            with open(true_object_path, "rb") as f:
                self.true_objects_train_global_dict = pickle.load(f)
        else:
            self.true_subjects_train_global_dict = dict()
            self.true_objects_train_global_dict = dict()
            for t, quads in self.model.time2quads_train.items():
                # print(t,len(quads))
                true_subjects_dict, true_objects_dict = get_true_subject_and_object_per_graph(quads[:, :3])
                self.true_subjects_train_global_dict[t] = true_subjects_dict
                self.true_objects_train_global_dict[t] = true_objects_dict
            with open(true_subject_path, 'wb') as fp:
                pickle.dump(self.true_subjects_train_global_dict, fp)
            with open(true_object_path, 'wb') as fp:
                pickle.dump(self.true_objects_train_global_dict, fp)
    '''

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
            s, r, o, t = quadruples[i]
            s, r, o, t = s.item(), r.item(), o.item(), t.item()
            known_entities = self.known_entities if use_fixed_known_entities else self.all_known_entities[t]
            tail_samples = self.corrupt_triple(s, r, o, t, negative_rate, self.true_objects_train_global_dict, known_entities, corrupt_object=True)
            head_samples = self.corrupt_triple(s, r, o, t, negative_rate, self.true_subjects_train_global_dict, known_entities, corrupt_object=False)
            neg_object_samples[i][0] = o
            neg_subject_samples[i][0] = s
            neg_object_samples[i, 1:] = tail_samples
            neg_subject_samples[i, 1:] = head_samples

        neg_object_samples, neg_subject_samples = torch.from_numpy(neg_object_samples), torch.from_numpy(neg_subject_samples)
        if self.use_cuda:
            neg_object_samples, neg_subject_samples, labels = \
                cuda(neg_object_samples, self.args.n_gpu), cuda(neg_subject_samples, self.args.n_gpu), cuda(labels, self.args.n_gpu)
        return neg_object_samples.long(), neg_subject_samples.long(), labels

    def corrupt_triple(self, s, r, o, t, negative_rate, other_true_entities_dict, known_entities, corrupt_object=True):
        negative_sample_list = []
        negative_sample_size = 0

        true_entities = other_true_entities_dict["{}+{}+{}".format(t, s, r)] if \
            corrupt_object else other_true_entities_dict["{}+{}+{}".format(t, o, r)]

        while negative_sample_size < negative_rate:
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