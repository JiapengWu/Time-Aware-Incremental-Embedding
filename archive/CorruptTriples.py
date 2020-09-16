import numpy as np
import torch
from utils.utils import cuda, get_true_subject_and_object_per_graph


class CorruptTriples:
    def __init__(self, model):
        self.model = model
        self.args = model.args
        self.negative_rate = self.args.negative_rate
        self.use_cuda = self.args.use_cuda
        self.graph_dict_train = model.graph_dict_train
        self.get_true_subject_object()

    def get_true_subject_object(self):
        self.true_subjects_train = dict()
        self.true_objects_train = dict()
        for t, g in self.graph_dict_train.items():
            triples = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
            true_subjects, true_objects = get_true_subject_and_object_per_graph(triples)
            self.true_subjects_train[t] = true_subjects
            self.true_objects_train[t] = true_objects

    def negative_sampling(self, quadruples):
        size_of_batch = quadruples.shape[0]
        negative_rate = min(self.negative_rate, len(self.model.known_entities))
        neg_object_samples = np.zeros((size_of_batch, 1 + negative_rate), dtype=int)
        neg_subject_samples = np.zeros((size_of_batch, 1 + negative_rate), dtype=int)
        neg_object_samples[:, 0] = quadruples[:, 2]
        neg_subject_samples[:, 0] = quadruples[:, 0]
        labels = torch.zeros(size_of_batch)
        for i in range(size_of_batch):
            s, r, o, t = quadruples[i]
            s, r, o, t = s.item(), r.item(), o.item(), t.item()
            idx_dict = self.model.graph_dict_train[t].ids
            tail_samples = self.corrupt_triple(s, r, o, negative_rate, self.true_objects_train[t], idx_dict, corrupt_object=True)
            head_samples = self.corrupt_triple(s, r, o, negative_rate, self.true_subjects_train[t], idx_dict, corrupt_object=False)
            neg_object_samples[i][0] = idx_dict[o]
            neg_subject_samples[i][0] = idx_dict[s]
            neg_object_samples[i, 1:] = tail_samples
            neg_subject_samples[i, 1:] = head_samples

        neg_object_samples, neg_subject_samples = torch.from_numpy(neg_object_samples), torch.from_numpy(neg_subject_samples)
        if self.use_cuda:
            neg_object_samples, neg_subject_samples, labels = \
                cuda(neg_object_samples, self.args.n_gpu), cuda(neg_subject_samples, self.args.n_gpu), cuda(labels, self.args.n_gpu)
        return neg_object_samples.long(), neg_subject_samples.long(), labels

    def corrupt_triple(self, s, r, o, negative_rate, other_true_entities, idx_dict, corrupt_object=True):
        negative_sample_list = []
        negative_sample_size = 0
        true_entities = other_true_entities[(s, r)] if corrupt_object else other_true_entities[(r, o)]
        while negative_sample_size < negative_rate:
            # import pdb; pdb.set_trace()
            if self.args.negative_sample_all_entities:
                negative_sample = np.random.randint(self.model.num_ents, size=negative_rate)
            else:
                negative_sample = np.random.choice(self.model.known_entities, size=negative_rate)
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

    def collect_negative_relations_kd(self, positive_triples, time):
        cur_global2local = {v: k for k, v in self.graph_dict_train[time].ids.items()}
        last_global2local = {v: k for k, v in self.graph_dict_train[time - 1].ids.items()}

        subjects = positive_triples[:, 0].tolist()
        objects = positive_triples[:, 2].tolist()
        subject_object_pairs = list(zip(subjects, objects))
        # import pdb; pdb.set_trace()

        last_known_relations = self.model.all_known_relations[time - 1]
        relation_global2local = {n: i for i, n in enumerate(last_known_relations)}
        relational_mask = positive_triples.new_zeros(len(subject_object_pairs), len(last_known_relations))

        for i, (s, o) in enumerate(subject_object_pairs):
            last_s, last_o = last_global2local[s], last_global2local[o]
            cur_s, cur_o = cur_global2local[s], cur_global2local[o]
            last_true_relations = self.graph_dict_train[time - 1].edges[last_s, last_o].data['type_s'].tolist()
            cur_true_relations = self.graph_dict_train[time].edges[cur_s, cur_o].data['type_s'].tolist()
            symmetrics_diff = list(set(cur_true_relations) ^ set(last_true_relations))

            for r in symmetrics_diff:
                relational_mask[i][relation_global2local[r]] = 1

        # subject_tensor = torch.tensor([s for s, o in subject_object_pairs])
        relation_tensor = torch.from_numpy(last_known_relations)
        # object_tensor = torch.tensor([o for s, o in subject_object_pairs])
        return relation_tensor, relational_mask.byte()

    def collect_negative_entities_kd(self, positive_triples, time, negative_rate=50):
        cur_local2global = self.graph_dict_train[time].ids
        last_local2global = self.graph_dict_train[time - 1].ids
        cur_global2local = {v: k for k, v in self.graph_dict_train[time].ids.items()}
        last_global2local = {v: k for k, v in self.graph_dict_train[time - 1].ids.items()}
        last_known_entities = self.model.last_known_entities
        # import pdb; pdb.set_trace()
        # sub_rel_pairs = list(zip(subjects, relations))
        # rel_obj_pairs = list(zip(relations, objects))

        cur_true_objects_dict = self.true_objects_train[time]
        last_true_objects_dict = self.true_objects_train[time - 1]
        cur_true_subjects_dict = self.true_subjects_train[time]
        last_true_subjects_dict = self.true_subjects_train[time - 1]
        neg_subject_samples = np.zeros((len(positive_triples), 1 + negative_rate), dtype=int)
        neg_object_samples = np.zeros((len(positive_triples), 1 + negative_rate), dtype=int)
        for i in range(len(positive_triples)):
            s, r, o, = positive_triples[i]
            s, r, o = s.item(), r.item(), o.item()

            last_s = last_global2local[s]
            cur_s = cur_global2local[s]
            last_true_objects = [last_local2global[o] for o in last_true_objects_dict[(last_s, r)]]
            cur_true_objects = [cur_local2global[o] for o in cur_true_objects_dict[(cur_s, r)]]
            union = list(set(cur_true_objects) | set(last_true_objects))
            neg_object_samples[i][0] = o
            neg_object_samples[i][1:] = self.sample_negative_entities(last_known_entities, union, negative_rate)

            last_o = last_global2local[o]
            cur_o = cur_global2local[o]
            last_true_subjects = [last_local2global[s] for s in last_true_subjects_dict[(r, last_o)]]
            cur_true_subjects = [cur_local2global[s] for s in cur_true_subjects_dict[(r, cur_o)]]
            union = list(set(cur_true_subjects) | set(last_true_subjects))
            neg_subject_samples[i][0] = s
            neg_subject_samples[i][1:] = self.sample_negative_entities(last_known_entities, union, negative_rate)

        neg_subject_samples, neg_object_samples = torch.from_numpy(neg_subject_samples), torch.from_numpy(neg_object_samples)
        if self.use_cuda:
            neg_subject_samples, neg_object_samples = cuda(neg_subject_samples, self.args.n_gpu), cuda(neg_object_samples, self.args.n_gpu)
        return neg_subject_samples, neg_object_samples

    def sample_negative_entities(self, entity_population, other_true_entities, negative_rate):
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < negative_rate:
            negative_sample = np.random.choice(entity_population, size=negative_rate)
            mask = np.in1d(
                negative_sample,
                other_true_entities,
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        return np.concatenate(negative_sample_list)[:negative_rate]
