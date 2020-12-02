import torch
from utils.util_functions import cuda, get_true_subject_and_object_per_graph, sort_and_rank
import numpy as np
import pdb


class EvaluationFilterGlobal:
    def __init__(self, model):
        self.model = model
        self.args = model.args
        self.calc_score = model.calc_score
        self.get_true_subject_and_object_global()
        self.get_true_subject_and_object_val()

    def get_true_subject_and_object_global(self):
        self.true_subject_global_dict = dict()
        self.true_object_global_dict = dict()

        for t in self.model.total_time:
            quads = torch.cat([self.model.time2quads_train[t],
                                 self.model.time2quads_val[t],
                                 self.model.time2quads_test[t]])
            true_head_dict, true_tail_dict = get_true_subject_and_object_per_graph(quads[:, :3])
            self.true_subject_global_dict[t] = true_head_dict
            self.true_object_global_dict[t] = true_tail_dict

    def get_true_subject_and_object_val(self):
        self.true_subject_val_dict = dict()
        self.true_object_val_dict = dict()
        self.true_subject_test_dict = dict()
        self.true_object_test_dict = dict()

        for t in self.model.total_time:
            val_quads = self.model.time2quads_val[t]
            test_quads = self.model.time2quads_test[t]
            true_subject_dict_val, true_object_dict_val = get_true_subject_and_object_per_graph(val_quads[:, :3])
            true_subject_dict_test, true_object_dict_test = get_true_subject_and_object_per_graph(test_quads[:, :3])

            self.true_subject_val_dict[t] = true_subject_dict_val
            self.true_object_val_dict[t] = true_object_dict_val

            self.true_subject_test_dict[t] = true_subject_dict_test
            self.true_object_test_dict[t] = true_object_dict_test

    def calc_relative_rank(self, quadruples, known_entities, global2known_func, test_set):
        true_subject_dict = self.true_subject_val_dict if not test_set else self.true_subject_test_dict
        true_object_dict = self.true_object_val_dict if not test_set else self.true_object_test_dict

        subject_tensor = quadruples[:, 0]
        relation_tensor = quadruples[:, 1]
        object_tensor = quadruples[:, 2]
        time_tensor = quadruples[:, 3]
        relation_embedding = self.model.get_rel_embeds(relation_tensor, time_tensor)
        object_scores = self.get_unmasked_score(subject_tensor, relation_embedding, time_tensor, known_entities, mode='tail')
        subject_scores = self.get_unmasked_score(object_tensor, relation_embedding, time_tensor, known_entities, mode='head')

        _, object_rank = torch.sort(object_scores, dim=1, descending=True)
        _, subject_rank = torch.sort(subject_scores, dim=1, descending=True)

        rank_length = object_rank.shape[1]

        raw_ranks = []
        relative_ranks = []
        deleted_facts_ranks = []
        both_positive_ranks = []
        first_positive_ranks = []

        subject_relation_dict = self.model.eval_subject_relation_dict
        object_relation_dict = self.model.eval_object_relation_dict

        for i in range(len(quadruples)):
            s, r, o, t = quadruples[i]
            s, r, o, t = s.item(), r.item(), o.item(), t.item()
            prev_true_object = []
            prev_true_subject = []

            for prev_t in range(max(0, t - self.args.eval_seq_len), t):
                if (s, r) in self.true_object_global_dict[prev_t]:
                    prev_true_object += self.true_object_global_dict[prev_t][(s, r)].tolist()
                if (r, o) in self.true_subject_global_dict[prev_t]:
                    prev_true_subject += self.true_subject_global_dict[prev_t][(r, o)].tolist()

            # true subject or true object: global -> known
            if (s, r) not in subject_relation_dict:
                subject_relation_dict[(s, r)] = 0
                cur_true_object_idx = global2known_func(true_object_dict[t][(s, r)])
                all_cur_true_object_idx = global2known_func(self.true_object_global_dict[t][(s, r)])
                cur_object_rank = object_rank[i].unsqueeze(0).cpu()
                positive_object_rank = torch.nonzero(cur_object_rank.expand(len(cur_true_object_idx), rank_length) == torch.tensor(cur_true_object_idx).view(-1, 1))[:, 1]
                raw_ranks.append(positive_object_rank)
                self.calc_relative_rank_per_query_type(first_positive_ranks, relative_ranks, deleted_facts_ranks, both_positive_ranks, prev_true_object,
                                              positive_object_rank, cur_true_object_idx, cur_object_rank, all_cur_true_object_idx, rank_length, global2known_func)

            if (o, r) not in object_relation_dict:
                object_relation_dict[(o, r)] = 0
                cur_true_subject_idx = global2known_func(true_subject_dict[t][(r, o)])
                all_cur_true_subject_idx = global2known_func(self.true_subject_global_dict[t][(r, o)])
                cur_subject_rank = subject_rank[i].unsqueeze(0).cpu()
                positive_subject_rank = torch.nonzero(cur_subject_rank.expand(len(cur_true_subject_idx), rank_length) == torch.tensor(cur_true_subject_idx).view(-1, 1))[:, 1]
                raw_ranks.append(positive_subject_rank)
                self.calc_relative_rank_per_query_type(first_positive_ranks, relative_ranks, deleted_facts_ranks, both_positive_ranks, prev_true_subject,
                                              positive_subject_rank, cur_true_subject_idx, cur_subject_rank, all_cur_true_subject_idx, rank_length, global2known_func)

        raw_ranks_tensor = 1 + torch.cat(raw_ranks) if len(raw_ranks) > 0 else torch.tensor([]).long()
        first_positive_ranks_tensor = 1 + torch.cat(first_positive_ranks) if len(first_positive_ranks) > 0 else torch.tensor([]).long()
        both_positive_ranks_tensor = 1 + torch.cat(both_positive_ranks) if len(both_positive_ranks) > 0 else torch.tensor([]).long()
        deleted_facts_ranks_tensor = 1 + torch.cat(deleted_facts_ranks) if len(deleted_facts_ranks) > 0 else torch.tensor([]).long()
        relative_ranks_tensor = torch.cat(relative_ranks) if len(relative_ranks) > 0 else torch.tensor([]).float()
        # pdb.set_trace()

        return raw_ranks_tensor, first_positive_ranks_tensor, both_positive_ranks_tensor, relative_ranks_tensor, deleted_facts_ranks_tensor

    def calc_relative_rank_per_query_type(self, first_positive_ranks, relative_ranks, deleted_facts_ranks, both_positive_ranks, prev_true_entity,
                                          positive_entity_rank, cur_true_entity_idx, cur_entity_rank, all_cur_true_entity_idx, rank_length, vfunc):
        if len(prev_true_entity) == 0:
            first_positive_ranks.append(positive_entity_rank)
            return

        prev_true_entity_idx = vfunc(np.unique(prev_true_entity))

        # first positive ranks
        # it's OK here since cur_true_entity_idx is in val set, none of them can be in the training set of t.
        prev_false_cur_true_entity = np.setdiff1d(cur_true_entity_idx, prev_true_entity_idx, assume_unique=True)
        if len(prev_false_cur_true_entity) > 0:
            added_entity_rank = torch.nonzero(
                cur_entity_rank.expand(len(prev_false_cur_true_entity), rank_length) == torch.tensor(
                    prev_false_cur_true_entity).view(-1, 1))[:, 1]
            first_positive_ranks.append(added_entity_rank)

        # both positive rank
        both_positive_entity_idx = np.intersect1d(prev_true_entity_idx, cur_true_entity_idx, assume_unique=True)
        if len(both_positive_entity_idx) > 0:
            both_positive_entity_rank = torch.nonzero(
                cur_entity_rank.expand(len(both_positive_entity_idx), rank_length) == torch.tensor(
                    both_positive_entity_idx).view(-1, 1))[:, 1]
            both_positive_ranks.append(both_positive_entity_rank)

        # deleted edge relative rank
        prev_true_cur_false_entity = np.setdiff1d(prev_true_entity_idx, all_cur_true_entity_idx, assume_unique=True)

        if len(prev_true_cur_false_entity) > 0:
            deleted_entity_rank = torch.nonzero(
                cur_entity_rank.expand(len(prev_true_cur_false_entity), rank_length) == torch.tensor(
                    prev_true_cur_false_entity).view(-1, 1))[:, 1]

            relative_entity_rank = ((1 / (positive_entity_rank + 1).float()).unsqueeze(0) - (1 / (deleted_entity_rank + 1).float()).unsqueeze(1)).view(
                len(positive_entity_rank) * len(deleted_entity_rank))
            # if float('inf') in relative_entity_rank or -float("inf") in relative_entity_rank:
            #     pdb.set_trace()
            deleted_facts_ranks.append(deleted_entity_rank)
            relative_ranks.append(relative_entity_rank)

    def calc_metrics_quadruples(self, quadruples, known_entities, global2known_func, calc_mask):
        # import pdb; pdb.set_trace()

        subject_tensor = quadruples[:, 0]
        relation_tensor = quadruples[:, 1]
        object_tensor = quadruples[:, 2]
        time_tensor = quadruples[:, 3]

        relation_embedding = self.model.get_rel_embeds(relation_tensor, time_tensor)
        entity_length = len(known_entities)
        o_mask = self.mask_eval_set(quadruples, global2known_func, entity_length, calc_mask, mode="tail")
        s_mask = self.mask_eval_set(quadruples, global2known_func, entity_length, calc_mask, mode="head")

        # get object ranks
        ranks_o = self.perturb_and_get_rank(subject_tensor, relation_embedding, object_tensor, time_tensor,
                                            known_entities, o_mask, global2known_func, mode='tail')
        # subject ranks
        ranks_s = self.perturb_and_get_rank(object_tensor, relation_embedding, subject_tensor, time_tensor,
                                            known_entities, s_mask, global2known_func, mode='head')
        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1  # change to 1-indexed
        return ranks

    def mask_eval_set(self, quadruples, global2known_func, entity_length, calc_mask, mode='tail'):
        test_size = quadruples.shape[0]
        mask = quadruples.new_zeros(test_size, entity_length)
        if not calc_mask:
            return mask.byte()
        # filter setting
        for i in range(test_size):
            s, r, o, t = quadruples[i]
            s, r, o, t = s.item(), r.item(), o.item(), t.item()
            # true subject or true object: local -> global -> known
            if mode == 'tail':
                tails = self.true_object_global_dict[t][(s, r)]
                tail_idx = global2known_func(tails)
                mask[i][tail_idx] = 1
                mask[i][global2known_func(o)] = 0

            elif mode == 'head':
                heads = self.true_subject_global_dict[t][(r, o)]
                head_idx = global2known_func(heads)
                mask[i][head_idx] = 1
                mask[i][global2known_func(s)] = 0

        return mask.byte()

    def perturb_and_get_rank(self, anchor_entities, relation_embedding, target, time_tensor,
                             known_entities, mask, global2known_func, mode ='tail'):
        """ Perturb one element in the triplets
        """

        cur_target = torch.tensor(global2known_func(target.cpu()))

        if self.args.use_cuda:
            cur_target = cuda(cur_target, self.args.n_gpu)

        unmasked_score = self.get_unmasked_score(anchor_entities, relation_embedding, time_tensor, known_entities, mode)
        masked_score = torch.where(mask, -10e6 * unmasked_score.new_ones(unmasked_score.shape), unmasked_score)

        # mask: 1 for local id
        score = torch.sigmoid(masked_score)  # bsz, n_ent
        return sort_and_rank(score, cur_target)

    def get_unmasked_score(self, anchor_entities, relation_embedding, time_tensor, known_entities, mode ='tail'):
        anchor_embedding = self.model.get_ent_embeds_train_global(
                    anchor_entities, time_tensor)
        neg_entity_embeddings = self.model.get_ent_embeds_train_global(
                    known_entities, time_tensor, mode='neg')

        if mode == 'tail':
            subject_embedding = anchor_embedding
            object_embedding = neg_entity_embeddings

        else:
            subject_embedding = neg_entity_embeddings
            object_embedding = anchor_embedding

        return self.calc_score(subject_embedding, relation_embedding, object_embedding, mode=mode)
