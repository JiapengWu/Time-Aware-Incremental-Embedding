from models.TKG_Module_Global import TKG_Module_Global
import torch
from utils.util_functions import cuda, mse_loss, store_grad, overwrite_grad
import torch.nn.functional as F
import time
import numpy as np
import math
import pdb


class TKG_Embedding_Global(TKG_Module_Global):
    def __init__(self, args, num_ents, num_rels):
        super(TKG_Embedding_Global, self).__init__(args, num_ents, num_rels)

    def load_old_parameters(self):
        self.old_ent_embeds.data = self.ent_embeds.data.clone()
        self.old_rel_embeds.data = self.rel_embeds.data.clone()
        self.last_known_entities = self.all_known_entities[self.time - 1]

    def evaluation_func(self, quadruples, known_entities, calc_mask=True):
        global2known = dict({n: i for i, n in enumerate(known_entities)})
        return self.evaluater.calc_metrics_quadruples(quadruples, known_entities, np.vectorize(lambda x: global2known[x]), calc_mask)

    def relative_rank_eval_func(self, quadruples, known_entities, test_set=False):
        global2known = dict({n: i for i, n in enumerate(known_entities)})
        return self.evaluater.calc_relative_rank(quadruples, known_entities, np.vectorize(lambda x: global2known[x]), test_set)

    def evaluate(self, quadruples, batch_idx):
        # overfitting test
        if type(quadruples) == dict:
            quadruples = quadruples['train']

        if batch_idx == 0: # first time evaluating the at some epoch
            self.precompute_entity_time_embed()

        if quadruples.shape[0] == 0:
            return cuda(torch.tensor([]).long(), self.n_gpu) if self.use_cuda else torch.tensor([]).long(), 0

        known_entities = self.all_known_entities[self.args.end_time_step - 1] if self.args.train_base_model else self.known_entities
        return self.evaluation_func(quadruples, known_entities)

    def test_func(self, quadruples_dict, batch_idx):
        if batch_idx == 0: # first time evaluating the at some epoch
            self.precompute_entity_time_embed()

        # if 'deleted' in quadruples_dict:
        raw_ranks = first_positive_ranks = both_positive_ranks = relative_ranks = deleted_facts_ranks = None
        if self.time in quadruples_dict:
            raw_ranks, first_positive_ranks, both_positive_ranks, relative_ranks, deleted_facts_ranks = \
                self.relative_rank_eval_func(quadruples_dict[self.time], self.all_known_entities[self.time])

        times = []
        ranks = []

        for t, quadruples in quadruples_dict.items():
            cur_times = quadruples[:, 3]
            known_entities = self.all_known_entities[t]
            cur_ranks = self.evaluation_func(quadruples, known_entities)
            times.extend([cur_times, cur_times])
            ranks.append(cur_ranks)

        return raw_ranks, first_positive_ranks, both_positive_ranks, relative_ranks, deleted_facts_ranks, torch.cat(ranks), torch.cat(times)

    '''
    def eval_global_idx(self, triples):
        if triples.shape[0] == 0:
            return cuda(torch.tensor([]).long(), self.n_gpu) if self.use_cuda else torch.tensor([]).long(), 0

        self.eval_all_embeds_g = self.get_all_embeds_Gt()
        self.reduced_entity_embedding = self.eval_all_embeds_g[self.known_entities]

        global2known = dict({n: i for i, n in enumerate(self.known_entities)})
        rank = self.evaluater.calc_metrics_single_graph(self.rel_embeds, triples, global2known)
        return rank
    '''

    def inference(self, cur_time, start_time_step, end_time_step, test_set=False):
        test_time2quads = self.time2quads_val if not test_set else self.time2quads_test
        # print('current model : {}, evaluating {}'.format(self.time, time))

        self.precompute_entity_time_embed()
        cur_quadruples = cuda(test_time2quads[cur_time], self.n_gpu) if self.use_cuda else test_time2quads[cur_time]
        test_batch_size = self.args.test_batch_size

        num_batchs = math.ceil(len(cur_quadruples) / test_batch_size)
        raw_ranks_lst = []; first_positive_ranks_lst = []; both_positive_ranks_lst = []; relative_ranks_lst = []; deleted_facts_ranks_lst = []

        for batch_id in range(num_batchs):
            start, end = batch_id * test_batch_size, (batch_id + 1) * test_batch_size
            raw_ranks, first_positive_ranks, both_positive_ranks, relative_ranks, deleted_facts_ranks = \
                self.relative_rank_eval_func(cur_quadruples[start:end], self.all_known_entities[cur_time], test_set)
            raw_ranks_lst.append(raw_ranks)
            first_positive_ranks_lst.append(first_positive_ranks)
            both_positive_ranks_lst.append(both_positive_ranks)
            relative_ranks_lst.append(relative_ranks)
            deleted_facts_ranks_lst.append(deleted_facts_ranks)

        raw_ranks, first_positive_ranks, both_positive_ranks, relative_ranks, \
                deleted_facts_ranks = torch.cat(raw_ranks_lst), torch.cat(first_positive_ranks_lst), \
                torch.cat(both_positive_ranks_lst), torch.cat(relative_ranks_lst), torch.cat(deleted_facts_ranks_lst)

        rank_dict = {}
        for time in range(start_time_step, end_time_step):
            quadruples = test_time2quads[time]
            if self.use_cuda:
                quadruples = cuda(quadruples, self.n_gpu)
            num_batchs = math.ceil(len(quadruples) / test_batch_size)
            cur_tank_lst = []
            for batch_id in range(num_batchs):
                start, end = batch_id * test_batch_size, (batch_id + 1) * test_batch_size
                cur_tank_lst.append(self.evaluation_func(quadruples[start:end], self.all_known_entities[time]))
            rank_dict[time] = torch.cat(cur_tank_lst)

        return raw_ranks, first_positive_ranks, both_positive_ranks, relative_ranks, deleted_facts_ranks, rank_dict

    def single_step_inference(self, cur_time, test_set=False):
        test_time2quads = self.time2quads_val if not test_set else self.time2quads_test
        self.precompute_entity_time_embed()
        cur_quadruples = cuda(test_time2quads[cur_time], self.n_gpu) if self.use_cuda else test_time2quads[cur_time]
        ranks = self.evaluation_func(cur_quadruples, self.all_known_entities[cur_time])

        predictions = []
        num_quads = len(cur_quadruples)
        for i in range(num_quads):
            s, r, o, _ = cur_quadruples[i]
            s, r, o = s.item(), r.item(), o.item()
            predictions.append([s, r, o, cur_time, 'sub', ranks[i].item()])
            predictions.append([s, r, o, cur_time, 'obj', ranks[num_quads + i].item()])
        return predictions

    def forward_global(self, quadruples):
        neg_object_samples, neg_subject_samples, labels = self.corrupter.negative_sampling(quadruples.cpu(), self.args.negative_rate)
        torch.cuda.synchronize()
        self.batch_start_time = time.time()
        return self.learn_training_edges(quadruples, neg_subject_samples, neg_object_samples, labels)

    '''
    def backward(self, trainer, loss, optimizer, optimizer_idx):
        loss.backward()
        if self.args.a_gem:
            store_grad(self.named_parameters, self.grads)
            dotp = torch.sum(self.ref_grads * self.grads)
            # if self.args.plot_gradient:
            #     self.metrics_collector.update_gradient_similarity(self.time,
            #       dotp / (torch.norm(self.ref_grads, p=2) * torch.norm(self.grads, p=2)))
            #     pdb.set_trace()
            if dotp < 0:
                g_ref_square = torch.sum(self.ref_grads * self.ref_grads)
                g_projected = self.grads - (dotp / g_ref_square) * self.ref_grads
                overwrite_grad(self.named_parameters, g_projected)
                # store_grad_new(self.named_parameters, self.grads)
    '''

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        loss.backward()
        if self.args.a_gem and self.time > 0:
            store_grad(self.named_parameters, self.grads)
            g_projected = {}

            for k in self.grads:
                dot_p = torch.sum(self.ref_grads[k] * self.grads[k], -1)
                dot_p = torch.where(dot_p < 0, dot_p, self.zero_matrix[:dot_p.shape[0]])
                factor = dot_p / torch.sum(self.ref_grads[k] * self.ref_grads[k], -1)
                # 0 / 0 -> inf; n / 0 -> nan
                factor = torch.where(torch.logical_or(torch.isinf(factor), torch.isnan(factor)),
                                     self.zero_matrix[:factor.shape[0]], factor)
                g_projected[k] = self.grads[k] - factor.unsqueeze(-1) * self.ref_grads[k]
            overwrite_grad(self.named_parameters, g_projected)

    def forward(self, quadruples_dict):
        # insert timer
        loss = 0

        if 'train' in quadruples_dict:
            train_quadruples = quadruples_dict['train']
            neg_object_samples, neg_subject_samples, labels = \
                self.corrupter.negative_sampling(train_quadruples.cpu(), self.args.negative_rate)

        if self.reservoir_sampling and self.sample_neg_entity and self.time > 0:
            reservoir_samples = quadruples_dict['reservoir']
            pos_time_tensor = reservoir_samples[:, -1]

            self.neg_reservoir_object_samples, self.neg_reservoir_subject_samples, self.reservoir_labels \
                = self.corrupter.negative_sampling(reservoir_samples.cpu(),
                    self.args.negative_rate_reservoir, use_fixed_known_entities=False)

            self.cur_neg_subject_embeddings = self.get_ent_embeds_train_global(
                self.neg_reservoir_subject_samples, pos_time_tensor, mode='double-neg')
            self.cur_neg_object_embeddings = self.get_ent_embeds_train_global(
                self.neg_reservoir_object_samples, pos_time_tensor, mode='double-neg')

            if self.args.KD_reservoir:
                self.last_neg_subject_embeddings = self.get_ent_embeds_train_global_old(
                    self.neg_reservoir_subject_samples, pos_time_tensor, mode='double-neg')
                self.last_neg_object_embeddings = self.get_ent_embeds_train_global_old(
                    self.neg_reservoir_object_samples, pos_time_tensor, mode='double-neg')

        if self.use_cuda:
            torch.cuda.synchronize()

        self.batch_start_time = time.time()
        if self.args.a_gem and self.time > 0:

            res_loss = self.calc_quad_kd_loss(quadruples_dict['reservoir'])
            # if self.deletion and 'prev true' in quadruples_dict:
            #     res_loss += self.learn_prev_true_edges(quadruples_dict['prev true'])
            res_loss.backward()
            store_grad(self.named_parameters, self.ref_grads)
            self.zero_grad()

            if 'train' in quadruples_dict:
                loss = self.learn_training_edges(train_quadruples, neg_subject_samples, neg_object_samples, labels)

            if self.deletion and 'deleted' in quadruples_dict:
                loss += self.unlearn_deleted_edges(quadruples_dict['deleted'])

            if self.self_kd and self.time > 0:
                loss += self.self_kd_factor * self.calc_self_kd_loss()

        else:
            if 'train' in quadruples_dict:
                loss += self.learn_training_edges(train_quadruples, neg_subject_samples, neg_object_samples, labels)

            if self.deletion and 'deleted' in quadruples_dict:
                loss += self.unlearn_deleted_edges(quadruples_dict['deleted'])

            if self.self_kd and self.time > 0:
                loss += self.self_kd_factor * self.calc_self_kd_loss()

            if self.reservoir_sampling and 'reservoir' in quadruples_dict:
                loss += self.calc_quad_kd_loss(quadruples_dict['reservoir'])
        return loss

    def calc_quad_kd_loss(self, reservoir_samples):
        loss = 0
        pos_time_tensor = reservoir_samples[:, -1]
        cur_pos_relation_embeddings, cur_pos_subject_embeddings, cur_pos_object_embeddings = \
            self.get_cur_embedding_positive(reservoir_samples, pos_time_tensor)

        if self.args.KD_reservoir:

            last_pos_relation_embeddings, last_pos_subject_embeddings, last_pos_object_embeddings = \
                self.get_old_embedding_positive(reservoir_samples, pos_time_tensor)


            if self.sample_positive:
                loss += self.pos_kd(last_pos_subject_embeddings, last_pos_relation_embeddings, last_pos_object_embeddings,
                                    cur_pos_subject_embeddings, cur_pos_relation_embeddings, cur_pos_object_embeddings)

            if self.sample_neg_entity:
                loss += self.entity_kd(cur_pos_subject_embeddings, cur_pos_relation_embeddings, cur_pos_object_embeddings,
                                       last_pos_relation_embeddings, last_pos_subject_embeddings, last_pos_object_embeddings,
                                       self.cur_neg_subject_embeddings, self.cur_neg_object_embeddings,
                                       self.last_neg_subject_embeddings, self.last_neg_object_embeddings)

            if self.sample_neg_relation:
                raise NotImplementedError

        if self.args.CE_reservoir:

            if self.sample_positive:
                score = self.calc_score(cur_pos_subject_embeddings, cur_pos_relation_embeddings,
                                        cur_pos_object_embeddings)
                labels = torch.ones(len(reservoir_samples))
                labels = cuda(labels, self.n_gpu) if self.use_cuda else labels
                loss += F.binary_cross_entropy_with_logits(score, labels)

            if self.sample_neg_entity:
                loss_tail = self.train_link_prediction(cur_pos_subject_embeddings, cur_pos_relation_embeddings,
                                                       self.cur_neg_object_embeddings, self.reservoir_labels,
                                                       corrupt_tail=True)
                loss_head = self.train_link_prediction(self.cur_neg_subject_embeddings, cur_pos_relation_embeddings,
                                                       cur_pos_object_embeddings, self.reservoir_labels,
                                                       corrupt_tail=False)
                loss += loss_tail + loss_head

            if self.sample_neg_relation:
                raise NotImplementedError
        return loss

    def calc_self_kd_loss(self):
        entity_kd_loss = mse_loss(self.ent_embeds[self.last_known_entities],
                                  self.old_ent_embeds[self.last_known_entities])
        relation_kd_loss = mse_loss(self.rel_embeds, self.old_rel_embeds)
        return entity_kd_loss + relation_kd_loss

    def learn_training_edges(self, train_quadruples, neg_subject_samples, neg_object_samples, labels):
        time_tensor = train_quadruples[:, -1]
        relation_embeddings = self.get_rel_embeds(train_quadruples[:, 1], time_tensor)
        subject_embeddings = self.get_ent_embeds_train_global(train_quadruples[:, 0], time_tensor)
        object_embeddings = self.get_ent_embeds_train_global(train_quadruples[:, 2], time_tensor)
        neg_subject_embedding = self.get_ent_embeds_train_global(neg_subject_samples, time_tensor, mode='double-neg')
        neg_object_embedding = self.get_ent_embeds_train_global(neg_object_samples, time_tensor, mode='double-neg')

        loss_tail = self.train_link_prediction(subject_embeddings, relation_embeddings, neg_object_embedding, labels,
                                               corrupt_tail=True)
        loss_head = self.train_link_prediction(neg_subject_embedding, relation_embeddings, object_embeddings, labels,
                                               corrupt_tail=False)
        return loss_tail + loss_head

    def learn_prev_true_edges(self, prev_true_quads):
        subjects, relations, objects, prev_time_tensor = [prev_true_quads[:, i] for i in range(4)]

        subject_embeddings = self.get_ent_embeds_train_global(subjects, prev_time_tensor)
        object_embeddings = self.get_ent_embeds_train_global(objects, prev_time_tensor)
        relation_embeddings = self.get_rel_embeds(relations, prev_time_tensor)

        score = self.calc_score(subject_embeddings, relation_embeddings, object_embeddings)
        labels = cuda(torch.ones(len(prev_true_quads)), self.n_gpu) if self.use_cuda else torch.ones(len(prev_true_quads))
        return self.args.up_weight_factor * F.binary_cross_entropy_with_logits(score, labels)

    def unlearn_deleted_edges(self, deleted_quadruples):
        subjects, relations, objects, cur_time_tensor = [deleted_quadruples[:, i] for i in range(4)]
        cur_subject_embeddings = self.get_ent_embeds_train_global(subjects, cur_time_tensor)
        cur_object_embeddings = self.get_ent_embeds_train_global(objects, cur_time_tensor)
        cur_relation_embeddings = self.get_rel_embeds(relations, cur_time_tensor)

        score = self.calc_score(cur_subject_embeddings, cur_relation_embeddings, cur_object_embeddings)
        labels = cuda(torch.zeros(len(deleted_quadruples)), self.n_gpu) if self.use_cuda else torch.zeros(len(deleted_quadruples))
        return self.args.up_weight_factor * F.binary_cross_entropy_with_logits(score, labels)

    def get_cur_embedding_positive(self, reservoir_samples, pos_time_tensor):
        cur_pos_relation_embeddings = self.get_rel_embeds(reservoir_samples[:, 1], pos_time_tensor)
        cur_pos_subject_embeddings = self.get_ent_embeds_train_global(reservoir_samples[:, 0], pos_time_tensor)
        cur_pos_object_embeddings = self.get_ent_embeds_train_global(reservoir_samples[:, 2], pos_time_tensor)
        return cur_pos_relation_embeddings, cur_pos_subject_embeddings, cur_pos_object_embeddings

    def get_old_embedding_positive(self, reservoir_samples, pos_time_tensor):
        last_pos_relation_embeddings = self.get_old_rel_embeds(reservoir_samples[:, 1], pos_time_tensor)
        last_pos_subject_embeddings = self.get_ent_embeds_train_global_old(reservoir_samples[:, 0], pos_time_tensor)
        last_pos_object_embeddings = self.get_ent_embeds_train_global_old(reservoir_samples[:, 2], pos_time_tensor)
        return last_pos_relation_embeddings, last_pos_subject_embeddings, last_pos_object_embeddings

    def pos_kd(self, last_subject_embeddings, last_relation_embeddings, last_object_embeddings, cur_subject_embeddings, cur_relation_embeddings, cur_object_embeddings):
        last_triple_scores = self.calc_score(last_subject_embeddings, last_relation_embeddings, last_object_embeddings)
        current_triple_score = self.calc_score(cur_subject_embeddings, cur_relation_embeddings, cur_object_embeddings)
        return F.kl_div(F.logsigmoid(current_triple_score), torch.sigmoid(last_triple_scores), reduction='mean')

    def entity_kd(self, cur_pos_subject_embeddings, cur_pos_relation_embeddings, cur_pos_object_embeddings,
                         last_pos_relation_embeddings, last_pos_subject_embeddings, last_pos_object_embeddings,
                         cur_neg_subject_embeddings, cur_neg_object_embeddings, last_neg_subject_embeddings, last_neg_object_embeddings):
        last_neg_sub_triple_score = self.calc_score(last_neg_subject_embeddings, last_pos_relation_embeddings, last_pos_object_embeddings, mode='head')
        cur_neg_sub_triple_score = self.calc_score(cur_neg_subject_embeddings, cur_pos_relation_embeddings, cur_pos_object_embeddings, mode='head')

        last_neg_obj_triple_score = self.calc_score(last_pos_subject_embeddings, last_pos_relation_embeddings, last_neg_object_embeddings, mode='tail')
        cur_neg_obj_triple_score = self.calc_score(cur_pos_subject_embeddings, cur_pos_relation_embeddings, cur_neg_object_embeddings, mode='tail')
        return self.loss_fn_kd(cur_neg_sub_triple_score, last_neg_sub_triple_score) \
                              + self.loss_fn_kd(cur_neg_obj_triple_score, last_neg_obj_triple_score)
