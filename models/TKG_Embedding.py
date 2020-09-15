from models.TKG_Module import TKG_Module
import time
import torch
from utils.utils import cuda, mse_loss, collect_one_hot_neighbors
import torch.nn.functional as F
import time
import pdb


class TKG_Embedding(TKG_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(TKG_Embedding, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    def load_old_parameters(self):
        self.old_ent_embeds.data = self.ent_embeds.data.clone()
        self.old_rel_embeds.data = self.rel_embeds.data.clone()
        self.last_known_entities = self.all_known_entities[self.time - 1]

    def evaluate(self, quadruples, batch_idx):
        triples = quadruples[:, :-1]
        if triples.shape[0] == 0:
            return cuda(torch.tensor([]).long(), self.n_gpu) if self.use_cuda else torch.tensor([]).long(), 0

        if batch_idx == 0: # first time evaluating the at some epoch
            self.eval_ent_embed = self.get_graph_ent_embeds()
            self.eval_all_embeds_g = self.get_all_embeds_Gt()
            # reduced_entity_embedding[known_id] = self.eval_all_embeds_g[global_id]
            self.reduced_entity_embedding = self.eval_all_embeds_g[self.known_entities]

        local2global = self.train_graph.ids
        global2known = dict({n: i for i, n in enumerate(self.known_entities)})
        rank = self.evaluater.calc_metrics_single_graph(self.eval_ent_embed,
                        self.rel_embeds, self.eval_all_embeds_g, triples, local2global, self.time, global2known)
        return rank

    def evaluate_global(self, quadruples, batch_idx):
        if quadruples.shape[0] == 0:
            return cuda(torch.tensor([]).long(), self.n_gpu) if self.use_cuda else torch.tensor([]).long(), 0
        known_entities = self.all_known_entities[self.args.end_time_step]
        relation_embedding = self.rel_embeds[quadruples[:, 1]]

        if batch_idx == 0: # first time evaluating the at some epoch
            self.precompute_entity_time_embed()

        global2known = dict({n: i for i, n in enumerate(known_entities)})
        return self.evaluater.calc_metrics_quadruples(quadruples, relation_embedding, known_entities, global2known)

    def eval_global_idx(self, triples):
        if triples.shape[0] == 0:
            return cuda(torch.tensor([]).long(), self.n_gpu) if self.use_cuda else torch.tensor([]).long(), 0

        self.eval_all_embeds_g = self.get_all_embeds_Gt()
        self.reduced_entity_embedding = self.eval_all_embeds_g[self.known_entities]

        global2known = dict({n: i for i, n in enumerate(self.known_entities)})
        rank = self.evaluater.calc_metrics_single_graph(self.rel_embeds, triples, global2known)
        return rank

    def inference(self, quadruples, time):
        print('current model : {}, evaluating {}'.format(self.time, time))
        triples = quadruples[:, :-1]
        if triples.shape[0] == 0:
            return cuda(torch.tensor([]).long(), self.n_gpu) if self.use_cuda else torch.tensor([]).long(), 0

        eval_ent_embed = self.get_graph_ent_embeds(time)
        eval_all_embeds_g = self.get_all_embeds_Gt(time)
        self.reduced_entity_embedding = eval_all_embeds_g[self.all_known_entities[time]]

        print("# of known entities: {}, # of active entities: {}".format(len(self.all_known_entities[time]),
                                                                         eval_ent_embed.shape[0]))
        local2global = self.graph_dict_val[time].ids
        global2known = dict({n: i for i, n in enumerate(self.all_known_entities[time])})
        rank = self.evaluater.calc_metrics_single_graph(eval_ent_embed,
                        self.rel_embeds, eval_all_embeds_g, triples, local2global, time, global2known)
        return rank

    def forward_global(self, quadruples):
        neg_object_samples, neg_subject_samples, labels = self.corrupter.negative_sampling(quadruples.cpu())
        torch.cuda.synchronize()
        self.batch_start_time = time.time()
        # self.precompute_entity_time_embed()
        relation_embedding = self.rel_embeds[quadruples[:, 1]]
        time_tensor = quadruples[:, -1]
        subject_embedding = self.get_ent_embeds_train_global(quadruples[:, 0], time_tensor)
        object_embedding = self.get_ent_embeds_train_global(quadruples[:, 2], time_tensor)
        # import pdb; pdb.set_trace()
        neg_subject_embedding = self.get_ent_embeds_train_global(neg_subject_samples, time_tensor, mode='double-neg')
        neg_object_embedding = self.get_ent_embeds_train_global(neg_object_samples, time_tensor, mode='double-neg')

        loss_tail = self.train_link_prediction(subject_embedding, relation_embedding, neg_object_embedding, labels,
                                               corrupt_tail=True)
        loss_head = self.train_link_prediction(neg_subject_embedding, relation_embedding, object_embedding, labels,
                                               corrupt_tail=False)
        return loss_tail + loss_head

    def forward(self, quadruples):
        # import pdb; pdb.set_trace()
        neg_object_samples, neg_subject_samples, labels = self.corrupter.negative_sampling(quadruples.cpu())

        if self.time > 0:
            if self.positive_kd or self.neg_relation_kd or self.neg_entity_kd:
                positive_triples = self.get_positive_samples(quadruples)
                last_entity_embedding = self.get_all_embeds_Gt_old()
                last_subject_embeddings, last_object_embeddings = last_entity_embedding[positive_triples[:, 0]], last_entity_embedding[positive_triples[:, 2]]
                last_relation_embeddings = self.old_rel_embeds[positive_triples[:, 1]]
                cur_relation_embeddings = self.rel_embeds[positive_triples[:, 1]]
            if self.neg_relation_kd:
                relation_tensor, relational_mask = self.corrupter.collect_negative_relations_kd(positive_triples, self.time)

            if self.neg_entity_kd:
                neg_subject_samples, neg_object_samples = self.corrupter.collect_negative_entities_kd(positive_triples,
                                                                    self.time, negative_rate=self.args.negative_entity_kd_rate)

        torch.cuda.synchronize()
        self.batch_start_time = time.time()
        cur_entity_embedding = self.get_all_embeds_Gt()
        if self.time > 0 and (self.positive_kd or self.neg_relation_kd or self.neg_entity_kd):
            cur_subject_embeddings, cur_object_embeddings = cur_entity_embedding[positive_triples[:, 0]], cur_entity_embedding[positive_triples[:, 2]]

        relation_embedding = self.rel_embeds[quadruples[:, 1]]

        self.precompute_entity_time_embed()
        time_tensor = quadruples[:, -1]
        subject_embedding = self.get_ent_embeds_train_global(quadruples[:, 0], time_tensor)
        object_embedding = self.get_ent_embeds_train_global(quadruples[:, 2], time_tensor)
        neg_subject_embedding = self.get_ent_embeds_train_global(neg_subject_samples, time_tensor, mode='neg')
        neg_object_embedding = self.get_ent_embeds_train_global(neg_object_samples, time_tensor, mode='neg')

        loss_tail = self.train_link_prediction(subject_embedding, relation_embedding, neg_object_embedding, labels, corrupt_tail=True)
        loss_head = self.train_link_prediction(neg_subject_embedding, relation_embedding, object_embedding, labels, corrupt_tail=False)
        cross_entropy_loss = loss_tail + loss_head
        if self.deletion:
            cross_entropy_loss += self.unlearn_deleted_edges(cur_entity_embedding)

        if self.self_kd and self.time > 0:
            entity_kd_loss = mse_loss(self.ent_embeds[self.last_known_entities], self.old_ent_embeds[self.last_known_entities])
            relation_kd_loss = mse_loss(self.rel_embeds, self.old_rel_embeds)
            cross_entropy_loss += self.kd_factor * (entity_kd_loss + relation_kd_loss)
        if self.positive_kd and self.time > 0:
            cross_entropy_loss += self.pos_kd(last_subject_embeddings, last_relation_embeddings, last_object_embeddings,
                   cur_subject_embeddings, cur_relation_embeddings, cur_object_embeddings)

        if self.neg_relation_kd and self.time > 0:
            cross_entropy_loss += self.relational_kd(last_subject_embeddings, last_object_embeddings,
                    cur_subject_embeddings, cur_object_embeddings, relation_tensor, relational_mask)

        if self.neg_entity_kd and self.time > 0:
            cross_entropy_loss += self.entity_kd(last_subject_embeddings, last_relation_embeddings, last_object_embeddings,
                                                 cur_subject_embeddings, cur_relation_embeddings, cur_object_embeddings,
                                                 last_entity_embedding, cur_entity_embedding, neg_subject_samples, neg_object_samples)

        return cross_entropy_loss

    def unlearn_deleted_edges(self, all_embeds_g):
        deleted_triples = self.deleted_graph_triples_train[self.time]
        if type(deleted_triples) == type(None) or len(deleted_triples) == 0:
            return 0
        relation_embedding = self.rel_embeds[deleted_triples[:, 1]]
        subject_embedding, object_embedding = all_embeds_g[deleted_triples[:, 0]], all_embeds_g[deleted_triples[:, 2]]
        score = self.calc_score(subject_embedding, relation_embedding, object_embedding)
        labels = torch.zeros(len(deleted_triples))
        labels = cuda(labels, self.n_gpu) if self.use_cuda else labels
        return self.args.up_weight_factor * F.binary_cross_entropy_with_logits(score, labels)

    def get_positive_samples(self, quadruples):
        involved_entities = torch.unique(torch.cat([quadruples[:, 0], quadruples[:, 2]])).tolist()
        positive_triples = collect_one_hot_neighbors(self.common_triples_dict[self.time], involved_entities, self.train_graph.ids,
                                                     self.args.random_positive_sampling, self.args.train_batch_size)
        return cuda(positive_triples, self.n_gpu) if self.use_cuda else positive_triples

    def pos_kd(self, last_subject_embeddings, last_relation_embeddings, last_object_embeddings, cur_subject_embeddings, cur_relation_embeddings, cur_object_embeddings):
        last_triple_scores = self.calc_score(last_subject_embeddings, last_relation_embeddings, last_object_embeddings)
        current_triple_score = self.calc_score(cur_subject_embeddings, cur_relation_embeddings, cur_object_embeddings)
        # F.binary_cross_entropy_with_logits(current_triple_score, last_triple_scores)
        # F.binary_cross_entropy(torch.sigmoid(current_triple_score), torch.sigmoid(last_triple_scores))
        return F.kl_div(F.logsigmoid(current_triple_score), torch.sigmoid(last_triple_scores), reduction='mean')

    def relational_kd(self, last_subject_embeddings, last_object_embeddings, cur_subject_embeddings, cur_object_embeddings, relation_tensor, relational_mask):
        last_relation_embeddings = self.old_rel_embeds[relation_tensor]
        cur_relation_embeddings = self.rel_embeds[relation_tensor]

        last_triple_scores = self.calc_score(last_subject_embeddings, last_relation_embeddings, last_object_embeddings, mode="relation")
        current_triple_score = self.calc_score(cur_subject_embeddings, cur_relation_embeddings, cur_object_embeddings, mode="relation")

        masked_last_triple_scores = torch.where(relational_mask, -10e6 * relational_mask.new_ones(relational_mask.shape).float(), last_triple_scores)
        masked_current_triple_scores = torch.where(relational_mask, -10e6 * relational_mask.new_ones(relational_mask.shape).float(), current_triple_score)

        return self.loss_fn_kd(masked_current_triple_scores, masked_last_triple_scores)

    def entity_kd(self, last_subject_embeddings, last_relation_embeddings, last_object_embeddings,
                   cur_subject_embeddings, cur_relation_embeddings, cur_object_embeddings,
                   last_entity_embedding, cur_entity_embedding, neg_subject_samples, neg_object_samples):
        last_neg_subject_embeddings = last_entity_embedding[neg_subject_samples]
        last_neg_object_embeddings = last_entity_embedding[neg_object_samples]
        cur_neg_subject_embeddings = cur_entity_embedding[neg_subject_samples]
        cur_neg_object_embeddings = cur_entity_embedding[neg_object_samples]

        last_neg_sub_triple_score = self.calc_score(last_neg_subject_embeddings, last_relation_embeddings, last_object_embeddings, mode='head')
        cur_neg_sub_triple_score = self.calc_score(cur_neg_subject_embeddings, cur_relation_embeddings, cur_object_embeddings, mode='head')

        last_neg_obj_triple_score = self.calc_score(last_subject_embeddings, last_relation_embeddings, last_neg_object_embeddings, mode='tail')
        cur_neg_obj_triple_score = self.calc_score(cur_subject_embeddings, cur_relation_embeddings, cur_neg_object_embeddings, mode='tail')
        # pdb.set_trace()
        return self.loss_fn_kd(cur_neg_sub_triple_score, last_neg_sub_triple_score)\
               + self.loss_fn_kd(cur_neg_obj_triple_score, last_neg_obj_triple_score)

    '''
    def get_ent_embeds_multi_step(self):
        max_num_entities = max([len(self.graph_dict_train[t].nodes()) for t in range(max(0, self.time - self.train_seq_len), self.time + 1)])
        num_time_steps = self.time - max(0, self.time - self.train_seq_len) + 1
        ent_embed_all_time = self.ent_embeds.new_zeros(num_time_steps, max_num_entities, self.embed_size)
        all_embeds_g_all_time = self.ent_embeds.new_zeros(num_time_steps, self.num_ents, self.embed_size)
        for t in range(max(0, self.time - self.train_seq_len), self.time + 1):
            cur_local_ent_embedding = self.get_graph_ent_embeds(t)
            ent_embed_all_time[self.time - t][:len(cur_local_ent_embedding)] = cur_local_ent_embedding
            all_embeds_g_all_time[self.time - t] = self.get_all_embeds_Gt(t)

        return ent_embed_all_time, all_embeds_g_all_time

    def forward_multi_step(self, quadruples, all_embeds_g, neg_tail_samples, neg_head_samples, labels):
        ent_embed_all_time, all_embeds_g_all_time = self.get_ent_embeds_multi_step()
        loss_head = self.train_link_prediction_multi_step(ent_embed_all_time, all_embeds_g_all_time, quadruples,
                                                          neg_head_samples, labels, corrupt_tail=False)
        loss_tail = self.train_link_prediction_multi_step(ent_embed_all_time, all_embeds_g_all_time, quadruples,
                                                          neg_tail_samples, labels, corrupt_tail=True)
        return loss_tail + loss_head
    '''