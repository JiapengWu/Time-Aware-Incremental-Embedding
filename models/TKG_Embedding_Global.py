from models.TKG_Module_Global import TKG_Module_Global
import time
import torch
from utils.utils import cuda, mse_loss, collect_one_hot_neighbors_global
import torch.nn.functional as F
import time
import pdb


class TKG_Embedding_Global(TKG_Module_Global):
    def __init__(self, args, num_ents, num_rels):
        super(TKG_Embedding_Global, self).__init__(args, num_ents, num_rels)

    def load_old_parameters(self):
        self.old_ent_embeds.data = self.ent_embeds.data.clone()
        self.old_rel_embeds.data = self.rel_embeds.data.clone()
        self.last_known_entities = self.all_known_entities[self.time - 1]

    def evaluate(self, quadruples, batch_idx):
        if type(quadruples) == dict:
            quadruples = quadruples['train']

        if quadruples.shape[0] == 0:
            return cuda(torch.tensor([]).long(), self.n_gpu) if self.use_cuda else torch.tensor([]).long(), 0
        known_entities = self.all_known_entities[self.args.end_time_step] if self.args.train_base_model else self.known_entities
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
        neg_object_samples, neg_subject_samples, labels = self.corrupter.negative_sampling(quadruples.cpu(), self.args.negative_rate)
        torch.cuda.synchronize()
        self.batch_start_time = time.time()

        return self.learn_training_edges(quadruples, neg_subject_samples, neg_object_samples, labels)

    def forward(self, quadruples_dict):
        if 'train' in quadruples_dict:
            train_quadruples = quadruples_dict['train']
            neg_object_samples, neg_subject_samples, labels = self.corrupter.negative_sampling(train_quadruples.cpu(), self.args.negative_rate)

        if self.sample_neg_entity and self.reservoir_sampling:
            self.neg_reservoir_object_samples, self.neg_reservoir_subject_samples, self.reservoir_labels \
                = self.corrupter.negative_sampling(quadruples_dict['reservoir'].cpu(),
                        self.args.negative_rate_reservoir, use_fixed_known_entities=False)

        if self.use_cuda:
            torch.cuda.synchronize()

        self.batch_start_time = time.time()

        loss = 0
        if 'train' in quadruples_dict:
            loss += self.learn_training_edges(train_quadruples, neg_subject_samples, neg_object_samples, labels)

        if self.deletion and 'deleted' in quadruples_dict:
            loss += self.unlearn_deleted_edges(quadruples_dict['deleted'])

        if self.self_kd:
            loss += self.self_kd_factor * self.calc_self_kd_loss()

        if self.reservoir_sampling:
            loss += self.calc_quad_kd_loss(quadruples_dict['reservoir'])
        return loss

    def calc_quad_kd_loss(self, reservoir_samples):
        loss = 0
        pos_time_tensor = reservoir_samples[:, -1]
        cur_pos_relation_embeddings, cur_pos_subject_embeddings, cur_pos_object_embeddings = \
            self.get_cur_embedding_positive(reservoir_samples, pos_time_tensor)

        if self.sample_neg_entity:
            # pdb.set_trace()
            cur_neg_subject_embeddings = self.get_ent_embeds_train_global(
                self.neg_reservoir_subject_samples, pos_time_tensor, mode='double-neg')
            cur_neg_object_embeddings = self.get_ent_embeds_train_global(
                self.neg_reservoir_object_samples, pos_time_tensor, mode='double-neg')

        if self.args.KD_reservoir:

            last_pos_relation_embeddings, last_pos_subject_embeddings, last_pos_object_embeddings = \
                self.get_old_embedding_positive(reservoir_samples, pos_time_tensor)

            if self.sample_positive:
                loss += self.pos_kd(last_pos_subject_embeddings, last_pos_relation_embeddings,
                                    last_pos_object_embeddings,
                                    cur_pos_subject_embeddings, cur_pos_relation_embeddings,
                                    cur_pos_object_embeddings)

            if self.sample_neg_entity:
                last_neg_subject_embeddings = self.get_ent_embeds_train_global_old(self.neg_reservoir_subject_samples,
                                                                                   pos_time_tensor,
                                                                                   mode='double-neg')
                last_neg_object_embeddings = self.get_ent_embeds_train_global_old(self.neg_reservoir_object_samples,
                                                                                  pos_time_tensor,
                                                                                  mode='double-neg')
                loss += self.entity_kd(cur_pos_subject_embeddings, cur_pos_relation_embeddings,
                                       cur_pos_object_embeddings,
                                       last_pos_relation_embeddings, last_pos_subject_embeddings,
                                       last_pos_object_embeddings,
                                       cur_neg_subject_embeddings, cur_neg_object_embeddings,
                                       last_neg_subject_embeddings, last_neg_object_embeddings)
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
                                                       cur_neg_object_embeddings, self.reservoir_labels,
                                                       corrupt_tail=True)
                loss_head = self.train_link_prediction(cur_neg_subject_embeddings, cur_pos_relation_embeddings,
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
        relation_embeddings = self.rel_embeds[train_quadruples[:, 1]]
        subject_embeddings = self.get_ent_embeds_train_global(train_quadruples[:, 0], time_tensor)
        object_embeddings = self.get_ent_embeds_train_global(train_quadruples[:, 2], time_tensor)
        neg_subject_embedding = self.get_ent_embeds_train_global(neg_subject_samples, time_tensor, mode='double-neg')
        neg_object_embedding = self.get_ent_embeds_train_global(neg_object_samples, time_tensor, mode='double-neg')

        loss_tail = self.train_link_prediction(subject_embeddings, relation_embeddings, neg_object_embedding, labels,
                                               corrupt_tail=True)
        loss_head = self.train_link_prediction(neg_subject_embedding, relation_embeddings, object_embeddings, labels,
                                               corrupt_tail=False)
        return loss_tail + loss_head

    def unlearn_deleted_edges(self, deleted_quadruples):
        time_tensor = deleted_quadruples[:, -1]
        subject_embeddings = self.get_ent_embeds_train_global(deleted_quadruples[:, 0], time_tensor)
        object_embeddings = self.get_ent_embeds_train_global(deleted_quadruples[:, 2], time_tensor)
        relation_embedding = self.rel_embeds[deleted_quadruples[:, 1]]
        # subject_embedding, object_embedding = all_embeds_g[deleted_triples[:, 0]], all_embeds_g[deleted_triples[:, 2]]
        score = self.calc_score(subject_embeddings, relation_embedding, object_embeddings)
        labels = torch.zeros(len(deleted_quadruples))
        labels = cuda(labels, self.n_gpu) if self.use_cuda else labels
        return self.args.up_weight_factor * F.binary_cross_entropy_with_logits(score, labels)

    def get_cur_embedding_positive(self, reservoir_samples, pos_time_tensor):
        cur_pos_relation_embeddings = self.get_rel_embeds_train_global(reservoir_samples[:, 1], pos_time_tensor)
        cur_pos_subject_embeddings = self.get_ent_embeds_train_global(reservoir_samples[:, 0], pos_time_tensor)
        cur_pos_object_embeddings = self.get_ent_embeds_train_global(reservoir_samples[:, 2], pos_time_tensor)
        return cur_pos_relation_embeddings, cur_pos_subject_embeddings, cur_pos_object_embeddings

    def get_old_embedding_positive(self, reservoir_samples, pos_time_tensor):
        last_pos_relation_embeddings = self.get_rel_embeds_train_global_old(reservoir_samples[:, 1], pos_time_tensor)
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
        cur_neg_obj_triple_score = self.calc_score(cur_pos_subject_embeddings, cur_pos_relation_embeddings,  cur_neg_object_embeddings, mode='tail')
        return self.loss_fn_kd(cur_neg_sub_triple_score, last_neg_sub_triple_score) \
                              + self.loss_fn_kd(cur_neg_obj_triple_score, last_neg_obj_triple_score)

    def get_rel_embeds_train_global_old(self, relations, time_tensor):
        return self.rel_embeds[relations]

    def get_rel_embeds_train_global(self, relations, time_tensor):
        return self.rel_embeds[relations]