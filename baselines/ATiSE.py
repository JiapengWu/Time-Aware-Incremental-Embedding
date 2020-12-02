from torch import nn
from utils.scores import *
from utils.util_functions import cuda
from models.TKG_Embedding_Global import TKG_Embedding_Global
import math
import numpy as np
import torch.nn.functional as F
import pdb


class ATiSE(TKG_Embedding_Global):
    def __init__(self, args, num_ents, num_rels):
        super(ATiSE, self).__init__(args, num_ents, num_rels)
        self.weight_normalization()

        if self.reservoir_sampling or self.self_kd:
            self.old_alpha_ent = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size), requires_grad=False)
            self.old_alpha_rel = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size), requires_grad=False)
            self.old_w_ent = nn.Parameter(torch.Tensor(self.num_ents, 1), requires_grad=False)
            self.old_w_rel = nn.Parameter(torch.Tensor(self.num_rels * 2, 1), requires_grad=False)
            self.old_beta_ent = nn.Parameter(torch.Tensor(self.num_ents, 1), requires_grad=False)
            self.old_beta_rel = nn.Parameter(torch.Tensor(self.num_rels * 2, 1), requires_grad=False)
            self.old_omega_ent = nn.Parameter(torch.Tensor(self.num_ents, 1), requires_grad=False)
            self.old_omega_rel = nn.Parameter(torch.Tensor(self.num_rels * 2, 1), requires_grad=False)
            self.old_sigma_ent = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size), requires_grad=False)
            self.old_sigma_rel = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size), requires_grad=False)

    def load_old_parameters(self):
        super(ATiSE, self).load_old_parameters()
        self.old_alpha_ent.data = self.alpha_ent.data.clone()
        self.old_alpha_rel.data = self.alpha_rel.data.clone()
        self.old_w_ent.data = self.w_ent.data.clone()
        self.old_w_rel.data = self.w_rel.data.clone()
        self.old_beta_ent.data = self.beta_ent.data.clone()
        self.old_beta_rel.data = self.beta_rel.data.clone()
        self.old_omega_ent.data = self.omega_ent.data.clone()
        self.old_omega_rel.data = self.omega_rel.data.clone()
        self.old_sigma_ent.data = self.sigma_ent.data.clone()
        self.old_sigma_rel.data = self.sigma_rel.data.clone()

    def build_model(self):
        beta_size = 1 if True else self.embed_size
        self.w_ent = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.w_rel = nn.Parameter(torch.Tensor(self.num_rels*2, self.embed_size))
        nn.init.xavier_uniform_(self.w_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_rel, gain=nn.init.calculate_gain('relu'))

        self.alpha_ent = nn.Parameter(torch.Tensor(self.num_ents, 1))
        self.alpha_rel = nn.Parameter(torch.Tensor(self.num_rels*2, 1))
        nn.init.xavier_uniform_(self.alpha_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.alpha_rel, gain=nn.init.calculate_gain('relu'))

        self.beta_ent = nn.Parameter(torch.Tensor(self.num_ents, beta_size))
        self.beta_rel = nn.Parameter(torch.Tensor(self.num_rels*2, beta_size))
        nn.init.xavier_uniform_(self.beta_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.beta_rel, gain=nn.init.calculate_gain('relu'))

        self.omega_ent = nn.Parameter(torch.Tensor(self.num_ents, beta_size))
        self.omega_rel = nn.Parameter(torch.Tensor(self.num_rels*2, beta_size))
        nn.init.xavier_uniform_(self.omega_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.omega_rel, gain=nn.init.calculate_gain('relu'))

        self.sigma_ent = nn.Parameter(torch.ones(self.num_ents, self.embed_size).uniform_(0.005,0.5))
        self.sigma_rel = nn.Parameter(torch.ones(self.num_rels*2, self.embed_size).uniform_(0.005,0.5))
        nn.init.xavier_uniform_(self.sigma_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.sigma_rel, gain=nn.init.calculate_gain('relu'))

    def init_parameters(self):
        super(ATiSE, self).init_parameters()
        nn.init.xavier_uniform_(self.w_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_rel, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.alpha_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.alpha_rel, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.beta_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.beta_rel, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.omega_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.omega_rel, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.sigma_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.sigma_rel, gain=nn.init.calculate_gain('relu'))

    def weight_normalization(self):
        self.ent_embeds.data.copy_(self.ent_embeds / torch.norm(self.ent_embeds, dim=1).unsqueeze(1))
        self.rel_embeds.data.copy_(self.rel_embeds / torch.norm(self.rel_embeds, dim=1).unsqueeze(1))
        self.w_ent.data.copy_(self.w_ent / torch.norm(self.w_ent, dim=1).unsqueeze(1))        
        self.w_rel.data.copy_(self.w_rel / torch.norm(self.w_rel, dim=1).unsqueeze(1))

        self.sigma_ent.data.copy_(torch.clamp(input=self.sigma_ent.data, min=0.005, max=0.5))
        self.sigma_rel.data.copy_(torch.clamp(input=self.sigma_rel.data, min=0.005, max=0.5))    

    def precompute_entity_time_embed(self):
        time_tensor = torch.tensor(list(range(self.args.end_time_step))).unsqueeze(0).unsqueeze(2)
        if self.use_cuda:
            time_tensor = cuda(time_tensor, self.n_gpu)

        trend = self.alpha_ent.unsqueeze(1) * self.w_ent.unsqueeze(1) * time_tensor
        seasonality = self.beta_ent.unsqueeze(1) * torch.sin(2 * np.pi * self.omega_ent.unsqueeze(1) * time_tensor)
        self.temp_ent_embeds_all_times = self.ent_embeds.unsqueeze(1) + trend + seasonality

    def get_rel_embeds_train_global(self, relations, time_tensor):
        static_rel_embeds = self.rel_embeds[relations]
        trend = self.alpha_rel[relations] * self.w_rel[relations] * time_tensor.unsqueeze(-1)
        seasonality = self.beta_rel[relations] * torch.sin(2 * np.pi * self.omega_rel[relations] * time_tensor.unsqueeze(-1))
        return static_rel_embeds + trend + seasonality

    def get_ent_embeds_train_global(self, entities, time_tensor, mode='pos'):

        static_ent_embeds = self.ent_embeds[entities]
        if mode == 'pos':
            trend = self.alpha_ent[entities] * self.w_ent[entities] * time_tensor.unsqueeze(-1)
            seasonality = self.beta_ent[entities] * torch.sin(
                2 * np.pi * self.omega_ent[entities] * time_tensor.unsqueeze(-1))

            return static_ent_embeds + trend + seasonality

        elif mode == 'neg':
            return self.temp_ent_embeds_all_times[entities][:, time_tensor].transpose(0, 1)

        else:
            time_tensor = time_tensor.unsqueeze(-1).unsqueeze(-1)
            trend = self.alpha_ent[entities] * self.w_ent[entities] * time_tensor
            seasonality = self.beta_ent[entities] * torch.sin(2 * np.pi * self.omega_ent[entities])
            return static_ent_embeds + trend + seasonality

    def get_rel_embeds_train_global_old(self, relations, time_tensor):
        static_rel_embeds = self.old_rel_embeds[relations]
        trend = self.old_alpha_rel[relations] * self.old_w_rel[relations] * time_tensor.unsqueeze(-1)
        seasonality = self.old_beta_rel[relations] * torch.sin(
            2 * np.pi * self.old_omega_rel[relations] * time_tensor.unsqueeze(-1))
        return static_rel_embeds + trend + seasonality

    def get_ent_embeds_train_global_old(self, entities, time_tensor, mode='pos'):
        static_ent_embeds = self.old_ent_embeds[entities]
        if mode == 'pos':
            trend = self.old_alpha_ent[entities] * self.old_w_ent[entities] * time_tensor.unsqueeze(-1)
            seasonality = self.old_beta_ent[entities] * torch.sin(
                2 * np.pi * self.old_omega_ent[entities] * time_tensor.unsqueeze(-1))
        else:
            time_tensor = time_tensor.unsqueeze(-1).unsqueeze(-1)
            trend = self.old_alpha_ent[entities] * self.old_w_ent[entities] * time_tensor
            seasonality = self.old_beta_ent[entities] * torch.sin(2 * np.pi * self.old_omega_ent[entities])
        return static_ent_embeds + trend + seasonality

    def evaluate(self, quadruples, batch_idx):
        if type(quadruples) == dict:
            quadruples = quadruples['train']

        if quadruples.shape[0] == 0:
            return cuda(torch.tensor([]).long(), self.n_gpu) if self.use_cuda else torch.tensor([]).long(), 0
        known_entities = self.all_known_entities[
            self.args.end_time_step] if self.args.train_base_model else self.known_entities

        subjects, relations, objects, time_tensor = quadruples[:, 0], \
                                                    quadruples[:, 1], quadruples[:, 2], quadruples[:, 3]
        relation_mean = self.get_rel_embeds_train_global(relations, time_tensor)
        rel_cov = self.sigma_rel[relations]

        if batch_idx == 0:  # first time evaluating the at some epoch
            self.precompute_entity_time_embed()

        global2known = dict({n: i for i, n in enumerate(known_entities)})
        return self.evaluater.calc_metrics_quadruples(quadruples, relation_mean, rel_cov, known_entities, global2known)

    def learn_training_edges(self, train_quadruples, neg_subject_samples, neg_object_samples, labels):
        subjects, relations, objects, time_tensor = train_quadruples[:, 0], \
                    train_quadruples[:, 1], train_quadruples[:, 2], train_quadruples[:, 3]
        relation_mean = self.get_rel_embeds_train_global(relations, time_tensor)
        rel_cov = self.sigma_rel[relations]

        subject_mean = self.get_ent_embeds_train_global(subjects, time_tensor)
        object_mean = self.get_ent_embeds_train_global(objects, time_tensor)
        neg_subject_mean = self.get_ent_embeds_train_global(neg_subject_samples, time_tensor, mode='double-neg')
        neg_object_mean = self.get_ent_embeds_train_global(neg_object_samples, time_tensor, mode='double-neg')

        subject_cov = self.sigma_ent[subjects]
        object_cov = self.sigma_ent[objects]
        neg_subject_cov = self.sigma_ent[neg_subject_samples]
        neg_object_cov = self.sigma_ent[neg_object_samples]

        loss_tail = self.train_link_prediction(subject_mean, subject_cov, neg_object_mean, neg_object_cov,
                                               relation_mean, rel_cov, labels, corrupt_tail=True)
        loss_head = self.train_link_prediction(neg_subject_mean, neg_subject_cov, object_mean, object_cov,
                                               relation_mean, rel_cov, labels, corrupt_tail=False)
        self.weight_normalization()

        return loss_tail + loss_head
        # return torch.tensor(np.nan, requires_grad=True)

    def train_link_prediction(self, subject_mean, subject_cov, object_mean, object_cov, rel_mean, rel_cov, labels, corrupt_tail=True):
        # neg samples are in global idx
        score = self.calc_score(subject_mean, subject_cov, object_mean, object_cov, rel_mean, rel_cov, mode='tail' if corrupt_tail else 'head')
        return F.cross_entropy(score, labels.long())

    def unlearn_deleted_edges(self, deleted_quadruples):
        subjects, relations, objects, time_tensor = deleted_quadruples[:, 0], \
                        deleted_quadruples[:, 1], deleted_quadruples[:, 2], deleted_quadruples[:, 3]

        subject_mean = self.get_ent_embeds_train_global(subjects, time_tensor)
        object_mean = self.get_ent_embeds_train_global(objects, time_tensor)
        subject_cov = self.sigma_ent[subjects]
        object_cov = self.sigma_ent[objects]

        rel_mean = self.get_rel_embeds_train_global(relations, time_tensor)
        rel_cov = self.sigma_rel[relations]
        # subject_embedding, object_embedding = all_embeds_g[deleted_triples[:, 0]], all_embeds_g[deleted_triples[:, 2]]
        score = self.calc_score(subject_mean, subject_cov, object_mean, object_cov, rel_mean, rel_cov)

        labels = torch.zeros(len(deleted_quadruples))
        labels = cuda(labels, self.n_gpu) if self.use_cuda else labels
        return self.args.up_weight_factor * F.binary_cross_entropy_with_logits(score, labels)

    def calc_quad_kd_loss(self, reservoir_samples):

        loss = 0
        subjects, relations, objects, pos_time_tensor = reservoir_samples[:, 0], \
                        reservoir_samples[:, 1], reservoir_samples[:, 2], reservoir_samples[:, 3]
        cur_pos_relation_mean, cur_pos_subject_mean, cur_pos_object_mean = \
            self.get_cur_embedding_positive(reservoir_samples, pos_time_tensor)

        cur_pos_relation_cov, cur_pos_subject_cov, cur_pos_object_cov = \
            self.sigma_ent[relations], self.sigma_ent[subjects], self.sigma_ent[objects]

        if self.sample_neg_entity:
            # pdb.set_trace()
            cur_neg_subject_mean = self.get_ent_embeds_train_global(
                self.neg_reservoir_subject_samples, pos_time_tensor, mode='double-neg')
            cur_neg_object_mean = self.get_ent_embeds_train_global(
                self.neg_reservoir_object_samples, pos_time_tensor, mode='double-neg')
            cur_neg_subject_cov = self.sigma_ent[self.neg_reservoir_subject_samples]
            cur_neg_object_cov = self.sigma_ent[self.neg_reservoir_object_samples]

        if self.args.KD_reservoir:

            last_pos_relation_mean, last_pos_subject_mean, last_pos_object_mean = \
                self.get_old_embedding_positive(reservoir_samples, pos_time_tensor)

            last_pos_relation_cov, last_pos_subject_cov, last_pos_object_cov = \
                self.old_sigma_ent[relations], self.old_sigma_ent[subjects], self.old_sigma_ent[objects]

            if self.sample_positive:
                loss += self.pos_kd(reservoir_samples, last_pos_subject_mean, last_pos_relation_mean, last_pos_object_mean,
                                                        cur_pos_subject_mean, cur_pos_relation_mean, cur_pos_object_mean)

            if self.sample_neg_entity:
                last_neg_subject_mean = self.get_ent_embeds_train_global_old(self.neg_reservoir_subject_samples,
                                                                                   pos_time_tensor, mode='double-neg')
                last_neg_object_mean = self.get_ent_embeds_train_global_old(self.neg_reservoir_object_samples,
                                                                                  pos_time_tensor, mode='double-neg')
                last_neg_subject_cov = self.sigma_ent[self.neg_reservoir_subject_samples]
                last_neg_object_cov = self.sigma_ent[self.neg_reservoir_object_samples]

                last_neg_sub_triple_score = self.calc_score(last_neg_subject_mean, last_neg_subject_cov, last_pos_object_mean,
                                                            last_pos_object_cov, last_pos_relation_mean, last_pos_relation_cov, mode='head')
                cur_neg_sub_triple_score = self.calc_score(cur_neg_subject_mean, cur_neg_subject_cov, cur_pos_object_mean,
                                                            cur_pos_object_cov, cur_pos_relation_mean, cur_pos_relation_cov, mode='head')

                last_neg_obj_triple_score = self.calc_score(last_pos_subject_mean, last_pos_subject_cov, last_neg_object_mean,
                                                            last_neg_object_cov, last_pos_relation_mean, last_pos_relation_cov, mode='tail')
                cur_neg_obj_triple_score = self.calc_score(cur_pos_subject_mean, cur_pos_subject_cov, cur_neg_object_mean,
                                                           cur_neg_object_cov, cur_pos_relation_mean, cur_pos_relation_cov, mode='tail')

                loss += self.loss_fn_kd(cur_neg_sub_triple_score, last_neg_sub_triple_score) \
                       + self.loss_fn_kd(cur_neg_obj_triple_score, last_neg_obj_triple_score)
            if self.sample_neg_relation:
                raise NotImplementedError

        if self.args.CE_reservoir:

            if self.sample_positive:
                score = self.calc_score(cur_pos_subject_mean, cur_pos_subject_cov, cur_pos_object_mean,
                                        cur_pos_object_cov, cur_pos_relation_mean, cur_pos_relation_cov)
                labels = torch.ones(len(reservoir_samples))
                labels = cuda(labels, self.n_gpu) if self.use_cuda else labels
                loss += F.binary_cross_entropy_with_logits(score, labels)

            if self.sample_neg_entity:
                loss_tail = self.train_link_prediction(cur_pos_subject_mean, cur_pos_subject_cov, cur_neg_object_mean,
                                                       cur_neg_object_cov, cur_pos_relation_mean, cur_pos_relation_cov,
                                                       self.reservoir_labels, corrupt_tail=True)
                loss_head = self.train_link_prediction(cur_neg_subject_mean, cur_neg_subject_cov, cur_pos_object_mean,
                                                       cur_pos_object_cov, cur_pos_relation_mean, cur_pos_relation_cov,
                                                       self.reservoir_labels, corrupt_tail=False)
                loss += loss_tail + loss_head

            if self.sample_neg_relation:
                raise NotImplementedError
        return loss

    def pos_kd(self, reservoir_samples, last_subject_mean, last_relation_mean, last_object_mean, cur_subject_mean, cur_relation_mean, cur_object_mean):

        subjects, relations, objects, time_tensor = reservoir_samples[:, 0], \
                        reservoir_samples[:, 1], reservoir_samples[:, 2], reservoir_samples[:, 3]
        last_subject_cov, last_object_cov, last_rel_cov = self.old_sigma_ent[subjects], self.old_sigma_ent[objects], self.old_sigma_rel[relations]
        cur_subject_cov, cur_object_cov, cur_rel_cov = self.sigma_ent[subjects], self.sigma_ent[objects], self.sigma_rel[relations]
        last_triple_scores = self.calc_score(last_subject_mean, last_subject_cov, last_object_mean, last_object_cov, last_relation_mean, last_rel_cov)
        current_triple_score = self.calc_score(cur_subject_mean, cur_subject_cov, cur_object_mean, cur_object_cov, cur_relation_mean, cur_rel_cov)
        return F.kl_div(F.logsigmoid(current_triple_score), torch.sigmoid(last_triple_scores), reduction='mean')