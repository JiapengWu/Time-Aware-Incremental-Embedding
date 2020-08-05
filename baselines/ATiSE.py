from torch import nn
from utils.scores import *
from utils.utils import cuda
from baselines.TKG_Embedding import TKG_Embedding
import math
import numpy as np
import torch.nn.functional as F

class ATiSE(TKG_Embedding):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(ATiSE, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
        self.weight_normalization()

    def build_model(self):

        self.w_ent = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.w_rel = nn.Parameter(torch.Tensor(self.num_rels*2, self.embed_size))
        nn.init.xavier_uniform_(self.w_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_rel, gain=nn.init.calculate_gain('relu'))

        self.alpha_ent = nn.Parameter(torch.Tensor(self.num_ents, 1))
        self.alpha_rel = nn.Parameter(torch.Tensor(self.num_rels*2, 1))
        nn.init.xavier_uniform_(self.alpha_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.alpha_rel, gain=nn.init.calculate_gain('relu'))

        self.beta_ent = nn.Parameter(torch.Tensor(self.num_ents, 1))
        self.beta_rel = nn.Parameter(torch.Tensor(self.num_rels*2, 1))
        nn.init.xavier_uniform_(self.beta_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.beta_rel, gain=nn.init.calculate_gain('relu'))

        self.sigma_ent = nn.Parameter(torch.ones(self.num_ents, self.embed_size).uniform_(0.005,0.5))
        self.sigma_rel = nn.Parameter(torch.ones(self.num_rels*2, self.embed_size).uniform_(0.005,0.5))
        nn.init.xavier_uniform_(self.sigma_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.sigma_rel, gain=nn.init.calculate_gain('relu'))

    def get_all_embeds_Gt(self, time=None):
        if time is None: time = self.time
        static_ent_embeds = self.ent_embeds
        trend = self.alpha_ent * self.w_ent * time
        seasonality = self.beta_ent * torch.sin(2 * np.pi * self.w_ent * time)
        return static_ent_embeds + trend + seasonality 

    def get_graph_ent_embeds(self, time=None):
        if time is None: time = self.time
        node_idx = self.graph_dict_train[time].ndata['id']
        static_ent_embeds = self.ent_embeds[node_idx].view(-1, self.embed_size)
        trend = self.alpha_ent[node_idx].view(-1,1) * self.w_ent[node_idx].view(-1, self.embed_size) * time
        seasonality = self.beta_ent[node_idx].view(-1,1) * torch.sin(2 * np.pi * self.w_ent[node_idx].view(-1, self.embed_size) * time)
        return static_ent_embeds + trend + seasonality 

    def get_graph_ent_covs(self, time=None):
        if time is None: time = self.time
        node_idx = self.graph_dict_train[time].ndata['id']
        static_ent_covs = self.sigma_ent[node_idx].view(-1, self.embed_size)
        return static_ent_covs

    def get_all_rel_embeds_Gt(self, time=None):
        if time is None: time = self.time
        static_rel_embeds = self.rel_embeds
        trend = self.alpha_rel * self.w_rel * time
        seasonality = self.beta_rel * torch.sin(2 * np.pi * self.w_rel * time)
        return static_rel_embeds + trend + seasonality 

    def weight_normalization(self):
        self.ent_embeds.data.copy_(self.ent_embeds / torch.norm(self.ent_embeds, dim=1).unsqueeze(1))
        self.rel_embeds.data.copy_(self.rel_embeds / torch.norm(self.rel_embeds, dim=1).unsqueeze(1))
        self.w_ent.data.copy_(self.w_ent / torch.norm(self.w_ent, dim=1).unsqueeze(1))        
        self.w_rel.data.copy_(self.w_rel / torch.norm(self.w_rel, dim=1).unsqueeze(1))    

        self.sigma_ent.data.copy_(torch.clamp(input=self.sigma_ent.data, min=0.005, max=0.5))
        self.sigma_rel.data.copy_(torch.clamp(input=self.sigma_rel.data, min=0.005, max=0.5))    

    def forward_incremental(self, quadruples, neg_tail_samples, neg_head_samples, labels):
        ent_embed = self.get_graph_ent_embeds()
        all_embeds_g = self.get_all_embeds_Gt()
        loss_tail = self.train_link_prediction(ent_embed, quadruples, neg_tail_samples, labels, all_embeds_g, corrupt_tail=True, loss='CE')
        loss_head = self.train_link_prediction(ent_embed, quadruples, neg_head_samples, labels, all_embeds_g, corrupt_tail=False, loss='CE')
        self.weight_normalization()
        return loss_tail + loss_head

    def train_link_prediction(self, ent_embed, quadruples, neg_samples, labels, all_embeds_g, corrupt_tail=True, loss='CE'):
        # neg samples are in global idx
        head_cov = self.get_graph_ent_covs()[quadruples[:, 0]] if corrupt_tail else self.sigma_ent[neg_samples]
        tail_cov = self.sigma_ent[neg_samples] if corrupt_tail else self.get_graph_ent_covs()[quadruples[:, 2]]
        head_mean = ent_embed[quadruples[:, 0]] if corrupt_tail else all_embeds_g[neg_samples]
        tail_mean = all_embeds_g[neg_samples] if corrupt_tail else ent_embed[quadruples[:, 2]]

        rel_mean = self.get_all_rel_embeds_Gt()[quadruples[:, 1]]
        rel_cov = self.sigma_rel[quadruples[:, 1]]

        score = self.calc_score(head_mean, head_cov, tail_mean, tail_cov, rel_mean, rel_cov, mode='tail' if corrupt_tail else 'head')
        if loss == 'CE':
            return F.cross_entropy(score, labels.long())
        elif loss == 'margin':
            pos_score = score[:, 0].unsqueeze(-1).repeat(1, self.negative_rate)
            neg_score = score[:, 1:]
            return  torch.sum(- F.logsigmoid(1 - pos_score) - F.logsigmoid(neg_score - 1))
        else:
            raise NotImplementedError

    def evaluate(self, quadruples, batch_idx):
        triples = quadruples[:, :-1]
        if triples.shape[0] == 0:
            return cuda(torch.tensor([]).long(), self.args.n_gpu) if self.use_cuda else torch.tensor([]).long(), 0

        if batch_idx == 0: # first time evaluating the at some epoch
            self.eval_ent_embed = self.get_graph_ent_embeds()
            self.eval_all_embeds_g = self.get_all_embeds_Gt()
            self.reduced_entity_embedding = self.eval_all_embeds_g[self.known_entities]

            self.eval_ent_cov_embed = self.get_graph_ent_covs()
            self.reduced_entity_cov_embedding = self.sigma_ent[self.known_entities]

            self.eval_rel_embed = self.get_all_rel_embeds_Gt()

        local2global = self.train_graph.ids	
        global2known = dict({n: i for i, n in enumerate(self.known_entities)})	

        rank = self.evaluater.calc_metrics_single_graph(self.eval_ent_embed,
                        self.eval_rel_embed, self.eval_all_embeds_g, triples, local2global, self.time, global2known)
        return rank
