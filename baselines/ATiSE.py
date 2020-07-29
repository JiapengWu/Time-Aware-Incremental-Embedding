from torch import nn
from utils.scores import *
from utils.utils import cuda
from baselines.TKG_Embedding import TKG_Embedding
import math
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

class ATiSE(TKG_Embedding):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(ATiSE, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    def build_model(self):

        self.w_ent = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        nn.init.xavier_uniform_(self.w_ent, gain=nn.init.calculate_gain('relu'))

        self.alpha_ent = nn.Parameter(torch.Tensor(self.num_ents, 1))
        nn.init.xavier_uniform_(self.alpha_ent, gain=nn.init.calculate_gain('relu'))

        self.beta_ent = nn.Parameter(torch.Tensor(self.num_ents, 1))
        nn.init.xavier_uniform_(self.beta_ent, gain=nn.init.calculate_gain('relu'))

        self.sigma_ent = torch.randn(self.num_ents, self.embed_size, requires_grad=True)
        nn.init.xavier_uniform_(self.sigma_ent, gain=nn.init.calculate_gain('relu'))
        
        self.gaussian_noise = torch.tensor(torch.zeros(self.num_ents, self.embed_size), requires_grad=True)
        if self.args.use_cuda:
            self.gaussian_noise = cuda(self.gaussian_noise, self.args.n_gpu)

    def get_all_embeds_Gt(self, time=None):
        if time is None: time = self.time

        static_ent_embeds = self.ent_embeds
        trend = self.alpha_ent * self.w_ent * time
        seasonality = self.beta_ent * torch.sin(2 * np.pi * self.w_ent * time)
        randomness = self.gaussian_noise

        return static_ent_embeds + trend + seasonality #+ randomness

    def get_graph_ent_embeds(self, time=None):
        if time is None: time = self.time

        node_idx = self.graph_dict_train[time].ndata['id']
        static_ent_embeds = self.ent_embeds[node_idx].view(-1, self.embed_size)
        trend = self.alpha_ent[node_idx].view(-1,1) * self.w_ent[node_idx].view(-1, self.embed_size) * time
        seasonality = self.beta_ent[node_idx].view(-1,1) * torch.sin(2 * np.pi * self.w_ent[node_idx].view(-1, self.embed_size) * time)
        randomness = self.gaussian_noise[node_idx].view(-1, self.embed_size)

        return static_ent_embeds + trend + seasonality #+ randomness
    
    def sample_gaussian_noise(self):
        # ensure the covariance is semi-positive
        self.sigma_ent = torch.max(torch.ones(self.sigma_ent.shape)*0.005, torch.min(torch.ones(self.sigma_ent.shape)*0.5, self.sigma_ent))
        for i in range(self.num_ents):
            self.gaussian_noise[i] = MultivariateNormal(torch.zeros(self.embed_size), torch.diag(self.sigma_ent[i])).sample()

    def weight_normalization(self):
        self.ent_embeds = nn.Parameter(self.ent_embeds / torch.norm(self.ent_embeds, dim=1).unsqueeze(1))
        self.rel_embeds = nn.Parameter(self.rel_embeds / torch.norm(self.rel_embeds, dim=1).unsqueeze(1))
        self.w_ent = nn.Parameter(self.w_ent / torch.norm(self.w_ent, dim=1).unsqueeze(1))
    
    def forward_incremental(self, quadruples, neg_tail_samples, neg_head_samples, labels):
        self.sample_gaussian_noise()
        self.weight_normalization()
        ent_embed = self.get_graph_ent_embeds()
        all_embeds_g = self.get_all_embeds_Gt()
        loss_tail = self.train_link_prediction(ent_embed, quadruples, neg_tail_samples, labels, all_embeds_g, corrupt_tail=True, loss='margin')
        loss_head = self.train_link_prediction(ent_embed, quadruples, neg_head_samples, labels, all_embeds_g, corrupt_tail=False, loss='margin')
        return loss_tail + loss_head
