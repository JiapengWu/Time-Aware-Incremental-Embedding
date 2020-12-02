from torch import nn
from utils.scores import *
# from models.TKG_Embedding import TKG_Embedding
from models.TKG_Embedding_Global import TKG_Embedding_Global
import math
from utils.util_functions import cuda, mse_loss
import pdb


class DiachronicEmbedding(TKG_Embedding_Global):
    def __init__(self, args, num_ents, num_rels):
        super(DiachronicEmbedding, self).__init__(args, num_ents, num_rels)
        if self.reservoir_sampling or self.self_kd:
            self.old_w_temp_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.temporal_embed_size), requires_grad=False)
            self.old_b_temp_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.temporal_embed_size), requires_grad=False)

    def load_old_parameters(self):
        super(DiachronicEmbedding, self).load_old_parameters()
        self.old_w_temp_ent_embeds.data = self.w_temp_ent_embeds.data.clone()
        self.old_b_temp_ent_embeds.data = self.b_temp_ent_embeds.data.clone()

    def build_model(self):
        self.static_embed_size = math.floor(0.5 * self.embed_size)
        self.temporal_embed_size = self.embed_size - self.static_embed_size
        self.w_temp_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.temporal_embed_size))
        self.b_temp_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.temporal_embed_size))

    def init_parameters(self):
        super(DiachronicEmbedding, self).init_parameters()
        nn.init.xavier_uniform_(self.w_temp_ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.b_temp_ent_embeds, gain=nn.init.calculate_gain('relu'))

    '''
    def get_all_embeds_Gt(self, time=None):
        if time is None: time = self.time
        static_ent_embeds = self.ent_embeds
        temp_ent_embeds = torch.sin(time * self.w_temp_ent_embeds + self.b_temp_ent_embeds)
        return torch.cat([static_ent_embeds[:, :self.static_embed_size],
                              static_ent_embeds[:, self.static_embed_size:] * temp_ent_embeds], dim=-1)

    def get_graph_ent_embeds(self, time=None):
        if time is None: time = self.time
        node_idx = self.graph_dict_train[time].ndata['id']
        static_ent_embeds = self.ent_embeds[node_idx].view(-1, self.embed_size)
        ones = static_ent_embeds.new_ones(static_ent_embeds.shape[0], self.static_embed_size)
        temp_ent_embeds = torch.sin(time * self.w_temp_ent_embeds[node_idx].view(-1, self.temporal_embed_size) +
                                self.b_temp_ent_embeds[node_idx].view(-1, self.temporal_embed_size))
        return torch.cat([static_ent_embeds[:, :self.static_embed_size],
                              static_ent_embeds[:, self.static_embed_size:] * temp_ent_embeds], dim=-1)
    

    def get_all_embeds_Gt_old(self):
        static_ent_embeds = self.old_ent_embeds
        # ones = static_ent_embeds.new_ones(static_ent_embeds.shape[0], self.static_embed_size)
        temp_ent_embeds = torch.sin(self.time * self.old_w_temp_ent_embeds + self.old_b_temp_ent_embeds)
        return torch.cat([static_ent_embeds[:, :self.static_embed_size],
                              static_ent_embeds[:, self.static_embed_size:] * temp_ent_embeds], dim=-1)
    '''

    def get_old_rel_embeds(self, relations, time_tensor):
        return self.old_rel_embeds[relations]

    def get_rel_embeds(self, relations, time_tensor):
        return self.rel_embeds[relations]

    def precompute_entity_time_embed(self):
        time_tensor = torch.tensor(list(range(len(self.total_time)))).unsqueeze(0).unsqueeze(2)
        if self.use_cuda:
            time_tensor = cuda(time_tensor, self.n_gpu)
        self.temp_ent_embeds_all_times = torch.sin(time_tensor * self.w_temp_ent_embeds.unsqueeze(1)
                                                                + self.b_temp_ent_embeds.unsqueeze(1))

    def get_ent_embeds_train_global(self, entities, time_tensor, mode='pos'):
        # ones = static_ent_embeds.new_ones(entities.shape[0], self.static_embed_size)
        static_ent_embeds = self.ent_embeds[entities]
        if mode == 'pos':
            temp_ent_embeds = torch.sin(time_tensor.unsqueeze(-1) * self.w_temp_ent_embeds[entities] + self.b_temp_ent_embeds[entities])
            return torch.cat([static_ent_embeds[:, :self.static_embed_size],
                              static_ent_embeds[:, self.static_embed_size:] * temp_ent_embeds], dim=-1)
        elif mode == 'neg':
            static_ent_embeds = static_ent_embeds.unsqueeze(1)
            temp_ent_embeds = self.temp_ent_embeds_all_times[entities][:, time_tensor]
            return torch.cat([static_ent_embeds[:, :, :self.static_embed_size].expand(len(entities), len(time_tensor), self.static_embed_size),
                              static_ent_embeds[:, :, self.static_embed_size:] * temp_ent_embeds], dim=-1).transpose(0, 1)
        else:
            temp_ent_embeds = torch.sin(time_tensor.unsqueeze(-1).unsqueeze(-1) * self.w_temp_ent_embeds[entities] + self.b_temp_ent_embeds[entities])
            return torch.cat([static_ent_embeds[:, :, :self.static_embed_size],
                              static_ent_embeds[:, :, self.static_embed_size:] * temp_ent_embeds], dim=-1)

    def get_ent_embeds_train_global_old(self, entities, time_tensor, mode='pos'):
        static_ent_embeds = self.old_ent_embeds[entities]
        if mode == 'pos':
            temp_ent_embeds = torch.sin(time_tensor.unsqueeze(-1) * self.old_w_temp_ent_embeds[entities] + self.old_b_temp_ent_embeds[entities])
            return torch.cat([static_ent_embeds[:, :self.static_embed_size],
                              static_ent_embeds[:, self.static_embed_size:] * temp_ent_embeds], dim=-1)
        else:
            temp_ent_embeds = torch.sin(time_tensor.unsqueeze(-1).unsqueeze(-1) * self.old_w_temp_ent_embeds[entities] + self.old_b_temp_ent_embeds[entities])
            return torch.cat([static_ent_embeds[:, :, :self.static_embed_size],
                              static_ent_embeds[:, :, self.static_embed_size:] * temp_ent_embeds], dim=-1)

    def calc_self_kd_loss(self):
        first_loss = super().calc_self_kd_loss()
        w_kd_loss = mse_loss(self.w_temp_ent_embeds[self.last_known_entities],
                                  self.old_w_temp_ent_embeds[self.last_known_entities])
        b_kd_loss = mse_loss(self.b_temp_ent_embeds[self.last_known_entities],
                                  self.old_b_temp_ent_embeds[self.last_known_entities])
        return first_loss + w_kd_loss + b_kd_loss
