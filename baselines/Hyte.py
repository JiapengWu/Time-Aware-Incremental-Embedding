from torch import nn
from utils.scores import *
from models.TKG_Embedding_Global import TKG_Embedding_Global
import pdb
from utils.util_functions import cuda, mse_loss


class Hyte(TKG_Embedding_Global):
    def __init__(self, args, num_ents, num_rels):
        args.score_function = 'transE'
        super(Hyte, self).__init__(args, num_ents, num_rels)
        if self.reservoir_sampling or self.self_kd:
            self.old_time_embeddings = nn.Parameter(torch.Tensor(len(self.total_time), self.embed_size))
        # self.calc_score = transE

    def build_model(self):
        self.time_embeddings = nn.Parameter(torch.Tensor(len(self.total_time), self.embed_size))

    def get_per_graph_embeds(self, t, g):
            # n_ent, embed_size
        static_ent_embeds = self.ent_embeds[g.ndata['id']].view(-1, self.embed_size)
        # 1, embed_size
        time_embeddings = self.time_embeddings[t].unsqueeze(0)
        normalized_embedding = time_embeddings / torch.norm(time_embeddings, p=2, dim=-1)
        # n_ent, embed_size - (1, embed_size * n_ent, 1)
        projected_ent_embed = static_ent_embeds - normalized_embedding * (torch.sum(static_ent_embeds * normalized_embedding, dim=-1)).unsqueeze(1)
        projected_rel_embed = self.rel_embeds - normalized_embedding * (torch.sum(self.rel_embeds * normalized_embedding, dim=-1)).unsqueeze(1)
        return projected_ent_embed, projected_rel_embed

    def load_old_parameters(self):
        super(Hyte, self).load_old_parameters()
        self.old_time_embeddings.data = self.time_embeddings.data.clone()

    def init_parameters(self):
        super(Hyte, self).init_parameters()
        nn.init.xavier_uniform_(self.time_embeddings, gain=nn.init.calculate_gain('relu'))

    def weight_normalization(self):
        self.time_embeddings.data.copy_(self.time_embeddings / torch.norm(self.time_embeddings, p=2, dim=1).unsqueeze(1))

    def precompute_entity_time_embed(self):
        static_ent_embeds = self.ent_embeds.unsqueeze(1)
        # self.weight_normalization()
        time_tensor = torch.tensor(list(range(len(self.total_time))))
        time_embeddings = self.time_embeddings[time_tensor].unsqueeze(0)
        # n_ent, embed_size - (1, embed_size * n_ent, 1)
        self.temp_ent_embeds_all_times = static_ent_embeds - time_embeddings * (torch.sum(static_ent_embeds * time_embeddings, dim=-1)).unsqueeze(-1)

    def get_old_rel_embeds(self, relations, time_tensor):
        static_rel_embedding = self.old_rel_embeds[relations]
        time_embeddings = self.time_embeddings[time_tensor]
        return static_rel_embedding - time_embeddings * (torch.sum(
                static_rel_embedding * time_embeddings, dim=-1)).unsqueeze(1)

    def get_rel_embeds(self, relations, time_tensor):
        static_rel_embedding = self.rel_embeds[relations]
        time_embeddings = self.time_embeddings[time_tensor]
        return static_rel_embedding - time_embeddings * (torch.sum(
                static_rel_embedding * time_embeddings, dim=-1)).unsqueeze(1)

    def get_ent_embeds_train_global(self, entities, time_tensor, mode='pos'):
        # ones = static_ent_embeds.new_ones(entities.shape[0], self.static_embed_size)

        static_ent_embeds = self.ent_embeds[entities]
        self.weight_normalization()
        if mode == 'pos':

            time_embeddings = self.time_embeddings[time_tensor]
            return static_ent_embeds - time_embeddings * (torch.sum(static_ent_embeds * time_embeddings, dim=-1)).unsqueeze(1)

        elif mode == 'neg':
            return self.temp_ent_embeds_all_times[entities][:, time_tensor].transpose(0, 1)
        else:
            time_embeddings = self.time_embeddings[time_tensor].unsqueeze(1)
            # pdb.set_trace()
            return static_ent_embeds - time_embeddings * (torch.sum(static_ent_embeds * time_embeddings, dim=-1)).unsqueeze(-1)

    def get_ent_embeds_train_global_old(self, entities, time_tensor, mode='pos'):
        static_ent_embeds = self.old_ent_embeds[entities]
        if mode == 'pos':
            time_embeddings = self.old_time_embeddings[time_tensor]
            return static_ent_embeds - time_embeddings * (torch.sum(static_ent_embeds * time_embeddings, dim=-1)).unsqueeze(1)
        else:
            time_embeddings = self.old_time_embeddings[time_tensor].unsqueeze(1)
            return static_ent_embeds - time_embeddings * (torch.sum(static_ent_embeds * time_embeddings, dim=-1)).unsqueeze(-1)

    def calc_self_kd_loss(self):
        first_loss = super().calc_self_kd_loss()

        time_embedding_kd_loss = mse_loss(self.old_time_embeddings,
                                  self.time_embeddings)
        return first_loss + time_embedding_kd_loss

    '''
    def train_link_prediction(self, ent_embed, rel_embed, triplets, neg_samples, labels, corrupt_tail=True):
        r = rel_embed[triplets[:, 1]]
        if corrupt_tail:
            s = ent_embed[triplets[:, 0]]
            neg_o = ent_embed[neg_samples]
            score = self.calc_score(s, r, neg_o, mode='tail')
        else:
            neg_s = ent_embed[neg_samples]
            o = ent_embed[triplets[:, 2]]
            score = self.calc_score(neg_s, r, o, mode='head')

        predict_loss = F.cross_entropy(score, labels)
        return predict_loss

    def calc_metrics(self, per_graph_ent_embeds, per_graph_rel_embeds, t_list, triplets, labels):
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        i = 0
        for ent_embed, rel_embed in zip(per_graph_ent_embeds, per_graph_rel_embeds):
            if triplets[i].shape[0] == 0: continue
            mrr, hit_1, hit_3, hit_10 = self.evaluater.calc_metrics_single_graph(ent_embed, rel_embed, triplets[i], t_list[i])
            loss = self.link_classification_loss(ent_embed, rel_embed, triplets[i], labels[i])
            mrrs.append(mrr)
            hit_1s.append(hit_1)
            hit_3s.append(hit_3)
            hit_10s.append(hit_10)
            losses.append(loss.item())
            i += 1
        return np.mean(mrrs), np.mean(hit_1s), np.mean(hit_3s), np.mean(hit_10s), np.sum(losses)

    def evaluate(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        g_list = [graph_dict[i.item()] for i in t_list]
        per_graph_ent_embeds, per_graph_rel_embeds = self.get_per_graph_embeds(t_list, g_list)
        triplets, labels = self.corrupter.sample_labels_val(g_list)
        return self.calc_metrics(per_graph_ent_embeds, per_graph_rel_embeds, t_list, triplets, labels)

    def forward(self, t_list, reverse=False):
        reconstruct_loss = 0
        g_list = [self.graph_dict_train[i.item()] for i in t_list]

        for t, g in zip(t_list, g_list):
            ent_embed, rel_embed = self.get_per_graph_ent_embeds(t, g)
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.negative_sampling(t)
            loss_tail = self.train_link_prediction(ent_embed, rel_embed, triplets, neg_tail_samples, labels, corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed, rel_embed, triplets, neg_head_samples, labels, corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head

        return reconstruct_loss
    '''