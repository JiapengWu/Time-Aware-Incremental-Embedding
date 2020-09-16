from models.RGCN import RGCN
import numpy as np
from utils.utils import move_dgl_to_cuda
from utils.scores import *
from archive.TKG_Module import TKG_Module
from utils.utils import build_sampled_graph_from_triples


class StaticRGCN(TKG_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(StaticRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
        self.last_time_step = -1

    def build_model(self):
        self.ent_encoder = RGCN(self.args, self.hidden_size, self.embed_size, self.num_rels, self.total_time)

    def evaluate(self, triples, batch_idx):
        # import pdb; pdb.set_trace()
        if batch_idx == 0: # first time evaluating the at some epoch
            # import pdb; pdb.set_trace()
            self.eval_ent_embed = self.get_graph_ent_embeds(triples, val=True)
            self.eval_all_embeds_g = self.get_all_embeds_Gt(self.eval_ent_embed)

        if triples.shape[0] == 0:
            return self.cuda(torch.tensor([]).long(), self.n_gpu) if self.use_cuda else torch.tensor([]).long(), 0

        id_dict = self.train_graph.ids
        rank = self.evaluater.calc_metrics_single_graph(self.eval_ent_embed, self.rel_embeds, self.eval_all_embeds_g, triples, id_dict, self.time)
        return rank

    def forward(self, quadruples):
        # import pdb; pdb.set_trace()
        ent_embed = self.get_graph_ent_embeds(quadruples[:, :-1])
        all_embeds_g = self.get_all_embeds_Gt(ent_embed)
        neg_tail_samples, neg_head_samples, labels = \
            self.corrupter.negative_sampling(quadruples.cpu(), self.train_graph, self.num_ents)
        loss_tail = self.train_link_prediction(ent_embed, quadruples, neg_tail_samples, labels, all_embeds_g, corrupt_tail=True)
        loss_head = self.train_link_prediction(ent_embed, quadruples, neg_head_samples, labels, all_embeds_g, corrupt_tail=False)
        return loss_tail + loss_head

    def get_all_embeds_Gt(self, convoluted_embeds):
        all_embeds_g = self.ent_encoder.forward_isolated(self.ent_embeds, self.time)
        keys = np.array(list(self.train_graph.ids.keys()))
        values = np.array(list(self.train_graph.ids.values()))
        all_embeds_g[values] = convoluted_embeds[keys]
        return all_embeds_g

    def get_graph_ent_embeds(self, triples, val=False):
        g = self.train_graph if val else build_sampled_graph_from_triples(triples, self.train_graph)
        # g = self.train_graph
        g.ndata['h'] = self.ent_embeds[g.ndata['id']].view(-1, self.embed_size)
        if self.use_cuda:
            move_dgl_to_cuda(g, self.n_gpu)
        enc_ent_mean_graph = self.ent_encoder(g, self.time)
        ent_enc_embeds = enc_ent_mean_graph.ndata['h']
        return ent_enc_embeds
