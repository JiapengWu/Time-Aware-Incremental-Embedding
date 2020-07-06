from models.RGCN import RGCN
import dgl
import numpy as np
from utils.utils import comp_deg_norm, move_dgl_to_cuda
from utils.scores import *
from models.TKG_Module import TKG_Module
from utils.utils import cuda, node_norm_to_edge_norm


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
            return cuda(torch.tensor([]).long(), self.args.n_gpu) if self.use_cuda else torch.tensor([]).long(), 0

        id_dict = self.train_graph.ids
        rank = self.evaluater.calc_metrics_single_graph(self.eval_ent_embed, self.rel_embeds, self.eval_all_embeds_g, triples, id_dict, self.time)
        # loss = self.link_classification_loss(self.eval_ent_embed, self.rel_embeds, triples, label)
        return rank

    def forward(self, triples):
        ent_embed = self.get_graph_ent_embeds(triples)
        all_embeds_g = self.get_all_embeds_Gt(ent_embed)
        neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(triples.cpu(), self.time, self.train_graph, self.num_ents)
        loss_tail = self.train_link_prediction(ent_embed, triples, neg_tail_samples, labels, all_embeds_g, corrupt_tail=True)
        loss_head = self.train_link_prediction(ent_embed, triples, neg_head_samples, labels, all_embeds_g, corrupt_tail=False)
        return loss_tail + loss_head

    def get_all_embeds_Gt(self, convoluted_embeds):
        all_embeds_g = self.ent_encoder.forward_isolated(self.ent_embeds, self.time)
        keys = np.array(list(self.train_graph.ids.keys()))
        values = np.array(list(self.train_graph.ids.values()))
        all_embeds_g[values] = convoluted_embeds[keys]
        return all_embeds_g

    # '''
    def build_graph_from_triples(self, triples):
        sample_idx = np.random.choice(np.arange(len(triples)), size=int(0.5 * len(triples)), replace=False)
        src, rel, dst = triples[sample_idx].transpose(0, 1)
        g = dgl.DGLGraph()
        g.add_nodes(len(self.train_graph.nodes))
        g.add_edges(src, dst)
        node_norm = comp_deg_norm(g)
        g.ndata.update({'id': self.train_graph.ndata['id'], 'norm': torch.from_numpy(node_norm).view(-1, 1)})
        g.edata['norm'] = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        g.edata['type_s'] = rel
        g.ids = self.train_graph.ids
        return g

    '''
    def build_graph_from_triples(self, triples):
        g = self.train_graph
        src, rel, dst = g.edges()[0], g.edata['type_s'], g.edges()[1]
        total_idx = np.random.choice(np.arange(src.shape[0]), size=int(0.5 * src.shape[0]), replace=False)
        sg = g.edge_subgraph(total_idx, preserve_nodes=True)
        node_norm = comp_deg_norm(sg)
        sg.ndata.update({'id': g.ndata['id'], 'norm': torch.from_numpy(node_norm).view(-1, 1)})
        sg.edata['norm'] = node_norm_to_edge_norm(sg, torch.from_numpy(node_norm).view(-1, 1))
        sg.edata['type_s'] = rel[total_idx]
        sg.ids = g.ids
        return sg
    #'''

    def get_graph_ent_embeds(self, triples, val=False):
        g = self.train_graph if val else self.build_graph_from_triples(triples)
        # g = self.train_graph
        g.ndata['h'] = self.ent_embeds[g.ndata['id']].view(-1, self.embed_size)
        if self.use_cuda:
            move_dgl_to_cuda(g, self.args.n_gpu)
        enc_ent_mean_graph = self.ent_encoder(g, self.time)
        ent_enc_embeds = enc_ent_mean_graph.ndata['h']
        return ent_enc_embeds
