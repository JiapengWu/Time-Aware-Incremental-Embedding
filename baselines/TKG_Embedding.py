from models.TKG_Module import TKG_Module
import time
import torch
from utils.utils import cuda


class TKG_Embedding(TKG_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(TKG_Embedding, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    def evaluate(self, triples, batch_idx):
        # triples = triples[0]
        if triples.shape[0] == 0:
            return cuda(torch.tensor([]).long(), self.args.n_gpu) if self.use_cuda else torch.tensor([]).long(), 0

        if batch_idx == 0: # first time evaluating the at some epoch
            self.eval_ent_embed = self.get_graph_ent_embeds()
            self.eval_all_embeds_g = self.get_all_embeds_Gt()

        # label = torch.ones(triples.shape[0])
        # if self.use_cuda:
        #     label = cuda(label)

        id_dict = self.train_graph.ids
        rank = self.evaluater.calc_metrics_single_graph(self.eval_ent_embed,
                    self.rel_embeds, self.eval_all_embeds_g, triples, id_dict, self.time)
        # loss = self.link_classification_loss(ent_embed, self.rel_embeds, triples, label)
        return rank

    def forward(self, triples):
        ent_embed = self.get_graph_ent_embeds()
        all_embeds_g = self.get_all_embeds_Gt()
        neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(triples.cpu(), self.time, self.train_graph, self.num_ents)
        loss_tail = self.train_link_prediction(ent_embed, triples, neg_tail_samples, labels, all_embeds_g, corrupt_tail=True)
        loss_head = self.train_link_prediction(ent_embed, triples, neg_head_samples, labels, all_embeds_g, corrupt_tail=False)
        return loss_tail + loss_head



