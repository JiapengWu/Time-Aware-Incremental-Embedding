from models.TKG_Module import TKG_Module
import time
import torch
from utils.utils import cuda
import pdb

class TKG_Embedding(TKG_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(TKG_Embedding, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

        self.multi_step = args.multi_step

    def evaluate(self, quadruples, batch_idx):
        triples = quadruples[:, :-1]
        if triples.shape[0] == 0:
            return cuda(torch.tensor([]).long(), self.args.n_gpu) if self.use_cuda else torch.tensor([]).long(), 0

        if batch_idx == 0: # first time evaluating the at some epoch
            self.eval_ent_embed = self.get_graph_ent_embeds()
            self.eval_all_embeds_g = self.get_all_embeds_Gt()
            self.reduced_entity_embedding = self.eval_all_embeds_g[self.known_entities]
        id_dict = self.train_graph.ids
        rank = self.evaluater.calc_metrics_single_graph(self.eval_ent_embed,
                        self.rel_embeds, self.eval_all_embeds_g, triples, id_dict, self.time)
        return rank

    def forward(self, quadruples):
        neg_tail_samples, neg_head_samples, labels = self.corrupter.negative_sampling(quadruples.cpu())
        forward_func = self.forward_multi_step if self.multi_step else self.forward_incremental
        return forward_func(quadruples, neg_tail_samples, neg_head_samples, labels)

    def get_ent_embeds_multi_step(self):
        max_num_entities = max([len(self.graph_dict_train[t].nodes()) for t in range(max(0, self.time - self.train_seq_len), self.time + 1)])
        num_time_steps = self.time - max(0, self.time - self.train_seq_len) + 1
        ent_embed_all_time = self.ent_embeds.new_zeros(num_time_steps, max_num_entities, self.embed_size)
        all_embeds_g_all_time = self.ent_embeds.new_zeros(num_time_steps, self.num_ents, self.embed_size)
        # pdb.set_trace()
        for t in range(max(0, self.time - self.train_seq_len), self.time + 1):
            cur_local_ent_embedding = self.get_graph_ent_embeds(t)
            ent_embed_all_time[self.time - t][:len(cur_local_ent_embedding)] = cur_local_ent_embedding
            all_embeds_g_all_time[self.time - t] = self.get_all_embeds_Gt(t)

        return ent_embed_all_time, all_embeds_g_all_time

    def forward_multi_step(self, quadruples, neg_tail_samples, neg_head_samples, labels):
        ent_embed_all_time, all_embeds_g_all_time = self.get_ent_embeds_multi_step()
        loss_head = self.train_link_prediction_multi_step(ent_embed_all_time, all_embeds_g_all_time, quadruples,
                                                          neg_head_samples, labels, corrupt_tail=False)
        loss_tail = self.train_link_prediction_multi_step(ent_embed_all_time, all_embeds_g_all_time, quadruples,
                                                          neg_tail_samples, labels, corrupt_tail=True)
        return loss_tail + loss_head

    def forward_incremental(self, quadruples, neg_tail_samples, neg_head_samples, labels):
        ent_embed = self.get_graph_ent_embeds()
        all_embeds_g = self.get_all_embeds_Gt()
        loss_tail = self.train_link_prediction(ent_embed, quadruples, neg_tail_samples, labels, all_embeds_g, corrupt_tail=True)
        loss_head = self.train_link_prediction(ent_embed, quadruples, neg_head_samples, labels, all_embeds_g, corrupt_tail=False)
        return loss_tail + loss_head
