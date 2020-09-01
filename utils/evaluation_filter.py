import torch
from utils.utils import cuda, get_true_head_and_tail_per_graph, sort_and_rank
import numpy as np


class EvaluationFilter:
    def __init__(self, model):
        self.model = model
        self.args = model.args
        self.negative_sample_all_entities = self.args.negative_sample_all_entities
        self.calc_score = model.calc_score
        self.graph_dict_train = model.graph_dict_train
        self.graph_dict_val = model.graph_dict_val
        self.graph_dict_test = model.graph_dict_test
        self.graph_dict_total = {**self.graph_dict_train, **self.graph_dict_val, **self.graph_dict_test}
        self.get_true_head_and_tail_all()

    def get_true_head_and_tail_all(self):
        self.true_heads = dict()
        self.true_tails = dict()
        times = list(self.graph_dict_total.keys())
        for t in times:
            if self.args.dataset_dir == 'extrapolation':
                g = self.graph_dict_total[t]
                triples = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
            else:
                triples = []
                for g in self.graph_dict_train[t], self.graph_dict_val[t], self.graph_dict_test[t]:
                    triples.append(torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1))
                triples = torch.cat(triples, dim=0)
            true_head, true_tail = get_true_head_and_tail_per_graph(triples)
            self.true_heads[t] = true_head
            self.true_tails[t] = true_tail

    def calc_metrics_single_graph(self, ent_embeddings, rel_embeddings, all_ent_embeds, samples, local2global, time, global2known, eval_bz=100):
        with torch.no_grad():
            s = samples[:, 0]
            r = samples[:, 1]
            o = samples[:, 2]
            test_size = samples.shape[0]
            o_mask = self.mask_eval_set(samples, test_size, time, local2global, global2known, mode="tail")
            s_mask = self.mask_eval_set(samples, test_size, time, local2global, global2known, mode="head")

            # perturb object
            ranks_o = self.perturb_and_get_rank(ent_embeddings, rel_embeddings, s, r, o, test_size, o_mask, local2global, global2known, eval_bz, mode='tail')
            # perturb subject
            ranks_s = self.perturb_and_get_rank(ent_embeddings, rel_embeddings, s, r, o, test_size, s_mask, local2global, global2known, eval_bz, mode='head')
            ranks = torch.cat([ranks_s, ranks_o])
            ranks += 1 # change to 1-indexed
        return ranks

    def perturb_and_get_rank(self, ent_mean, rel_enc_means, s, r, o, test_size, mask, local2global, global2known, batch_size=100, mode ='tail'):
        """ Perturb one element in the triplets
        """
        # reduction_dict: from global id to reduced id
        n_batch = (test_size + batch_size - 1) // batch_size
        ranks = []
        for idx in range(n_batch):
            batch_start = idx * batch_size
            batch_end = min(test_size, (idx + 1) * batch_size)
            batch_r = rel_enc_means[r[batch_start: batch_end]]

            if self.calc_score.__name__ != 'ATiSE_score':
                if mode == 'tail':
                    batch_s = ent_mean[s[batch_start: batch_end]]
                    batch_o = self.model.eval_all_embeds_g if self.negative_sample_all_entities else self.model.reduced_entity_embedding
                    target = o[batch_start: batch_end]
                else:
                    batch_s = self.model.eval_all_embeds_g if self.negative_sample_all_entities else self.model.reduced_entity_embedding
                    batch_o = ent_mean[o[batch_start: batch_end]]
                    target = s[batch_start: batch_end]

                # target: local -> global -> known
                if self.negative_sample_all_entities:
                    target = torch.tensor([local2global[i.item()] for i in target])
                else:
                    target = torch.tensor([global2known[local2global[i.item()]] for i in target])

                if self.args.use_cuda:
                    target = cuda(target, self.args.n_gpu)
                unmasked_score = self.calc_score(batch_s, batch_r, batch_o, mode=mode)
            else:
                if mode == 'tail':
                    batch_s = ent_mean[s[batch_start: batch_end]]
                    batch_s_cov = self.model.eval_ent_cov_embed[s[batch_start: batch_end]]
                    batch_o =  self.model.eval_all_embeds_g if self.negative_sample_all_entities else self.model.reduced_entity_embedding
                    batch_o_cov = self.model.sigma_ent if self.negative_sample_all_entities else self.model.reduced_entity_cov_embedding
                    target = o[batch_start: batch_end]
                else:
                    batch_s = self.model.eval_all_embeds_g if self.negative_sample_all_entities else self.model.reduced_entity_embedding
                    batch_s_cov = self.model.sigma_ent if self.negative_sample_all_entities else self.model.reduced_entity_cov_embedding
                    batch_o = ent_mean[o[batch_start: batch_end]]
                    batch_o_cov = self.model.eval_ent_cov_embed[o[batch_start: batch_end]]
                    target = s[batch_start: batch_end]

                barch_r_cov = self.model.sigma_rel[r[batch_start: batch_end]]

                if self.negative_sample_all_entities:
                    target = torch.tensor([local2global[i.item()] for i in target])
                else:
                    target = torch.tensor([global2known[local2global[i.item()]] for i in target])

                if self.args.use_cuda:
                    target = cuda(target, self.args.n_gpu)
                unmasked_score = self.calc_score(batch_s, batch_s_cov, batch_o, batch_o_cov, batch_r, barch_r_cov, mode=mode)

            # mask: 1 for local id
            masked_score = torch.where(mask[batch_start: batch_end], -10e6 * unmasked_score.new_ones(unmasked_score.shape), unmasked_score)
            score = torch.sigmoid(masked_score)  # bsz, n_ent
            ranks.append(sort_and_rank(score, target))

        return torch.cat(ranks)

    def mask_eval_set(self, test_triplets, test_size, time, local2global, global2known, mode='tail'):
        mask = test_triplets.new_zeros(test_size, self.model.num_ents if self.negative_sample_all_entities else len(global2known))

        # filter setting
        for i in range(test_size):
            s, r, o = test_triplets[i]
            s, r, o = s.item(), r.item(), o.item()
            # true subject or true object: local -> global -> known
            if mode == 'tail':
                tails = self.true_tails[time][(s, r)]
                if self.negative_sample_all_entities:
                    tail_idx = np.array(list(map(lambda x: local2global[x], tails)))
                    mask[i][tail_idx] = 1
                    mask[i][local2global[o]] = 0
                else:
                    tail_idx = np.array(list(map(lambda x: global2known[local2global[x]], tails)))
                    mask[i][tail_idx] = 1
                    mask[i][global2known[local2global[o]]] = 0

            elif mode == 'head':
                heads = self.true_heads[time][(r, o)]
                if self.negative_sample_all_entities:
                    head_idx = np.array(list(map(lambda x: local2global[x], heads)))
                    mask[i][head_idx] = 1
                    mask[i][local2global[s]] = 0
                else:
                    head_idx = np.array(list(map(lambda x: global2known[local2global[x]], heads)))
                    mask[i][head_idx] = 1
                    mask[i][global2known[local2global[s]]] = 0

        return mask.byte()