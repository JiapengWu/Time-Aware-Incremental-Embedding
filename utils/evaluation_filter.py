import torch
from utils.utils import cuda, get_true_head_and_tail_per_graph, sort_and_rank
import numpy as np


class EvaluationFilter:
    def __init__(self, model):
        self.model = model
        self.args = model.args
        self.calc_score = model.calc_score
        self.graph_dict_train = model.graph_dict_train
        self.graph_dict_val = model.graph_dict_val
        self.graph_dict_test = model.graph_dict_test
        self.occurred_entity_positive_mask = model.occurred_entity_positive_mask
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

    def calc_metrics_single_graph(self, ent_embeddings, rel_embeddings, all_ent_embeds, samples, id_dict, time, eval_bz=100):
        with torch.no_grad():
            s = samples[:, 0]
            r = samples[:, 1]
            o = samples[:, 2]
            test_size = samples.shape[0]
            # num_ent = all_ent_embeds.shape[0]
            global2local = dict({n: i for i, n in enumerate(self.model.known_entities)})
            o_mask = self.mask_eval_set(samples, test_size, time, id_dict, global2local, mode="tail")
            s_mask = self.mask_eval_set(samples, test_size, time, id_dict, global2local, mode="head")
            # import pdb; pdb.set_trace()
            # perturb object
            ranks_o = self.perturb_and_get_rank(ent_embeddings, rel_embeddings, s, r, o, test_size, o_mask, id_dict, global2local, eval_bz, mode='tail')
            # perturb subject
            ranks_s = self.perturb_and_get_rank(ent_embeddings, rel_embeddings, s, r, o, test_size, s_mask, id_dict, global2local, eval_bz, mode='head')
            ranks = torch.cat([ranks_s, ranks_o])
            ranks += 1 # change to 1-indexed
        return ranks

    def perturb_and_get_rank(self, ent_mean, rel_enc_means, s, r, o, test_size, mask, id_dict, global2local, batch_size=100, mode ='tail'):
        """ Perturb one element in the triplets
        """
        # reduction_dict: from global id to reduced id
        n_batch = (test_size + batch_size - 1) // batch_size
        ranks = []
        for idx in range(n_batch):
            batch_start = idx * batch_size
            batch_end = min(test_size, (idx + 1) * batch_size)
            batch_r = rel_enc_means[r[batch_start: batch_end]]

            if mode == 'tail':
                batch_s = ent_mean[s[batch_start: batch_end]]
                batch_o = self.model.reduced_entity_embedding
                target = o[batch_start: batch_end]
            else:
                batch_s = self.model.reduced_entity_embedding
                batch_o = ent_mean[o[batch_start: batch_end]]
                target = s[batch_start: batch_end]

            target = torch.tensor([global2local[id_dict[i.item()]] for i in target])

            if self.args.use_cuda:
                target = cuda(target, self.args.n_gpu)

            unmasked_score = self.calc_score(batch_s, batch_r, batch_o, mode=mode)
            # mask: 1 for local id
            masked_score = torch.where(mask[batch_start: batch_end], -10e6 * unmasked_score.new_ones(unmasked_score.shape), unmasked_score)
            score = torch.sigmoid(masked_score)  # bsz, n_ent
            ranks.append(sort_and_rank(score, target))
        # print("Number of evaluated entities: ".format(score.shape[1]))
        return torch.cat(ranks)

    def mask_eval_set(self, test_triplets, test_size, time, id_dict, global2local, mode='tail'):
        # time = int(time.item())
        # len(self.known_entities)
        mask = test_triplets.new_zeros(test_size, len(global2local))
        # filter setting
        for i in range(test_size):
            h, r, t = test_triplets[i]
            h, r, t = h.item(), r.item(), t.item()
            if mode == 'tail':
                tails = self.true_tails[time][(h, r)]
                tail_idx = np.array(list(map(lambda x: global2local[id_dict[x]], tails)))
                # import pdb; pdb.set_trace()
                mask[i][tail_idx] = 1
                mask[i][global2local[id_dict[t]]] = 0
            elif mode == 'head':
                heads = self.true_heads[time][(r, t)]
                head_idx = np.array(list(map(lambda x: global2local[id_dict[x]], heads)))
                mask[i][head_idx] = 1
                mask[i][global2local[id_dict[h]]] = 0
        return mask.byte()