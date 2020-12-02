import torch
from utils.util_functions import cuda, sort_and_rank


class EvaluationRaw:
    def __init__(self, model):
        self.model = model
        self.args = model.args
        self.negative_sample_all_entities = self.args.negative_sample_all_entities
        self.calc_score = model.calc_score

    def calc_metrics_single_graph(self, rel_embeddings, samples, global2known, eval_bz=100):
        with torch.no_grad():
            s = samples[:, 0]
            r = samples[:, 1]
            o = samples[:, 2]
            test_size = samples.shape[0]
            # perturb object
            ranks_o = self.perturb_and_get_rank(rel_embeddings, s, r, o, test_size, global2known, eval_bz, mode='tail')
            # perturb subject
            ranks_s = self.perturb_and_get_rank(rel_embeddings, s, r, o, test_size, global2known, eval_bz, mode='head')
            # import pdb; pdb.set_trace()
            ranks = torch.cat([ranks_s, ranks_o])  # [head_ranks, tail_ranks]
            ranks += 1  # change to 1-indexed
        return ranks

    def perturb_and_get_rank(self, rel_enc_means, s, r, o, test_size, global2known, batch_size=100, mode ='tail'):
        """ Perturb one element in the triplets
        """
        # reduction_dict: from global id to reduced id
        n_batch = (test_size + batch_size - 1) // batch_size
        ranks = []
        for idx in range(n_batch):
            batch_start = idx * batch_size
            batch_end = min(test_size, (idx + 1) * batch_size)
            batch_r = rel_enc_means[r[batch_start: batch_end]]

            # ent_mean[local_id] = self.eval_all_embeds_g[local2global[local_id]]
            # reduced_entity_embedding[global2known[global_id]] = self.eval_all_embeds_g[global_id]
            if mode == 'tail':
                batch_s = self.model.eval_all_embeds_g[s[batch_start: batch_end]]
                batch_o = self.model.eval_all_embeds_g if self.negative_sample_all_entities else self.model.reduced_entity_embedding
                target = o[batch_start: batch_end]
            else:
                batch_s = self.model.eval_all_embeds_g if self.negative_sample_all_entities else self.model.reduced_entity_embedding
                batch_o = self.model.eval_all_embeds_g[o[batch_start: batch_end]]
                target = s[batch_start: batch_end]

            target = torch.tensor([global2known[i.item()] for i in target])
            if self.args.use_cuda:
                target = cuda(target, self.args.n_gpu)
            score = torch.sigmoid(self.calc_score(batch_s, batch_r, batch_o, mode=mode))  # bsz, n_ent

            ranks.append(sort_and_rank(score, target))

        return torch.cat(ranks)