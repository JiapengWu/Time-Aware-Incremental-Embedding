import torch
from utils.utils import cuda, get_true_subject_and_object_per_graph, sort_and_rank
import numpy as np
# import baselines


class EvaluationFilterGlobal:
    def __init__(self, model):
        self.model = model
        self.args = model.args
        self.calc_score = model.calc_score
        self.get_true_subject_and_object_global()

    def get_true_subject_and_object_global(self):
        self.true_heads_global_dict = dict()
        self.true_tails_global_dict = dict()

        for t in self.model.total_time:
            quads = torch.cat([self.model.time2quads_train[t],
                                 self.model.time2quads_val[t],
                                 self.model.time2quads_test[t]])
            true_head_dict, true_tail_dict = get_true_subject_and_object_per_graph(quads[:, :3])
            self.true_heads_global_dict[t] = true_head_dict
            self.true_tails_global_dict[t] = true_tail_dict

    def calc_metrics_quadruples(self, quadruples, relation_embedding, known_entities, global2known, eval_bz=100):
        # import pdb; pdb.set_trace()
        test_size = quadruples.shape[0]
        subject_tensor = quadruples[:, 0]
        object_tensor = quadruples[:, 2]
        time_tensor = quadruples[:, 3]
        o_mask = self.mask_eval_set(quadruples, test_size, global2known, mode="tail")
        s_mask = self.mask_eval_set(quadruples, test_size, global2known, mode="head")
        # perturb object
        ranks_o = self.perturb_and_get_rank(subject_tensor, relation_embedding, object_tensor, time_tensor,
                                            known_entities, test_size, o_mask, global2known, eval_bz, mode='tail')
        # perturb subject
        ranks_s = self.perturb_and_get_rank(object_tensor, relation_embedding, subject_tensor, time_tensor,
                                            known_entities, test_size, s_mask, global2known, eval_bz, mode='head')
        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1  # change to 1-indexed
        return ranks

    def mask_eval_set(self, test_triplets, test_size, global2known, mode='tail'):
        mask = test_triplets.new_zeros(test_size, len(global2known))

        # filter setting
        for i in range(test_size):
            s, r, o, t = test_triplets[i]
            s, r, o, t = s.item(), r.item(), o.item(), t.item()
            # true subject or true object: local -> global -> known
            if mode == 'tail':
                tails = self.true_tails_global_dict[t][(s, r)]
                tail_idx = np.array(list(map(lambda x: global2known[x], tails)))
                mask[i][tail_idx] = 1
                mask[i][global2known[o]] = 0

            elif mode == 'head':
                heads = self.true_heads_global_dict[t][(r, o)]
                head_idx = np.array(list(map(lambda x: global2known[x], heads)))
                mask[i][head_idx] = 1
                mask[i][global2known[s]] = 0

        return mask.byte()

    def perturb_and_get_rank(self, anchor_entities, relation_embedding, target, time_tensor,
                             known_entities, test_size, mask, global2known, batch_size=100, mode ='tail'):
        """ Perturb one element in the triplets
        """

        # import pdb; pdb.set_trace()
        anchor_embedding = self.model.get_ent_embeds_train_global(
                    anchor_entities, time_tensor)
        neg_entity_embeddings = self.model.get_ent_embeds_train_global(
                    known_entities, time_tensor, mode='neg')

        if mode == 'tail':
            subject_embedding = anchor_embedding
            object_embedding = neg_entity_embeddings

        else:
            subject_embedding = neg_entity_embeddings
            object_embedding = anchor_embedding

        cur_target = torch.tensor([global2known[i.item()] for i in target])

        if self.args.use_cuda:
            cur_target = cuda(cur_target, self.args.n_gpu)

        unmasked_score = self.calc_score(subject_embedding, relation_embedding, object_embedding, mode=mode)
        masked_score = torch.where(mask, -10e6 * unmasked_score.new_ones(unmasked_score.shape), unmasked_score)

        # mask: 1 for local id
        score = torch.sigmoid(masked_score)  # bsz, n_ent
        return sort_and_rank(score, cur_target)