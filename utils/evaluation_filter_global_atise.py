from utils.evaluation_filter_global import EvaluationFilterGlobal
import torch
from utils.utils import cuda, sort_and_rank
import pdb

class EvaluationFilterGlobalAtiSE(EvaluationFilterGlobal):
    def __init__(self, model):
        super().__init__(model)

    def calc_metrics_quadruples(self, quadruples, relation_mean, rel_cov, known_entities, global2known, eval_bz=100):
        # import pdb; pdb.set_trace()
        test_size = quadruples.shape[0]
        subject_tensor, object_tensor, time_tensor = quadruples[:, 0], quadruples[:, 2], quadruples[:, 3]
        o_mask = self.mask_eval_set(quadruples, test_size, global2known, mode="tail")
        s_mask = self.mask_eval_set(quadruples, test_size, global2known, mode="head")
        # perturb object
        ranks_o = self.perturb_and_get_rank(subject_tensor, relation_mean, rel_cov, object_tensor, time_tensor,
                                            known_entities, test_size, o_mask, global2known, eval_bz, mode='tail')
        # perturb subject
        ranks_s = self.perturb_and_get_rank(object_tensor, relation_mean, rel_cov, subject_tensor, time_tensor,
                                            known_entities, test_size, s_mask, global2known, eval_bz, mode='head')
        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1  # change to 1-indexed
        return ranks

    def perturb_and_get_rank(self, anchor_entities, relation_mean, rel_cov, target, time_tensor,
                             known_entities, test_size, mask, global2known, batch_size=100, mode ='tail'):
        """ Perturb one element in the triplets
        """

        anchor_mean = self.model.get_ent_embeds_train_global(anchor_entities, time_tensor)
        neg_entity_mean = self.model.get_ent_embeds_train_global(known_entities, time_tensor, mode='neg')
        anchor_cov = self.model.sigma_ent[anchor_entities]
        neg_entity_cov = self.model.sigma_ent[known_entities]

        if mode == 'tail':
            subject_mean, object_mean = anchor_mean, neg_entity_mean
            subject_cov, object_cov = anchor_cov, neg_entity_cov
        else:
            subject_mean, object_mean = neg_entity_mean, anchor_mean
            subject_cov, object_cov = neg_entity_cov, anchor_cov

        cur_target = torch.tensor([global2known[i.item()] for i in target])

        if self.args.use_cuda:
            cur_target = cuda(cur_target, self.args.n_gpu)
        unmasked_score = self.calc_score(subject_mean, subject_cov, object_mean, object_cov, relation_mean, rel_cov, mode=mode)
        masked_score = torch.where(mask, -10e6 * unmasked_score.new_ones(unmasked_score.shape), unmasked_score)

        # mask: 1 for local id
        score = torch.sigmoid(masked_score)  # bsz, n_ent
        return sort_and_rank(score, cur_target)