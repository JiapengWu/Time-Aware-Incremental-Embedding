from models.TKG_Embedding_Global import TKG_Embedding_Global


class Static(TKG_Embedding_Global):
    def __init__(self, args, num_ents, num_rels):
        super(Static, self).__init__(args, num_ents, num_rels)

    def build_model(self):
        pass

    def get_all_embeds_Gt(self, time=None):
        return self.ent_embeds

    def get_graph_ent_embeds(self, time=None):
        if time is None: time = self.time
        return self.ent_embeds[self.graph_dict_train[time].ndata['id']].view(-1, self.embed_size)

    def forward_multi_step(self, quadruples, neg_tail_samples, neg_head_samples, labels):
        return self.forward_incremental(quadruples, neg_tail_samples, neg_head_samples, labels)

    def get_all_embeds_Gt_old(self):
        return self.old_ent_embeds

    def precompute_entity_time_embed(self):
        pass

    def get_ent_embeds_train_global(self, entities, time_tensor, mode='pos'):
        # import pdb; pdb.set_trace()
        return self.ent_embeds[entities]

    def get_ent_embeds_train_global_old(self, entities, time_tensor, mode='pos'):
        return self.old_ent_embeds[entities]