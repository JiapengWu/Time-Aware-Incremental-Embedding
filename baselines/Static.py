from baselines.TKG_Embedding import TKG_Embedding

class Static(TKG_Embedding):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(Static, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    def build_model(self):
        pass

    def get_all_embeds_Gt(self, time=None):
        return self.ent_embeds

    def get_graph_ent_embeds(self, time=None):
        if time is None: time = self.time
        return self.ent_embeds[self.graph_dict_train[time].ndata['id']].view(-1, self.embed_size)

    def forward_multi_step(self, quadruples, neg_tail_samples, neg_head_samples, labels):
        return self.forward_incremental(quadruples, neg_tail_samples, neg_head_samples, labels)