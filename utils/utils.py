import numpy as np
import torch
from pytorch_lightning.logging import TestTubeLogger
import os
import json
import dgl
import torch
import pdb


def build_sampled_graph_from_triples(triples, train_graph):
    sample_idx = np.random.choice(np.arange(len(triples)), size=int(0.5 * len(triples)), replace=False)
    return build_graph_from_triples(triples[sample_idx], train_graph)


def build_graph_from_triples(triples, train_graph):
    src, rel, dst = triples.transpose(0, 1)
    g = dgl.DGLGraph()
    g.add_nodes(len(train_graph.nodes))
    g.add_edges(src, dst)
    node_norm = comp_deg_norm(g)
    g.ndata.update({'id': train_graph.ndata['id'], 'norm': torch.from_numpy(node_norm).view(-1, 1)})
    g.edata['norm'] = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
    g.edata['type_s'] = rel
    g.ids = train_graph.ids
    return g


def get_edges(g):
    triples = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
    return triples, [(g.ids[s], r, g.ids[o]) for s, r, o in triples.tolist()]


def sort_and_rank(score, target):
    # pdb.set_trace()
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def edge_difference(pre_edges, cur_edges, time=None):
    pre_ind_dict = {k: i for i, k in enumerate(pre_edges)}
    cur_ind_dict = {k: i for i, k in enumerate(cur_edges)}
    pre_set = set(pre_edges)
    cur_set = set(cur_edges)
    added = cur_set - pre_set
    deleted = pre_set - cur_set
    added_idx = [cur_ind_dict[x] for x in added]
    deleted_idx = [pre_ind_dict[x] for x in deleted]
    # print("At time step {}, total number of edges: {}, number of shared edges: {}, number of added edges: {}, number of deleted edges: {}".
    #       format(time, len(cur_set), len(pre_set & cur_set), len(added), len(deleted)))
    return added_idx, deleted_idx


def sanity_check(graph_dict_train):
    for time, graph in graph_dict_train.items():
        edges = get_edges(graph)
        edge_set = set(edges)
        if len(edges) - len(edge_set) > 0:
            print("At time {}. there are {} duplicated edges".format(time, len(edges) - len(edge_set)))
    print("Sanity check done.")


def get_add_del_graph(graph_dict_train):
    # sanity_check(graph_dict_train)
    last_graph = last_triples = last_edges = None

    appended_graphs = {}
    deleted_graphs = {}

    for time, g in graph_dict_train.items():
        cur_triples, cur_edges = get_edges(g)
        if not last_edges:
            appended_graphs[time] = g
            deleted_graphs[time] = None
        else:
            added_idx, deleted_idx = edge_difference(last_edges, cur_edges, time)
            appended_graph = build_graph_from_triples(cur_triples[added_idx], g)
            deleted_graph = build_graph_from_triples(last_triples[deleted_idx], last_graph)
            appended_graphs[time] = appended_graph
            deleted_graphs[time] = deleted_graph
        last_edges = cur_edges
        last_triples = cur_triples
        last_graph = g

    return appended_graphs, deleted_graphs


def move_dgl_to_cuda(g, device):
    g.ndata.update_eval_metrics({k: cuda(g.ndata[k], device) for k in g.ndata})
    g.edata.update_eval_metrics({k: cuda(g.edata[k], device) for k in g.edata})


def cuda(tensor, device):
    # pdb.set_trace()
    torch.device(device[0])
    if tensor.device == torch.device('cpu'):
        return tensor.cuda(device[0])
    else:
        return tensor


def filter_none(l):
    return list(filter(lambda x: x is not None, l))


def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']


def get_true_head_and_tail_per_graph(triples):
    true_head = {}
    true_tail = {}
    for head, relation, tail in triples:
        head, relation, tail = head.item(), relation.item(), tail.item()
        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    # this is correct
    for relation, tail in true_head:
        true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
    for head, relation in true_tail:
        true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

    return true_head, true_tail


class MyTestTubeLogger(TestTubeLogger):
    def __init__(self, *args, **kwargs):
        super(MyTestTubeLogger, self).__init__(*args, **kwargs)

    def log_hyperparams(self, args):
        config_path = self.experiment.get_data_path(self.experiment.name, self.experiment.version)
        with open(os.path.join(config_path, 'config.json'), 'w') as configfile:
            configfile.write(json.dumps(args.__dict__, indent=2, sort_keys=True))


def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm

