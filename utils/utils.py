import numpy as np
import torch
import pdb
from pytorch_lightning.logging import TestTubeLogger
import os
import json
import dgl

def move_dgl_to_cuda(g, device):
    g.ndata.update({k: cuda(g.ndata[k], device) for k in g.ndata})
    g.edata.update({k: cuda(g.edata[k], device) for k in g.edata})

def filter_none(l):
    return list(filter(lambda x: x is not None, l))

def cuda(tensor, device):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda(device[0])
    else:
        return tensor

def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']

class MyTestTubeLogger(TestTubeLogger):
    def __init__(self, *args, **kwargs):
        super(MyTestTubeLogger, self).__init__(*args, **kwargs)

    def log_hyperparams(self, args):
        config_path = self.experiment.get_data_path(self.experiment.name, self.experiment.version)
        with open(os.path.join(config_path, 'config.json'), 'w') as configfile:
            configfile.write(json.dumps(vars(args), indent=2, sort_keys=True))


def reparametrize(mean, std):
    """using std to sample"""
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mean)


def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm


def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    return g, rel, norm