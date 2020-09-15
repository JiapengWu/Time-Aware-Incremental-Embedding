from utils.dataset import id2entrel, get_total_number, build_interpolation_graphs
from utils.args import process_args
import torch
import pdb

def get_edges(g):
    triples = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
    return [(g.ids[s], r, g.ids[o]) for s, r, o in triples.tolist()]


def edge_difference(pre_edges_global, cur_edges_global):
    pre_set = set(pre_edges_global)
    cur_set = set(cur_edges_global)
    added = cur_set - pre_set
    deleted = pre_set - cur_set
    return added, deleted


def print_add_del_graph(graph_dict_train):
    # sanity_check(graph_dict_train)
    # last_graph = last_edges_local_id
    last_edges_global_id = None
    for time, g in graph_dict_train.items():
        cur_edges_global = get_edges(g)
        if type(last_edges_global_id) == type(None):
            added, deleted = cur_edges_global, None
        else:
            added, deleted = edge_difference(last_edges_global_id, cur_edges_global)
        last_edges_global_id = cur_edges_global
        print("At time step {}, the added edges are: ")
        for s, r, o in added:
            try:
                print(id2ent[s], id2rel[r], id2ent[o], time)
            except:
                pdb.set_trace()
        # pdb.set_trace()
        print("The following edges are deleted from time step t-1: ")

        if deleted != None:
            try:
                print(id2ent[s], id2rel[r], id2ent[o], time)
            except:
                pdb.set_trace()
        pdb.set_trace()


if __name__ == '__main__':
    args = process_args()
    graph_dict_train, graph_dict_val, graph_dict_test = build_interpolation_graphs(args)

    num_ents, num_rels = get_total_number(args.dataset, 'stat.txt')
    id2ent, id2rel = id2entrel(args.dataset, num_rels)
    print_add_del_graph(graph_dict_train)
