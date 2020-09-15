import numpy as np
import os
import pickle
import dgl
from torch.utils.data import Dataset, ConcatDataset
import torch
from utils.args import process_args
from utils.utils import node_norm_to_edge_norm, comp_deg_norm, sort_dict, get_true_subject_and_object_per_graph, cuda
from collections import defaultdict
from functools import reduce
import pdb


def load_quadruples(dataset_path, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(dataset_path, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)

    if fileName2 is not None:
        with open(os.path.join(dataset_path, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)

    if fileName3 is not None:
        with open(os.path.join(dataset_path, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()
    return np.asarray(quadrupleList), np.asarray(times)


def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)


def get_total_number(dataset_path, fileName="stat.txt"):
    with open(os.path.join(dataset_path, fileName), 'r') as fr:
        line_split = fr.readline().split()
    return int(line_split[0]), int(line_split[1]), int(line_split[2])


def get_big_graph(data, num_rels):

    add_reverse = True
    if add_reverse:

        src, rel, dst = data.transpose() # node ids
        # uniq_v: range from 0 to the number of nodes acting as g.nodes();
        # edges: uniq_v[edges] = np.unique((src, dst)), mapping from (o, len(nodes)) to the original node idx
        uniq_v, edges = np.unique((src, dst), return_inverse=True)

        src, dst = np.reshape(edges, (2, -1))
        g = dgl.DGLGraph()
        g.add_nodes(len(uniq_v))
        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))

        rel_o = np.concatenate((rel + num_rels, rel))
        rel_s = np.concatenate((rel, rel + num_rels))
        g.add_edges(src, dst)
        norm = comp_deg_norm(g)
        # import pdb; pdb.set_trace()
        g.ndata.update_eval_metrics({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': torch.from_numpy(norm).view(-1, 1)})
        g.edata['type_s'] = torch.LongTensor(rel_s)
        g.edata['type_o'] = torch.LongTensor(rel_o)
        g.ids = {}
        in_graph_idx = 0
        # graph.ids: node id in the entire node set -> node index
        for id in uniq_v:
            g.ids[in_graph_idx] = id
            in_graph_idx += 1
    else:
        src, rel, dst = data.transpose() # node ids
        # uniq_v: range from 0 to the number of nodes acting as g.nodes();
        # edges: uniq_v[edges] = np.unique((src, dst)), mapping from (o, len(nodes)) to the original node idx
        uniq_v, edges = np.unique((src, dst), return_inverse=True)

        src, dst = np.reshape(edges, (2, -1))
        g = dgl.DGLGraph()
        g.add_nodes(len(uniq_v))
        g.add_edges(src, dst)
        norm = comp_deg_norm(g)
        g.ndata.update_eval_metrics({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': torch.from_numpy(norm).view(-1, 1)})
        g.edata['type_s'] = torch.LongTensor(rel)
        g.ids = {}
        in_graph_idx = 0
        for id in uniq_v:
            g.ids[in_graph_idx] = id
            in_graph_idx += 1
    return g


def get_train_val_test_graph_at_t(triples, num_rels):
    train_triples, val_triples, test_triples = \
        np.array(triples['train']), np.array(triples['valid']), np.array(triples['test'])
    try:
        total_triples = np.concatenate([train_triples, val_triples, test_triples], axis=0)
    except:
        # import pdb; pdb.set_trace()
        if test_triples.shape[0] == 0 and val_triples.shape[0] == 0:
            total_triples = train_triples
        elif test_triples.shape[0] == 0:
            total_triples = np.concatenate([train_triples, val_triples], axis=0)
        elif val_triples.shape[0] == 0:
            total_triples = np.concatenate([train_triples, test_triples], axis=0)

    src_total, rel_total, dst_total = total_triples.transpose()  # node ids
    # g.nodes() = len(uniq_v), uniq_v are the idx of nodes
    # edges: uniq_v[edges] = np.concat((src, dst)), mapping from (0, len(nodes)) to the original node idx
    uniq_v, edges = np.unique((src_total, dst_total), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))

    g_train = dgl.DGLGraph()
    g_val = dgl.DGLGraph()
    g_test = dgl.DGLGraph()

    # for training, add reverse tuples (o, r-1, s); not for val and test graphs
    src_train, rel_train, dst_train = src[:len(train_triples)], rel_total[:len(train_triples)], dst[:len(train_triples)]

    src_val, rel_val, dst_val = src[len(train_triples): len(train_triples) + len(val_triples)], \
                                rel_total[len(train_triples): len(train_triples) + len(val_triples)], \
                                dst[len(train_triples): len(train_triples) + len(val_triples)]

    src_test, rel_test, dst_test = src[len(train_triples) + len(val_triples):], \
                                   rel_total[len(train_triples) + len(val_triples):], \
                                   dst[len(train_triples) + len(val_triples):]

    add_reverse = False
    if add_reverse:
        src_train, dst_train = np.concatenate((src_train, dst_train)), np.concatenate((dst_train, src_train))
        g_train.add_nodes(len(uniq_v))
        g_train.add_edges(src_train, dst_train)
        norm = comp_deg_norm(g_train)

        rel_o = np.concatenate((rel_train + num_rels, rel_train))
        rel_s = np.concatenate((rel_train, rel_train + num_rels))

        g_train.ndata.update_eval_metrics({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)})
        g_train.edata['type_s'] = torch.LongTensor(rel_s)
        g_train.edata['type_o'] = torch.LongTensor(rel_o)
        g_train.ids = {}
        in_graph_idx = 0
        for id in uniq_v:
            g_train.ids[id] = in_graph_idx
            in_graph_idx += 1

        g_list, src_list, rel_list, dst_list = [g_test, g_val], [src_test, src_val], [rel_test, rel_val], [dst_test, dst_val]
    else:
        g_list, src_list, rel_list, dst_list = [g_train, g_test, g_val], [src_train, src_test, src_val], \
                                               [rel_train, rel_test, rel_val], [dst_train, dst_test, dst_val]

    for graph, cur_src, cur_rel, cur_dst in zip(
        g_list, src_list, rel_list, dst_list
    ):
        graph.add_nodes(len(uniq_v))
        graph.add_edges(cur_src, cur_dst)
        node_norm = comp_deg_norm(graph)
        graph.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': torch.from_numpy(node_norm).view(-1, 1)})
        # import pdb; pdb.set_trace()
        graph.edata['norm'] = node_norm_to_edge_norm(graph, torch.from_numpy(node_norm).view(-1, 1))
        graph.edata['type_s'] = torch.LongTensor(cur_rel)
        graph.ids = {}
        in_graph_idx = 0
        # graph.ids: node id in the entire node set -> node index
        for id in uniq_v:
            graph.ids[in_graph_idx] = id
            in_graph_idx += 1
    return g_train, g_val, g_test


def load_quadruples_interpolation(dataset_path, train_fname, valid_fname, test_fname, total_times):
    time2triples = {}

    for tim in total_times:
        time2triples[tim] = {"train": [], "valid": [], "test": []}

    for fname, mode in zip([train_fname, valid_fname, test_fname], ["train", "valid", "test"]):
        with open(os.path.join(dataset_path, fname), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                rel = int(line_split[1])
                tail = int(line_split[2])
                time = int(line_split[3])
                time2triples[time][mode].append((head, rel, tail))

    return time2triples


def load_quadruples_tensor(dataset_path, train_fname, valid_fname, test_fname):
    time2quads_train = defaultdict(list)
    time2quads_val = defaultdict(list)
    time2quads_test = defaultdict(list)
    for fname, time2quads in zip([train_fname, valid_fname, test_fname],
                           [time2quads_train, time2quads_val, time2quads_test]):
        with open(os.path.join(dataset_path, fname), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                rel = int(line_split[1])
                tail = int(line_split[2])
                time = int(line_split[3])
                time2quads[time].append([head, rel, tail, time])
    # pdb.set_trace()
    for time2quads in time2quads_train, time2quads_val, time2quads_test:
        for t in time2quads.keys():
            time2quads[t] = torch.tensor(time2quads[t])
    return sort_dict(time2quads_train), sort_dict(time2quads_val), sort_dict(time2quads_test)


def get_per_entity_time_sequence(time2triples):
    interaction_time_sequence = defaultdict(set)
    for tim, triple_dict in time2triples.items():
        for h, r, t in triple_dict['train']:
            interaction_time_sequence[h].add(tim)
            interaction_time_sequence[t].add(tim)

    for k in interaction_time_sequence.keys():
        interaction_time_sequence[k] = sorted(list(interaction_time_sequence[k]))

    return interaction_time_sequence


def build_interpolation_graphs_given_dataset(dataset):
    train_graph_dict_path = os.path.join(dataset, 'train_graphs.txt')
    dev_graph_dict_path = os.path.join(dataset, 'dev_graphs.txt')
    test_graph_dict_path = os.path.join(dataset, 'test_graphs.txt')
    # time_sequence = os.path.join(args.dataset, 'interaction_time_sequence.txt')
    if not os.path.isfile(train_graph_dict_path) or not os.path.isfile(dev_graph_dict_path) or not os.path.isfile(
            test_graph_dict_path):

        total_data, total_times = load_quadruples(dataset, 'train.txt', 'valid.txt', 'test.txt')
        time2triples = load_quadruples_interpolation(dataset, 'train.txt', 'valid.txt', 'test.txt', total_times)
        num_e, num_r = get_total_number(dataset, 'stat.txt')

        # interaction_time_sequence = get_per_entity_time_sequence(time2triples)

        graph_dict_train = {}
        graph_dict_dev = {}
        graph_dict_test = {}
        for tim in total_times:
            print(str(tim) + '\t' + str(max(total_times)))
            g_train, g_val, g_test = get_train_val_test_graph_at_t(time2triples[tim], num_r)
            graph_dict_train[tim] = g_train
            graph_dict_dev[tim] = g_val
            graph_dict_test[tim] = g_test

        for graph_dict, path in zip(
                [graph_dict_train, graph_dict_dev, graph_dict_test],
                [train_graph_dict_path, dev_graph_dict_path, test_graph_dict_path]
        ):
            with open(path, 'wb') as fp:
                pickle.dump(graph_dict, fp)

    else:
        graph_dicts = []
        for path in train_graph_dict_path, dev_graph_dict_path, test_graph_dict_path:
            with open(path, 'rb') as f:
                graph_dicts.append(pickle.load(f))
        graph_dict_train, graph_dict_dev, graph_dict_test = graph_dicts

    return graph_dict_train, graph_dict_dev, graph_dict_test


def build_interpolation_graphs(args):
    return build_interpolation_graphs_given_dataset(args.dataset)


def id2entrel(dataset_path, num_rels):
    id2ent = {}; id2rel = {}
    with open(os.path.join(dataset_path, "entity2id.txt"), 'r') as fr:
        for line in fr:
            line_split = line.strip().split("\t")
            name = line_split[0]
            id = int(line_split[1])
            id2ent[id] = name
    with open(os.path.join(dataset_path, "relation2id.txt"), 'r') as fr:
        for line in fr:
            line_split = line.strip().split("\t")
            name = line_split[0]
            id = int(line_split[1])
            id2rel[id] = name
            id2rel[id + num_rels] = "{}_inv".format(name)
    return id2ent, id2rel


flatten = lambda l: [item for sublist in l for item in sublist]


class FullBatchDataset(Dataset):
    def __init__(self, graph_dict, time):

        graph = graph_dict[time]
        self.quadruples = self.get_quadruples(graph, time)

    def get_quadruples(self, graph, time):
        return torch.stack([graph.edges()[0], graph.edata['type_s'], graph.edges()[1],
                                  torch.ones(len(graph.edges()[0]), dtype=int) * time]).transpose(0, 1)

    def __getitem__(self, index):
        return self.quadruples[index]

    def __len__(self):
        return len(self.quadruples)


class BaseModelDataset(Dataset):
    def __init__(self, time2triples, end_time_step):
        # import pdb; pdb.set_trace()
        self.quadruples = torch.cat([triples for time, triples in time2triples.items() if time < end_time_step])
    def __getitem__(self, index):
        return self.quadruples[index]

    def __len__(self):
        return len(self.quadruples)


class ValDataset(Dataset):
    def __init__(self, time2triples, time):
        self.quadruples = torch.tensor(time2triples[time])

    def __getitem__(self, index):
        # pdb.set_trace()
        return self.quadruples[index]

    def __len__(self):
        return len(self.quadruples)


class TrainDataset(ValDataset):
    def __init__(self, quadruples):
        self.quadruples = quadruples


class DataLoaderIterWrapper:
    def __init__(self, data_loader_iters_dict):
        self.data_loader_iters_dict = data_loader_iters_dict

    def __next__(self):
        next_data = {}
        for key, data_loader_iters in self.data_loader_iters_dict.items():
            next_data[key] = next(data_loader_iters)
        return next_data

    def __iter__(self):
        return self


class DataLoaderWrapper:
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict

    def __iter__(self):
        return DataLoaderIterWrapper({key: dataloader.__iter__() for key, dataloader in self.dataloader_dict.items()})


if __name__ == '__main__':
    args = process_args()
    build_interpolation_graphs(args)