import numpy as np
import torch
from pytorch_lightning.logging import TestTubeLogger
import os
import json
import dgl
import torch
import networkx as nx
import pdb
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import describe


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


sort_dict = lambda x: dict(sorted(x.items()))


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def get_edges(g):
    triples = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
    return triples, [(g.ids[s], r, g.ids[o]) for s, r, o in triples.tolist()]


def edge_difference(pre_edges_global, cur_edges_global, time=None):
    pre_ind_dict = {k: i for i, k in enumerate(pre_edges_global)}
    cur_ind_dict = {k: i for i, k in enumerate(cur_edges_global)}
    pre_set = set(pre_edges_global)
    cur_set = set(cur_edges_global)
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
    # last_graph = last_edges_local_id
    last_edges_global_id = None
    appended_graphs = {}
    deleted_edges_dict = {}

    for time, g in graph_dict_train.items():
        cur_edges_local, cur_edges_global = get_edges(g)
        if type(last_edges_global_id) == type(None):
            appended_graphs[time] = g
            deleted_edges_dict[time] = None
        else:
            added_idx, deleted_idx = edge_difference(last_edges_global_id, cur_edges_global, time)
            appended_graph = build_graph_from_triples(cur_edges_local[added_idx], g)
            deleted_edges_global = torch.tensor(last_edges_global_id)[deleted_idx]
            appended_graphs[time] = appended_graph
            deleted_edges_dict[time] = deleted_edges_global

        last_edges_global_id = cur_edges_global
        # last_edges_local_id = cur_edges_local_id
        # last_graph = g
    return appended_graphs, deleted_edges_dict


def get_add_del_graph_global(time2quads_train):
    added_edges_dict = {}
    deleted_edges_dict = {}

    last_edge_set = None
    for time, quads in time2quads_train.items():
        # pdb.set_trace()
        cur_edge_set = set([(s.item(), r.item(), o.item()) for s, r, o, t in quads])
        if type(last_edge_set) == type(None):
            added_edges_dict[time] = quads
            deleted_edges_dict[time] = None
        else:
            added_idx, deleted_idx = cur_edge_set - last_edge_set, last_edge_set - cur_edge_set
            # pdb.set_trace()
            added_edges_dict[time] = torch.tensor([list(elem) + [time] for elem in added_idx])
            deleted_edges_dict[time] = torch.tensor([list(elem) + [time] for elem in deleted_idx])

        last_edge_set = cur_edge_set
    return added_edges_dict, deleted_edges_dict


def get_common_triples_adjacent_time(graph_dict_train):
    common_triples_dict = {}
    last_edges = None
    for time, g in graph_dict_train.items():
        triples = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
        cur_edges = [(g.ids[s], r, g.ids[o]) for s, r, o in triples.tolist()]
        if type(last_edges) == type(None):
            common_triples_dict[time] = None
            last_edges = cur_edges
            continue
        common_triples_dict[time] = build_simple_graph_from_triples(list(set(cur_edges) & set(last_edges)))
        last_edges = cur_edges

    return common_triples_dict


def get_common_triples_adjacent_time_global(time2quads_train):
    common_triples_dict = {}
    last_edges = None
    for time, quads in time2quads_train.items():
        cur_edges = [(s, r, o) for s, r, o, t in quads]
        if type(last_edges) == type(None):
            common_triples_dict[time] = None
            last_edges = cur_edges
            continue
        common_triples_dict[time] = build_simple_graph_from_triples(list(set(cur_edges) & set(last_edges)))
        last_edges = cur_edges

    return common_triples_dict


def collect_one_hot_neighbors(common_triple_graphs, involved_entities, local2global, random_sample, batch_size):
    triples = []
    forward_g, backward_g = common_triple_graphs
    global_involved_entities = [local2global[i] for i in involved_entities]
    # print(len(forward_g.edges))
    if len(forward_g.edges) > batch_size:
        if random_sample:
            for node1, node2, data in forward_g.edges(data=True):
                triples.append([node1, data['type_s'], node2])
            sample_idx = np.random.randint(len(triples), size=batch_size)
            triples = np.array(triples)[sample_idx]
        else:
            for e in global_involved_entities:
                if e in forward_g.nodes:
                    for obj in forward_g[e]:
                        triples.append([e, forward_g[e][obj]['type_s'], obj])
                    for sub in backward_g[e]:
                        triples.append([sub, backward_g[e][sub]['type_o'], e])
            # print("sampling one hot neighbors, number of triples: {}".format(len(triples)))
    else:
        for node1, node2, data in forward_g.edges(data=True):
            triples.append([node1, data['type_s'], node2])
    return torch.tensor(triples)


def collect_one_hot_neighbors_global(common_triple_graphs, involved_entities, one_hop_positive_sampling, batch_size):
    triples = []
    forward_g, backward_g = common_triple_graphs
    # print(len(forward_g.edges))
    if len(forward_g.edges) > batch_size:
        if one_hop_positive_sampling:
            for e in involved_entities:
                if e in forward_g.nodes:
                    for obj in forward_g[e]:
                        triples.append([e, forward_g[e][obj]['type_s'], obj])
                    for sub in backward_g[e]:
                        triples.append([sub, backward_g[e][sub]['type_o'], e])
        else:
            for node1, node2, data in forward_g.edges(data=True):
                triples.append([node1, data['type_s'], node2])
            sample_idx = np.random.randint(len(triples), size=batch_size)
            triples = np.array(triples)[sample_idx]
            # print("sampling one hot neighbors, number of triples: {}".format(len(triples)))
    else:
        for node1, node2, data in forward_g.edges(data=True):
            triples.append([node1, data['type_s'], node2])
    return torch.tensor(triples)


def build_simple_graph_from_triples(common_triples):
    src_list = [triple[0] for triple in common_triples]
    # rel_list = [triple[1] for triple in common_triples]
    dst_list = [triple[2] for triple in common_triples]
    forward_g = nx.DiGraph()
    forward_g.add_nodes_from(src_list + dst_list)
    forward_g.add_edges_from(zip(src_list, dst_list))
    nx.set_edge_attributes(forward_g, {(s, o): r for s, r, o in common_triples}, 'type_s')
    backward_g = nx.DiGraph()
    backward_g.add_nodes_from(src_list + dst_list)
    backward_g.add_edges_from(zip(dst_list, src_list))
    nx.set_edge_attributes(backward_g, {(o, s): r for s, r, o in common_triples}, 'type_o')
    return forward_g, backward_g


def move_dgl_to_cuda(g, device):
    g.ndata.update({k: cuda(g.ndata[k], device) for k in g.ndata})
    g.edata.update({k: cuda(g.edata[k], device) for k in g.edata})


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


def get_true_subject_and_object_per_graph(triples):
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

    def log_args(self, args):
        config_path = self.experiment.get_data_path(self.experiment.name, self.experiment.version)
        with open(os.path.join(config_path, 'config.json'), 'w') as configfile:
            configfile.write(json.dumps(args.__dict__, indent=2, sort_keys=True))


def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm


def get_metrics(ranks):
    mrr = torch.mean(1.0 / ranks.float())
    hit_1 = torch.mean((ranks <= 1).float())
    hit_3 = torch.mean((ranks <= 3).float())
    hit_10 = torch.mean((ranks <= 10).float())
    return mrr, hit_1, hit_3, hit_10


def mse_loss(input, target):
    return torch.sum((input - target) ** 2)


def get_known_entities_per_time_step(graph_dict_train, num_ents):
    all_known_entities = {}
    occurred_entity_positive_mask = np.zeros(num_ents)
    for t in range(len(graph_dict_train)):
        occurred_entity_positive_mask[list(graph_dict_train[t].ids.values())] = 1
        known_entities = occurred_entity_positive_mask.nonzero()[0]
        all_known_entities[t] = known_entities
    return all_known_entities


def get_known_relations_per_time_step(graph_dict_train, num_rels):
    all_known_relations = {}
    occurred_relation_positive_mask = np.zeros(num_rels)
    for t in range(len(graph_dict_train)):
        prev_known_relations = occurred_relation_positive_mask.nonzero()[0]
        if len(prev_known_relations) == num_rels:
            for next_t in range(t, len(graph_dict_train)):
                all_known_relations[next_t] = prev_known_relations
            break
        relation_set = graph_dict_train[t].edata['type_s'].unique().tolist()
        occurred_relation_positive_mask[relation_set] = 1
        known_relations = occurred_relation_positive_mask.nonzero()[0]
        all_known_relations[t] = known_relations
    return all_known_relations


def get_known_entities_relations_per_time_step_global(time2quads_train,
                    time2quads_val, time2quads_test, num_ents, num_rels):
    all_known_entities = {}
    all_known_relations = {}

    occurred_entity_positive_mask = np.zeros(num_ents)
    occurred_relation_positive_mask = np.zeros(num_rels)
    for t in time2quads_train.keys():
        for quads in time2quads_train[t], time2quads_val[t], time2quads_test[t]:
            for quad in quads:
                occurred_entity_positive_mask[quad[0]] = 1
                occurred_entity_positive_mask[quad[2]] = 1
                occurred_relation_positive_mask[quad[1]] = 1
        all_known_entities[t] = occurred_entity_positive_mask.nonzero()[0]
        all_known_relations[t] = occurred_relation_positive_mask.nonzero()[0]
    return all_known_entities, all_known_relations


def plot_frequency_stats(target_triple_freq_lst, target_ent_pair_freq_lst, target_sub_rel_freq_lst,
                         target_rel_obj_freq_lst, target_sub_freq_lst, target_obj_freq_lst, all_time=False, historical=True):

    plot_dict = {k: v for k, v in sorted(Counter(target_triple_freq_lst).items(), key=lambda item: item[0])}
    plt.plot(np.log([x + 1 for x in list(plot_dict.keys())]), list(plot_dict.values()), '-o', markersize=2)
    plt.ylabel('# triple occurrence with such frequency value')
    plt.xlabel('{}{}log triple frequency value'.format("historical " if historical else "current ", "all time " if all_time else ""))
    plt.show()
    plt.clf()

    plot_dict = {k: v for k, v in sorted(Counter(target_ent_pair_freq_lst).items(), key=lambda item: item[0])}
    plt.plot(np.log([x + 1 for x in list(plot_dict.keys())]), list(plot_dict.values()), '-o', markersize=2)
    plt.ylabel('# entity pair occurrence with such frequency value')
    plt.xlabel('{}{}log entity pair frequency value'.format("historical " if historical else "current ", "all time " if all_time else ""))
    plt.show()
    plt.clf()

    plot_dict = {k: v for k, v in sorted(Counter(target_sub_rel_freq_lst).items(), key=lambda item: item[0])}
    plt.plot(np.log([x + 1 for x in list(plot_dict.keys())]), list(plot_dict.values()), '-o', markersize=2)
    plt.ylabel('# subject relation occurrence with such frequency value')
    plt.xlabel('{}{}log subject relation frequency value'.format("historical " if historical else "current ", "all time " if all_time else ""))
    plt.show()
    plt.clf()

    plot_dict = {k: v for k, v in sorted(Counter(target_rel_obj_freq_lst).items(), key=lambda item: item[0])}
    plt.plot(np.log([x + 1 for x in list(plot_dict.keys())]), list(plot_dict.values()), '-o', markersize=2)
    plt.ylabel('# object relation occurrence with such frequency value')
    plt.xlabel('{}{}log object relation frequency value'.format("historical " if historical else "current ", "all time " if all_time else ""))
    plt.show()
    plt.clf()

    plot_dict = {k: v for k, v in sorted(Counter(target_sub_freq_lst).items(), key=lambda item: item[0])}
    plt.plot(np.log([x + 1 for x in list(plot_dict.keys())]), list(plot_dict.values()), '-o', markersize=2)
    plt.ylabel('# subject occurrence with such frequency value')
    plt.xlabel('{}{}log subject frequency value'.format("historical " if historical else "current ", "all time " if all_time else ""))
    plt.show()
    plt.clf()

    plot_dict = {k: v for k, v in sorted(Counter(target_obj_freq_lst).items(), key=lambda item: item[0])}
    plt.plot(np.log([x + 1 for x in list(plot_dict.keys())]), list(plot_dict.values()), '-o', markersize=2)
    plt.ylabel('# object occurrence with such frequency value')
    plt.xlabel('{}{}log object frequency value'.format("historical " if historical else "current ", "all time " if all_time else ""))
    plt.show()
    plt.clf()


def count_frequency_value_lst(quads, target_triple_freq, target_ent_pair_freq, target_sub_rel_freq,
                              target_rel_obj_freq, target_sub_freq, target_obj_freq):
    target_triple_freq_lst = []
    target_ent_pair_freq_lst = []
    target_sub_rel_freq_lst = []
    target_rel_obj_freq_lst = []
    target_sub_freq_lst = []
    target_obj_freq_lst = []

    for s, r, o, t in quads:
        s, r, o, t = s.item(), r.item(), o.item(), t.item()
        target_triple_freq_lst.append(target_triple_freq[(s, r, o)])
        target_ent_pair_freq_lst.append(target_ent_pair_freq[(s, o)])
        target_sub_rel_freq_lst.append(target_sub_rel_freq[(s, r)])
        target_rel_obj_freq_lst.append(target_rel_obj_freq[(r, o)])
        target_sub_freq_lst.append(target_sub_freq[s])
        target_obj_freq_lst.append(target_obj_freq[o])

    return target_triple_freq_lst, target_ent_pair_freq_lst, target_sub_rel_freq_lst, \
           target_rel_obj_freq_lst, target_sub_freq_lst, target_obj_freq_lst


def analyze_top_samples(reservoir_sampler, time, all_hist_quads, sample_size, id2ent, id2rel):
    hist_target_triple_freq = reservoir_sampler.triple_freq_per_time_step_agg[time]
    hist_target_ent_pair_freq = reservoir_sampler.ent_pair_freq_per_time_step_agg[time]
    hist_target_sub_rel_freq = reservoir_sampler.sub_rel_freq_per_time_step_agg[time]
    hist_target_rel_obj_freq = reservoir_sampler.rel_obj_freq_per_time_step_agg[time]
    hist_target_sub_freq = reservoir_sampler.sub_freq_per_time_step_agg[time]
    hist_target_obj_freq = reservoir_sampler.obj_freq_per_time_step_agg[time]

    cur_target_triple_freq = reservoir_sampler.triple_freq_per_time_step[time]
    cur_target_ent_pair_freq = reservoir_sampler.ent_pair_freq_per_time_step[time]
    cur_target_sub_rel_freq = reservoir_sampler.sub_rel_freq_per_time_step[time]
    cur_target_rel_obj_freq = reservoir_sampler.rel_obj_freq_per_time_step[time]
    cur_target_sub_freq = reservoir_sampler.sub_freq_per_time_step[time]
    cur_target_obj_freq = reservoir_sampler.obj_freq_per_time_step[time]

    sample_rate_array = np.array(reservoir_sampler.sample_rate_cache[time])

    probability_array = sample_rate_array / np.sum(sample_rate_array)
    sorted_index = np.argsort(-probability_array) if reservoir_sampler.frequency_sampling or reservoir_sampler.inverse_frequency_sampling\
                        else torch.randperm(all_hist_quads.size(0))
    # sorted_probability_array = -np.sort(-probability_array)
    print(describe(probability_array))

    hist_target_triple_freq_lst = []
    hist_target_ent_pair_freq_lst = []
    hist_target_sub_rel_freq_lst = []
    hist_target_rel_obj_freq_lst = []
    hist_target_sub_freq_lst = []
    hist_target_obj_freq_lst = []

    cur_target_triple_freq_lst = []
    cur_target_ent_pair_freq_lst = []
    cur_target_sub_rel_freq_lst = []
    cur_target_rel_obj_freq_lst = []
    cur_target_sub_freq_lst = []
    cur_target_obj_freq_lst = []

    triple_lst = []
    ent_pair_lst = []
    subject_rel_lst = []
    rel_object_lst = []
    subject_lst = []
    object_lst = []
    time_lst = []
    n = 0
    for i in sorted_index[:sample_size]:
        s, r, o, t = all_hist_quads[i]
        s, r, o, t = s.item(), r.item(), o.item(), t.item()
        s_string = id2ent[s] if s in id2ent else s
        o_string = id2ent[o] if o in id2ent else o
        r_string = id2rel[r]
        if n < 500:
            print("{}\t{}\t{}\t{}, score: {}".format(s_string, r_string, o_string, t, sample_rate_array[i]))
        n += 1
        triple_lst.append((s_string, r_string, o_string))
        ent_pair_lst.append((s_string, o_string))
        subject_rel_lst.append((s_string, r_string))
        rel_object_lst.append((r_string, o_string))
        subject_lst.append(s_string)
        object_lst.append(o_string)
        time_lst.append(t)

        hist_target_triple_freq_lst.append(hist_target_triple_freq[(s, r, o)])
        hist_target_ent_pair_freq_lst.append(hist_target_ent_pair_freq[(s, o)])
        hist_target_sub_rel_freq_lst.append(hist_target_sub_rel_freq[(s, r)])
        hist_target_rel_obj_freq_lst.append(hist_target_rel_obj_freq[(r, o)])
        hist_target_sub_freq_lst.append(hist_target_sub_freq[s])
        hist_target_obj_freq_lst.append(hist_target_obj_freq[o])

        cur_target_triple_freq_lst.append(cur_target_triple_freq[(s, r, o)])
        cur_target_ent_pair_freq_lst.append(cur_target_ent_pair_freq[(s, o)])
        cur_target_sub_rel_freq_lst.append(cur_target_sub_rel_freq[(s, r)])
        cur_target_rel_obj_freq_lst.append(cur_target_rel_obj_freq[(r, o)])
        cur_target_sub_freq_lst.append(cur_target_sub_freq[s])
        cur_target_obj_freq_lst.append(cur_target_obj_freq[o])

    '''
    triple_dict = {k: v for k, v in sorted(Counter(Counter(triple_lst).values()).items(), key=lambda item: item[0])}
    plt.plot(np.log([x + 1 for x in list(triple_dict.keys())]), list(triple_dict.values()), '-o', markersize=2)
    plt.ylabel('# triple occurrence with such frequency value')
    plt.xlabel('log of triples in the sampled facts')
    plt.show()
    plt.clf()

    ent_pair_dict = {k: v for k, v in sorted(Counter(Counter(ent_pair_lst).values()).items(), key=lambda item: item[0])}
    plt.plot(np.log([x + 1 for x in list(ent_pair_dict.keys())]), list(ent_pair_dict.values()), '-o', markersize=2)
    plt.ylabel('# entity pair occurrence with such frequency value')
    plt.xlabel('log of entity pairs in the sampled facts')
    plt.show()
    plt.clf()

    subject_rel_dict = {k: v for k, v in sorted(Counter(Counter(subject_rel_lst).values()).items(), key=lambda item: item[0])}
    plt.plot(np.log([x + 1 for x in list(subject_rel_dict.keys())]), list(subject_rel_dict.values()), '-o', markersize=2)
    plt.ylabel('# subject-relation occurrence with such frequency value')
    plt.xlabel('log of subject-relation in the sampled facts')
    plt.show()
    plt.clf()

    rel_object_dict = {k: v for k, v in sorted(Counter(Counter(rel_object_lst).values()).items(), key=lambda item: item[0])}
    plt.plot(np.log([x + 1 for x in list(rel_object_dict.keys())]), list(rel_object_dict.values()), '-o', markersize=2)
    plt.ylabel('# triple object-relation with such frequency value')
    plt.xlabel('current log of object-relation in the sampled facts')
    plt.show()
    plt.clf()

    subject_dict = {k: v for k, v in sorted(Counter(Counter(subject_lst).values()).items(), key=lambda item: item[0])}
    plt.plot(np.log([x + 1 for x in list(subject_dict.keys())]), list(subject_dict.values()), '-o', markersize=2)
    plt.ylabel('# subject occurrence with such frequency value')
    plt.xlabel('log of subject in the sampled facts')
    plt.show()
    plt.clf()

    object_dict = {k: v for k, v in sorted(Counter(Counter(object_lst).values()).items(), key=lambda item: item[0])}
    plt.plot(np.log([x + 1 for x in list(object_dict.keys())]), list(object_dict.values()), '-o', markersize=2)
    plt.ylabel('# object occurrence with such frequency value')
    plt.xlabel('log of object in the sampled facts')
    plt.show()
    plt.clf()

    time_dict = {k: v for k, v in sorted(Counter(time_lst).items(), key=lambda item: item[0])}
    plt.plot(list(time_dict.keys()), list(time_dict.values()), '-o', markersize=2)
    plt.ylabel('# time occurrence with such frequency value')
    plt.xlabel('number of time in the sampled facts')
    plt.show()
    plt.clf()
    '''

    plot_frequency_stats(hist_target_triple_freq_lst, hist_target_ent_pair_freq_lst, hist_target_sub_rel_freq_lst,
                         hist_target_rel_obj_freq_lst, hist_target_sub_freq_lst, hist_target_obj_freq_lst, all_time=False)

    plot_frequency_stats(cur_target_triple_freq_lst, cur_target_ent_pair_freq_lst, cur_target_sub_rel_freq_lst,
                         cur_target_rel_obj_freq_lst, cur_target_sub_freq_lst, cur_target_obj_freq_lst, all_time=True, historical=False)


def print_per_step_top_patterns(reservoir_sampler, time, all_hist_quads, sample_size, id2ent, id2rel):
    sample_rate_array = np.array(reservoir_sampler.sample_rate_cache[time])

    probability_array = sample_rate_array / np.sum(sample_rate_array)
    sorted_index = np.argsort(
        -probability_array) if reservoir_sampler.frequency_sampling or reservoir_sampler.inverse_frequency_sampling \
        else torch.randperm(all_hist_quads.size(0))

    triple_lst = []
    ent_pair_lst = []
    subject_rel_lst = []
    rel_object = []
    subject_lst = []
    object_lst = []
    relation_lst = []
    time_lst = []
    for i in sorted_index[:sample_size]:
        s, r, o, t = all_hist_quads[i]
        s, r, o, t = s.item(), r.item(), o.item(), t.item()
        s_string = id2ent[s] if s in id2ent else s
        o_string = id2ent[o] if o in id2ent else o
        r_string = id2rel[r]
        # if n < 500:
        #     print("{}\t{}\t{}\t{}, score: {}".format(s_string, r_string, o_string, t, sample_rate_array[i]))
        triple_lst.append((s_string, r_string, o_string))
        ent_pair_lst.append((s_string, o_string))
        subject_rel_lst.append((s_string, r_string))
        rel_object.append((r_string, o_string))
        subject_lst.append(s_string)
        object_lst.append(o_string)
        relation_lst.append(r_string)
        time_lst.append(t)

    print(time)
    print("triples")
    print_dict(Counter(triple_lst).most_common(10))
    print("entity pairs")
    print_dict(Counter(ent_pair_lst).most_common(10))
    print("subject-relation")
    print_dict(Counter(subject_rel_lst).most_common(10))
    print("relation-object")
    print_dict(Counter(rel_object).most_common(10))
    print("subject")
    print_dict(Counter(subject_lst).most_common(10))
    print("object")
    print_dict(Counter(object_lst).most_common(10))
    print("relation")
    print_dict(Counter(relation_lst).most_common(30))
    print()

def print_dict(inp_lst):
    for pattern, freq in inp_lst:
        print("{}\t{}".format(pattern, freq))
    print()
