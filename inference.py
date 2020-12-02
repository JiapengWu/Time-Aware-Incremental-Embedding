from utils.dataset import *
from utils.args import process_args
from baselines.Static import Static
from baselines.Simple import SimplE
from baselines.Hyte import Hyte
from baselines.DiachronicEmbedding import DiachronicEmbedding
from baselines.StaticRGCN import StaticRGCN
import glob
import json
import pickle
import os.path
from utils.util_functions import get_metrics, get_add_del_graph


def get_predictions(quadruples, ranks):
    predictions = []
    num_quadruples = len(quadruples)
    for i in range(num_quadruples):
        s, r, o, t = quadruples[i]
        s, r, o, t = s.item(), r.item(), o.item(), t.item()
        idx_dict = graph_dict_val[t].ids
        predictions.append([idx_dict[s], r, idx_dict[o], t, 'head', ranks[i]])
        predictions.append([idx_dict[s], r, idx_dict[o], t, 'tail', ranks[num_quadruples + i]])
    return predictions


def get_predictions_at_time(triples, ranks, time):
    predictions = []
    num_triples = len(triples)
    for i in range(num_triples):
        s, r, o = triples[i]
        s, r, o = s.item(), r.item(), o.item()
        predictions.append([s, r, o, time, 'head', ranks[i]])
        predictions.append([s, r, o, time, 'tail', ranks[num_triples + i]])
    return predictions


def get_quadruples(graph_dict, time):
    graph = graph_dict[time]
    quadruples = torch.stack([graph.edges()[0], graph.edata['type_s'], graph.edges()[1],
                        torch.ones(len(graph.edges()[0]), dtype=int) * time]).transpose(0, 1)
    return quadruples.cuda(gpus[0]) if use_cuda else quadruples


def deleted_edges_inference(time2checkpoint, model):
    prediction_file = os.path.join(experiment_path, "deleted-edges-predictions.pk")
    _, deleted_edges_dict = get_add_del_graph(graph_dict_train)
    all_time_predictions = []
    for time, checkpoint_path in time2checkpoint.items():
        if time == 1:
            continue
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.on_load_checkpoint(checkpoint)
        model.on_time_step_start(time)
        triples = deleted_edges_dict[time]
        # quadruples = torch.cat([quadruples, torch.zeros(quadruples.shape[0])], dim=1)
        ranks = model.eval_global_idx(triples)
        predictions = get_predictions_at_time(triples, ranks, time)
        mrr, hit_1, hit_3, hit_10 = get_metrics(ranks)
        print("Deleted edges metrics at time step {}, MRR: {}, hit 1: {}, hit 3: {}, hit 10: {}"
              .format(time, mrr.item(), hit_1.item(), hit_3.item(), hit_10.item()))
        all_time_predictions.extend(predictions)

    with open(prediction_file, 'wb') as filehandle:
        pickle.dump(all_time_predictions, filehandle)

    return all_time_predictions


def single_step_inference(time2checkpoint, model):
    prediction_file = os.path.join(experiment_path, "predictions.pk")
    print(prediction_file)
    all_time_predictions = []
    for time, checkpoint_path in time2checkpoint.items():
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.on_load_checkpoint(checkpoint)
        model.on_time_step_start(time)
        quadruples = get_quadruples(graph_dict, time)
        ranks = model.evaluate(quadruples, 0)
        predictions = get_predictions(quadruples, ranks)
        mrr, hit_1, hit_3, hit_10 = get_metrics(ranks)
        print("Metrics at time step {}, MRR: {}, hit 1: {}, hit 3: {}, hit 10: {}"
              .format(time, mrr.item(), hit_1.item(), hit_3.item(), hit_10.item()))
        all_time_predictions.extend(predictions)

    with open(prediction_file, 'wb') as filehandle:
        pickle.dump(all_time_predictions, filehandle)

    return all_time_predictions


def multi_step_inference(time2checkpoint, model):
    prediction_file = os.path.join(experiment_path, "multi-step-predictions.pk")
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for time, checkpoint_path in time2checkpoint.items():
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.on_load_checkpoint(checkpoint)
        model.on_time_step_start(time)
        for t in range(time + 1):
            quadruples = get_quadruples(graph_dict, t)
            ranks = model.inference(quadruples, t)
            mrr, hit_1, hit_3, hit_10 = get_metrics(ranks)
            metrics[time][t]['mrr'] = mrr.item()
            metrics[time][t]['hit_1'] = hit_1.item()
            metrics[time][t]['hit_3'] = hit_3.item()
            metrics[time][t]['hit_10'] = hit_10.item()

    with open(prediction_file, 'wb') as filehandle:
        pickle.dump(default_to_regular(metrics), filehandle)
    return metrics


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def inference():
    snapshot_paths = glob.glob(os.path.join(experiment_path, "snapshot-*"))
    time2checkpoint = {}
    for snapshot_path in snapshot_paths:
        time_step = int(snapshot_path.split('-')[-1])
        try:
            checkpoint_path = glob.glob(os.path.join(snapshot_path, "*.ckpt"))[0]
            time2checkpoint[time_step] = checkpoint_path
        except:
            continue
    # time2checkpoint = dict(zip(time_steps, checkpoint_paths))
    time2checkpoint = dict(sorted(time2checkpoint.items(), key=lambda kv: kv[0]))
    # prediction_path_prefix = "-".join(experiment_path.split('/')[1:])
    args.inference = True
    args.n_gpu = gpus

    num_ents, num_rels, num_time_steps = get_total_number(args.dataset, 'stat.txt')
    module = {
              "Simple": SimplE,
              "Static": Static,
              "DE": DiachronicEmbedding,
              "Hyte": Hyte,
              "SRGCN": StaticRGCN
              }[args.module]

    model = module(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
    model.get_known_entities_per_time_step()
    if use_cuda:
        model = model.cuda(gpus[0])
    if do_multi_step_inference:
        multi_step_inference(time2checkpoint, model)
    else:
        if do_deleted_edges_inference:
            deleted_edges_inference(time2checkpoint, model)
        else:
            single_step_inference(time2checkpoint, model)


if __name__ == '__main__':
    args = process_args()
    experiment_path, do_multi_step_inference, gpus, eval_on_test_set, do_deleted_edges_inference = \
        args.checkpoint_path, args.multi_step_inference, args.n_gpu, args.eval_on_test_set, args.deleted_edges_inference
    use_cuda = args.use_cuda = len(args.n_gpu) >= 0 and torch.cuda.is_available()

    config_path = os.path.join(experiment_path, "config.json")
    args_json = json.load(open(config_path))
    args.__dict__.update(dict(args_json))
    torch.manual_seed(args.seed)
    graph_dict_train, graph_dict_val, graph_dict_test = build_interpolation_graphs(args)
    graph_dict = graph_dict_test if eval_on_test_set else graph_dict_val
    inference()
