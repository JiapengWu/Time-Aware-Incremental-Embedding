from utils.dataset import *
from utils.args import process_args
from baselines.Static import Static
from baselines.DiachronicEmbedding import DiachronicEmbedding
import glob
import json
import pickle
import os.path
from utils.util_functions import get_metrics, get_add_del_graph
from utils.metrics_collection import eval_metric_collection
from baselines.Hyte import Hyte



def inference_func(model, metrics_collector, cur_time, start_time_step, end_time_step):
    model.time = cur_time
    model.eval_subject_relation_dict = defaultdict(int)
    model.eval_object_relation_dict = defaultdict(int)
    raw_ranks, first_positive_ranks, both_positive_ranks, relative_ranks, \
    deleted_facts_ranks, rank_dict = model.inference(cur_time, start_time_step, end_time_step, test_set=test_set)

    mrr, hit_1, hit_3, hit_10 = get_metrics(raw_ranks)
    metrics_collector.update_raw_ranks(cur_time, mrr.item(), hit_1.item(), hit_3.item(), hit_10.item())

    mrr, hit_1, hit_3, hit_10 = get_metrics(first_positive_ranks)
    metrics_collector.update_first_positive_ranks(cur_time, mrr.item(), hit_1.item(), hit_3.item(), hit_10.item())

    mrr, hit_1, hit_3, hit_10 = get_metrics(both_positive_ranks)
    metrics_collector.update_both_positive_ranks(cur_time, mrr.item(), hit_1.item(), hit_3.item(), hit_10.item())

    mrr, hit_1, hit_3, hit_10 = get_metrics(deleted_facts_ranks)
    metrics_collector.update_deleted_facts_ranks(cur_time, mrr.item(), hit_1.item(), hit_3.item(), hit_10.item())

    metrics_collector.update_mean_relative_ranks(cur_time, torch.mean(relative_ranks.float()).item())

    for t in range(start_time_step, end_time_step):
        mrr, hit_1, hit_3, hit_10 = get_metrics(rank_dict[t])
        if t == cur_time:
            metrics_collector.update_eval_metrics(cur_time, mrr.item(), hit_1.item(), hit_3.item(), hit_10.item())
        metrics_collector.update_diff_time_eval_results(cur_time, t, mrr, hit_1, hit_3, hit_10)


def inference_single_model():
    snapshot_paths = glob.glob(os.path.join(experiment_path, "snapshot-0"))[0]
    checkpoint_path = glob.glob(os.path.join(snapshot_paths, "*.ckpt"))[0]
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    for cur_time in range(start_time_step, end_time_step):
        print("Inference at time step {}".format(cur_time))
        inference_func(model, metrics_collector, cur_time, start_time_step, end_time_step)

    metrics_collector.copy()
    metrics_collector.save()


def load_multiple_models():
    snapshot_paths = glob.glob(os.path.join(experiment_path, "snapshot-*"))

    time2checkpoint = {}
    for snapshot_path in snapshot_paths:
        time_step = int(snapshot_path.split('-')[-1])
        try:
            checkpoint_path = glob.glob(os.path.join(snapshot_path, "*.ckpt"))[0]
            time2checkpoint[time_step] = checkpoint_path
        except:
            time2checkpoint[time_step] = None
            continue

    for t, checkpoint in time2checkpoint.items():
        if type(checkpoint) == type(None):
            print("Missing checkpoint for time step {}, using the checkpoint of time step ".format(t, t - 1))
            time2checkpoint[t] = time2checkpoint[t - 1]
    return time2checkpoint


def inference_multi_model():
    time2checkpoint = load_multiple_models()
    for cur_time in range(start_time_step, end_time_step):
        print("Inference at time step {}".format(cur_time))
        checkpoint = torch.load(time2checkpoint[cur_time], map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.on_load_checkpoint(checkpoint)
        inference_func(model, metrics_collector, cur_time, start_time_step, end_time_step)

    metrics_collector.copy()
    metrics_collector.save()


def predict_multi_model():
    time2checkpoint = load_multiple_models()
    prediction_file = os.path.join(experiment_path, "predictions.pk")
    all_time_predictions = []
    for cur_time in range(start_time_step, end_time_step):
        print("Inference at time step {}".format(cur_time))
        checkpoint = torch.load(time2checkpoint[cur_time], map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.on_load_checkpoint(checkpoint)
        predictions = model.single_step_inference(cur_time, test_set=test_set)

        all_time_predictions.extend(predictions)

    with open(prediction_file, 'wb') as filehandle:
        pickle.dump(all_time_predictions, filehandle)

    return all_time_predictions

if __name__ == '__main__':
    args = process_args()
    experiment_path, gpus, test_set = args.checkpoint_path, args.n_gpu, args.test_set
    use_cuda = args.use_cuda = len(args.n_gpu) >= 0 and torch.cuda.is_available()

    config_path = os.path.join(experiment_path, "config.json")
    args_json = json.load(open(config_path))
    args.__dict__.update(dict(args_json))
    torch.manual_seed(args.seed)
    args.n_gpu = gpus
    args.historical_sampling = args.reservoir_sampling = args.self_kd = False
    print(args)

    num_ents, num_rels, num_time_steps = get_total_number(args.dataset, 'stat.txt')

    module = {
              "Static": Static,
              "DE": DiachronicEmbedding,
              "hyte": Hyte,
              }[args.module]

    model = module(args, num_ents, num_rels)
    if use_cuda:
        model = model.cuda(gpus[0])

    dataset = 'wiki' if 'wiki' in args.dataset else 'yago'
    start_time_step = {'wiki': 54, "yago": 42}[dataset]
    end_time_step = len(model.total_time)
    metrics_collector = eval_metric_collection(start_time_step, end_time_step, experiment_path, test_set)

    if args.base_model:
        inference_single_model()
    else:
        if True:
            predict_multi_model()
        else:
            inference_multi_model()

