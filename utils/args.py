import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='TKG-VAE')
    parser.add_argument("--dataset-dir", type=str, default='interpolation')
    parser.add_argument("-d", "--dataset", type=str, default='wiki')
    parser.add_argument("--score-function", type=str, default='complex')
    parser.add_argument("--module", type=str, default='DE')

    parser.add_argument('--n-gpu', nargs='+', type=int, default=[0])
    parser.add_argument("--distributed-backend", type=str, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--embed-size", type=int, default=128)
    parser.add_argument("--max-nb-epochs", type=int, default=100)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    # parser.add_argument("--n-bases", type=int, default=128, help="number of weight blocks for each relation")

    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--test-batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--np-seed", type=int, default=0)
    parser.add_argument("--negative-rate", type=int, default=500)
    parser.add_argument('--log-gpu-memory', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot-gradient', action='store_true')

    parser.add_argument('--test-set', action='store_true')
    parser.add_argument("--cold-start", action='store_true', help='Not loading models from the last time step')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--multi-step-inference', action='store_true')
    parser.add_argument('--deleted-edges-inference', action='store_true')
    parser.add_argument('--prediction-file-path',type=str, default='')

    parser.add_argument('--analysis', action='store_true')
    parser.add_argument("--overfit", action='store_true')
    parser.add_argument("--negative-sample-all-entities", action='store_true')

    parser.add_argument("--train-base-model", action='store_true')
    parser.add_argument("--addition", action='store_true')
    parser.add_argument("--deletion", action='store_true')

    # reservoir sampling parameters
    parser.add_argument("--all-prev-time-steps", action='store_true')
    parser.add_argument("--frequency-sampling", action='store_true')
    parser.add_argument("--inverse-frequency-sampling", action='store_true')
    parser.add_argument("--lambda-triple", type=int, default=2)
    parser.add_argument("--lambda-ent-pair", type=int, default=1.5)
    parser.add_argument("--lambda-ent-rel", type=int, default=1.3)
    parser.add_argument("--lambda-ent", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=10)
    parser.add_argument("--cur-frequency-discount-factor", type=float, default=0.5)

    # positive vs negative entities vs negative relations
    parser.add_argument("--a-gem", action='store_true')
    parser.add_argument("--sample-positive", action='store_true')
    parser.add_argument("--sample-neg-entity", action='store_true')
    parser.add_argument("--sample-neg-relation", action='store_true')
    parser.add_argument("--negative-rate-reservoir", type=int, default=50)

    # history versus present
    parser.add_argument("--historical-sampling", action='store_true')
    parser.add_argument("--train-seq-len", type=int, default=10)
    parser.add_argument("--eval-seq-len", type=int, default=10)
    parser.add_argument("--num-samples-each-time-step", type=int, default=1000)
    parser.add_argument("--deleted-edge-sample-size", type=int, default=10000)
    parser.add_argument("--present-sampling", action='store_true')
    parser.add_argument("--one-hop-positive-sampling", action='store_true')
    # parser.add_argument("--max-num-pos-samples", type=float, default=2048)

    # after reservoir sampling, decide whether to use KD loss, CE loss or both
    parser.add_argument("--KD-reservoir", action='store_true')
    parser.add_argument("--CE-reservoir", action='store_true')

    # KD parameters
    parser.add_argument("--self-kd-factor", type=float, default=1)
    parser.add_argument("--up-weight-factor", type=float, default=1)
    parser.add_argument("--self-kd", action='store_true', help='use self knowledge distillation')
    parser.add_argument("--load-base-model", action='store_true', help='load a base model')
    parser.add_argument("--base-model-path", action='store_true', help='path of the base model')
    parser.add_argument("--start-time-step", type=int, default=0)
    parser.add_argument("--end-time-step", type=int, default=10000, help='stop training after this number of time steps, used for base model training')

    parser.add_argument("--fast", action='store_true')
    parser.add_argument('--config', '-c', type=str, default=None, help='JSON file with argument for the run.')
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--base-model", action='store_true')

    parser.add_argument("--cpu", action='store_true')
    return parser.parse_args()


def process_args():
    args = get_args()
    dataset = args.dataset
    args.dataset = os.path.join(args.dataset_dir, args.dataset)

    assert not (args.sample_positive and args.sample_neg_entity)
    assert not (args.sample_positive and args.sample_neg_relation)
    assert not (args.addition and args.present_sampling)

    if args.KD_reservoir or args.CE_reservoir:
        assert args.historical_sampling or args.present_sampling
        assert args.sample_positive or args.sample_neg_entity or args.sample_neg_relation
    if args.a_gem:
        args.KD_reservoir = False
    if args.load_base_model:
        start_time_step_dict = {'wiki': 54, "yago": 42, "icews14": 292, 'gdelt': 292, 'wikidata': 1}
        myhost = os.uname()[1]
        if myhost == 'gdl':
            args.base_model_path = "/media/data/jiapeng-yishi/experiments/base-model/{}-{}".format(args.module, dataset)
        elif myhost == 'curie':
            args.base_model_path = "/data/jwu558/experiments/base-model/{}-{}".format(args.module, dataset)
        args.start_time_step = start_time_step_dict[dataset]
        print(args.base_model_path)
    if 'wikidata' in args.dataset:
        args.test_batch_size = 1
    if args.cold_start:
        args.load_base_model = False
    if args.module == 'atise':
        args.score_function = 'atise'
    return args
