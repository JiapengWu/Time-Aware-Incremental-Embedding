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
    parser.add_argument("--n-bases", type=int, default=128, help="number of weight blocks for each relation")
    parser.add_argument("--train-seq-len", type=int, default=0)
    parser.add_argument("--test-seq-len", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--test-batch-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--negative-rate", type=int, default=500)
    parser.add_argument('--log-gpu-memory', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test', action='store_true')

    parser.add_argument("--overfit", action='store_true')
    parser.add_argument("--multi-step", action='store_true')
    parser.add_argument("--addition", action='store_true')

    parser.add_argument("--load-base-model", action='store_true', help='load a base model')
    parser.add_argument("--base-model-path", action='store_true', help='path of the base model')
    parser.add_argument("--start-time-step", type=int, default=0)
    parser.add_argument("--end-time-step", type=int, default=10000, help='stop training after this number of time steps, used for base model training')

    parser.add_argument("--fast", action='store_true')
    parser.add_argument('--config', '-c', type=str, default=None, help='JSON file with argument for the run.')
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--cpu", action='store_true')
    return parser.parse_args()


def process_args():
    args = get_args()
    dataset = args.dataset
    args.dataset = os.path.join(args.dataset_dir, args.dataset)
    if args.load_base_model:
        args.base_model_path = "experiments/base-model/{}-{}".format(args.module, dataset)
        args.start_time_step = 1
        print(args.base_model_path)
    return args
