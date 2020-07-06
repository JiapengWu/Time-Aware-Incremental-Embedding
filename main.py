# from comet_ml import Experiment, ExistingExperiment
from utils.dataset import *
from utils.args import process_args
from baselines.Static import Static
from baselines.Simple import SimplE
from baselines.Hyte import Hyte
from baselines.DiachronicEmbedding import DiachronicEmbedding
from baselines.StaticRGCN import StaticRGCN
import time
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from utils.utils import MyTestTubeLogger
import json
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import sys
import glob


if __name__ == '__main__':
    args = process_args()
    torch.manual_seed(args.seed)

    if args.config:
        args_json = json.load(open(args.config))
        args.__dict__.update(dict(args_json))

    use_cuda = args.use_cuda = len(args.n_gpu) >= 0 and torch.cuda.is_available() and not args.cpu
    args.n_gpu = 0 if args.cpu else args.n_gpu
    num_ents, num_rels = get_total_number(args.dataset, 'stat.txt')

    graph_dict_train, graph_dict_val, graph_dict_test = build_interpolation_graphs(args)

    total_time = np.array(list(graph_dict_train.keys()))
    module = {
              "Simple": SimplE,
              "Static": Static,
              "DE": DiachronicEmbedding,
              "Hyte": Hyte,
              "SRGCN": StaticRGCN
              }[args.module]

    name = "{}-{}-{}-{}".format(args.module, args.dataset.split('/')[-1], args.score_function,
                                  args.train_seq_len)
    version = time.strftime('%Y%m%d%H%M')
    log_file = "logs/log-{}-{}".format(name, version)

    tt_logger = MyTestTubeLogger(
        save_dir="experiments",
        name=name,
        debug=False,
        version=version,
        create_git_tag=True
    )

    args.base_path = tt_logger.experiment.get_data_path(tt_logger.experiment.name, tt_logger.experiment.version)

    if not args.debug:
        sys.stdout = open(log_file, 'w')
        sys.stderr = open(log_file, 'w')

    model = module(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
    if args.resume:
        assert args.model_name, args.version
        tt_logger = MyTestTubeLogger(
            save_dir='experiments',
            name=args.model_name,
            debug=False,
            version=args.version  # An existing version with a saved checkpoint
        )

    accumulative_test_result = {"mrr": 0, "hit_1": 0, "hit_3": 0, "hit_10": 0}

    for time in total_time:
        early_stop_callback = EarlyStopping(
            monitor='hit_10',
            min_delta=0.00,
            patience=args.patience,
            verbose=False,
            mode='max'
        )

        checkpoint_path = os.path.join(args.base_path, "snapshot-{}".format(time), "checkpoints")
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=True,
            monitor='hit_10',
            mode='max',
            prefix=''
        )

        tt_logger.log_hyperparams(args)
        tt_logger.save()

        trainer = Trainer(logger=tt_logger, gpus=args.n_gpu,
                          gradient_clip_val=args.gradient_clip_val,
                          max_epochs=args.max_nb_epochs,
                          fast_dev_run=args.fast,
                          num_processes=args.num_processes,
                          distributed_backend=args.distributed_backend,
                          num_sanity_val_steps=1 if args.debug else 1,
                          early_stop_callback=early_stop_callback,
                          limit_train_batches=1 if args.debug else 1.0,
                          overfit_batches=1 if args.overfit else 0,
                          show_progress_bar=True,
                          checkpoint_callback=checkpoint_callback)

        model.reset_time(time)
        trainer.fit(model)
        trainer.use_ddp = False
        # model.test_dataloader = model.val_dataloader
        load_path = glob.glob(os.path.join(os.path.join(args.base_path, "snapshot-{}").format(time), "*.ckpt"))[0]
        # print(load_path)
        checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

        test_res = trainer.test(model=model)
        print("Accumulative results:")
        for i in "mrr", "hit_1", "hit_3", "hit_10":
            print("{}: {}".format(i, model.accumulative_val_result[i]))
