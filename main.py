# from comet_ml import Experiment, ExistingExperiment
from utils.dataset import *
from utils.args import process_args
from baselines.Static import Static
from baselines.DiachronicEmbedding import DiachronicEmbedding
from baselines.ATiSE import ATiSE
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
        args.__dict__.update_eval_metrics(dict(args_json))

    use_cuda = args.use_cuda = len(args.n_gpu) >= 0 and torch.cuda.is_available() and not args.cpu
    args.n_gpu = 0 if args.cpu else args.n_gpu
    name = "{}-{}-{}-{}{}{}{}{}".format(args.module, args.dataset.split('/')[-1], args.score_function,
                                        args.train_seq_len,
                                        "-multi-step" if args.multi_step else "",
                                        "-addition" if args.addition else "",
                                        "-debug" if args.debug else "",
                                        "-overfit" if args.overfit else "")

    version = time.strftime('%Y%m%d%H%M')
    log_file_out = "logs/log-{}-{}".format(name, version)
    log_file_err = "errs/log-{}-{}".format(name, version)

    if not args.debug:
        sys.stdout = open(log_file_out, 'w')
        sys.stderr = open(log_file_err, 'w')

    tt_logger = MyTestTubeLogger(
        save_dir="experiments",
        name=name,
        debug=False,
        version=version,
        create_git_tag=True
    )

    args.base_path = tt_logger.experiment.get_data_path(tt_logger.experiment.name, tt_logger.experiment.version)

    num_ents, num_rels = get_total_number(args.dataset, 'stat.txt')

    graph_dict_train, graph_dict_val, graph_dict_test = build_interpolation_graphs(args)

    total_time = np.array(list(graph_dict_train.keys()))
    module = {
              "Static": Static,
              "DE": DiachronicEmbedding,
              "ATiSE": ATiSE,
              "SRGCN": StaticRGCN
              }[args.module]

    print("Model checkpoint path: {}".format(args.base_path))
    print(args)

    if not args.debug:
        tt_logger.log_hyperparams(args)
        tt_logger.save()

    model = module(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
    if args.load_base_model:
        base_model_path = glob.glob(os.path.join(args.base_model_path, "*.ckpt"))[0]
        base_model_checkpoint = torch.load(base_model_path, map_location=lambda storage, loc: storage)
        # import pdb; pdb.set_trace()
        model.load_state_dict(base_model_checkpoint['state_dict'], strict=False)

    end_time_step = min(len(total_time), args.end_time_step + 1)
    for time in range(args.start_time_step, end_time_step):
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

        trainer = Trainer(logger=tt_logger, gpus=args.n_gpu,
                          gradient_clip_val=args.gradient_clip_val,
                          max_epochs=args.max_nb_epochs,
                          fast_dev_run=args.fast,
                          num_processes=args.num_processes,
                          distributed_backend=args.distributed_backend,
                          num_sanity_val_steps=1,
                          early_stop_callback=early_stop_callback,
                          overfit_batches=1 if args.overfit else 0,
                          show_progress_bar=True,
                          checkpoint_callback=checkpoint_callback)

        model.on_time_step_start(time)
        trainer.fit(model)
        trainer.use_ddp = False
        test_res = trainer.test(model=model)
        model.on_time_step_end()
