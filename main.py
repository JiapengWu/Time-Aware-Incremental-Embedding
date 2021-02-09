# from comet_ml import Experiment, ExistingExperiment
from utils.dataset import *
from utils.args import process_args
from baselines.Static import Static
from baselines.DiachronicEmbedding import DiachronicEmbedding
from baselines.Hyte import Hyte
import time
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from utils.util_functions import MyTestTubeLogger
import json
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import sys
import glob


if __name__ == '__main__':
    args = process_args()
    debug = args.debug
    fast = args.fast
    overfit = args.overfit
    torch.manual_seed(args.seed)

    if args.config:
        args_json = json.load(open(args.config))
        args.__dict__.update(dict(args_json))
        args.debug = debug
        args.fast = fast
        args.overfit = overfit

    use_cuda = args.use_cuda = len(args.n_gpu) >= 0 and torch.cuda.is_available() and not args.cpu
    args.n_gpu = 0 if args.cpu else args.n_gpu
    if use_cuda:
        torch.cuda.set_device(args.n_gpu[0])

    name = "{}-{}-{}-patience-{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}".format(args.module, args.dataset.split('/')[-1],
                                                            args.score_function, args.patience,
                                        "-addition" if args.addition else "",
                                        "-deletion" if args.deletion else "",
                                        "-up-weight-factor-{}".format(args.up_weight_factor) if args.deletion else "",
                                                             # "-multi-step" if args.multi_step else "",
                                                             # '-length-{}'.format(args.train_seq_len) if args.multi_step else '',
                                        '-self-kd-factor-{}'.format(args.self_kd_factor) if args.self_kd else '',
                                        "-debug" if args.debug else "",
                                        "-overfit" if args.overfit else "",
                                        "-cold-start" if args.cold_start else "",
                                        "-all-prev-time-steps" if args.all_prev_time_steps else ""
                                        "-KD" if args.KD_reservoir else "",
                                        "-CE" if args.CE_reservoir else "",
                                        "-a-gem" if args.a_gem else "",
                                        "-historical-sampling" if args.historical_sampling else "",
                                        "-train-seq-len-{}".format(args.train_seq_len) if args.historical_sampling else "",
                                        "-num-samples-each-time-step-{}".format(args.num_samples_each_time_step) if args.historical_sampling else "",
                                        "-present-sampling" if args.present_sampling else "",
                                        "-one-hop-positive-sampling" if args.one_hop_positive_sampling else "",
                                        "-sample-positive" if args.sample_positive else "",
                                        "-sample-neg-relation" if args.sample_neg_relation else "",
                                        "-sample-neg-entity" if args.sample_neg_entity else "",
                                        "-neg-rate-reservoir-{}".format(args.negative_rate_reservoir) if args.sample_neg_entity else "",
                                        "-frequency-sampling" if args.frequency_sampling else "",
                                        "-inverse-frequency-sampling" if args.inverse_frequency_sampling else "",
                                        "-seed-{}".format(args.seed),
                                        "{}".format("-end-time-step-{}".format(args.end_time_step) if not args.load_base_model else "")
                                    )
    # TODO: adjust the naming function

    version = time.strftime('%Y%m%d%H%M')
    log_file_out = "logs/log-{}-{}".format(name, version)
    log_file_err = "errs/log-{}-{}".format(name, version)

    myhost = os.uname()[1]
    if myhost == 'gdl':
        experiment_path = "/media/data/jiapeng-yishi/"
    elif myhost == 'curie':
        experiment_path = "/data/jwu558/"

    if not args.debug:
        sys.stdout = open(log_file_out, 'w')
        sys.stderr = open(log_file_err, 'w')

    tt_logger = MyTestTubeLogger(
        save_dir=os.path.join(experiment_path, "experiments"),
        name=name,
        debug=False,
        version=version,
        create_git_tag=True
    )

    args.base_path = tt_logger.experiment.get_data_path(tt_logger.experiment.name, tt_logger.experiment.version)

    num_ents, num_rels, num_time_steps = get_total_number(args.dataset, 'stat.txt')
    # graph_dict_train, graph_dict_val, graph_dict_test = build_interpolation_graphs(args)

    total_time = np.array(list(range(num_time_steps)))
    module = {
              "Static": Static,
              "DE": DiachronicEmbedding,
              "hyte": Hyte
              }[args.module]

    print("\'{}\',".format(args.base_path))
    print(args)

    tt_logger.log_args(args)
    tt_logger.save()

    args.end_time_step = min(len(total_time), args.end_time_step + 1)
    args.train_base_model = args.train_base_model or args.end_time_step < len(total_time)
    # import pdb; pdb.set_trace()
    # print(len(total_time))
    # print(end_time_step)
    # print(args.train_base_model)
    # model = module(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
    model = module(args, num_ents, num_rels)
    if args.load_base_model:
        base_model_path = glob.glob(os.path.join(args.base_model_path, "*.ckpt"))[0]
        base_model_checkpoint = torch.load(base_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(base_model_checkpoint['state_dict'], strict=False)
        # pdb.set_trace()

    for time in range(args.start_time_step, args.end_time_step):
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
                          print_nan_grads=True,
                          # terminate_on_nan=True,
                          checkpoint_callback=checkpoint_callback
                          )

        if args.train_base_model:
            model.on_time_step_start(time)
            try:
                trainer.fit(model)
            except ValueError:
                pass

            model.load_best_checkpoint()
            test_res = trainer.test(model=model)
            model.on_time_step_end()
            break
        else:
            model.on_time_step_start(time)
            if not model.should_skip_training():
                trainer.fit(model)
                model.load_best_checkpoint()
            trainer.use_ddp = False
            test_res = trainer.test(model=model)
            model.on_time_step_end()
