import numpy as np
from utils.scores import *
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from collections import OrderedDict
from torch.utils.data import DataLoader
from utils.dataset import BaseModelDataset, TrainDataset, ValDataset, DataLoaderWrapper
from utils.CorruptTriplesGlobal import CorruptTriplesGlobal
import torch.nn.functional as F
from utils.evaluation_filter_global import EvaluationFilterGlobal
from utils.metrics_collection import metric_collection, counter_gauge
import torch.nn as nn
from utils.utils import get_add_del_graph_global, get_metrics, collect_one_hot_neighbors_global, \
    get_known_entities_relations_per_time_step_global, get_common_triples_adjacent_time_global
from utils.dataset import load_quadruples_tensor
import os
import glob
import torch
import time
import math
from utils.reservoir_sampler import time_window_random_historical_sampling
import pdb


class TKG_Module_Global(LightningModule):
    def __init__(self, args, num_ents, num_rels):
        super(TKG_Module_Global, self).__init__()
        self.args = self.hparams = args
        self.known_entities = None
        self.time2quads_train, self.time2quads_val, self.time2quads_test = \
            load_quadruples_tensor(args.dataset, 'train.txt', 'valid.txt', 'test.txt')
        self.total_time = np.array(list(self.time2quads_train.keys()))
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.use_cuda = args.use_cuda
        self.negative_rate = args.negative_rate
        self.calc_score = {'distmult': distmult, 'complex': complex, 'transE': transE, 'atise':ATiSE_score}[args.score_function]
        self.build_model()
        if not self.args.inference:
            self.init_metrics_collection()

        self.corrupter = CorruptTriplesGlobal(self)
        self.evaluater = EvaluationFilterGlobal(self)
        self.addition = args.addition
        self.deletion = args.deletion
        self.sample_positive = self.args.sample_positive
        self.sample_neg_entity = self.args.sample_neg_entity
        self.sample_neg_relation = self.args.sample_neg_relation
        self.n_gpu = self.args.n_gpu
        if self.addition or self.deletion:
            self.added_edges_dict, self.deleted_edges_dict = get_add_del_graph_global(self.time2quads_train)
        if self.addition and self.args.present_sampling:
            self.common_triples_dict = get_common_triples_adjacent_time_global(self.time2quads_train)
        self.ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))

        self.init_parameters()

        self.get_known_entities_relation_per_time_step()
        self.self_kd = args.self_kd
        self.reservoir_sampling = self.args.KD_reservoir or self.args.CE_reservoir
        # self.use_kd = self.self_kd or self.positive_kd or self.sample_neg_entity or self.sample_neg_relation
        self.self_kd_factor = self.args.self_kd_factor

        if self.reservoir_sampling or self.self_kd :
            self.old_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size), requires_grad=False)
            self.old_rel_embeds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size), requires_grad=False)

    def init_metrics_collection(self):
        self.accumulative_val_result = {"mrr": 0, "hit_1": 0, "hit_3": 0, "hit_10": 0, "all_ranks": None}
        self.metrics_collector = metric_collection(self.args.base_path)
        self.epoch_time_gauge = counter_gauge()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))

    def load_best_checkpoint(self):
        load_path = glob.glob(os.path.join(os.path.join(self.args.base_path, "snapshot-{}").format(self.time), "*.ckpt"))[0]
        checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['state_dict'])

    def on_time_step_start(self, time):
        self.time = time
        # self.train_graph = self.graph_dict_train[time]
        self.known_entities = self.all_known_entities[time]
        self.known_relations = self.all_known_relations[time]
        self.corrupter.set_known_entities()
        if (self.reservoir_sampling or self.self_kd) and time > 0:
            self.load_old_parameters()
        print("Number of known entities up to time step {}: {}".format(self.time, len(self.known_entities)))
        # self.reduced_ent_embeds = self.ent_embeds[self.known_entities]

    def on_time_step_end(self):
        if self.args.cold_start:
            self.init_parameters()

        self.metrics_collector.update_eval_accumulated_metrics(self.accumulative_val_result)
        self.metrics_collector.update_time(self.time, self.epoch_time_gauge)
        self.metrics_collector.save()
        self.epoch_time_gauge.reset()

        print("Accumulative results:")
        for i in "mrr", "hit_1", "hit_3", "hit_10":
            print("{}: {}".format(i, self.accumulative_val_result[i]))

    def on_epoch_start(self):
        self.epoch_time = 0

    def on_epoch_end(self):
        self.epoch_time_gauge.add(self.epoch_time)

    def on_batch_end(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.epoch_time += time.time() - self.batch_start_time

    def training_step(self, quadruples, batch_idx):
        forward_func = self.forward_global if self.args.all_prev_time_steps or self.args.train_base_model else self.forward
        loss = forward_func(quadruples)
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        self.logger.experiment.log(output)
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, quadruples, batch_idx):
        # gc.collect()
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        ranks = self.evaluate(quadruples, batch_idx)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        # if self.trainer.use_dp or self.trainer.use_ddp2:
        #     loss = loss.unsqueeze(0)

        log_output = OrderedDict({
            'mean_ranks': ranks.float().mean().item(),
            # 'val_loss': loss,
        })
        output = OrderedDict({
            'ranks': ranks,
            # 'val_loss': loss
        })
        self.logger.experiment.log(log_output)
        return output

    def validation_epoch_end(self, outputs):
        # avg_val_loss = np.mean([x['val_loss'].item() for x in outputs])
        all_ranks = torch.cat([x['ranks'] for x in outputs])
        mrr, hit_1, hit_3, hit_10 = get_metrics(all_ranks)

        return {'mrr': mrr,
                # 'avg_val_loss': avg_val_loss,
                'hit_10': hit_10,
                'hit_3': hit_3,
                'hit_1': hit_1
                }

    def test_step(self, quadruples, batch_idx):
        ranks = self.evaluate(quadruples, batch_idx)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        # if self.trainer.use_dp or self.trainer.use_ddp2:
        #     loss = loss.unsqueeze(0)

        log_output = OrderedDict({
            'mean_ranks': ranks.float().mean().item(),
            # 'test_loss': loss,
        })

        output = OrderedDict({
            'ranks': ranks,
            # 'test_loss': loss,
        })
        self.logger.experiment.log(log_output)

        return output

    def test_epoch_end(self, outputs):
        # avg_test_loss = np.mean([x['test_loss'].item() for x in outputs])
        all_ranks = torch.cat([x['ranks'] for x in outputs])
        mrr, hit_1, hit_3, hit_10 = get_metrics(all_ranks)

        self.metrics_collector.update_eval_metrics(self.time, mrr.item(), hit_1.item(), hit_3.item(), hit_10.item())

        test_result = {
                        'mrr': mrr.item(),
                        # 'avg_test_loss': avg_test_loss.item(),
                        'hit_10': hit_10.item(),
                        'hit_3': hit_3.item(),
                        'hit_1': hit_1.item()
                        }

        self.update_accumulator(all_ranks)
        return test_result

    def update_accumulator(self, test_ranks):
        if type(self.accumulative_val_result['all_ranks']) == type(None):
            self.accumulative_val_result['all_ranks'] = test_ranks
        else:
            self.accumulative_val_result['all_ranks'] = torch.cat([self.accumulative_val_result['all_ranks'], test_ranks])
        mrr, hit_1, hit_3, hit_10 = get_metrics(self.accumulative_val_result['all_ranks'])
        self.accumulative_val_result.update({
                       'mrr': mrr.item(),
                       'hit_10': hit_10.item(),
                       'hit_3': hit_3.item(),
                       'hit_1': hit_1.item()
                       })

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=0.0001)
        return optimizer

    def should_skip_training(self):
        return self.addition and len(self.added_edges_dict[self.time]) == 0 \
               and not self.deletion and not self.reservoir_sampling

    def _dataloader(self, dataset, batch_size, should_shuffle):

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
        )

    def _concat_dataloader(self, train_dataset_dict, train_dataset_sizes, original_batch_size):
        dataloader_dict = {}

        quadruple_size_sum = np.sum(list(train_dataset_sizes.values()))
        num_batch = math.ceil(quadruple_size_sum / original_batch_size)
        for key, dataset in train_dataset_dict.items():
            batch_size = math.ceil(len(dataset) / num_batch)
            dataloader_dict[key] = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
            )
        return DataLoaderWrapper(dataloader_dict)

    def _dataloader_train(self):
        # when using multi-node (ddp) we need to add the datasampler
        # TODO: reservoir sampling
        train_dataset_dict = {}
        train_dataset_sizes = {}
        train_quads = self.added_edges_dict[self.time] if self.addition else self.time2quads_train[self.time]

        if len(train_quads) > 0:
            train_dataset_dict['train'] = TrainDataset(train_quads)
            train_dataset_sizes['train'] = len(train_quads)
            # train_dataset_sizes['train'] = len(train_quads) * self.args.negative_rate * 2

        # pdb.set_trace()
        if self.deletion and len(self.deleted_edges_dict[self.time]) > 0:
            train_dataset_dict['deleted'] = TrainDataset(self.deleted_edges_dict[self.time])
            train_dataset_sizes['deleted'] = len(self.deleted_edges_dict[self.time])

        if self.reservoir_sampling:
            reservoir_quads = []
            if self.args.historical_sampling:

                historical_quads = time_window_random_historical_sampling(self.time2quads_train,
                        self.time, self.args.train_seq_len, self.args.num_samples_each_time_step)
                reservoir_quads.append(historical_quads)

            if self.args.present_sampling:
                involved_entities = torch.unique(torch.cat([self.added_edges_dict[self.time][:, 0],
                                                    self.added_edges_dict[self.time][:, 2]])).tolist()
                present_quads = collect_one_hot_neighbors_global(self.common_triples_dict[self.time], involved_entities,
                                                                    self.args.one_hop_positive_sampling, self.args.train_batch_size)
                reservoir_quads.append(present_quads)

            train_dataset_dict['reservoir'] = TrainDataset(torch.cat(reservoir_quads))
            train_dataset_sizes['reservoir'] = len(reservoir_quads)
            # train_dataset_sizes['reservoir'] = len(reservoir_quads) * self.args.negative_rate_reservoir * 2

        return self._concat_dataloader(train_dataset_dict, train_dataset_sizes, self.args.train_batch_size)

    def _dataloader_val(self, quads_dict):
        # when using multi-node (ddp) we need to add the datasampler
        dataset = ValDataset(quads_dict, self.time)
        return self._dataloader(dataset, self.args.test_batch_size, False)

    def _dataloader_base(self, time2quads, batch_size, end_time_step, train=False):
        # when using multi-node (ddp) we need to add the datasampler
        dataset = BaseModelDataset(time2quads, end_time_step)
        should_shuffle = train and not self.use_ddp
        return self._dataloader(dataset, batch_size, should_shuffle)

    @pl.data_loader
    def train_dataloader(self):
        if self.args.train_base_model:
            return self._dataloader_base(self.time2quads_train, self.args.train_batch_size, self.args.end_time_step, train=True)
        else:
            if self.args.all_prev_time_steps:
                return self._dataloader_base(self.time2quads_train, self.args.train_batch_size, self.time + 1, train=True)
            else:
                return self._dataloader_train()

    @pl.data_loader
    def val_dataloader(self):
        if self.args.train_base_model:
            return self._dataloader_base(self.time2quads_val, self.args.test_batch_size, self.args.end_time_step)
        else:
            return self._dataloader_val(self.time2quads_val)

    @pl.data_loader
    def test_dataloader(self):
        time2triples_test = self.time2quads_test if self.args.eval_on_test_set else self.time2quads_val
        if self.args.train_base_model:
            return self._dataloader_base(time2triples_test, self.args.test_batch_size, self.args.end_time_step)
        else:
            return self._dataloader_val(time2triples_test)

    def train_link_prediction(self, subject_embedding, relation_embedding, object_embedding, labels, corrupt_tail=True, loss='CE'):
        # neg samples are in global idx
        score = self.calc_score(subject_embedding, relation_embedding, object_embedding, mode='tail' if corrupt_tail else 'head')
        if loss == 'CE':
            return F.cross_entropy(score, labels.long())
        elif loss == 'margin':
            pos_score = score[:, 0].unsqueeze(-1).repeat(1, self.negative_rate)
            neg_score = score[:, 1:]
            return torch.sum(- F.logsigmoid(1 - pos_score) - F.logsigmoid(neg_score - 1))
        else:
            raise NotImplementedError

    def link_classification_loss(self, ent_embed, rel_embeds, triplets, labels):
        # triplets is a list of extrapolation samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        s = ent_embed[triplets[:, 0]]
        r = rel_embeds[triplets[:, 1]]
        o = ent_embed[triplets[:, 2]]
        score = self.calc_score(s, r, o)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        return predict_loss

    def loss_fn_kd(self, outputs, teacher_outputs):
        return F.kl_div(F.log_softmax(outputs, dim=1), F.softmax(teacher_outputs, dim=1), reduction='mean')

    def get_known_entities_relation_per_time_step(self):
        self.all_known_entities, self.all_known_relations = \
            get_known_entities_relations_per_time_step_global(self.time2quads_train, self.time2quads_val, self.time2quads_test)
