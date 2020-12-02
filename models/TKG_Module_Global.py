import numpy as np
from utils.scores import *
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from collections import OrderedDict
from utils.dataset import TrainDataset, ValDataset, \
    load_quadruples_tensor, init_data_loader, base_model_data_loader, dataloader_wrapper
from utils.CorruptTriplesGlobal import CorruptTriplesGlobal
import torch.nn.functional as F
from utils.evaluation_filter_global import EvaluationFilterGlobal
from utils.evaluation_filter_global_atise import EvaluationFilterGlobalAtiSE
from utils.metrics_collection import metric_collection, counter_gauge
import torch.nn as nn
from utils.util_functions import get_add_del_graph_global, get_metrics, collect_one_hot_neighbors_global, \
    get_known_entities_relations_per_time_step_global, get_common_triples_adjacent_time_global
import os
import glob
import torch
import time
from utils.reservoir_sampler import DeletedEdgeReservoir, ReservoirSampler
import baselines
from collections import defaultdict


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
        # self.evaluater = EvaluationFilterGlobal(self) if not isinstance(self, baselines.ATiSE.ATiSE) else EvaluationFilterGlobalAtiSE(self)

        self.addition = args.addition
        self.deletion = args.deletion

        self.sample_positive = self.args.sample_positive
        self.sample_neg_entity = self.args.sample_neg_entity
        self.sample_neg_relation = self.args.sample_neg_relation
        self.n_gpu = self.args.n_gpu

        self.deleted_edges_reservoir = DeletedEdgeReservoir(args, self.time2quads_train)
        if self.addition:
            self.added_edges_dict, _ = get_add_del_graph_global(self.time2quads_train)
        if self.addition and self.args.present_sampling:
            self.common_triples_dict = get_common_triples_adjacent_time_global(self.time2quads_train)

        self.ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))
        self.init_parameters()

        self.get_known_entities_relation_per_time_step()
        self.self_kd = args.self_kd
        self.reservoir_sampling = self.args.KD_reservoir or self.args.CE_reservoir
        self.self_kd_factor = self.args.self_kd_factor

        self.frequency_sampling = args.frequency_sampling
        self.inverse_frequency_sampling = args.inverse_frequency_sampling

        if self.args.historical_sampling:

            self.reservoir_sampler = ReservoirSampler(args, self.time2quads_train)

        if self.reservoir_sampling or self.self_kd :
            self.old_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size), requires_grad=False)
            self.old_rel_embeds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size), requires_grad=False)

    def init_metrics_collection(self):
        self.accumulative_val_result = {"mrr": 0, "hit_1": 0, "hit_3": 0, "hit_10": 0, "all_ranks": None}
        self.metrics_collector = metric_collection(self.args, self.args.base_path)
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

        self.eval_subject_relation_dict = defaultdict(int)
        self.eval_object_relation_dict = defaultdict(int)
        # print("Number of known entities up to time step {}: {}".format(self.time, len(self.known_entities)))
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
        print(self.epoch_time)
        self.epoch_time_gauge.add(self.epoch_time)

    # def on_batch_start(self, batch):

    def on_batch_end(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.epoch_time += time.time() - self.batch_start_time
        # print(time.time() - self.batch_start_time)

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

        log_output = OrderedDict({
            'mean_ranks': ranks.float().mean().item(),
            # 'val_loss': loss,
        })
        output = OrderedDict({
            'ranks': ranks,
            # 'val_loss': loss
        })

        # if self.args.debug:
        #     output['quads'] = torch.cat([quadruples, quadruples])

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
        output = {}

        if not self.args.train_base_model:
            raw_ranks, first_positive_ranks, both_positive_ranks, relative_ranks, \
                    deleted_facts_ranks, ranks, times = self.test_func(quadruples, batch_idx)
            output['times'] = times
            if type(first_positive_ranks) != type(None):
                output['raw_ranks'] = raw_ranks
                output['first_positive_ranks'] = first_positive_ranks
                output['both_positive_ranks'] = both_positive_ranks
                output['relative_ranks'] = relative_ranks
                output['deleted_facts_ranks'] = deleted_facts_ranks

        else:
            ranks = self.evaluate(quadruples, batch_idx)

        output['ranks'] = ranks
        log_output = OrderedDict({
            'mean_ranks': ranks.float().mean().item(),
        })

        self.logger.experiment.log(log_output)
        return output

    def test_epoch_end(self, outputs):
        # avg_test_loss = np.mean([x['test_loss'].item() for x in outputs])
        all_ranks = torch.cat([x['ranks'] for x in outputs])
        if not self.args.train_base_model:
            raw_ranks = torch.cat([x['raw_ranks'] for x in outputs if 'raw_ranks' in x])
            first_positive_ranks = torch.cat([x['first_positive_ranks'] for x in outputs if 'first_positive_ranks' in x])
            both_positive_ranks = torch.cat([x['both_positive_ranks'] for x in outputs if 'both_positive_ranks' in x])
            relative_ranks = torch.cat([x['relative_ranks'] for x in outputs if 'relative_ranks' in x])
            deleted_facts_ranks = torch.cat([x['deleted_facts_ranks'] for x in outputs if 'deleted_facts_ranks' in x])

            mrr, hit_1, hit_3, hit_10 = get_metrics(raw_ranks)
            self.metrics_collector.update_raw_ranks(self.time,
                       mrr.item(), hit_1.item(), hit_3.item(), hit_10.item())

            mrr, hit_1, hit_3, hit_10 = get_metrics(first_positive_ranks)
            self.metrics_collector.update_first_positive_ranks(self.time,
                       mrr.item(), hit_1.item(), hit_3.item(), hit_10.item())

            mrr, hit_1, hit_3, hit_10 = get_metrics(both_positive_ranks)
            self.metrics_collector.update_both_positive_ranks(self.time,
                       mrr.item(), hit_1.item(), hit_3.item(), hit_10.item())

            mrr, hit_1, hit_3, hit_10 = get_metrics(deleted_facts_ranks)
            self.metrics_collector.update_deleted_facts_ranks(self.time,
                       mrr.item(), hit_1.item(), hit_3.item(), hit_10.item())

            self.metrics_collector.update_mean_relative_ranks(self.time, torch.mean(relative_ranks.float()).item())

            all_times = torch.cat([x['times'] for x in outputs])
            for t in range(self.args.start_time_step, self.args.end_time_step):
                ranks_t = all_ranks[all_times == t]
                mrr, hit_1, hit_3, hit_10 = get_metrics(ranks_t)
                if t == self.time:
                    # pdb.set_trace()
                    self.metrics_collector.update_eval_metrics(self.time, mrr.item(),
                                                   hit_1.item(), hit_3.item(), hit_10.item())

                    self.update_accumulator(ranks_t)
                    test_result = {
                        'mrr': mrr.item(),
                        'hit_10': hit_10.item(),
                        'hit_3': hit_3.item(),
                        'hit_1': hit_1.item()
                    }
                self.metrics_collector.update_diff_time_eval_results(self.time, t, mrr, hit_1, hit_3, hit_10)

        else:
            mrr, hit_1, hit_3, hit_10 = get_metrics(all_ranks)
            self.metrics_collector.update_eval_metrics(self.time, mrr.item(), hit_1.item(), hit_3.item(), hit_10.item())
            test_result = {
                'mrr': mrr.item(),
                'hit_10': hit_10.item(),
                'hit_3': hit_3.item(),
                'hit_1': hit_1.item()
            }

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
        return torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=0.0001)

    def should_skip_training(self):
        return self.addition and len(self.added_edges_dict[self.time]) == 0 \
               and not self.deletion and not self.reservoir_sampling

    def _dataloader_train(self):
        # when using multi-node (ddp) we need to add the datasampler
        # TODO: reservoir sampling
        train_dataset_dict = {}
        train_quads = self.added_edges_dict[self.time] if self.addition else self.time2quads_train[self.time]
        training_data_size = 0
        if len(train_quads) > 0:
            train_dataset_dict['train'] = TrainDataset(train_quads)
            training_data_size += len(train_quads) * self.negative_rate * 2

        if self.deletion:
            # and len(self.deleted_edges_dict[self.time]) > 0:
            # train_dataset_dict['deleted'] = TrainDataset(self.deleted_edges_dict[self.time])
            # we hope to get all deleted edges here
            train_dataset_dict['deleted'] = self.deleted_edges_reservoir.sample_deleted_edges_train(self.time)
            training_data_size += len(train_dataset_dict['deleted'])

        if self.reservoir_sampling:
            reservoir_quads = []
            if self.args.historical_sampling:

                historical_quads = self.reservoir_sampler.sample(self.time)
                reservoir_quads.append(historical_quads)

            if self.args.present_sampling:
                involved_entities = torch.unique(torch.cat([self.added_edges_dict[self.time][:, 0],
                                                    self.added_edges_dict[self.time][:, 2]])).tolist()
                present_quads = collect_one_hot_neighbors_global(self.common_triples_dict[self.time], involved_entities,
                                                                    self.args.one_hop_positive_sampling, self.args.train_batch_size)
                reservoir_quads.append(present_quads)

            train_dataset_dict['reservoir'] = TrainDataset(torch.cat(reservoir_quads))
            multiple = 1 if not self.sample_neg_entity else self.args.negative_rate_reservoir

            training_data_size += len(train_dataset_dict['reservoir']) * multiple * 2
        self.metrics_collector.update_training_data_size(self.time, training_data_size)

        return dataloader_wrapper(train_dataset_dict, self.args.train_batch_size)

    @pl.data_loader
    def train_dataloader(self):
        if self.args.train_base_model:
            return base_model_data_loader(self.time2quads_train, self.args.train_batch_size, self.args.end_time_step, train=True)
        else:
            if self.args.all_prev_time_steps:
                return base_model_data_loader(self.time2quads_train, self.args.train_batch_size, self.time + 1, train=True)
            else:
                return self._dataloader_train()

    @pl.data_loader
    def val_dataloader(self):
        if self.args.train_base_model:
            return base_model_data_loader(self.time2quads_val, self.args.test_batch_size, self.args.end_time_step)
        else:
            dataset = ValDataset(self.time2quads_val, self.time)
            return init_data_loader(dataset, self.args.test_batch_size, False)

    @pl.data_loader
    def test_dataloader(self):
        time2triples_test = self.time2quads_test if self.args.test_set else self.time2quads_val
        if self.args.train_base_model:
            return base_model_data_loader(time2triples_test, self.args.test_batch_size, self.args.end_time_step)
        else:
            dataset_dict = {t: ValDataset(time2triples_test, t) for t in range(self.args.start_time_step, self.args.end_time_step)}
            # dataset_dict['deleted'] = TrainDataset(self.deleted_edges_reservoir.get_deleted_edges_val(self.time))
            return dataloader_wrapper(dataset_dict, self.args.test_batch_size, False)

    def train_link_prediction(self, subject_embedding, relation_embedding, object_embedding, labels, corrupt_tail=True):
        # neg samples are in global idx
        score = self.calc_score(subject_embedding, relation_embedding, object_embedding, mode='tail' if corrupt_tail else 'head')
        return F.cross_entropy(score, labels.long())

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
            get_known_entities_relations_per_time_step_global(self.time2quads_train,
                            self.time2quads_val, self.time2quads_test, self.num_ents, self.num_rels)
