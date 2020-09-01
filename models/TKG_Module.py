import numpy as np
from utils.scores import *
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from collections import OrderedDict
from torch.utils.data import DataLoader
from utils.dataset import FullBatchDataset
from utils.CorrptTriples import CorruptTriples
import torch.nn.functional as F
from utils.evaluation_filter import EvaluationFilter
from utils.metrics_collection import metric_collection, counter_gauge
import torch.nn as nn
from utils.utils import get_add_del_graph, get_metrics, get_know_entities_per_time_step
import os
import glob
import torch
import time


class TKG_Module(LightningModule):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(TKG_Module, self).__init__()
        self.args = self.hparams = args
        self.graph_dict_train = graph_dict_train
        self.graph_dict_val = graph_dict_val
        self.graph_dict_test = graph_dict_test
        self.known_entities = None
        self.total_time = np.array(list(graph_dict_train.keys()))
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
        self.multi_step = self.args.multi_step
        self.train_seq_len = self.args.train_seq_len if self.multi_step else 0
        self.corrupter = CorruptTriples(self)
        self.evaluater = EvaluationFilter(self)
        self.addition = args.addition
        self.deletion = args.deletion
        self.n_gpu = self.args.n_gpu
        if self.addition or self.deletion:
            self.added_graph_dict_train, self.deleted_graph_triples_train = get_add_del_graph(graph_dict_train)

        self.ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))
        self.init_parameters()

        self.self_kd = args.self_kd
        if self.self_kd:
            self.kd_factor = self.args.kd_factor
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
        self.train_graph = self.graph_dict_train[time]
        self.known_entities = self.inference_know_entities[time]

        if self.self_kd and time > 0:
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
        torch.cuda.synchronize()
        self.epoch_time += time.time() - self.batch_start_time

    def training_step(self, quadruples, batch_idx):
        loss = self.forward(quadruples)
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

        test_result = {'mrr': mrr.item(),
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
        self.accumulative_val_result.update({'mrr': mrr.item(),
                       'hit_10': hit_10.item(),
                       'hit_3': hit_3.item(),
                       'hit_1': hit_1.item()
                       })

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=0.0001)
        return optimizer

    def _dataloader(self, graph_dict, batch_size, train_seq_len, train=False):
        # when using multi-node (ddp) we need to add the datasampler

        dataset = FullBatchDataset(graph_dict, self.time, train_seq_len)

        should_shuffle = train and not self.use_ddp
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=0
        )

        return loader

    @pl.data_loader
    def train_dataloader(self):
        train_graph_dict = self.added_graph_dict_train if self.addition else self.graph_dict_train
        # print("At time step {}, number of edges: {}, number of added edges: {}"
        #       .format(self.time, len(self.graph_dict_train[self.time].edges()[0]),
        #               len(self.added_graph_dict_train[self.time].edges()[0])))
        return self._dataloader(train_graph_dict, self.args.train_batch_size, self.train_seq_len, train=True)
        # train_graph = self.added_train_graph if self.addition else self.train_graph
        # return self._dataloader(train_graph, self.args.train_batch_size, train=True)

    @pl.data_loader
    def val_dataloader(self):
        return self._dataloader(self.graph_dict_val, self.args.test_batch_size, 0)

    @pl.data_loader
    def test_dataloader(self):
        test_graph_dict = self.graph_dict_test if self.args.eval_on_test_set else self.graph_dict_val
        return self._dataloader(test_graph_dict, self.args.test_batch_size, 0)

    def collect_embedding_corrupt_tail(self, quadruples, neg_samples, ent_embed_all_time, all_embeds_g_all_time):
        subject_embedding = self.ent_embeds.new_zeros(len(quadruples), self.embed_size)
        neg_object_embedding = self.ent_embeds.new_zeros(*neg_samples.shape, self.embed_size)
        for i in range(len(quadruples)):
            s, _, _, t = quadruples[i]
            neg_o = neg_samples[i]
            s, t = s.item(), t.item()
            subject_embedding[i] = ent_embed_all_time[self.time - t][s]
            neg_object_embedding[i] = all_embeds_g_all_time[self.time - t][neg_o]
        return subject_embedding, neg_object_embedding

    def collect_embedding_corrupt_head(self, quadruples, neg_samples, ent_embed_all_time, all_embeds_g_all_time):
        object_embedding = self.ent_embeds.new_zeros(len(quadruples), self.embed_size)
        neg_subject_embedding = self.ent_embeds.new_zeros(*neg_samples.shape, self.embed_size)
        for i in range(len(quadruples)):
            _, _, o, t = quadruples[i]
            o, t = o.item(), t.item()
            neg_s = neg_samples[i]
            object_embedding[i] = ent_embed_all_time[self.time - t][o]
            neg_subject_embedding[i] = all_embeds_g_all_time[self.time - t][neg_s]
        return neg_subject_embedding, object_embedding

    def train_link_prediction_multi_step(self, ent_embed_all_time, all_embeds_g_all_time, quadruples, neg_samples, labels, corrupt_tail=True):
        relation_embedding = self.rel_embeds[quadruples[:, 1]]
        embed_collector = self.collect_embedding_corrupt_tail if corrupt_tail else self.collect_embedding_corrupt_head
        subject_embedding, object_embedding = embed_collector(quadruples, neg_samples, ent_embed_all_time, all_embeds_g_all_time)
        score = self.calc_score(subject_embedding, relation_embedding, object_embedding, mode='tail' if corrupt_tail else "head")
        return F.cross_entropy(score, labels.long())

    def train_link_prediction(self, ent_embed, quadruples, neg_samples, labels, all_embeds_g, corrupt_tail=True, loss='CE'):
        # neg samples are in global idx
        relation_embedding = self.rel_embeds[quadruples[:, 1]]
        subject_embedding = ent_embed[quadruples[:, 0]] if corrupt_tail else all_embeds_g[neg_samples]
        object_embedding = all_embeds_g[neg_samples] if corrupt_tail else ent_embed[quadruples[:, 2]]
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

    def set_known_entities_per_time_step(self, inference_know_entities):
        self.inference_know_entities = inference_know_entities

    def get_know_entities_per_time_step(self):
        self.inference_know_entities = get_know_entities_per_time_step(self.graph_dict_train, self.num_ents)