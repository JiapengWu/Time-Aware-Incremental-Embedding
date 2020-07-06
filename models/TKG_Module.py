import numpy as np
from utils.scores import *
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from collections import OrderedDict
from torch.utils.data import DataLoader
from utils.dataset import TimeDataset
from utils.CorrptTriples import CorruptTriples
import torch.nn.functional as F
from utils.evaluation import EvaluationFilter
from utils.metrics_collection import metric_collection
import torch.nn as nn


class TKG_Module(LightningModule):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type=EvaluationFilter):
        super(TKG_Module, self).__init__()
        self.args = self.hparams = args
        self.graph_dict_train = graph_dict_train
        self.graph_dict_val = graph_dict_val
        self.graph_dict_test = graph_dict_test

        self.total_time = np.array(list(graph_dict_train.keys()))
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.use_cuda = args.use_cuda
        self.negative_rate = args.negative_rate
        self.calc_score = {'distmult': distmult, 'complex': complex, 'transE': transE}[args.score_function]
        self.build_model()
        self.train_seq_len = self.args.train_seq_len
        self.corrupter = CorruptTriples(self.args, graph_dict_train)
        self.evaluater = evaluater_type(args, self.calc_score, graph_dict_train, graph_dict_val, graph_dict_test)
        self.accumulative_val_result = {"mrr": 0, "hit_1": 0, "hit_3": 0, "hit_10": 0, "all_ranks": None}
        self.metrics_collector = metric_collection(args.base_path)

        self.ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))

        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))

    def reset_time(self, time):
        self.time = time
        self.train_graph = self.graph_dict_train[time]
        self.val_graph = self.graph_dict_val[time]
        self.test_graph = self.graph_dict_test[time]

    def training_step(self, triples, batch_idx):
        loss = self.forward(triples)
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

    def validation_step(self, triples, batch_idx):
        # gc.collect()
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        ranks = self.evaluate(triples, batch_idx)

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
        mrr, hit_1, hit_3, hit_10 = self.get_metrics(all_ranks)

        return {'mrr': mrr,
                # 'avg_val_loss': avg_val_loss,
                'hit_10': hit_10,
                'hit_3': hit_3,
                'hit_1': hit_1
                }

    def test_step(self, triples, batch_idx):
        ranks = self.evaluate(triples, batch_idx)

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
        mrr, hit_1, hit_3, hit_10 = self.get_metrics(all_ranks)

        self.metrics_collector.update(self.time, mrr.item(), hit_1.item(), hit_3.item(), hit_10.item())
        self.metrics_collector.save()

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
        mrr, hit_1, hit_3, hit_10 = self.get_metrics(self.accumulative_val_result['all_ranks'])
        self.accumulative_val_result.update({'mrr': mrr.item(),
                       'hit_10': hit_10.item(),
                       'hit_3': hit_3.item(),
                       'hit_1': hit_1.item()
                       })

    def get_metrics(self, ranks):
        mrr = torch.mean(1.0 / ranks.float())
        hit_1 = torch.mean((ranks <= 1).float())
        hit_3 = torch.mean((ranks <= 3).float())
        hit_10 = torch.mean((ranks <= 10).float())
        return mrr, hit_1, hit_3, hit_10

    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=0.0001)
        return optimizer

    def _dataloader(self, graph, batch_size, train=False):
        # when using multi-node (ddp) we need to add the datasampler
        dataset = TimeDataset(graph)

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
        return self._dataloader(self.train_graph, self.args.train_batch_size, train=True)

    @pl.data_loader
    def val_dataloader(self):
        return self._dataloader(self.val_graph, self.args.test_batch_size)

    @pl.data_loader
    def test_dataloader(self):
        return self._dataloader(self.val_graph, self.args.test_batch_size)

    def train_link_prediction(self, ent_embed, triplets, neg_samples, labels, all_embeds_g, corrupt_tail=True):
        r = self.rel_embeds[triplets[:, 1]]
        if corrupt_tail:
            s = ent_embed[triplets[:, 0]]
            neg_o = all_embeds_g[neg_samples]
            score = self.calc_score(s, r, neg_o, mode='tail')
        else:
            neg_s = all_embeds_g[neg_samples]
            o = ent_embed[triplets[:, 2]]
            score = self.calc_score(neg_s, r, o, mode='head')
        predict_loss = F.cross_entropy(score, labels.long())
        return predict_loss

    def link_classification_loss(self, ent_embed, rel_embeds, triplets, labels):
        # triplets is a list of extrapolation samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        s = ent_embed[triplets[:, 0]]
        r = rel_embeds[triplets[:, 1]]
        o = ent_embed[triplets[:, 2]]
        score = self.calc_score(s, r, o)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        return predict_loss

    def get_batch_graph_list(self, t_list, seq_len, graph_dict):
        times = list(graph_dict.keys())
        # time_unit = times[1] - times[0]  # compute time unit
        time_list = []

        t_list = t_list.sort(descending=True)[0]
        g_list = []
        # s_lst = [t/15 for t in times]
        # print(s_lst)
        for tim in t_list:
            # length = int(tim / time_unit) + 1
            # cur_seq_len = seq_len if seq_len <= length else length
            length = times.index(tim) + 1
            time_seq = times[length - seq_len:length] if seq_len <= length else times[:length]
            time_list.append(([None] * (seq_len - len(time_seq))) + time_seq)
            g_list.append(([None] * (seq_len - len(time_seq))) + [graph_dict[t] for t in time_seq])
        t_batched_list = [list(x) for x in zip(*time_list)]
        g_batched_list = [list(x) for x in zip(*g_list)]
        return g_batched_list, t_batched_list

