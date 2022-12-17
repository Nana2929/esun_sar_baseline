from pprint import pprint

import numpy as np
import torch
import pickle
import os


from tqdm import tqdm

from base import BaseTrainer
from utils import inf_loop, MetricTracker, to_device
from model.metric import recall_n


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.training = data_loader.training
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        bs = data_loader.batch_size
        self.accumulation_step = 1 if bs > 256 else np.ceil(256 / bs)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    # def to_device(self, x, training=True):
    #     (b_idx, s_idx, data, target) = x
    #     return [
    #         [b.to(self.device) for b in b_idx],
    #         [s.to(self.device) for s in s_idx],
    #         [d.to(self.device) for d in data],
    #         target.to(self.device) if training else target
    #     ]

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()

        self.train_metrics.reset()
        embed_path = f'./save_embeds/epoch{epoch}'
        os.makedirs(embed_path, exist_ok=True)

        with tqdm(total=self.len_epoch) as pbar:
            targets = []
            outputs = []

            cust_embeds = []
            last_batch_idx = len(self.data_loader)
            for batch_idx, (batch) in enumerate(self.data_loader):
                # data, target = to_device(data, self.device), to_device(target, self.device)
                b_idx, c_idx, s_idx, data, target = to_device(batch, device=self.device, training=True)
                b2cmap = {}
                for bids, cids in zip(b_idx, c_idx):
                    for bid, cid in zip(bids, cids):
                        bid = bid.item()
                        if bid in b2cmap:
                            assert b2cmap[bid] == cid
                        b2cmap[bid] = cid
                # pprint(b2cmap)
                # c_idx is not tensor, just list of list of cust_ids
                self.optimizer.zero_grad()
                output, tempagg_outs = self.model(b_idx, s_idx, data)
                tempagg_outs = tempagg_outs.detach().cpu().numpy().tolist()

                pklpath = f'{embed_path}/cust_embeds_epoch{epoch}_batch{batch_idx}.pkl'
                cust_embeds = {b2cmap[id]: np.array(v) for id, v in enumerate(tempagg_outs)}
                with open(pklpath, 'wb') as f:
                    self.logger.info(f'{epoch} Saving batch {batch_idx} cust_embeddings ...')
                    pickle.dump(cust_embeds, f)

                loss = self.criterion(output, target)
                loss.backward()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss.item())

                pbar.set_description(
                    f"Train Epoch: {epoch} Loss: {loss.item():.6f}"
                )

                if (batch_idx + 1) % self.accumulation_step == 0 or batch_idx == last_batch_idx:
                    self.optimizer.step()
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()

                pbar.update()

                if batch_idx == self.len_epoch:
                    break

                targets += target.detach().cpu().numpy().tolist()
                outputs += output.detach().cpu().numpy().tolist()

            self.train_metrics.update('recall_n', recall_n(outputs, targets))
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            targets = []
            outputs = []
            for batch_idx, (batch) in enumerate(self.valid_data_loader):
                b_idx, c_idx, s_idx, data, target = to_device(batch, device=self.device, training=True)

                output, tempagg_outs = self.model(b_idx, s_idx, data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                targets += target.detach().cpu().numpy().tolist()
                outputs += output.detach().cpu().numpy().tolist()
            self.valid_metrics.update('recall_n', recall_n(outputs, targets))

            # for met in self.metric_ftns:
            #     self.valid_metrics.update(met.__name__, met(output, target))

            # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    # def to_device(self, data):
    #     if isinstance(data, tuple) or isinstance(data, list):
    #         return (d.to(self.device) for d in data)
    #     return data.to(self.device)

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
