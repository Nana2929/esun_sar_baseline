from ast import get_source_segment
from os import device_encoding
from pdb import set_trace as bp


from easydict import EasyDict as edict

import torch
from torch.utils.data import Dataset

from base import BaseDataLoader
from process_data.data_config import CONFIG_MAP, DATA_SOURCES
from process_data.utils import load_pickle, get_feats_name

# def padding_mask_collate(batch):
#     """
#     return
#     """


def batch_index_collate(data):
    data = list(zip(*data))
    y = torch.stack(data[1], 0)

    batch_indices = []
    for i, d in enumerate(data[0]):
        batch_indices += [i] * int(d.size()[0])

    return (
        (torch.Tensor(batch_indices), torch.cat(data[0]).float()),
        y.float()
    )


class MaxLenDataLoader(BaseDataLoader):
    class InnerDataset(Dataset):
        def __init__(self, data_path, max_len=512, training=True, num_data=-1):
            self.training = training
            self.max_len = max_len
            self.num_data = num_data if num_data != -1 else 30_000
            self.load_xs(data_path)

        def load_xs(self, data_path):
            print(f'loading data from {data_path}')
            pkl = load_pickle(data_path)
            self.data = []
            data_count = 0
            # 7708 unique customers, 7264 in training data
            for cust_id, v in pkl.items():
                masks = v.train_mask if self.training else v.test_mask
                for e in masks:
                    e += 1
                    s = max(e - self.max_len, 0)
                    self.data.append(edict({
                        'sources': v.sources[s:e],
                        'cust_data': v.cust_data[s:e],
                        'cust_id': cust_id,
                    }))
                    data_count += 1
                    if data_count >= self.num_data:
                        print(f'num of data: {len(self.data)}')
                        return
            print(f'num of data: {len(self.data)}')

        def __len__(self):
            return len(self.data)

        def get_source_data(self, datas, sources, source_type):
            ret = []
            config = CONFIG_MAP[source_type]
            max_seq_idx = len(sources) - 1
            feats_name = get_feats_name(config)
            for seq_idx, (source, data) in enumerate(zip(sources, datas)):
                # seq_idx: 該feature_row在cust_data中的位置（和其他table混雜在一起）
                # source: 該feature_row來自哪個table
                if source != source_type:
                    continue
                # 最後一個seq_idx 對應的feature row應該是 DataSource.CUSTINFO
                # 在src/analysis_preprocess.ipynb中有把sar_flag merge進來
                # 為最後一個feature
                # 故這段是在檢查是否是 cust_info 中最後一個 feature row 且 feature name 是 sar_flag
                # 如果是的話則給2的值 （unknown）
                d = [data[feat_name] if not (max_seq_idx == seq_idx and feat_name ==
                                             'sar_flag') else 2 for feat_name in feats_name]
                ret.append((seq_idx, d))  # feature_row_index, feature_row as a list
            return ret

        def __getitem__(self, i):
            data = self.data[i]
            # an alert-key transaction
            # the associated customer's row sources: a list of table sources [cdtx, custinfo, dp, dp, ...]
            sources = data.sources
            # the associated customer's feature rows: a list of feature rows [{a row in cdtx}, {a row in custinfo}, ...]
            cust_data = data.cust_data
            cust_id = data.cust_id
            # src/process_data/data_config.py
            # DATA_SOURCES = [DataSource.CCBA, DataSource.CDTX, DataSource.DP, DataSource.REMIT, DataSource.CUSTINFO]
            # flatten the feature rows
            x = [self.get_source_data(cust_data, sources, ds) for ds in DATA_SOURCES]

            if self.training:
                y = cust_data[-1].sar_flag
            else:
                y = cust_data[-1].alert_key

            return [x, y, cust_id]

    class BatchCollate:
        def __init__(self, max_len=512, training=True):
            self.max_len = max_len
            self.training = training

        def __call__(self, datas):
            xs, ys, cids = list(zip(*datas))
            batch_idxs = [[] for _ in range(5)]
            cust_ids = [[] for _ in range(5)]  # 12/16 for idx recovery and embedding lookup
            seq_idxs = [[] for _ in range(5)]
            ret_xs = [[] for _ in range(5)]

            for batch_idx, (x, cid) in enumerate(zip(xs, cids)):
                for i, xi in enumerate(x):
                    for seq_idx, v in xi:
                        # v: a feature row of a customer corresponding to the alert_key
                        batch_idxs[i].append(batch_idx)
                        cust_ids[i].append(cid)
                        seq_idxs[i].append(seq_idx)
                        ret_xs[i].append(v)

            if self.training:
                ys = torch.tensor(ys).float()
            return [
                [torch.tensor(b).long() for b in batch_idxs],
                cust_ids,
                [torch.tensor(s).long() for s in seq_idxs],
                [torch.tensor(x).float() for x in ret_xs],  # (ccba, cdtx, dp, remit, cinfo),
                ys
            ]

    def __init__(self,
                 data_path, max_len=512,
                 batch_size=128, shuffle=True, fold_idx=-1, validation_split=0.0, num_workers=1, training=True):
        cls = self.__class__
        self.dataset = cls.InnerDataset(data_path, training=training)
        self.training = training
        super().__init__(
            self.dataset,
            batch_size, shuffle, fold_idx, validation_split, num_workers,
            collate_fn=cls.BatchCollate(max_len, training)
        )
