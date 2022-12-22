import argparse
import os
from pdb import set_trace as bp
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
# my lib
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from utils import to_device
import pickle
from model.metric import recall_n


def run_test_of_single_fold(config, output_dir, fold_idx, data_loader, logger):
    print(f'=== run fold {fold_idx} ===')

    # load model
    model = config.init_obj('arch', module_arch)
    # model = config.init_obj('arch', module_arch)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = f'{output_dir}/fold{fold_idx}/model_best.pth'
    print(f'load checkpoint from {model_path}')
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    ret = {}

    for batch_idx, batch in tqdm(enumerate(data_loader)):
        b_idx, c_idx, s_idx, data, alert_keys = to_device(batch, device=device, training=False)
        b2cmap = {}
        for bids, cids in zip(b_idx, c_idx):
            for bid, cid in zip(bids, cids):
                bid = bid.item()
                if bid in b2cmap:
                    assert b2cmap[bid] == cid
                b2cmap[bid] = cid

        outputs, tempagg_outs = model(b_idx, s_idx, data)
        tempagg_outs = tempagg_outs.detach().cpu().numpy().tolist()
        outputs = outputs.detach().cpu().numpy().tolist()
        for alert_key, output in zip(alert_keys, outputs):
            ret[alert_key] = output

        embed_path = f'./save_embeds/test'
        os.makedirs(embed_path, exist_ok=True)
        pklpath = f'{embed_path}/cust_embeds_batch{batch_idx}.pkl'
        cust_embeds = {b2cmap[id]: np.array(v) for id, v in enumerate(tempagg_outs)}
        with open(pklpath, 'wb') as f:
            pickle.dump(cust_embeds, f)
        logger.info(f'Testing: batch {batch_idx} saved to {pklpath}')
    return ret


def main(config, output_dir, fold_idx, answer_path):
    logger = config.get_logger('test')

    # setup data_loader instances
    config['data_loader']['args']['validation_split'] = False
    config['data_loader']['args']['training'] = False
    config['data_loader']['args']['batch_size'] = 32
    config['data_loader']['args']['num_workers'] = 2

    data_loader = getattr(module_data, config['data_loader']['type'])(
        **config['data_loader']['args'],
    )
    # get output of 5fold
    outputs = {}


    out = run_test_of_single_fold(config, output_dir, fold_idx, data_loader, logger)
    for k, v in out.items():
        if k not in outputs:
            outputs[k] = []
        outputs[k] = v

    # generate submission for this particular fold and calc scores (public answer is out)
    answer_df = pd.read_csv(answer_path)
    sample_df = pd.read_csv(sample_path)
    both_alert_keys = sample_df.alert_key
    for alert_key in both_alert_keys:
        if alert_key not in outputs:
            outputs[alert_key] = 0


    # calculate on public test set only
    outprobs = []
    targets = []
    for rid, row in answer_df.iterrows():
        alert_key = row.alert_key
        targets.append(row.sar_flag)
        outprobs.append(outputs[alert_key])
    recn = recall_n(outprobs, targets)
    print(f'* [fold {fold_idx}] public test Precision of recall@n-1: {recn}')


    # submit on public and private tests
    submit = pd.DataFrame(
        data={
            'alert_key': list(outputs.keys()),
            'probability': list(outputs.values())
        }
    )


    submit['alert_key'] = submit['alert_key'].astype(int)
    submit.sort_values(by='probability', inplace=True)
    assert len(submit) == len(both_alert_keys)
    assert set(submit.alert_key) == set(both_alert_keys)
    submit.to_csv(f'{output_dir}/submission_{args.suffix}.csv', index=None)
    # calculate scores on public test for this fold
    print(f'Public test submission saved to {output_dir}/submission_{args.suffix}.csv')



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--output_type', type=str, default='top3_indices', choices=['top3_indices', 'logits'])
    args.add_argument('-o', '--output_dir', default='/home/nanaeilish/projects/Github/esun_sar_baseline/save_dir/bigger_tp99', type=str,
                      help='output_dir')
    args.add_argument('-s', '--suffix', default='5fold-ensemble', type=str,
                      help='suffix information of the output file')
    args.add_argument('-sp', '--sample_path', default='/home/nanaeilish/projects/esun_sar_baseline/sample_submission.csv',
                      type=str, help='sample submission file for public and private Leaderboard')
    args.add_argument('-f', '--fold_idx', default=0, type=int)
    args.add_argument('-a', '--answer_path', default='/home/nanaeilish/projects/esun_sar_baseline/train_first/24_ESun_public_y_answer.csv')
    config = ConfigParser.from_args(args, test=True)
    args = args.parse_args()

    print(f'Run test for {args.output_dir}, information: {args.suffix}.')
    output_type = args.output_type
    output_dir = args.output_dir
    sample_path = args.sample_path
    fold_idx = args.fold_idx
    answer_path = args.answer_path

    os.makedirs(output_dir, exist_ok=True)
    main(config, output_dir, fold_idx=fold_idx, answer_path=answer_path)
# mb activate esun-ai-open
# 1. process_data/data_config.py sar_flag那行要改成 TARGET type
# 2. 要確認test.py那裡有吃到
# py test.py -c /home/nanaeilish/projects/Github/esun_sar_baseline/save_dir/bigger_tp99/fold0/config.json -d 0 -s 5fold-ensemble-1202
