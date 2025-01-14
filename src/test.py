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

def run_test_of_single_fold(config, output_dir, fold_idx, data_loader):
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
        b_idx, s_idx, data, alert_keys = to_device(batch, device=device, training=False)

        outputs = model(b_idx, s_idx, data)
        outputs = outputs.detach().cpu().numpy().tolist()
        for alert_key, output in zip(alert_keys, outputs):
            ret[alert_key] = output

    return ret


def main(config, output_dir, num_fold=5):
    logger = config.get_logger('test')

    # setup data_loader instances
    config['data_loader']['args']['validation_split'] = False
    config['data_loader']['args']['training'] = False
    # config['data_loader']['args']['batch_size'] = 512
    config['data_loader']['args']['num_workers'] = 2

    data_loader = getattr(module_data, config['data_loader']['type'])(
        **config['data_loader']['args'],
    )

    # get output of 5fold
    outputs = {}
    for fold_idx in range(num_fold):

        out = run_test_of_single_fold(config, output_dir, fold_idx, data_loader)
        for k, v in out.items():
            if k not in outputs:
                outputs[k] = []
            outputs[k].append(v)

    # mean
    for k, v in outputs.items():
        outputs[k] = sum(v) / num_fold

    # generate submission by ensemble of 5folds
    all_alert_keys = pd.read_csv(sample_path).alert_key
    for alert_key in all_alert_keys:
        if alert_key not in outputs:
            outputs[alert_key] = 0

    submit = pd.DataFrame(
        data={
            'alert_key': list(outputs.keys()),
            'probability': list(outputs.values())
        }
    )
    submit['alert_key'] = submit['alert_key'].astype(int)
    submit.sort_values(by='probability', inplace=True)
    submit.to_csv(f'{output_dir}/submission_{args.suffix}.csv', index=None)
    print(f'Submission saved to {output_dir}/submission_{args.suffix}.csv')


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
    args.add_argument('-sp', '--sample_path', default='/home/nanaeilish/projects/Github/esun_sar_baseline/sample_submission.csv',
                      type=str, help='sample submission file for public and private Leaderboard')
    args.add_argument('-nf', '--n_fold', default=5, type=int)
    config = ConfigParser.from_args(args, test=True)
    args = args.parse_args()

    print(f'Run test for {args.output_dir}, information: {args.suffix}.')
    output_type = args.output_type
    output_dir = args.output_dir
    sample_path = args.sample_path
    n_fold = args.n_fold

    os.makedirs(output_dir, exist_ok=True)
    main(config, output_dir, num_fold=n_fold)
# mb activate esun-ai-open
# 1. process_data/data_config.py sar_flag那行要改成 TARGET type
# 2. 要確認test.py那裡有吃到
# py test.py -c /home/nanaeilish/projects/Github/esun_sar_baseline/save_dir/bigger_tp99/fold0/config.json -d 0 -s 5fold-ensemble-1202
