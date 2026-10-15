#!/usr/bin/env python3
import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np

from train.train_configs import get_cfg
from train.dataloader import Dataset
from train.trainer import TrainingModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg_file', required=True, help='path to yaml config file')
    p.add_argument('--checkpoint', required=True, help='path to checkpoint .ckpt or .pth')
    p.add_argument('--device', default='cuda', help='device to run on (cuda or cpu)')
    p.add_argument('--max_samples', type=int, default=-1, help='max number of samples to evaluate (<=0 for all)')
    return p.parse_args()


def main():
    args = parse_args()
    configs = get_cfg(args.cfg_file)

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

    # Datasets
    valid_dataset = Dataset(configs, configs.DATASET.VALID_DATA)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load module
    module = TrainingModule(configs)
    chk = torch.load(args.checkpoint, map_location='cpu')
    state = chk['state_dict'] if 'state_dict' in chk else chk
    try:
        module.load_state_dict(state)
    except Exception:
        new_state = {k.replace('model.', ''): v for k, v in state.items()}
        module.load_state_dict(new_state)

    module.to(device)
    module.eval()

    total_loss = 0.0
    total_error = 0.0
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            if 0 < args.max_samples <= i:
                break

            color_img, pcloud, intrinsics, extrinsics, states, target_trav, trav_weights, depth_target, depth_mask = batch
            color_img = color_img.to(device)
            pcloud = pcloud.to(device)
            intrinsics = intrinsics.to(device)
            extrinsics = extrinsics.to(device)
            target_trav = target_trav.to(device)
            trav_weights = trav_weights.to(device)

            trav_map, pred_depth, debug = module.model(color_img, pcloud, intrinsics, extrinsics, depth_target)

            # Use TrainingModule's trav_criterion to compute loss and error consistent with training
            loss, err = module.trav_criterion(states.to(device), trav_map, target_trav, trav_weights)
            total_loss += loss.item()
            total_error += err.item()
            count += 1

            if i % 50 == 0:
                print(f'Processed {i} samples — loss {loss.item():.4f} error {err.item():.4f}')

    if count > 0:
        print('Evaluation results:')
        print(f'  Samples: {count}')
        print(f'  Avg trav loss: {total_loss/count:.6f}')
        print(f'  Avg trav error: {total_error/count:.6f}')
    else:
        print('No samples evaluated.')


if __name__ == '__main__':
    main()
