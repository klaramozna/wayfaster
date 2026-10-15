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
    p.add_argument('--output_dir', default='outputs/infer', help='where to save predicted maps')
    p.add_argument('--max_batches', type=int, default=-1, help='max number of batches to run (<=0 for all)')
    p.add_argument('--device', default='cuda', help='device to run on (cuda or cpu)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    configs = get_cfg(args.cfg_file)

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

    # Prepare dataset and loader (validation)
    valid_dataset = Dataset(configs, configs.DATASET.VALID_DATA)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Build module and load weights
    module = TrainingModule(configs)
    chk = torch.load(args.checkpoint, map_location='cpu')
    # Try to support both Lightning and pure-state checkpoints
    if 'state_dict' in chk:
        state = chk['state_dict']
    else:
        state = chk
    try:
        module.load_state_dict(state)
    except Exception:
        # Try to remove "model." prefix if present (saved from LightningModule.model)
        new_state = {}
        for k,v in state.items():
            nk = k.replace('model.', '')
            new_state[nk] = v
        module.load_state_dict(new_state)

    module.eval()
    module.to(device)

    idx = 0
    with torch.no_grad():
        for batch in valid_loader:
            if 0 <= args.max_batches == idx:
                break

            # Unpack batch
            color_img, pcloud, intrinsics, extrinsics, states, target_trav, trav_weights, depth_target, depth_mask = batch

            # Move to device
            color_img = color_img.to(device)
            pcloud = pcloud.to(device)
            intrinsics = intrinsics.to(device)
            extrinsics = extrinsics.to(device)

            trav_map, pred_depth, debug = module.model(color_img, pcloud, intrinsics, extrinsics, depth_target)

            # Save traversability map (B,2,X,Y) as numpy
            trav_np = trav_map.detach().cpu().numpy()
            for b in range(trav_np.shape[0]):
                out_file = os.path.join(args.output_dir, f'trav_map_{idx:06d}.npz')
                np.savez_compressed(out_file, trav=trav_np[b])
                print('Saved', out_file)
                idx += 1


if __name__ == '__main__':
    main()
