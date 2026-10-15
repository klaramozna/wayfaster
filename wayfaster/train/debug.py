import torch
import numpy as np
import cv2

from types import SimpleNamespace
from train.dataloader import Dataset


# ============================================================
# Minimal config needed by Dataset
# ============================================================

configs = SimpleNamespace(
    TRAINING=SimpleNamespace(
        DT=0.1,
        HORIZON=300,
        VERBOSE=True,
    ),

    MODEL=SimpleNamespace(
        INPUT_SIZE=(320, 180),
        DOWNSAMPLE=8,
        TIME_LENGTH=1,
        PREDICT_DEPTH=True,

        GRID_BOUNDS={
            'xbound': [-10.0, 10.0, 0.2],
            'ybound': [-10.0, 10.0, 0.2],
            'zbound': [-2.0, 2.0, 0.2],
            'dbound': [0.0, 10.0, 0.2],
        },
    ),

    AUGMENTATIONS=SimpleNamespace(
        HORIZ_FLIP=0.0,
        PCLOUD_DROPOUT=0.0,
        MAX_TRANSLATION=0.0,
        MAX_ROTATION=0.0,
    ),

    DATASET=SimpleNamespace(
        CSV_FILE='rosbags.csv',
    ),
)


# ============================================================
# Helper
# ============================================================

def describe(name, x):

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    print(f"\n--- {name} ---")
    print("shape:", x.shape)
    print("dtype:", x.dtype)

    if x.size == 0:
        print("EMPTY")
        return

    print("min:", np.nanmin(x))
    print("max:", np.nanmax(x))
    print("mean:", np.nanmean(x))
    print("std:", np.nanstd(x))
    print("NaN:", np.isnan(x).sum())
    print("Inf:", np.isinf(x).sum())
    print("zero:", np.sum(x == 0))
    print("nonzero:", np.sum(x != 0))


# ============================================================
# Dataset paths
# ============================================================

train_paths = [
    "../dataset/mydata/data_train",
]


# ============================================================
# Create dataset
# ============================================================

print("\n==============================")
print("CREATING DATASET")
print("==============================")

dataset = Dataset(
    configs,
    train_paths,
    train=False,
)

print("\nDataset length:", len(dataset))


# ============================================================
# Examine ONE sample
# ============================================================

idx = 0

print("\n\n########################################")
print("SAMPLE", idx)
print("########################################")

sample = dataset[idx]

(
    color_img,
    pcloud,
    intrinsics,
    extrinsics,
    states,
    traversability,
    trav_weights,
    depth_target,
    depth_mask,
) = sample


# ============================================================
# Print statistics
# ============================================================

describe("color_img", color_img)
describe("pcloud", pcloud)
describe("intrinsics", intrinsics)
describe("extrinsics", extrinsics)
describe("states", states)
describe("traversability", traversability)
describe("trav_weights", trav_weights)
describe("depth_target", depth_target)
describe("depth_mask", depth_mask)


# ============================================================
# Depth target values
# ============================================================

print("\n==============================")
print("DEPTH TARGET VALUES")
print("==============================")

depth_np = depth_target.numpy()

vals = np.unique(depth_np)

print("Number of unique values:", len(vals))

print("\nFirst 30:")
print(vals[:30])

print("\nLast 30:")
print(vals[-30:])


# ============================================================
# Point cloud occupancy
# ============================================================

print("\n==============================")
print("POINT CLOUD")
print("==============================")

pcloud_np = pcloud.numpy()

print("Pcloud shape:", pcloud_np.shape)
print("Pcloud nonzero:", np.count_nonzero(pcloud_np))
print("Pcloud total:", pcloud_np.size)

if pcloud_np.size > 0:
    print(
        "Pcloud occupancy:",
        np.count_nonzero(pcloud_np) / pcloud_np.size
    )


# ============================================================
# RAW DEPTH + CAMERA PARAMETERS
# ============================================================

print("\n==============================")
print("RAW DEPTH CHECK")
print("==============================")

depth_file = dataset.rosbags[0]['depth_image'][0]

raw_depth = cv2.imread(depth_file, -1)

print("Depth file:")
print(depth_file)

if raw_depth is None:
    print("ERROR: cv2.imread returned None!")
else:
    print("shape:", raw_depth.shape)
    print("dtype:", raw_depth.dtype)
    print("min:", raw_depth.min())
    print("max:", raw_depth.max())
    print("mean:", raw_depth.mean())
    print("nonzero:", np.count_nonzero(raw_depth))
    print("total:", raw_depth.size)

    print("\nFirst 20 unique raw depth values:")
    raw_vals = np.unique(raw_depth)
    print(raw_vals[:20])

    print("\nLast 20 unique raw depth values:")
    print(raw_vals[-20:])


print("\n==============================")
print("INTRINSICS")
print("==============================")

print(intrinsics[0].numpy())


print("\n==============================")
print("EXTRINSICS")
print("==============================")

print(extrinsics[0].numpy())

print("\n==============================")
print("PCLOUD OCCUPIED VOXELS")
print("==============================")

pcloud_np = pcloud.numpy()[0]

occupied = np.argwhere(pcloud_np > 0)

print("Number occupied:", len(occupied))

if len(occupied) > 0:
    print("\nFirst occupied voxel indices [z, x, y]:")
    print(occupied[:30])

    print("\nIndex ranges:")
    print("z:", occupied[:, 0].min(), "to", occupied[:, 0].max())
    print("x:", occupied[:, 1].min(), "to", occupied[:, 1].max())
    print("y:", occupied[:, 2].min(), "to", occupied[:, 2].max())

print("\n==============================")
print("PCLOUD VOXEL -> METERS")
print("==============================")

xb = configs.MODEL.GRID_BOUNDS['xbound']
yb = configs.MODEL.GRID_BOUNDS['ybound']
zb = configs.MODEL.GRID_BOUNDS['zbound']

if len(occupied) > 0:
    z_idx = occupied[:, 0]
    x_idx = occupied[:, 1]
    y_idx = occupied[:, 2]

    z_m = zb[0] + (z_idx + 0.5) * zb[2]
    x_m = xb[0] + (x_idx + 0.5) * xb[2]
    y_m = yb[0] + (y_idx + 0.5) * yb[2]

    print("x meters:", x_m.min(), "to", x_m.max())
    print("y meters:", y_m.min(), "to", y_m.max())
    print("z meters:", z_m.min(), "to", z_m.max())    


print("\n==============================")
print("DONE")
print("==============================")