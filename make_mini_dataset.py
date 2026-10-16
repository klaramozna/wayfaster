import os
import shutil
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

SOURCE_ROOT = "dataset/mydata"
OUTPUT_ROOT = "dataset/mini-data"

# Use this recording
RECORDING = "recording_20260817_113035"

# Number of matched image/state samples to keep
N_SAMPLES = 80

# Train/validation split
TRAIN_RATIO = 0.8


# ============================================================
# Paths
# ============================================================

source_recording = os.path.join(
    SOURCE_ROOT,
    "data_train",
    RECORDING
)

source_images_csv = os.path.join(
    source_recording,
    "images.csv"
)

source_states_csv = os.path.join(
    source_recording,
    "states.csv"
)


# ============================================================
# Check input
# ============================================================

if not os.path.exists(source_images_csv):
    raise FileNotFoundError(
        f"Could not find:\n{source_images_csv}"
    )

if not os.path.exists(source_states_csv):
    raise FileNotFoundError(
        f"Could not find:\n{source_states_csv}"
    )


# ============================================================
# Read CSV files
# ============================================================

print("Reading:")
print(source_images_csv)
print(source_states_csv)

images_df = pd.read_csv(source_images_csv)
states_df = pd.read_csv(source_states_csv)

print("\nOriginal image rows:", len(images_df))
print("Original state rows:", len(states_df))


# ============================================================
# Find timestamps that exist in BOTH CSV files
# ============================================================

state_timestamps = set(states_df["timestamp"])

matched_images = images_df[
    images_df["timestamp"].isin(state_timestamps)
].copy()

# Preserve original order
matched_images = matched_images.reset_index(drop=True)


print("\nMatched image/state rows:", len(matched_images))


if len(matched_images) < N_SAMPLES:
    raise RuntimeError(
        f"Only {len(matched_images)} matched samples exist, "
        f"but N_SAMPLES={N_SAMPLES}."
    )


# ============================================================
# Take first N matched samples
# ============================================================

selected_images = matched_images.iloc[:N_SAMPLES].copy()

selected_timestamps = set(selected_images["timestamp"])

selected_states = states_df[
    states_df["timestamp"].isin(selected_timestamps)
].copy()


# Keep state rows in timestamp order corresponding to images
state_order = {
    timestamp: i
    for i, timestamp in enumerate(selected_images["timestamp"])
}

selected_states["_order"] = selected_states["timestamp"].map(state_order)
selected_states = (
    selected_states
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)


print("\nSelected samples:", len(selected_images))
print("Selected states:", len(selected_states))


# ============================================================
# 80 / 20 split
# ============================================================

n_train = int(N_SAMPLES * TRAIN_RATIO)
n_valid = N_SAMPLES - n_train

train_images = selected_images.iloc[:n_train].copy()
valid_images = selected_images.iloc[n_train:].copy()

train_timestamps = set(train_images["timestamp"])
valid_timestamps = set(valid_images["timestamp"])

train_states = selected_states[
    selected_states["timestamp"].isin(train_timestamps)
].copy()

valid_states = selected_states[
    selected_states["timestamp"].isin(valid_timestamps)
].copy()


print("\n==============================")
print("SPLIT")
print("==============================")
print("Train images:", len(train_images))
print("Train states:", len(train_states))
print("Valid images:", len(valid_images))
print("Valid states:", len(valid_states))


# ============================================================
# Helper: create one recording
# ============================================================

def create_recording(split_name, images_df, states_df):

    output_split = os.path.join(
        OUTPUT_ROOT,
        split_name
    )

    output_recording = os.path.join(
        output_split,
        RECORDING
    )

    output_image_dir = os.path.join(
        output_recording,
        "image"
    )

    output_depth_dir = os.path.join(
        output_recording,
        "depth"
    )

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_depth_dir, exist_ok=True)


    # --------------------------------------------------------
    # Copy image/depth files
    # --------------------------------------------------------

    new_image_rows = []

    for _, row in images_df.iterrows():

        timestamp = row["timestamp"]

        image_rel = row["image"]
        depth_rel = row["depth"]

        source_image = os.path.join(
            source_recording,
            image_rel
        )

        source_depth = os.path.join(
            source_recording,
            depth_rel
        )

        if not os.path.exists(source_image):
            print("WARNING: missing image:", source_image)
            continue

        if not os.path.exists(source_depth):
            print("WARNING: missing depth:", source_depth)
            continue

        # Keep only the filename, e.g. 002682.tif
        image_name = os.path.basename(image_rel)
        depth_name = os.path.basename(depth_rel)

        destination_image = os.path.join(
            output_image_dir,
            image_name
        )

        destination_depth = os.path.join(
            output_depth_dir,
            depth_name
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        shutil.copy2(
            source_depth,
            destination_depth
        )

        # Rebuild CSV paths relative to recording
        new_row = row.copy()

        new_row["image"] = os.path.join(
            "image",
            image_name
        )

        new_row["depth"] = os.path.join(
            "depth",
            depth_name
        )

        new_image_rows.append(new_row)


    # --------------------------------------------------------
    # Write images.csv
    # --------------------------------------------------------

    new_images_df = pd.DataFrame(new_image_rows)

    images_csv = os.path.join(
        output_recording,
        "images.csv"
    )

    new_images_df.to_csv(
        images_csv,
        index=False
    )


    # --------------------------------------------------------
    # Write states.csv
    # --------------------------------------------------------

    states_csv = os.path.join(
        output_recording,
        "states.csv"
    )

    states_df.to_csv(
        states_csv,
        index=False
    )


    # --------------------------------------------------------
    # Write rosbags.csv
    # --------------------------------------------------------

    rosbags_csv = os.path.join(
        output_split,
        "rosbags.csv"
    )

    pd.DataFrame({
        "recording": [RECORDING]
    }).to_csv(
        rosbags_csv,
        index=False
    )


    print("\nCreated:", output_recording)
    print("  images.csv:", len(new_images_df), "rows")
    print("  states.csv:", len(states_df), "rows")


# ============================================================
# Create train
# ============================================================

create_recording(
    "data_train",
    train_images,
    train_states
)


# ============================================================
# Create validation
# ============================================================

create_recording(
    "data_valid",
    valid_images,
    valid_states
)


# ============================================================
# Done
# ============================================================

print("\n==============================")
print("MINI DATASET CREATED")
print("==============================")

print(f"Location: {OUTPUT_ROOT}")

print("\nTrain:")
print(f"  {OUTPUT_ROOT}/data_train/{RECORDING}")

print("\nValidation:")
print(f"  {OUTPUT_ROOT}/data_valid/{RECORDING}")

print("\nTotal:")
print("  samples:", N_SAMPLES)
print("  train:", n_train)
print("  valid:", n_valid)
