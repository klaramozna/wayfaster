# WayFASTER Evaluation

This guide evaluates a trained checkpoint without using Weights & Biases.
Run the commands from the `wayfaster` repository root.

## 1. Select A Checkpoint

Checkpoint filenames contain the validation loss recorded at the end of an epoch.
For the latest run, the best checkpoint is:

```text
checkpoints/checkpoint_epoch=0-valid_loss=0.2040.ckpt
```

`last-v8.ckpt` is the final saved state, but it is not the best validation checkpoint.
Quote checkpoint paths because the filename contains `=` characters.

## 2. Activate The Training Environment

Use the same Python environment that was used for training. Verify that it has
PyTorch installed:

```bash
python -c "import torch; print(torch.__version__)"
```

The repository layout requires the inner `wayfaster` package on `PYTHONPATH`.

Linux/macOS:

```bash
export PYTHONPATH="$PWD/wayfaster"
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\wayfaster"
```

## 3. Evaluate Validation Data

Evaluate the complete ordered validation set:

```bash
python scripts/evaluate.py \
  --cfg_file configs/temporal_model.yaml \
  --checkpoint "checkpoints/checkpoint_epoch=0-valid_loss=0.2040.ckpt" \
  --device cuda
```

Use CPU when CUDA is unavailable:

```bash
python scripts/evaluate.py \
  --cfg_file configs/temporal_model.yaml \
  --checkpoint "checkpoints/checkpoint_epoch=0-valid_loss=0.2040.ckpt" \
  --device cpu
```

For a quick smoke test, evaluate only 20 samples:

```bash
python scripts/evaluate.py \
  --cfg_file configs/temporal_model.yaml \
  --checkpoint "checkpoints/checkpoint_epoch=0-valid_loss=0.2040.ckpt" \
  --device cuda \
  --max_samples 20
```

The script prints:

- Number of evaluated validation samples
- Average traversability loss
- Average traversability error

The `valid_loss` in the checkpoint filename is the total training objective,
which also includes the weighted depth loss. The current evaluator reports the
traversability component, so its printed loss is not necessarily identical to
the filename value.

## 4. Generate Predicted Maps

Generate predictions for the validation set and save them as compressed NumPy
files:

```bash
python scripts/infer.py \
  --cfg_file configs/temporal_model.yaml \
  --checkpoint "checkpoints/checkpoint_epoch=0-valid_loss=0.2040.ckpt" \
  --output_dir outputs/infer_best \
  --max_batches 20 \
  --device cuda
```

Remove `--max_batches 20` to process the complete validation set. The output
files are written to:

```text
outputs/infer_best/trav_map_000000.npz
outputs/infer_best/trav_map_000001.npz
...
```

Each file contains a `trav` array with shape `(2, X, Y)`:

- Channel `0`: `mu`
- Channel `1`: `nu`
- Values are in the range `[0, 1]`

## 5. Render Predictions As PNGs

Convert the NumPy predictions into viewable images:

```bash
python scripts/visualize_predictions.py outputs/infer_best --max-maps 20
```

PNG files are saved under:

```text
outputs/infer_best/png/
```

Each PNG shows the predicted `mu` and `nu` bird's-eye-view maps.

## 6. Compare Checkpoints

Run the evaluation command once for each checkpoint you want to compare. For
example:

```bash
python scripts/evaluate.py \
  --cfg_file configs/temporal_model.yaml \
  --checkpoint "checkpoints/checkpoint_epoch=0-valid_loss=0.2040.ckpt" \
  --device cuda

python scripts/evaluate.py \
  --cfg_file configs/temporal_model.yaml \
  --checkpoint "checkpoints/checkpoint_epoch=1-valid_loss=0.2456.ckpt" \
  --device cuda
```

Use the same dataset, device, and sample limit for every checkpoint.

## Troubleshooting

### `No module named torch`

The active Python interpreter is not the training environment. Activate the
training environment and verify it with:

```bash
python -c "import torch; print(torch.__version__)"
```

### `No module named train`

Set `PYTHONPATH` as shown in Section 2, then run the command from the repository
root.

### No prediction files are created

Check that the checkpoint exists and that the command points to the ordered
validation data configured in `configs/temporal_model.yaml`.

## Future Training Logs

Future runs use Lightning's local CSV logger instead of Weights & Biases. Their
metrics will be stored under:

```text
logs/wayfaster/version_*/metrics.csv
```

These files contain epoch-level training and validation metrics for plotting or
comparison after a run.
