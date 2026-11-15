# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in 2048 x 2048 GONG H-alpha images (MAGFiLO 1.0),
for the [Kaggle competition](https://www.kaggle.com/competitions/filament-segmentation-2026) (IEEE BigData Cup 2026).

## Method in one paragraph

A U-Net (ResNet34 encoder, ImageNet weights adapted to 1 input channel) predicts a filament
probability map. Inputs are preprocessed by masking the solar disk and dividing by a smooth
background (removes limb darkening). Targets are soft labels: the fraction of annotators who
marked each pixel. Training uses random 1024 px crops at full resolution with BCE + Dice.
Inference uses overlapping 1024 px tiles with Hann blending and flip TTA. The probability map
becomes instances by thresholding, grouping nearby fragments with a morphological closing
(the closing only decides grouping; closed pixels are not submitted), and removing small or
low-confidence instances. All post-processing parameters are tuned on out-of-fold PQ.

## Metric

`src/filament/metrics.py` reproduces the organizers' self-evaluation notebook: Panoptic
Quality pooled over annotator-images, a match when IoU > 0.5,
PQ = sum(IoU of TP) / (TP + 0.5 FP + 0.5 FN). It computes IoU on RLEs directly, so it is much
faster than the notebook, and `scripts/sanity_check.py` checks it against a dense version.

## Layout

| Path | Purpose |
| --- | --- |
| `src/filament/data.py` | COCO JSON to GT table (official id format), date-grouped folds, label masks |
| `src/filament/rle.py` | Mask and polygon to COCO RLE counts |
| `src/filament/metrics.py` | Official PQ, diagnostics (1:n, n:1, Dice) |
| `src/filament/preprocess.py` | Solar disk detection, limb-darkening flattening |
| `src/filament/dataset.py` | Random-crop training dataset with augmentation |
| `src/filament/model.py` | Model, loss, checkpoint I/O |
| `src/filament/infer.py` | Sliding-window inference with TTA |
| `src/filament/postprocess.py` | Probability map to disjoint instances |
| `src/filament/submission.py` | CSV writer and validator (no overlaps, no quotes) |
| `src/filament/train.py` | Training loop with PQ validation |
| `scripts/` | Command-line entry points (below) |
| `configs/` | Experiment configs (YAML) |

## Reproduce

```bash
pip install -r requirements.txt

# 1. Prepare caches, GT table and folds (once)
python scripts/prepare.py --data /path/MAGFiLO_1.0_Kaggle_2026 --cache cache --preprocess flat --labels soft
python scripts/sanity_check.py --cache cache

# 2. Train (fold 0), then tune post-processing on its validation probabilities
python scripts/train.py configs/baseline_r34.yaml data_root=/path/MAGFiLO_1.0_Kaggle_2026
python scripts/tune_postprocess.py --runs runs/r34_f0 --cache cache

# 3. Export compact weights and predict the test set
python scripts/export_weights.py runs/r34_f0/best.pt weights/r34_f0.pt
python scripts/predict.py --checkpoints weights/r34_f0.pt --post runs/r34_f0/best_post.json \
    --images /path/MAGFiLO_1.0_Kaggle_2026/test/test_images --out submission.csv
```

On a SLURM cluster: `sbatch slurm/train.sbatch configs/baseline_r34.yaml fold=0 work_dir=runs/r34_f0`.

## Experiment log

| Date | Run | Change | Val PQ (fold 0) | Public LB |
| --- | --- | --- | --- | --- |
| | r34_f0 | baseline | | |

## License

MIT
