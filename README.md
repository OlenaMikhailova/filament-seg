# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in 2048 x 2048 GONG H-alpha images (MAGFiLO 1.0),
for the [Kaggle competition](https://www.kaggle.com/competitions/filament-segmentation-2026).

**Authors:** Olena Mikhailova (University of Bordeaux, IPCV-AI), Sofiia Mikhailova (Université Jean Monnet, iPSRS)

## Final result

| Submission | Model | Out-of-fold PQ (707 images) | Public LB |
| --- | --- | --- | --- |
| `sub_ens5_oofpost.csv` | 5 x U-Net ResNet34 (one per fold) | 0.448 (95% CI 0.440 to 0.457, fold std 0.009) | 0.39 |

## Reproduce the submission (no training needed)

The weights of the five final models are in `weights/` (`r34_f0_e80.pt`, `r34_f1.pt` ... `r34_f4.pt`),
and the post-processing parameters are in `configs/final_post.json`.

```bash
pip install -r requirements.txt
```

Then either open **`notebooks/pipeline.ipynb`**, set `DATA_ROOT` to the unzipped Kaggle data
folder and run all cells, or use the command line:

```bash
python scripts/predict.py \
    --checkpoints weights/r34_f0_e80.pt weights/r34_f1.pt weights/r34_f2.pt weights/r34_f3.pt weights/r34_f4.pt \
    --post configs/final_post.json \
    --images /path/MAGFiLO_1.0_Kaggle_2026/test/test_images \
    --out submission.csv
```

A GPU is recommended (we used an NVIDIA RTX 4000 Ada, 20 GB); CPU works but is slow.

## Method

1. **Preprocessing:** detect the solar disk (Otsu + minimum enclosing circle), divide by a masked
   Gaussian background to remove limb darkening, z-score inside the disk, map to 128 + 32z, 0 outside.
2. **Labels:** soft targets, the fraction of annotators who marked each pixel.
3. **Splits:** 5 folds with `StratifiedGroupKFold`, grouped by observation date, stratified by year.
4. **Model:** U-Net with ImageNet ResNet34 encoder (1 input channel), `segmentation_models_pytorch`.
5. **Training:** random 1024 px crops at full resolution (60% centred near filaments), rotations,
   flips, contrast/brightness/noise jitter, BCE + Dice, AdamW (lr 3e-4), cosine schedule, AMP, batch 8.
6. **Inference:** 1024 px tiles with 256 px overlap, Hann blending, flip TTA, average of the 5 fold models.
7. **Post-processing:** threshold 0.4, closing (9 x 9 ellipse) only to group fragments, connected
   components, drop instances with area < 200 px or mean probability < 0.9. Tuned for PQ on
   out-of-fold predictions of all 707 training images.

## Experiment log

Fold 0 (142 validation images), same post-processing; PQ with 95% bootstrap CI (`scripts/uncertainty.py`).

| Variant | PQ [95% CI] | SQ | RQ | Public LB |
| --- | --- | --- | --- | --- |
| U-Net ResNet34 (baseline) | 0.440 [0.419, 0.459] | 0.684 | 0.643 | 0.37 to 0.38 |
| + ridge / top-hat input channels | 0.439 [0.419, 0.459] | 0.678 | 0.648 | 0.38 |
| UNet++ decoder | 0.434 [0.415, 0.455] | 0.679 | 0.640 | 0.38 |
| + Lovász fine-tune (thr 0.5, no conf. filter) | 0.425 [0.406, 0.444] | 0.677 | 0.627 | 0.38 |
| DeepLabV3+ decoder | 0.412 [0.389, 0.433] | 0.681 | 0.604 | 0.37 |

All 5 folds (707 out-of-fold images):

| Setting | PQ [95% CI] | Fold mean ± std | Public LB |
| --- | --- | --- | --- |
| 5-fold U-Net ResNet34, scale 1 (**submitted**) | 0.448 [0.440, 0.457] | 0.448 ± 0.009 | **0.39** |
| Multi-scale TTA (scales 1 + 1.5) | 0.438 [0.428, 0.448] | 0.438 ± 0.010 | |
| 10-model ensemble (+ 5 x EfficientNet-B4) | | | 0.38 |
| 3 seeds trained on all 707 images | | | 0.39 |

Annotator agreement (296 images with 2 to 3 annotators): annotator vs annotator PQ 0.340,
model vs each annotator 0.447.

## Repository layout

| Path | Purpose |
| --- | --- |
| `notebooks/pipeline.ipynb` | End-to-end reproduction of the submission from the stored weights |
| `weights/` | Trained weights (fp16): final `r34_f*` models and all experiment variants |
| `configs/` | Training configs (YAML) and post-processing parameters (`final_post.json`) |
| `src/filament/preprocess.py` | Solar disk detection, limb-darkening flattening |
| `src/filament/data.py` | COCO JSON to GT table, date-grouped folds, label masks |
| `src/filament/dataset.py` | Random-crop training dataset with augmentation |
| `src/filament/model.py` | Models, losses, checkpoint I/O |
| `src/filament/train.py` | Training loop with PQ validation and resume |
| `src/filament/infer.py` | Tiled inference with Hann blending and TTA |
| `src/filament/postprocess.py` | Probability map to disjoint instances |
| `src/filament/metrics.py` | Official PQ (matches the organisers' notebook) |
| `src/filament/rle.py`, `submission.py` | COCO RLE and submission CSV writer/validator |
| `scripts/prepare.py` | Build preprocessed image and label caches, folds |
| `scripts/train.py` | Train one model from a config |
| `scripts/tune_postprocess.py` | Grid search of post-processing on out-of-fold PQ |
| `scripts/predict.py` | Predict a folder of images and write a submission |
| `scripts/uncertainty.py` | PQ / SQ / RQ, fold std and bootstrap CI per run |
| `scripts/make_report_figure.py`, `make_results_plot.py` | Figures for the report |

## Retrain from scratch

```bash
python scripts/prepare.py --data /path/MAGFiLO_1.0_Kaggle_2026 --cache cache --preprocess flat --labels soft
python scripts/train.py configs/baseline_r34.yaml fold=0 work_dir=runs/r34_f0   # repeat for folds 0 to 4
python scripts/tune_postprocess.py --runs runs/r34_f0 runs/r34_f1 runs/r34_f2 runs/r34_f3 runs/r34_f4 --cache cache
python scripts/export_weights.py runs/r34_f0/best.pt weights/r34_f0.pt
```

## License

MIT, see `LICENSE`.
