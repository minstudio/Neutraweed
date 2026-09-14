# NeutraWeed

Code for a controlled study of whether AI-generated weed imagery has to look
real to be useful as detector training data.

The short answer is that it does not. Interventions on generator and compositor
quality did not produce a consistent improvement in detection, while increasing
the number of distinct composited scenes did, at identical synthetic volume.

## Layout

The Python package lives under `src/`; `scripts/` holds the analysis and
orchestration entry points and is not part of the package. Everything resolves
relative to the repository root, which also holds `configs/`, `data/` and
`results/`:

```
Neutraweed/
├── src/            the package (see below)
├── scripts/        analysis entry points, plus slurm/ job files
├── configs/        base.yaml + experiment YAMLs
├── data/
│   ├── real/       images/ and labels/ per split, plus split.json
│   ├── synthetic/  <pool>/images, <pool>/labels
│   └── datasets/   assembled manifests: train.txt, val.txt, test.txt, dataset.yaml
└── results/        checkpoints/, tables/, tables/preds/, figures/
```

Override the root with `WEED_GENAI_ROOT` if the data lives elsewhere.

Package modules run with `-m` from the repository root; scripts run directly:

```bash
python -m src.datasets.build_hybrid --help
python scripts/ap_analysis.py --help
```

### Packages

| Package | Stage | Contents |
| --- | --- | --- |
| `src/data_prep` | A | VOC ingestion, split construction, tiling |
| `src/generators` | B | SD 3.5 cut-out and ControlNet backends, FLUX, GAN baseline, conditioning, scene compositing |
| `src/annotate` | C | label routes for synthetic scenes: BiRefNet matte, SAM2, ControlNet mask, real-cut-out compositing |
| `src/qc` | C | FID, CLIP-IQA, dedup, mask-IoU validation of auto-annotations |
| `src/datasets` | D | hybrid set assembly, size-matched controls |
| `src/detect` | E | YOLO11 / YOLO26 and RT-DETRv2 training under a frozen protocol |
| `src/eval` | F | Ultralytics metrics, paired seed statistics, plots |

### Configs

`configs/base.yaml` holds everything shared: the split, the tiling parameters,
the class list and the frozen detector block. An experiment under
`configs/experiments/` sets `extends: ../base.yaml` and overrides only what it
changes; `src/common/config.py` resolves one level of `extends` and deep-merges.

Split keys are `"<source>/<VOC folder>"`. Only `test_fields` and `val_fields`
are listed, and every field not named goes to training.

One experiment departs from the frozen protocol on purpose.
`real_only_rtdetrv2.yaml` trains with AdamW at `lr0: 0.0001` instead of the
shared `optimizer: auto` at `lr0: 0.01`, because DETR-family detectors
undertrain badly under the YOLO settings. Image size, epochs, patience and batch
stay shared. Report that arm as trained with its architecture's recommended
optimiser, not under identical hyperparameters.

`real_only_yolo26_rot.yaml` is also outside the frozen protocol: it enables
`degrees: 180.0` and `flipud: 0.5`, which are label-preserving on nadir imagery
and left at the Ultralytics defaults of zero everywhere else. Runs using it
compare only against each other and need their own real-only baseline.

## Data

The real imagery is the tomato-weed dataset of Gómez et al. (2025) and is **not**
redistributed here. Obtain it from its authors.

Five detection targets, in frozen class-id order: `SOLNI` (*Solanum nigrum*),
`POROL` (*Portulaca oleracea*), `SETVE` (*Setaria verticillata*), `CYPRO`
(*Cyperus rotundus*), `ECHCG` (*Echinochloa crus-galli*). The crop (`LYPES`) and
the "not recognised" placeholder (`NR`) are excluded by design; `LYPES` is
labelled only in the 2021 season, so keeping it would introduce a cross-season
annotation inconsistency. The order is frozen in `src/common/classes.py`: changing
it invalidates every checkpoint and label file.

The split holds out whole fields, so every reported figure measures
generalisation to a field the detector has never seen. Both seasons stay in
training because *Echinochloa crus-galli* is almost absent from the 2021
imagery. The fields assigned to validation and testing, and the tiling
parameters, are set in `configs/base.yaml`.

## Pipeline

**A. Prepare the real data.** `prepare_data` tiles from the raw VOC sources.
On a cluster holding only the converted tree, tile in place instead. Tiling
parameters come from `configs/base.yaml`, and the step is one-shot: running it
twice tiles the tiles.

```bash
python -m src.data_prep.make_splits
python -m src.data_prep.tile_dataset
```

**B/C. Generate and label.** Fine-tune the adapter on training-field instances
only, render single plants, matte them, and composite them onto real soil
backgrounds drawn from the training fields. No pixel from the validation or test
parcels reaches the generator at any stage.

**D. Assemble a training set.** `--ratio` is synthetic entries per real tile, so
`--ratio 2.00` requests two synthetic entries for every real tile.
`--strict-unique` refuses to build when that would require repeating pool
entries, which upweights images rather than adding information.

```bash
python -m src.datasets.build_hybrid --name real_only --composition real-only

python -m src.datasets.build_hybrid --name hybrid_v2b_r200 \
    --composition hybrid --generator sd35cut_lora_v2b --ratio 2.00 --seed 0

python -m src.datasets.build_hybrid --name hybrid_v9_r125 \
    --composition hybrid --generator sd35cut_v9 --ratio 1.25 \
    --max-overlap 0.50 --seed 0
```

`--max-overlap` drops training tiles where more than that fraction of the plants
overlap another at IoU >= 0.10. It applies to the real and the synthetic half
alike, and it must be matched by `--drop-overlap` at scoring time; filtering one
side only would train on dense clusters and then decline to score them.

Validation and test always point at the real split. That invariant is enforced
in `build_hybrid`, not left to the caller.

**E. Train.** Every run shares one frozen config; only the dataset and the seed
change. Set `--epochs` explicitly when comparing arms, because a walltime kill
stops a larger dataset at fewer epochs and silently confounds the ratio sweep
with training length.

```bash
python -m src.detect.train_yolo --config experiments/yolo26.yaml \
    --data data/datasets/hybrid_v2b_r200/dataset.yaml --name hybrid_v2b_r200 \
    --seed 0 --epochs 25
```

Interrupted runs resume from `last.pt`; a checkpoint truncated by a walltime
kill is detected and discarded rather than crashing the array cell.

**F. Evaluate.** Evaluation is two steps, because the test field is small and
cannot be made larger without breaking the split. The detections are cached once
on GPU, then re-analysed for free on CPU.

```bash
python scripts/cache_preds.py --pattern 'hybrid_yolo26_sd35cut_lora_v2b_r200_seed*' --split test
```

That writes `results/tables/preds/<run>_<split>.npz` holding predicted and
ground-truth boxes in original image pixels. Everything below reads those
caches, touches no checkpoint, and is safe on a login node.

```bash
# AP by class and object size, bootstrap CIs over (tile, seed) units
python scripts/ap_analysis.py --runs 'hybrid_yolo26_sd35cut_lora_v2b_r200_seed*' --split test

# paired A/B: the same tiles are resampled for both arms, so tile difficulty cancels
python scripts/ap_analysis.py --runs 'hybrid_yolo26_sd35cut_lora_v2b_r200_seed*' \
    --vs 'realonly_yolo26_seed*' --split test

# where each class fails: FN, FP-Cls, FP-Loc, FP-Bkg
python scripts/error_breakdown.py --runs 'realonly_yolo26_seed*' --split test
```

Options that carry the measurement decisions:

| Flag | Effect |
| --- | --- |
| `--bins` | object-size boundaries in px |
| `--boot` | bootstrap resamples; `0` disables |
| `--match-mode ioa` | match on intersection over ground-truth area, used for *Cyperus rotundus* |
| `--fp-bkg-verified P` | credit a fraction `P` of background false positives as real plants |
| `--drop-overlap F` | exclude crowded tiles from scoring; must match `--max-overlap` at build time |
| `--valid-regions` | restrict scoring to the annotated inter-row region |

`src/eval` is the lighter path: `metrics` wraps Ultralytics `val` and appends a
row per run to `results/tables/metrics.csv`, and `stats` runs a paired t-test
and a Wilcoxon signed-rank test over that file across seeds.

The whole sweep can be driven end to end. `--dry-run` prints the plan without
training, and `--build-only` assembles the datasets so a job array can train
them.

```bash
python scripts/run_sweep.py --generator sd35cut_lora_v2b --detector yolo26 \
    --ratios 0.25:2.0:0.25 --seeds 0 1 2 --max-overlap 0.50 --dry-run
```

`scripts/slurm/` holds the job files these were run under, including
`prebuild_sweep.slurm` and `train_sweep_array.slurm`.

## Not yet in this repository

A dependency manifest pinning at least `ultralytics==8.4.68`, since the
optimiser selection described below is version-specific.

## Measurement conventions

Two properties of this benchmark change the numbers and are worth stating
whenever results from it are reported.

**Nutsedge is boxed inconsistently.** A single *Cyperus rotundus* rosette is
often annotated as several boxes. A prediction covering the whole clump then
overlaps each box weakly and scores as a false positive while the annotations
score as missed. Scoring this species on intersection over the *ground-truth
area* instead of intersection over union raises its AP substantially, with no
change to the model.

**Annotation is restricted to the inter-row area.** Plants outside it are real
but unlabelled, so detections on them count as false positives. Blind inspection
of random samples found that the large majority contain the weed the model
named. Results are reported both under the benchmark convention and with respect
to the plants actually present, across a range of assumed verification rates
rather than at one assumed value.

## Environment

Training and evaluation used Ultralytics 8.4.68. In this version
`optimizer=auto` selects MuSGD above 10,000 optimiser steps and AdamW at or
below, where steps are `ceil(n_train / 64) x epochs`. Since `n_train` depends on
the synthetic-to-real ratio, two arms of one contrast can resolve to different
optimisers. Comparisons in the paper are matched on this, and the few that are
not are excluded and identified.

## Citation

Please cite the paper (see `CITATION.cff`) and the underlying dataset:

> Gómez, S. et al. (2025). Agricultural Systems.
> https://doi.org/10.1016/j.agsy.2025.104394

## Funding

Carried out under the NeutraWeed project, funded by the European Union under the
HORIZON-MSCA-2023-SE-01-01 MSCA Staff Exchanges 2023 call, Grant Agreement
No 101182891. Views and opinions expressed are those of the author only and do
not necessarily reflect those of the European Union or the granting authority.
Neither the European Union nor the granting authority can be held responsible
for them.

## Licence

MIT. See `LICENSE`.
