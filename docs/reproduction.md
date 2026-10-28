# Running the Released Code

## Scope

The release supports dataset checks, baseline training, compatible-checkpoint
inference, S-norm candidate generation, and fusion of aligned score lists.
It does not contain winning checkpoints, original component scores, full run
configurations, or the complete primary-ensemble recipe. It is not a turnkey
reproduction of 8.00% EER.

## Environment and Data

Use Python 3.10+ and `pip install -r requirements.txt` in a virtual environment.
Install PyTorch for your hardware. Dependencies are unpinned, so this is not a
locked historical environment. CPU inference uses `--device cpu`.

```text
/path/to/competition/
  1 - Training/
    001/BF/W1/metadata.csv
    001/BF/W1/pipeline_1.npz
    ...
    150/...
  2 - Reference/
    reference_data.npz
    reference_metadata.csv
  3 - Probe/
    probe_data.npz
    probe_metadata.csv
```

The loader expects all 150 training participants, footwear groups `BF`, `ST`,
`P1`, `P2`, and speeds `W1`-`W4`. Obtain data via the organizers. No dataset or
data credentials are included. Cohort inference can require substantial RAM and time.

## Commands

```bash
python code/stepup_runner.py check --dataset-root /path/to/competition

python code/stepup_runner.py train \
  --dataset-root /path/to/competition \
  --output-dir runs/circle_baseline \
  --sample-size 64,64,32 --loss circle \
  --epochs 10 --batch-size 128 --lr 0.0001 --lr-gamma 0.95 \
  --num-workers 0 --seed 1337

python code/stepup_runner.py infer \
  --dataset-root /path/to/competition \
  --checkpoint runs/circle_baseline/best.pt \
  --scoring prototype --threshold 0.7 \
  --output runs/prototype.zip

python code/make_v2_submissions.py \
  --dataset-root /path/to/competition \
  --checkpoint runs/circle_baseline/best.pt \
  --output-dir runs/circle_baseline/submissions \
  --sample-size 64,64,32 --batch-size 128 \
  --cohort-batch-size 128 --num-workers 0 --top-k 100 200 400
```

Start with zero data-loader workers and increase only after checking memory and
throughput. `--sample-size` is a tensor shape, not a number of training examples.
The CLI does not expose a tiny-subset training flag. Training logs validation
EER at epoch completion and writes `best.pt` and `last.pt`.

Despite its historical `v2` filename, the public candidate generator imports
`FusionNet` from `stepup_runner.py`. Checkpoints must match that architecture and
the expected 140-class output head. Historical v2 checkpoints with different
heads are not directly compatible.

The default cohort uses training strides from participants 1-140. Changing
`--cohort-size` changes scores. With `--reuse-embeds`, use a separate output
directory for each checkpoint and preprocessing configuration to avoid mixing
cached embeddings. The generator exports prototype/nearest scores and S-norm
mean/max candidates with raw and min-max variants.

## Historical Final Step

```bash
python code/fuse_rank_scores.py \
  --primary /path/to/primary_ensemble.zip \
  --secondary /path/to/secondary_top400_mean_snorm.zip \
  --secondary-weight 0.115 --threshold 0.5 \
  --output runs/SUBMIT_TOMORROW_ms400_weight_115.zip
```

Inputs are ZIPs containing `scores.txt` or plain numeric files. They must refer
to the same claims in the same order. Length equality is checked; alignment is
the caller's responsibility. Two-dimensional inputs are averaged row-wise.
The original submission has 10,000 scores.

The output ZIP contains `scores.txt` and `threshold.txt`; sidecar text files are
also written. Threshold `0.5` preserves historical packaging and is not a newly
optimized deployment threshold. The original input score streams are not bundled.

## Checks Without Competition Data

```bash
python -m compileall -q code
python -m unittest discover -s tests -v
```

Focused tests use synthetic scores to check weights, ties, and ZIP contents.
They do not verify the historical EER, model convergence, or checkpoint equivalence.
