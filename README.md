# StepUP 2026 Footstep Recognition

Code for the ArogyaPandit Research Team submission to the 2nd International
StepUP Competition for Biometric Footstep Recognition at IJCB 2026.

The final visible CodaBench result for our best submission,
`SUBMIT_TOMORROW_ms400_weight_115.zip`, was 8.00% EER, pending the organizers'
final leaderboard review.

## Team

- Anant Gupta
- Gourav Gupta

Affiliation: Arogyapandit Private Limited, India

## Method Summary

The system learns stride-level biometric embeddings from paired left and right
pressure footsteps. Each foot is processed by a shared 3D CNN backbone. The two
foot embeddings are concatenated, projected to a 512-dimensional stride
embedding, and L2-normalized for verification.

Training used the provided competition training data only. Participants 1-140
were used for training/cohort generation and participants 141-150 were reserved
for subject-disjoint validation. The strongest models used Circle Loss
(`m=0.25`, `gamma=256`) and prototype scoring against each claimed user's
reference templates.

At inference, cosine claim scores were normalized with symmetric score
normalization (S-norm) using cohort embeddings from the training participants.
The final submitted file, `SUBMIT_TOMORROW_ms400_weight_115.zip`, used weighted
empirical-rank fusion of two S-norm score lists:

```text
final_score =
  0.885 * rank(primary_circle_snorm_ensemble)
+ 0.115 * rank(secondary_top400_mean_snorm_model)
```

No external biometric datasets or pretrained backbones were used.

## Repository Layout

```text
code/stepup_runner.py          Train/infer/check CLI for stride embeddings
code/make_v2_submissions.py    Builds prototype, nearest, and S-norm submission candidates
code/fuse_rank_scores.py       Recreates SUBMIT_TOMORROW_ms400_weight_115.zip score fusion
code/snorm.py                  S-norm helper
docs/approach.md               Short paper-style approach description
```

Competition data, checkpoints, embeddings, and submitted score files are not
included.

## Expected Dataset Layout

```text
data/competition/
  1 - Training/
  2 - Reference/
    reference_data.npz
    reference_metadata.csv
  3 - Probe/
    probe_data.npz
    probe_metadata.csv
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA environment if the default
package is not appropriate for your GPU.

## Basic Usage

Check the dataset:

```bash
python code/stepup_runner.py check --dataset-root data/competition
```

Train a Circle Loss baseline:

```bash
python code/stepup_runner.py train \
  --dataset-root data/competition \
  --output-dir runs/circle_baseline \
  --loss circle \
  --epochs 10 \
  --batch-size 128
```

Generate S-norm submission candidates:

```bash
python code/make_v2_submissions.py \
  --dataset-root data/competition \
  --checkpoint runs/circle_baseline/best.pt \
  --output-dir runs/circle_baseline/submissions \
  --top-k 100 200 400
```

Fuse two S-norm score lists with the final `SUBMIT_TOMORROW_ms400_weight_115.zip`
weighting:

```bash
python code/fuse_rank_scores.py \
  --primary path/to/primary_submission.zip \
  --secondary path/to/secondary_submission.zip \
  --secondary-weight 0.115 \
  --output runs/final_rank_fusion.zip
```
