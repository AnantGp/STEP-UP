#!/usr/bin/env python3
"""Build StepUP S-norm submission zips from a trained checkpoint.

The script writes several candidates because public LB calibration has been
noisy for this dataset:

* prototype / nearest claim scores
* S-norm scores averaged over the claimed user's reference templates
* raw and min-max variants for each S-norm top-k
"""

from __future__ import annotations

import argparse
import pathlib
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from stepup_runner import (
    CNN3D,
    DatasetPaths,
    FusionNet,
    TestDataset,
    TrainingDataset,
    get_pairs,
    infer_embeddings,
    load_training_metadata,
    make_transform,
    package_predictions,
    require_dataset,
    score_claims,
    set_seed,
)


class EmbeddingArrayDataset(Dataset):
    def __init__(self, embeds: np.ndarray) -> None:
        self.embeds = embeds

    def __len__(self) -> int:
        return len(self.embeds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.embeds[idx])


def parse_size(value: str) -> tuple[int, int, int]:
    parts = tuple(int(x) for x in value.replace("x", ",").split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("size must be like 64,64,32")
    return parts


def load_model(checkpoint_path: pathlib.Path, device: torch.device, n_classes: int) -> FusionNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint["clf"]
    model = FusionNet(CNN3D(), n_classes=n_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_zip(scores: np.ndarray, threshold: float, output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    scores_path = output.parent / f"{output.stem}_scores.txt"
    threshold_path = output.parent / f"{output.stem}_threshold.txt"
    np.savetxt(scores_path, scores)
    np.savetxt(threshold_path, np.array([threshold]))
    with zipfile.ZipFile(output, mode="w") as zf:
        zf.write(scores_path, arcname="scores.txt")
        zf.write(threshold_path, arcname="threshold.txt")


def normalize_minmax(scores: np.ndarray) -> np.ndarray:
    lo = float(scores.min())
    hi = float(scores.max())
    if hi <= lo:
        return scores.copy()
    return (scores - lo) / (hi - lo)


def load_or_infer_reference_probe(
    args,
    paths: DatasetPaths,
    model: FusionNet,
    device: torch.device,
    transform,
):
    ref_embed_path = args.output_dir / "ref_embeds.npy"
    probe_embed_path = args.output_dir / "probe_embeds.npy"
    if args.reuse_embeds and ref_embed_path.exists() and probe_embed_path.exists():
        print("Loading cached reference/probe embeddings", flush=True)
        metadata_ref = pd.read_csv(paths.ref / "reference_metadata.csv", index_col="Index")
        metadata_probe = pd.read_csv(paths.probe / "probe_metadata.csv", index_col="Index")
        return (
            np.load(ref_embed_path),
            metadata_ref["ParticipantID"].to_numpy(),
            np.load(probe_embed_path),
            metadata_probe["ClaimID"].to_numpy(),
        )

    print("Inferring reference embeddings", flush=True)
    metadata_ref = pd.read_csv(paths.ref / "reference_metadata.csv", index_col="Index")
    with np.load(paths.ref / "reference_data.npz") as f:
        footsteps_ref = {k: f[k].copy() for k in f.files}
    dataset_ref = TestDataset(footsteps_ref, metadata_ref, transform=transform, ref=True)
    ref_loader = DataLoader(
        dataset_ref,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    ref_embeds = infer_embeddings(model, ref_loader, device)

    print("Inferring probe embeddings", flush=True)
    metadata_probe = pd.read_csv(paths.probe / "probe_metadata.csv", index_col="Index")
    with np.load(paths.probe / "probe_data.npz") as f:
        footsteps_probe = {k: f[k].copy() for k in f.files}
    dataset_probe = TestDataset(footsteps_probe, metadata_probe, transform=transform, ref=False)
    probe_loader = DataLoader(
        dataset_probe,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    probe_embeds = infer_embeddings(model, probe_loader, device)

    np.save(ref_embed_path, ref_embeds.astype(np.float32))
    np.save(probe_embed_path, probe_embeds.astype(np.float32))
    return (
        ref_embeds,
        dataset_ref.labels,
        probe_embeds,
        dataset_probe.labels,
    )


def load_or_infer_cohort(args, paths: DatasetPaths, model: FusionNet, device: torch.device):
    cohort_path = args.output_dir / f"cohort_embeds_{args.cohort_size or 'all'}.npy"
    if args.reuse_embeds and cohort_path.exists():
        print(f"Loading cached cohort embeddings: {cohort_path}", flush=True)
        return np.load(cohort_path)

    print("Preparing cohort dataset", flush=True)
    metadata = load_training_metadata(paths.train)
    train_users = np.arange(1, 141)
    metadata_train = get_pairs(metadata.query("ParticipantID in @train_users"), overlapping=True)
    dataset_train = TrainingDataset(
        paths.train,
        metadata_train,
        transform=make_transform(args.sample_size),
        cache_trials=0,
    )

    dataset: Dataset = dataset_train
    if args.cohort_size and args.cohort_size < len(dataset_train):
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(dataset_train), size=args.cohort_size, replace=False)
        indices.sort()
        dataset = Subset(dataset_train, indices.tolist())
        print(f"Using cohort subset: {len(dataset)} / {len(dataset_train)}", flush=True)
    else:
        print(f"Using full cohort: {len(dataset_train)}", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.cohort_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    cohort_embeds = infer_embeddings(model, loader, device).astype(np.float32)
    np.save(cohort_path, cohort_embeds)
    return cohort_embeds


def topk_stats(query: np.ndarray, cohort: np.ndarray, top_k: int, chunk_size: int):
    means = np.zeros((len(query), 1), dtype=np.float32)
    stds = np.zeros((len(query), 1), dtype=np.float32)
    cohort_t = cohort.T
    for start in range(0, len(query), chunk_size):
        end = min(start + chunk_size, len(query))
        scores = query[start:end] @ cohort_t
        topk = np.partition(scores, -top_k, axis=1)[:, -top_k:]
        means[start:end] = topk.mean(axis=1, keepdims=True)
        stds[start:end] = topk.std(axis=1, keepdims=True) + 1e-6
        print(f"  cohort stats {start}:{end}", flush=True)
    return means, stds


def snorm_matrix(
    ref_embeds: np.ndarray,
    probe_embeds: np.ndarray,
    cohort_embeds: np.ndarray,
    top_k: int,
    chunk_size: int,
) -> np.ndarray:
    top_k = min(top_k, cohort_embeds.shape[0])
    raw = probe_embeds @ ref_embeds.T
    print(f"S-norm top_k={top_k}: probe cohort stats", flush=True)
    probe_mean, probe_std = topk_stats(probe_embeds, cohort_embeds, top_k, chunk_size)
    print(f"S-norm top_k={top_k}: ref cohort stats", flush=True)
    ref_mean, ref_std = topk_stats(ref_embeds, cohort_embeds, top_k, chunk_size)
    z_probe = (raw - probe_mean) / probe_std
    z_ref = (raw - ref_mean.T) / ref_std.T
    return (0.5 * (z_probe + z_ref)).astype(np.float32)


def flatten_by_claim(
    score_matrix: np.ndarray,
    ref_labels: np.ndarray,
    claim_labels: np.ndarray,
    reduction: str,
) -> np.ndarray:
    out = np.zeros(len(claim_labels), dtype=np.float32)
    for i, claim in enumerate(claim_labels):
        user_idx = ref_labels == claim
        if not np.any(user_idx):
            continue
        user_scores = score_matrix[i, user_idx]
        if reduction == "mean":
            out[i] = float(np.mean(user_scores))
        elif reduction == "max":
            out[i] = float(np.max(user_scores))
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=pathlib.Path, default=pathlib.Path("/home/anant/stepup2026/data/competition"))
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--sample-size", type=parse_size, default=(64, 64, 32))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cohort-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cohort-size", type=int, default=0, help="0 means full cohort")
    parser.add_argument("--top-k", type=int, nargs="+", default=[100, 200, 400])
    parser.add_argument("--snorm-chunk-size", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--reuse-embeds", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = DatasetPaths(args.dataset_root)
    require_dataset(paths)
    transform = make_transform(args.sample_size)
    device = torch.device(args.device)

    model = load_model(args.checkpoint, device, n_classes=140)
    ref_embeds, ref_labels, probe_embeds, claim_labels = load_or_infer_reference_probe(
        args, paths, model, device, transform
    )

    print("Writing direct scoring submissions", flush=True)
    proto_scores = score_claims(
        ref_embeds, ref_labels, probe_embeds, claim_labels, method="prototype"
    )
    near_scores = score_claims(
        ref_embeds, ref_labels, probe_embeds, claim_labels, method="nearest"
    )
    package_predictions(proto_scores, args.threshold, args.output_dir / "submission_v2_prototype.zip")
    package_predictions(near_scores, args.threshold, args.output_dir / "submission_v2_nearest.zip")

    cohort_embeds = load_or_infer_cohort(args, paths, model, device)
    print(f"Reference embeddings: {ref_embeds.shape}", flush=True)
    print(f"Probe embeddings: {probe_embeds.shape}", flush=True)
    print(f"Cohort embeddings: {cohort_embeds.shape}", flush=True)

    for top_k in args.top_k:
        matrix = snorm_matrix(
            ref_embeds,
            probe_embeds,
            cohort_embeds,
            top_k=top_k,
            chunk_size=args.snorm_chunk_size,
        )
        np.save(args.output_dir / f"snorm_matrix_top{top_k}.npy", matrix)
        for reduction in ("mean", "max"):
            raw_flat = flatten_by_claim(matrix, ref_labels, claim_labels, reduction=reduction)
            save_zip(
                raw_flat,
                0.0,
                args.output_dir / f"submission_v2_snorm_top{top_k}_{reduction}_raw.zip",
            )
            save_zip(
                normalize_minmax(raw_flat),
                args.threshold,
                args.output_dir / f"submission_v2_snorm_top{top_k}_{reduction}_minmax.zip",
            )
            print(
                f"top_k={top_k} reduction={reduction} "
                f"raw_range=({raw_flat.min():.4f}, {raw_flat.max():.4f})",
                flush=True,
            )

    print("Done. Submission candidates written to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
