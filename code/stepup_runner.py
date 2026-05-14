#!/usr/bin/env python3
"""Runnable StepUP 2026 baseline and submission helper.

This script converts the public notebook workflow into a repeatable CLI:

  python stepup_runner.py check --dataset-root /path/to/competition
  python stepup_runner.py train --dataset-root /path/to/competition
  python stepup_runner.py infer --dataset-root /path/to/competition --checkpoint runs/best.pt

Expected dataset layout:

  dataset_root/
    1 - Training/
    2 - Reference/
    3 - Probe/
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import pathlib
import random
import zipfile
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn.modules.utils import _triple
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_size(value: str) -> tuple[int, int, int]:
    parts = tuple(int(x) for x in value.replace("x", ",").split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("size must be like 64,64,32")
    return parts


@dataclass(frozen=True)
class DatasetPaths:
    root: pathlib.Path

    @property
    def train(self) -> pathlib.Path:
        return self.root / "1 - Training"

    @property
    def ref(self) -> pathlib.Path:
        return self.root / "2 - Reference"

    @property
    def probe(self) -> pathlib.Path:
        return self.root / "3 - Probe"


def require_dataset(paths: DatasetPaths) -> None:
    missing = [p for p in (paths.train, paths.ref, paths.probe) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset folders:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\nExpected: '1 - Training', '2 - Reference', '3 - Probe'."
        )


def load_training_metadata(datapath_train: pathlib.Path) -> pd.DataFrame:
    participant_ids = np.arange(1, 151)
    speed_ids = ["W1", "W2", "W3", "W4"]
    footwear_ids = ["BF", "ST", "P1", "P2"]

    frames = []
    for participant_id, footwear_id, speed_id in itertools.product(
        participant_ids, footwear_ids, speed_ids
    ):
        metadata_path = (
            datapath_train
            / f"{participant_id:03}"
            / footwear_id
            / speed_id
            / "metadata.csv"
        )
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing training metadata: {metadata_path}")
        frames.append(pd.read_csv(metadata_path))

    metadata = pd.concat(frames).reset_index(drop=True)
    if "Exclude" in metadata.columns:
        metadata = metadata.query("Exclude == False").reset_index(drop=True)
    return metadata


def remove_overlap(pair_idx: np.ndarray) -> np.ndarray:
    used: set[int] = set()
    keep_idx: list[int] = []
    for i, idx in enumerate(pair_idx):
        left_id = int(idx[0])
        right_id = int(idx[1])
        if left_id not in used and right_id not in used:
            keep_idx.append(i)
            used.update({left_id, right_id})
    return pair_idx[keep_idx]


def get_pairs(metadata: pd.DataFrame, overlapping: bool = True) -> pd.DataFrame:
    pair_idx = []

    for _, pass_metadata in metadata.groupby(
        ["ParticipantID", "PassID", "Footwear", "Speed"], sort=False
    ):
        pass_idx = pass_metadata.index.to_numpy()
        if pass_idx.shape[0] < 2:
            continue

        pairs = np.stack((pass_idx[:-1], pass_idx[1:])).T
        for i, j in pairs:
            side_i = pass_metadata.loc[i, "Side"]
            side_j = pass_metadata.loc[j, "Side"]
            is_diff = side_i != side_j
            is_consec = abs(
                int(pass_metadata.loc[i, "FootstepID"])
                - int(pass_metadata.loc[j, "FootstepID"])
            ) == 1
            if is_diff and is_consec:
                pair_idx.append([i, j] if side_i == "Left" else [j, i])

    pair_idx = np.array(pair_idx)
    if pair_idx.size == 0:
        raise ValueError("No left/right stride pairs found.")
    if not overlapping:
        pair_idx = remove_overlap(pair_idx)

    left_ids = metadata.loc[pair_idx[:, 0], "FootstepID"].to_numpy()
    right_ids = metadata.loc[pair_idx[:, 1], "FootstepID"].to_numpy()
    pair_metadata = metadata[
        ["ParticipantID", "Footwear", "Speed", "Side", "FootstepID"]
    ].loc[pair_idx[:, 0]].copy()
    pair_metadata["FootstepID"] = list(np.stack((left_ids, right_ids), axis=1))
    pair_metadata["Side"] = list(np.tile(["Left", "Right"], (pair_idx.shape[0], 1)))
    pair_metadata.sort_index(inplace=True)
    pair_metadata.reset_index(drop=True, inplace=True)
    return pair_metadata


class Normalize:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        denom = x.max() - x.min()
        if denom == 0:
            return torch.zeros_like(x)
        x = (x - x.min()) / denom
        x[torch.isnan(x)] = 0
        return x


class Resize:
    def __init__(self, size: tuple[int, int, int]) -> None:
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x.unsqueeze(0), size=self.size, mode="trilinear")
        return x.squeeze(0)


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x


class TrialCacheMixin:
    def _init_cache(self, cache_trials: int) -> None:
        self.cache_trials = cache_trials
        self._trial_cache: OrderedDict[pathlib.Path, dict[str, np.ndarray]] = OrderedDict()

    def _load_npz(self, path: pathlib.Path) -> dict[str, np.ndarray]:
        if self.cache_trials <= 0:
            with np.load(path) as data:
                return {k: data[k] for k in data.files}

        cached = self._trial_cache.get(path)
        if cached is not None:
            self._trial_cache.move_to_end(path)
            return cached

        with np.load(path) as data:
            loaded = {k: data[k] for k in data.files}
        self._trial_cache[path] = loaded
        while len(self._trial_cache) > self.cache_trials:
            self._trial_cache.popitem(last=False)
        return loaded


class TrainingDataset(Dataset, TrialCacheMixin):
    def __init__(
        self,
        datapath: pathlib.Path,
        metadata: pd.DataFrame,
        transform=None,
        cache_trials: int = 16,
    ) -> None:
        self.datapath = datapath
        self.transform = transform
        self.participant_ids = metadata["ParticipantID"].to_numpy()
        self.footwear_ids = metadata["Footwear"].to_numpy()
        self.speed_ids = metadata["Speed"].to_numpy()
        self.side = metadata["Side"].to_numpy()
        self.footstep_ids = metadata["FootstepID"].to_numpy()
        self.labels = LabelEncoder().fit_transform(self.participant_ids)
        self.n_classes = np.unique(self.labels).shape[0]
        self._init_cache(cache_trials)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        participant_id = int(self.participant_ids[idx])
        footwear_id = self.footwear_ids[idx]
        speed_id = self.speed_ids[idx]
        footstep_id = self.footstep_ids[idx]
        side = self.side[idx]

        sample_path = (
            self.datapath / f"{participant_id:03}" / footwear_id / speed_id / "pipeline_1.npz"
        )
        footsteps = self._load_npz(sample_path)

        if isinstance(footstep_id, np.ndarray) or isinstance(footstep_id, list):
            step_iter = zip(side, footstep_id)
        else:
            step_iter = zip([side], [footstep_id])

        xs = []
        for step_side, step_id in step_iter:
            footstep = footsteps[str(int(step_id))]
            if step_side == "Right":
                footstep = np.flip(footstep, axis=2)
            tensor = torch.from_numpy(footstep[None, :, :, :].astype(np.float32))
            if self.transform:
                tensor = self.transform(tensor)
            xs.append(tensor)

        x = xs[0] if len(xs) == 1 else tuple(xs)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class TestDataset(Dataset):
    def __init__(self, data_dict, metadata: pd.DataFrame, transform=None, ref: bool = True):
        self.data_dict = data_dict
        self.footstep_id = list(data_dict.keys())
        self.transform = transform
        self.labels = (
            metadata["ParticipantID"].to_numpy()
            if ref
            else metadata["ClaimID"].to_numpy()
        )
        self.n_classes = np.unique(self.labels).shape[0]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        left_footstep, right_footstep = self.data_dict[self.footstep_id[idx]]
        right_footstep = np.flip(right_footstep, axis=2)
        x_left = torch.from_numpy(left_footstep[None, :, :, :].astype(np.float32))
        x_right = torch.from_numpy(right_footstep[None, :, :, :].astype(np.float32))
        if self.transform:
            x_left = self.transform(x_left)
            x_right = self.transform(x_right)
        return (x_left, x_right), torch.tensor(self.labels[idx])


class SpatioTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]
        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]
        intermed_channels = int(
            math.floor(
                (kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)
                / (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)
            )
        )
        self.spatial_conv = nn.Conv3d(
            in_channels,
            intermed_channels,
            spatial_kernel_size,
            stride=spatial_stride,
            padding=spatial_padding,
            bias=bias,
        )
        self.norm = nn.InstanceNorm3d(intermed_channels)
        self.relu = nn.ReLU()
        self.temporal_conv = nn.Conv3d(
            intermed_channels,
            out_channels,
            temporal_kernel_size,
            stride=temporal_stride,
            padding=temporal_padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm(self.spatial_conv(x)))
        return self.temporal_conv(x)


class CNN3D(nn.Module):
    def __init__(self, n_channels: int = 1):
        super().__init__()
        self.conv1 = SpatioTemporalConv(n_channels, 32, [5, 5, 5], stride=[2, 2, 2])
        self.conv2 = SpatioTemporalConv(32, 64, [3, 3, 3], stride=[1, 1, 1])
        self.conv3 = SpatioTemporalConv(64, 128, [3, 3, 3], stride=[1, 1, 1])
        self.conv4 = SpatioTemporalConv(128, 256, [3, 3, 3], stride=[2, 2, 2])
        self.norm1 = nn.InstanceNorm3d(32)
        self.norm2 = nn.InstanceNorm3d(64)
        self.norm3 = nn.InstanceNorm3d(128)
        self.norm4 = nn.InstanceNorm3d(256)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))
        x = self.relu(self.norm4(self.conv4(x)))
        x = self.pool(x)
        return x.view(-1, 256)


class FusionNet(nn.Module):
    def __init__(self, backbone: nn.Module, n_classes: int, in_features: int = 256, out_features: int = 512):
        super().__init__()
        self.backbone = backbone
        self.fusionlayer = nn.Linear(in_features * 2, out_features)
        self.outputlayer = nn.Linear(out_features, n_classes)

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor):
        embeds_left = self.backbone(x_left)
        embeds_right = self.backbone(x_right)
        embeds = torch.cat((embeds_left, embeds_right), dim=1)
        embeds = self.fusionlayer(embeds)
        logits = self.outputlayer(embeds)
        return logits, embeds


def make_loss(name: str):
    if name == "circle":
        return losses.CircleLoss(m=0.25, gamma=256)
    if name == "triplet":
        return losses.TripletMarginLoss(margin=0.2)
    if name == "supcon":
        return losses.SupConLoss(temperature=0.1)
    raise ValueError(f"Unsupported loss: {name}")


def verification_comparisons(embeds: np.ndarray, labels: np.ndarray):
    matched_scores = []
    non_matched_scores = []
    for user in np.unique(labels):
        pos_idx = labels == user
        neg_idx = ~pos_idx
        dist_matrix = cdist(embeds[pos_idx], embeds[pos_idx], "cosine")
        mask = ~np.eye(*dist_matrix.shape, dtype=bool)
        matched_scores.append(1 - dist_matrix[mask].flatten())
        dist_matrix = cdist(embeds[pos_idx], embeds[neg_idx], "cosine")
        non_matched_scores.append(1 - dist_matrix.flatten())
    matched_scores = np.hstack(matched_scores)
    non_matched_scores = np.hstack(non_matched_scores)
    scores = np.hstack((matched_scores, non_matched_scores))
    ground_truth = np.hstack(
        (np.ones(matched_scores.shape), np.zeros(non_matched_scores.shape))
    )
    return scores, ground_truth


def compute_eer(ground_truth: np.ndarray, scores: np.ndarray) -> float:
    fmr, tmr, _ = roc_curve(ground_truth, scores, pos_label=1)
    return 100 * brentq(lambda x: 1.0 - x - interp1d(fmr, tmr)(x), 0.0, 1.0)


def infer_embeddings(model, dataloader, device: torch.device) -> np.ndarray:
    model.eval()
    embeds_all = []
    with torch.no_grad():
        for (x_left, x_right), _ in dataloader:
            x_left = x_left.to(device, non_blocking=True)
            x_right = x_right.to(device, non_blocking=True)
            _, embeds = model(x_left, x_right)
            embeds = F.normalize(embeds, p=2, dim=1)
            embeds_all.append(embeds.cpu().numpy())
    return np.vstack(embeds_all)


def score_claims(
    ref_embeds: np.ndarray,
    ref_labels: np.ndarray,
    probe_embeds: np.ndarray,
    claim_labels: np.ndarray,
    method: str,
) -> np.ndarray:
    scores = []
    for claim, probe_embed in zip(claim_labels, probe_embeds):
        user_idx = ref_labels == claim
        if not np.any(user_idx):
            scores.append(0.0)
            continue
        user_refs = ref_embeds[user_idx]
        if method == "nearest":
            dist = cdist(probe_embed[None, :], user_refs, "cosine").min()
            score = ((1 - dist) + 1) / 2
        elif method == "prototype":
            proto = user_refs.mean(axis=0)
            proto = proto / max(np.linalg.norm(proto), 1e-12)
            score = float(np.dot(probe_embed, proto))
            score = (score + 1) / 2
        else:
            raise ValueError(f"Unsupported scoring method: {method}")
        scores.append(score)
    return np.asarray(scores)


def package_predictions(scores: np.ndarray, threshold: float, output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_scores = output.parent / "scores.txt"
    tmp_threshold = output.parent / "threshold.txt"
    np.savetxt(tmp_scores, scores)
    np.savetxt(tmp_threshold, np.array([threshold]))
    with zipfile.ZipFile(output, mode="w") as zf:
        zf.write(tmp_scores, arcname="scores.txt")
        zf.write(tmp_threshold, arcname="threshold.txt")
    tmp_scores.unlink()
    tmp_threshold.unlink()


def make_transform(sample_size: tuple[int, int, int]):
    return Compose([Resize(sample_size), Normalize()])


def cmd_check(args) -> None:
    paths = DatasetPaths(args.dataset_root)
    require_dataset(paths)
    print("Dataset root:", paths.root)
    print("Training:", paths.train)
    print("Reference:", paths.ref)
    print("Probe:", paths.probe)
    for required in [
        paths.ref / "reference_metadata.csv",
        paths.ref / "reference_data.npz",
        paths.probe / "probe_metadata.csv",
        paths.probe / "probe_data.npz",
    ]:
        print("OK" if required.exists() else "MISSING", required)
    metadata = load_training_metadata(paths.train)
    paired = get_pairs(metadata, overlapping=True)
    paired_val = get_pairs(metadata.query("ParticipantID >= 141"), overlapping=False)
    print("Filtered footsteps:", len(metadata))
    print("Overlapping train-style stride pairs:", len(paired))
    print("Non-overlapping validation stride pairs for users 141-150:", len(paired_val))


def cmd_train(args) -> None:
    set_seed(args.seed)
    paths = DatasetPaths(args.dataset_root)
    require_dataset(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transform = make_transform(args.sample_size)
    metadata = load_training_metadata(paths.train)

    train_users = np.arange(1, 141)
    val_users = np.arange(141, 151)
    metadata_train = get_pairs(metadata.query("ParticipantID in @train_users"), overlapping=True)
    metadata_val = get_pairs(metadata.query("ParticipantID in @val_users"), overlapping=False)

    dataset_train = TrainingDataset(
        paths.train, metadata_train, transform=transform, cache_trials=args.cache_trials
    )
    dataset_val = TrainingDataset(
        paths.train, metadata_val, transform=transform, cache_trials=args.cache_trials
    )
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = FusionNet(CNN3D(), dataset_train.n_classes).to(device)
    loss_func = make_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)

    best_eer = float("inf")
    for epoch in range(args.epochs):
        model.train()
        losses_epoch = []
        for (x_left, x_right), y in train_loader:
            x_left = x_left.to(device, non_blocking=True)
            x_right = x_right.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _, embeds = model(x_left, x_right)
            embeds = F.normalize(embeds, p=2, dim=1)
            loss = loss_func(embeds, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses_epoch.append(float(loss.item()))
        scheduler.step()

        val_embeds = infer_embeddings(model, val_loader, device, args.sample_size)
        val_scores, val_truth = verification_comparisons(val_embeds, dataset_val.participant_ids)
        eer = compute_eer(val_truth, val_scores)
        mean_loss = float(np.mean(losses_epoch))
        print(f"epoch={epoch + 1:03d} loss={mean_loss:.5f} val_eer={eer:.2f}")

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "val_eer": eer,
            "args": vars(args),
        }
        torch.save(state, args.output_dir / "last.pt")
        if eer < best_eer:
            best_eer = eer
            torch.save(state, args.output_dir / "best.pt")
    print(f"best_val_eer={best_eer:.2f}")


def cmd_infer(args) -> None:
    paths = DatasetPaths(args.dataset_root)
    require_dataset(paths)
    transform = make_transform(args.sample_size)
    device = torch.device(args.device)

    metadata_ref = pd.read_csv(paths.ref / "reference_metadata.csv", index_col="Index")
    footsteps_ref = np.load(paths.ref / "reference_data.npz")
    dataset_ref = TestDataset(footsteps_ref, metadata_ref, transform=transform, ref=True)
    ref_loader = DataLoader(dataset_ref, batch_size=args.batch_size, shuffle=False)

    metadata_probe = pd.read_csv(paths.probe / "probe_metadata.csv", index_col="Index")
    footsteps_probe = np.load(paths.probe / "probe_data.npz")
    dataset_probe = TestDataset(footsteps_probe, metadata_probe, transform=transform, ref=False)
    probe_loader = DataLoader(dataset_probe, batch_size=args.batch_size, shuffle=False)

    n_classes = int(args.n_classes)
    model = FusionNet(CNN3D(), n_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint["clf"]
    model.load_state_dict(state_dict)

    ref_embeds = infer_embeddings(model, ref_loader, device, args.sample_size)
    probe_embeds = infer_embeddings(model, probe_loader, device, args.sample_size)
    scores = score_claims(
        ref_embeds,
        dataset_ref.labels,
        probe_embeds,
        dataset_probe.labels,
        method=args.scoring,
    )
    package_predictions(scores, args.threshold, args.output)
    print("Reference Feature Set:", ref_embeds.shape)
    print("Probe Feature Set:", probe_embeds.shape)
    print("Score range:", float(scores.min()), float(scores.max()))
    print("Submission:", args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dataset_root = pathlib.Path("/home/anant/stepup2026/data/competition")
    parser.add_argument("--dataset-root", type=pathlib.Path, default=default_dataset_root)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check")
    check.add_argument("--dataset-root", type=pathlib.Path, default=default_dataset_root)

    train = sub.add_parser("train")
    train.add_argument("--dataset-root", type=pathlib.Path, default=default_dataset_root)
    train.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("runs/baseline"))
    train.add_argument("--sample-size", type=parse_size, default=(64, 64, 32))
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--lr-gamma", type=float, default=0.95)
    train.add_argument("--loss", choices=["circle", "triplet", "supcon"], default="circle")
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--cache-trials", type=int, default=16)
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    infer = sub.add_parser("infer")
    infer.add_argument("--dataset-root", type=pathlib.Path, default=default_dataset_root)
    infer.add_argument("--checkpoint", type=pathlib.Path, required=True)
    infer.add_argument("--output", type=pathlib.Path, default=pathlib.Path("runs/submission.zip"))
    infer.add_argument("--sample-size", type=parse_size, default=(64, 64, 32))
    infer.add_argument("--batch-size", type=int, default=128)
    infer.add_argument("--threshold", type=float, default=0.7)
    infer.add_argument("--scoring", choices=["nearest", "prototype"], default="prototype")
    infer.add_argument("--n-classes", type=int, default=140)
    infer.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "check":
            cmd_check(args)
        elif args.command == "train":
            cmd_train(args)
        elif args.command == "infer":
            cmd_infer(args)
        else:
            parser.error(f"Unknown command: {args.command}")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
