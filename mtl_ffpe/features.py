"""Read already-extracted slide feature tensors and training manifests."""

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .model import TASKS


def load_feature_bag(path, feature_dim=1024):
    """Load a UNI feature tensor shaped [num_patches, feature_dim]."""
    bag = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(bag, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}")
    if bag.ndim != 2 or bag.shape[0] == 0 or bag.shape[1] != feature_dim:
        raise ValueError(f"Expected [num_patches, {feature_dim}] in {path}")
    return bag.float()


class FeatureBagDataset(Dataset):
    """Read slide paths and labels; image processing is outside this module."""

    def __init__(self, manifest, feature_dim=1024):
        manifest = Path(manifest).resolve()
        self.feature_dim = feature_dim
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"slide_id", "feature_path", *TASKS}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Manifest requires columns: {', '.join(sorted(required))}")
            self.items = []
            for row in reader:
                path = Path(row["feature_path"])
                if not path.is_absolute():
                    path = manifest.parent / path
                labels = [int(row[task]) for task in TASKS]
                if any(label not in (0, 1) for label in labels):
                    raise ValueError(f"Labels must be 0 or 1: {row['slide_id']}")
                self.items.append((row["slide_id"], path, labels))
        if not self.items:
            raise ValueError("Manifest contains no slides")

    @property
    def labels(self):
        return torch.tensor([item[2] for item in self.items], dtype=torch.long)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        slide_id, path, labels = self.items[index]
        return load_feature_bag(path, self.feature_dim), torch.tensor(labels), slide_id


def collate_bags(batch):
    bags, labels, slide_ids = zip(*batch)
    return list(bags), torch.stack(labels), list(slide_ids)
