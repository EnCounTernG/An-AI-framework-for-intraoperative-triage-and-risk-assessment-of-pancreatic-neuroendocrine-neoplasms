"""Predict three slide-level tasks from one precomputed UNI feature tensor."""

import argparse
import json

import torch

from mtl_ffpe import MTLFFPE, TASKS, predict_proba
from mtl_ffpe.features import load_feature_bag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True, help="One [num_patches, 1024] .pt tensor")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["model_config"]
    model = MTLFFPE(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    bag = load_feature_bag(args.features, config["feature_dim"])
    probabilities = predict_proba(model, [bag])
    result = {
        task: {"class_0": probabilities[task][0, 0].item(),
               "class_1": probabilities[task][0, 1].item()}
        for task in TASKS
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
