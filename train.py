"""Train only the MTL-FFPE module from precomputed UNI slide features."""

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mtl_ffpe import MTLFFPE, TASKS, UncertaintyWeighting, make_task_criteria, multitask_loss
from mtl_ffpe.features import FeatureBagDataset, collate_bags


def mean_macro_f1(truth, prediction):
    """Average binary macro F1 across the three tasks."""
    scores = []
    for column in range(len(TASKS)):
        class_scores = []
        for label in (0, 1):
            tp = ((truth[:, column] == label) & (prediction[:, column] == label)).sum().item()
            fp = ((truth[:, column] != label) & (prediction[:, column] == label)).sum().item()
            fn = ((truth[:, column] == label) & (prediction[:, column] != label)).sum().item()
            denominator = 2 * tp + fp + fn
            class_scores.append(2 * tp / denominator if denominator else 0.0)
        scores.append(sum(class_scores) / 2)
    return sum(scores) / len(scores)


def run_epoch(loader, model, task_weighting, criteria, optimizer=None, distill_weight=0.1):
    training = optimizer is not None
    model.train(training)
    task_weighting.train(training)
    total_loss, total_slides = 0.0, 0
    truths, predictions = [], []

    for bags, labels, _ in loader:
        with torch.set_grad_enabled(training):
            bag_logits, instance_logits, _ = model(bags)
            loss = multitask_loss(
                bag_logits, instance_logits, labels, criteria, task_weighting,
                distill_weight=distill_weight,
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_slides += batch_size
        truths.append(labels)
        predictions.append(torch.stack(
            [bag_logits[task].argmax(dim=1).detach().cpu() for task in TASKS], dim=1
        ))

    return total_loss / total_slides, mean_macro_f1(
        torch.cat(truths), torch.cat(predictions)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output", required=True, help="Best checkpoint path (.pt)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--distill-weight", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = FeatureBagDataset(args.train_manifest, args.feature_dim)
    val_set = FeatureBagDataset(args.val_manifest, args.feature_dim)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_bags
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_bags
    )

    config = {
        "feature_dim": args.feature_dim,
        "num_experts": args.num_experts,
        "hidden_dim": args.hidden_dim,
    }
    model = MTLFFPE(**config).to(device)
    task_weighting = UncertaintyWeighting().to(device)
    criteria = make_task_criteria(train_set.labels, args.label_smoothing).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(task_weighting.parameters()), lr=args.lr
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_f1, stale_epochs = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1 = run_epoch(
            train_loader, model, task_weighting, criteria, optimizer, args.distill_weight
        )
        val_loss, val_f1 = run_epoch(
            val_loader, model, task_weighting, criteria,
            distill_weight=args.distill_weight,
        )
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f} "
            f"train_macro_f1={train_f1:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1, stale_epochs = val_f1, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "task_weighting_state_dict": task_weighting.state_dict(),
                "model_config": config,
                "best_epoch": epoch,
                "best_val_macro_f1": best_f1,
            }, output)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    print(f"Saved best MTL-FFPE checkpoint to {output}")


if __name__ == "__main__":
    main()
