"""Class-balanced task losses, distillation and learned uncertainty weights."""

import torch
from torch import nn
from torch.nn import functional as F

from .model import TASKS


class WeightedCEWithLabelSmoothing(nn.Module):
    def __init__(self, class_weights, label_smoothing=0.1):
        super().__init__()
        self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes).to(logits.dtype)
        soft_labels = one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        per_sample = -(soft_labels * log_probs).sum(dim=1)
        weights = self.class_weights[targets]
        return (per_sample * weights).sum() / weights.sum()


class UncertaintyWeighting(nn.Module):
    """sum_t [L_t/(2 sigma_t^2) + log(sigma_t)]."""

    def __init__(self, num_tasks=len(TASKS)):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        if len(losses) != len(self.log_vars):
            raise ValueError("Expected one loss per task")
        return sum(
            0.5 * (torch.exp(-self.log_vars[i]) * loss + self.log_vars[i])
            for i, loss in enumerate(losses)
        )


def make_task_criteria(train_labels, label_smoothing=0.1):
    """Calculate per-task class weights from training labels [num_slides, 3]."""
    labels = torch.as_tensor(train_labels, dtype=torch.long)
    if labels.ndim != 2 or labels.shape[1] != len(TASKS):
        raise ValueError("train_labels must have shape [num_slides, 3]")
    if labels.shape[0] == 0 or torch.any((labels < 0) | (labels > 1)):
        raise ValueError("Expected nonempty binary training labels")

    criteria = {}
    for i, task in enumerate(TASKS):
        counts = torch.bincount(labels[:, i], minlength=2).float().clamp_min(1)
        weights = labels.shape[0] / (2 * counts)
        weights = weights / weights.sum() * 2
        criteria[task] = WeightedCEWithLabelSmoothing(weights, label_smoothing)
    return nn.ModuleDict(criteria)


def multitask_loss(bag_logits, instance_logits, labels, criteria, task_weighting,
                   distill_weight=0.1):
    """Bag classification plus detached instance-to-bag distillation."""
    labels = torch.as_tensor(labels, dtype=torch.long, device=bag_logits[TASKS[0]].device)
    if labels.ndim != 2 or labels.shape[1] != len(TASKS):
        raise ValueError("labels must have shape [batch, 3]")

    task_losses = []
    for i, task in enumerate(TASKS):
        classification = criteria[task](bag_logits[task], labels[:, i])
        teacher = F.softmax(instance_logits[task].detach(), dim=1)
        distillation = F.kl_div(
            F.log_softmax(bag_logits[task], dim=1), teacher, reduction="batchmean"
        )
        task_losses.append(classification + distill_weight * distillation)
    return task_weighting(task_losses)
