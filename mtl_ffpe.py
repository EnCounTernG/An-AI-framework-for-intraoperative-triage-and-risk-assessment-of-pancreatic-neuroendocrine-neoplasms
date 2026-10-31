"""Core MTL-FFPE architecture and training objective for RAPID-Pan.

Input: one [num_patches, 1024] UNI feature tensor per slide.
Tasks: NEN/non-NEN triage, Ki67 grading, and LNM prediction.
"""

import torch
from torch import nn
from torch.nn import functional as F


TASKS = ("NEN", "HI", "LN")
NUM_CLASSES = {task: 2 for task in TASKS}


class MTLFFPE(nn.Module):
    """Shared experts, task-specific gates, and dual-branch MIL heads."""

    def __init__(self, feature_dim=1024, num_experts=3, hidden_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts

        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.25),
            )
            for _ in range(num_experts)
        )
        self.gates = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(feature_dim, hidden_dim // 2), nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_experts), nn.Softmax(dim=-1),
            )
            for task in TASKS
        })
        self.attention = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            for task in TASKS
        })
        self.instance_heads = nn.ModuleDict({
            task: nn.Linear(hidden_dim, NUM_CLASSES[task]) for task in TASKS
        })
        self.bag_heads = nn.ModuleDict({
            task: nn.Linear(hidden_dim, NUM_CLASSES[task]) for task in TASKS
        })

    def _forward_bag(self, features):
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected [num_patches, {self.feature_dim}] features")
        if features.shape[0] == 0:
            raise ValueError("A slide must contain at least one patch")

        bag_logits, instance_logits, mean_gates = {}, {}, {}
        for task in TASKS:
            expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=1)
            gate_weights = self.gates[task](features)
            task_features = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)
            patch_logits = self.instance_heads[task](task_features)
            attention = torch.softmax(self.attention[task](task_features).squeeze(-1), dim=0)
            slide_features = (attention.unsqueeze(-1) * task_features).sum(dim=0)

            bag_logits[task] = self.bag_heads[task](slide_features)
            instance_logits[task] = patch_logits.max(dim=0).values
            mean_gates[task] = gate_weights.mean(dim=0).detach()
        return bag_logits, instance_logits, mean_gates

    def forward(self, bags):
        """Return task-indexed [batch, 2] bag and instance logits, plus mean gates."""
        if not bags:
            raise ValueError("At least one slide bag is required")
        device = next(self.parameters()).device
        outputs = [self._forward_bag(bag.to(device)) for bag in bags]
        return tuple(
            {task: torch.stack([slide[branch][task] for slide in outputs]) for task in TASKS}
            for branch in range(3)
        )


class WeightedCEWithLabelSmoothing(nn.Module):
    """Class-weighted cross entropy used for each of the three tasks."""

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
    """Learned uncertainty weighting: sum_t [L_t/(2 sigma_t^2) + log(sigma_t)]."""

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
    """Build per-task criteria from [num_slides, 3] training labels."""
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
    """Bag classification plus instance-to-bag distillation for each task."""
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


@torch.no_grad()
def predict_proba(model, bags):
    """Return task-indexed [batch, 2] slide-level probabilities."""
    was_training = model.training
    model.eval()
    try:
        bag_logits, _, _ = model(bags)
        return {task: F.softmax(bag_logits[task], dim=1) for task in TASKS}
    finally:
        model.train(was_training)
