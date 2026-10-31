"""Shared experts, task gates and dual-branch MIL for three slide-level tasks."""

import torch
from torch import nn
from torch.nn import functional as F


TASKS = ("NEN", "HI", "LN")
NUM_CLASSES = {task: 2 for task in TASKS}


class MTLFFPE(nn.Module):
    """MTL-FFPE model operating on one bag of UNI features per slide."""

    def __init__(self, feature_dim=1024, num_experts=3, hidden_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

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
            # The source implementation evaluates the expert pool separately
            # for each task, including independent dropout during training.
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
        """Return task-indexed [batch, 2] bag logits, instance logits and gates."""
        if not bags:
            raise ValueError("At least one slide bag is required")
        device = next(self.parameters()).device
        outputs = [self._forward_bag(bag.to(device)) for bag in bags]
        return tuple(
            {task: torch.stack([slide[branch][task] for slide in outputs]) for task in TASKS}
            for branch in range(3)
        )


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
