# MTL-FFPE: multi-task prediction core of RAPID-Pan

This repository contains the **ours** multi-task predictive module used for NEN/non-NEN triage, Ki67 grading, and lymph node metastasis (LNM) prediction. It implements shared experts with task-specific gates, an attention-based slide branch, an instance-response branch, and learned uncertainty weighting of the three task losses.

The input is a list of per-slide UNI feature tensors, each with shape `[number_of_patches, 1024]`. The model outputs three slide-level probabilities, one for each binary task. `HI` is the Ki67 grading task and `LN` is the LNM task in the original training code.

## Core code

`mtl_ffpe.py` contains the architecture (`MTLFFPE`), the class-weighted label-smoothed loss, the instance-to-bag distillation objective, and the prediction interface. The architecture and hyperparameter defaults follow the local `final.py` implementation used for the three-task model: 3 experts, 512-dimensional hidden representations, dropout 0.25, label smoothing 0.1, and distillation weight 0.1.

```python
import torch
from mtl_ffpe import MTLFFPE, UncertaintyWeighting, make_task_criteria, multitask_loss, predict_proba

model = MTLFFPE()
task_weighting = UncertaintyWeighting()
train_labels = torch.tensor([[0, 0, 0], [1, 1, 1]])  # replace with training cohort labels
criteria = make_task_criteria(train_labels)

bags = [torch.randn(16, 1024), torch.randn(20, 1024)]  # replace with UNI slide features
labels = train_labels
bag_logits, instance_logits, _ = model(bags)
loss = multitask_loss(bag_logits, instance_logits, labels, criteria, task_weighting)
loss.backward()

probabilities = predict_proba(model, bags)
```

For GPU training, move `model`, `task_weighting`, and `criteria` to the same device. An optimizer should receive the parameters of both `model` and `task_weighting`.

This release contains source code only. It does not include patient data, image patches, extracted features, trained weights, stain translation, data processing, or comparison experiments. The UNI encoder is available from [mahmoodlab/UNI](https://github.com/mahmoodlab/UNI); this repository expects its extracted features as input.
