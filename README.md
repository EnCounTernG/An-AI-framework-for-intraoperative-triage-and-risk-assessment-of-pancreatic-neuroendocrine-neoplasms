# RAPID-Pan：MTL-FFPE 多任务预测模块

本仓库公开论文中 **ours 方法的多任务预测核心代码**。MTL-FFPE 接收由 UNI 提取的切片 patch 特征，联合输出三个切片级任务：NEN／非 NEN 分诊、Ki67 分级和淋巴结转移（LNM）预测。代码从本研究的三任务训练实现中提取并整理成独立模块。

## 方法结构

每张切片表示为一个特征包 `X ∈ R^(N×1024)`，其中 `N` 为该切片的 patch 数。三个任务共享一组专家网络，但各自使用门控网络，为每个 patch 计算专家权重。任务特异性特征随后进入两个分支：

1. **Attention bag branch**：对 patch 特征计算注意力权重，聚合成切片级表征，再生成分类 logits。
2. **Instance-response branch**：计算每个 patch 的分类 logits，并取逐类最大响应。训练时将其作为蒸馏教师信号。

每个任务的损失由类别加权、标签平滑的交叉熵和 instance-to-bag KL 蒸馏组成；三个任务再通过可学习的不确定性权重组合。模型默认使用 3 个专家、512 维隐藏层、0.25 dropout、0.1 标签平滑及 0.1 蒸馏权重，与原始三任务实现一致。

## 代码结构

| 文件 | 作用 |
| --- | --- |
| `mtl_ffpe/model.py` | 共享专家、任务门控、双分支 MIL 和切片级概率输出 |
| `mtl_ffpe/losses.py` | 类别加权损失、蒸馏损失和不确定性任务加权 |
| `mtl_ffpe/features.py` | 读取已提取的 `.pt` 特征和训练清单 |
| `train.py` | 基于训练集与验证集特征训练，保存验证集表现最好的权重 |
| `predict.py` | 加载权重，对单张切片的特征包输出三项预测 |

仓库只包含多任务预测所需代码，不包含切片图像、UNI 模型实现、虚拟染色、WSI 处理或对比实验。UNI 特征提取器见 [mahmoodlab/UNI](https://github.com/mahmoodlab/UNI)。

## 环境

建议使用 Python 3.9 或更新版本。安装适用于本机 CPU／CUDA 环境的 PyTorch 后，执行：

```bash
pip install -r requirements.txt
```

## 输入格式

每张切片对应一个 PyTorch `.pt` 文件，内容必须是单个二维浮点 Tensor，形状为 `[patch 数, 1024]`。例如 `features/slide_001.pt`。训练入口只读取这些已提取的特征，不负责生成 patch 或运行 UNI。

分别准备训练集和验证集 CSV，列名必须为：

```csv
slide_id,feature_path,NEN,HI,LN
slide_001,features/slide_001.pt,1,0,0
slide_002,features/slide_002.pt,0,1,1
```

`feature_path` 可以是绝对路径，也可以是相对于该 CSV 文件的路径。三个任务的标签均为 `0` 或 `1`；概率输出中的 `class_0`、`class_1` 与清单中的编码一一对应。`HI` 对应 Ki67 分级任务，`LN` 对应 LNM 任务。请在自己的数据字典中明确两类的临床定义。

## 训练

```bash
python train.py --train-manifest train.csv --val-manifest val.csv --output checkpoints/mtl_ffpe_best.pt
```

训练脚本在训练集上计算每个任务的类别权重，使用 Adam 优化模型和任务权重，以验证集的三任务平均 macro-F1 选择最佳 epoch；默认最多 60 个 epoch，连续 10 个 epoch 无提升则停止。`--batch-size`、`--lr`、`--num-experts`、`--seed` 等参数可通过 `python train.py --help` 查看。

保存的 checkpoint 包含模型参数、任务权重、模型结构参数和最佳验证 epoch。将训练集与验证集在病例层面分开；这里的脚本直接使用你提供的两份清单，不执行数据划分。

## 预测

```bash
python predict.py --checkpoint checkpoints/mtl_ffpe_best.pt --features features/slide_003.pt
```

输出为 JSON，每个任务分别给出 `class_0` 和 `class_1` 的切片级概率。例如：

```json
{
  "NEN": {"class_0": 0.12, "class_1": 0.88},
  "HI":  {"class_0": 0.74, "class_1": 0.26},
  "LN":  {"class_0": 0.61, "class_1": 0.39}
}
```

以上数字仅示意输出格式。仓库不提供患者数据、特征文件或训练权重。
