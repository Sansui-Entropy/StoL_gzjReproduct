# CrossModalDistillation — 训练代码实现说明

本文件记录在原始推理仓库基础上新增/修改的全部内容，说明与论文的对应关系，以及接入真实数据集所需的步骤。

---

## 1. 文件变更清单

### 1.1 新增文件

| 文件路径 | 职责 |
|---|---|
| `cross_modal_distillation/models/spike_mae.py` | Spike MAE 教师模型（论文 Fig.7 / Appendix A.1） |
| `cross_modal_distillation/models/distillation.py` | LFP 学生 + 蒸馏模块（论文 Fig.2 / Eq.1） |
| `cross_modal_distillation/data/paired_dataset.py` | 配对数据集抽象基类（接口定义） |
| `cross_modal_distillation/train/__init__.py` | train 包标识 |
| `cross_modal_distillation/train/losses.py` | Poisson MAE 损失、蒸馏损失、全监督蒸馏损失 |
| `cross_modal_distillation/train/optim.py` | AdamW + warmup + 指数衰减调度（论文 A.2） |
| `cross_modal_distillation/train/train_mae.py` | 教师 MAE 预训练 / 微调训练脚本 |
| `cross_modal_distillation/train/train_distill.py` | LFP 学生蒸馏训练脚本 |
| `cross_modal_distillation/configs/train/teacher_mae.yaml` | 教师训练 Hydra 配置（论文超参） |
| `cross_modal_distillation/configs/train/distill.yaml` | 蒸馏训练 Hydra 配置（论文超参） |

### 1.2 修改文件

| 文件路径 | 改动内容 |
|---|---|
| `cross_modal_distillation/models/tokenizer.py` | 新增 `get_spatial_pad_mask()` 辅助方法，其余接口不变 |
| `cross_modal_distillation/data/collate.py` | 新增 `PairedBatchItem` 命名元组与 `collate_paired_fn` 函数 |

### 1.3 未改动（推理入口保持原样）

- `cross_modal_distillation/models/model.py`
- `cross_modal_distillation/inference_generalization.ipynb`
- `cross_modal_distillation/data/makin_dataset.py`
- `cross_modal_distillation/data/flint_dataset.py`
- `cross_modal_distillation/data/base_dataset.py`
- `cross_modal_distillation/build.py`

---

## 2. 架构与论文对应关系

### 2.1 Spike Tokenizer（论文 Fig.1，Section 3.1）

对应实现：`PatchTokenizer`（`use_embedding_for_input=True`，`learn_patch_embedding=True`）

- 只沿空间（神经元）切 patch，patch size `S=64`，不能整除则 pad
- Count embedding：每个 count 值 `0..k`（`k=5`）学一个 `R^{d/S}` 向量，patch 内 concat 成 `R^d`
- 每个 session 一套可学习 space embedding `E^i_j ∈ R^d`，与 value embedding 相加得到最终 token
- 新 session 通过 `tokenizer.update_for_new_sessions()` 注册新 space embedding，微调时全部参数可更新

### 2.2 Spike MAE 教师（论文 Fig.7 / Section 3.2 / Appendix A.1）

对应实现：`SpikeMAEModel`（`spike_mae.py`）

数据流：

```
spikes (B,T,N)
  → PatchTokenizer                 # count embed + space embed → tokens (B,T*P,256)
  → 随机丢弃 60% token             # 时间与空间均匀 drop (paper A.1)
  → RoPE encoder (10层, d=256)     # 只处理可见 token
  → 插回 mask_token + space embed  # paper A.1: "add their corresponding space embeddings"
  → Linear(256→192)                # 补全维度接合层（论文未命名，维度对齐必要）
  → RoPE predictor (4层, d=192)    # paper A.2
  → Linear(192→64)                 # "64-dimensional down-projection" (paper A.2)
  → softplus(·)                    # Poisson rate λ（本实现约定）
  → Poisson NLL                    # 仅 mask 位置、非 pad 维
```

`encode()` 方法：不 mask，tokenizer + encoder + 时间步 mean-pool，返回 `(B,T,D)` 表征，供冻结教师使用。

### 2.3 LFP 学生与蒸馏（论文 Fig.2 / Section 3.3 / Eq.1）

对应实现：`LFPStudentModel` + `DistillationModel`（`distillation.py`）

```
LFP (B,T,N_lfp)
  → PatchTokenizer (use_conv_for_input=True)   # 因果卷积 value embed，S=32
  → RoPE encoder (10层, d=256)
  → mean-pool 同时间步 patch                   # z_lfp: (B,T,D)
  → f_phi: Linear(256→N_lfp)                  # ŷ: (B,T,N_lfp)

Spike (B,T,N_spike)
  → 冻结教师 encode()                          # z_spike: (B,T,D), no_grad

L = MSE(y, ŷ) + 5·(1 - mean_t cosine(z_lfp_t, z_spike_t))   # Eq.1
```

- 教师在 `DistillationModel.__init__` 内全程冻结（`requires_grad=False` + `eval()`）
- `lambda_align=5`（论文 A.2）

### 2.4 损失函数（`train/losses.py`）

| 函数 | 对应 |
|---|---|
| `poisson_mae_loss` | 教师 MAE 训练损失（paper A.1，Poisson NLL，mask 且非 pad） |
| `distillation_loss` | Eq.1（MSE 重建 + 余弦对齐） |
| `supervised_distillation_loss` | Eq.3（行为回归 + 余弦对齐，Appendix A.11） |

### 2.5 优化器与调度（`train/optim.py`）

论文 A.2 参数：

| 超参 | 值 |
|---|---|
| 优化器 | AdamW |
| 最大学习率 | 6.25e-4 |
| 初始学习率因子 | 0.3 |
| Warmup epochs | 30 |
| 指数衰减因子 | 0.995（每 epoch） |
| Weight decay | 0.1（起始） |

### 2.6 数据接口（`data/paired_dataset.py`，`data/collate.py`）

抽象基类 `PairedSpikeLFPDataset` 定义 `__getitem__` 返回格式：

```python
{
    "spikes"          : Tensor[T, N_spike],   # 非负整数 count
    "lfp"             : Tensor[T, N_lfp],     # z-scored
    "subject_session" : str,
    "segment_filename": str,
}
```

`collate_paired_fn` 将 batch 收为 `PairedBatchItem(spikes, lfp, subject_sessions, position_ids, segment_filenames)`。

---

## 3. 实现时的补全约定

### 3.1 encoder→predictor 维度接合

论文 A.2 写了 encoder `d=256`、predictor `d=192`，但没有命名两者之间的连接层。本实现在 `SpikeMAEModel.__init__` 中添加了：

```python
self.enc_to_pred = nn.Linear(d_encoder, d_predictor, bias=True)  # 256→192
```

### 3.2 Poisson 重建头参数化

论文只写了"64-dimensional down-projection"，没有说明如何从 64 维数值得到 Poisson 参数 λ。本实现约定：

```python
logits = self.recon_head(pred_out)   # Linear(192→64)
lambda_ = F.softplus(logits)         # λ = softplus(z)，严格正
loss = F.poisson_nll_loss(lambda_, counts, log_input=False, ...)
```

64 维直接对应同一 patch 内 64 个神经元，每个神经元一个 λ。

### 3.3 空间 pad mask

论文 A.1：「padded dimensions … are excluded」。`PatchTokenizer.get_spatial_pad_mask(d_input)` 返回形状 `(num_patches, S)` 的 bool mask（`True=有效`），`SpikeMAEModel._mae_loss()` 用它在 Poisson NLL 中排除 pad 维。

---

## 4. 接入真实数据集的步骤

1. **实现配对数据集子类**：在 `data/paired_dataset.py`（或新文件）中继承 `PairedSpikeLFPDataset`，实现 `__len__` 和 `__getitem__`：
   - LFP：可复用 `MakinRTDataset` / `FlintCODataset` 已处理的 `data["lfp"]`
   - Spike：从原始 `.nwb` 加载宽带信号，binning 为 10 ms count，丢弃均值 < 1 Hz 单元，对齐到同一时间轴
   - 填充 `session_d_spike_dict` 和 `session_d_lfp_dict`

2. **修改训练脚本**：在 `train_mae.py` 和 `train_distill.py` 的 `# DATASET` 注释块中，将 `raise NotImplementedError` 替换为实际数据集构建代码。

3. **更新 yaml 配置**：在 `configs/train/teacher_mae.yaml` 和 `distill.yaml` 的 `dataset:` 节中填入 `lfp_metadata_path`、`spike_segment_dir`、`session` 等字段。

4. **运行**：

```bash
# Step 1：多 session 教师预训练
python -m cross_modal_distillation.train.train_mae mode=pretrain

# Step 2：单 session 教师微调（Monkey I 20160622_01 为例）
python -m cross_modal_distillation.train.train_mae \
    mode=finetune \
    teacher_ckpt=./results/checkpoints/teacher/best_pretrain.ckpt \
    dataset.new_session=MonkeyI_20160622_01

# Step 3：LFP 学生蒸馏
python -m cross_modal_distillation.train.train_distill \
    teacher_ckpt=./results/checkpoints/teacher/best_finetune.ckpt \
    dataset.session=MonkeyI_20160622_01
```

---

## 5. 本实现未涵盖的内容（明确列出）

以下内容在论文中存在，但超出本次实现范围，后续需要时再补充：

| 内容 | 论文位置 |
|---|---|
| Spike 数据的真实下载与预处理（MakinRT、Perich 等） | Appendix A.3 |
| Gallego-Carracedo LFP-power 信号支持（S=288，3s 段） | Appendix A.3 |
| LFP MAE 预训练（MS-LFP 基线） | Section 3.1-3.2 |
| 多 GPU 分布式训练 | Appendix A.2 |
| 监督微调（MAE + 行为回归联合损失） | Appendix A.1 |
| 全监督蒸馏（Eq.3，替换 AE 项为行为回归） | Appendix A.11 |
| 多 session 蒸馏（MS-Distilled LFP） | Appendix A.12 |
| SS-MM 多模态基线（spike+LFP 输入拼接） | Section 4 |
| Fig.5 跨 session 推理协议（复用蒸馏 session embedding） | Section 4.3 |
| NDT2 / MSID / BRANT 对比基线 | Appendix A.6-A.7 |
| t-SNE、CKA、检索指标分析 | Section 4.2 / Appendix A.5 |

---

## 6. 目录结构（新增部分）

```
cross_modal_distillation/
├── configs/
│   └── train/
│       ├── teacher_mae.yaml       # 教师 MAE 超参配置
│       └── distill.yaml           # 蒸馏超参配置
├── data/
│   ├── collate.py                 # 新增 PairedBatchItem + collate_paired_fn
│   └── paired_dataset.py          # 新增 PairedSpikeLFPDataset 抽象基类
├── models/
│   ├── tokenizer.py               # 新增 get_spatial_pad_mask()
│   ├── spike_mae.py               # 新增 SpikeMAEModel
│   └── distillation.py            # 新增 LFPStudentModel + DistillationModel
└── train/
    ├── __init__.py
    ├── losses.py                   # Poisson MAE / 蒸馏 / 全监督蒸馏损失
    ├── optim.py                    # AdamW + warmup + 指数衰减
    ├── train_mae.py                # 教师预训练 / 微调脚本
    └── train_distill.py            # 学生蒸馏脚本
```
