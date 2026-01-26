---
title: 条件-结构引导的ProteinMPNN序列设计
mathjax: true
date: 2026/01/21 20:46:25
img: https://www.researchgate.net/publication/363608659/figure/fig1/AS:11431281127577573@1679077202382/ProteinMPNN-architecture.jpg
excerpt: SF-MPNN-RL
---

#### A. 输入特征的简化与增强

早期的深度学习模型常使用二面角等特征，但 ProteinMPNN 的研究发现，使用**原子间的距离**作为输入特征效果更好 。

- **输入：** 仅依赖蛋白质骨架坐标（N, Ca, C, O）以及计算出的虚拟 Cb 原子位置。
- **特征：** 主要是原子间的距离（编码为高斯径向基函数）和相对位置编码 。模型明确排除了二面角特征，因为实验显示这并没有带来性能提升 。

#### B. 随机解码顺序 (Random Decoding Order)

这是该模型灵活性的核心。通常的自回归模型是从 N 端到 C 端（从左到右）解码。

- **创新：** ProteinMPNN 在训练时随机打乱解码顺序 。
- **优势：** 这使得模型在推断（Inference）时可以适应各种场景。例如，你可以“固定”蛋白质中间的某段序列（作为已知上下文），然后让模型设计其余部分。这对于设计结合蛋白（Binder design）非常关键 。

#### C. 引入噪声训练 (Training with Noise)

- **方法：** 在训练过程中，研究人员向真实的 PDB 骨架坐标中添加了高斯噪声（标准差 0.02Å） 。
- **目的：** 这提高了模型的鲁棒性。在实际应用中，设计的目标骨架往往不是完美的晶体结构，而是计算机生成的模型（如 AlphaFold 生成的结构）。噪声训练防止了模型过度拟合晶体结构中的微小细节，使其能更好地处理粗糙的或计算机生成的骨架 。

#### D. 对称性与多链处理 (Tied Decoding)

为了设计多聚体或对称结构（如同源二聚体、纳米笼），模型允许将不同位置的权重“绑定”（Tied）。

- **实现：** 对于对称位置（例如链 A 的位置 1 和链 B 的位置 1），模型会先分别预测 Logits（未归一化的概率），然后将它们平均，最后从这个平均分布中采样。这确保了生成的序列在对称单元中是完全一致的 。

### 3. 模型架构 (Architecture)

ProteinMPNN 采用了**编码器-解码器（Encoder-Decoder）**架构 。

- **骨架编码器 (Backbone Encoder):**
  - 输入：蛋白质骨架的几何特征（主要是原子间距离）。
  - 结构：3 层消息传递层。它包含节点（Node）更新和边（Edge）更新机制。节点代表氨基酸残基，边代表残基间的空间关系 。
  - 输出：包含骨架结构信息的嵌入向量。
- **序列解码器 (Sequence Decoder):**
  - 输入：编码器的输出 + 当前已生成的局部序列（Masked sequence）。
  - 结构：3 层标准的 MPNN 层。它是自回归的，意味着它基于骨架和已知的氨基酸来预测下一个氨基酸 。
  - 操作：通过迭代的方式，根据随机或指定的顺序填补序列中的未知部分。

### 4. 实验与评估 (Experiments & Evaluation)

论文通过计算机模拟（In silico）和湿实验（In vitro）全方位验证了模型。

#### A. 计算机模拟评估 (In Silico)

1. **序列恢复率 (Sequence Recovery):** 在天然骨架上，ProteinMPNN 的序列恢复率达到 **52.4%**，而 Rosetta 只有 32.9% 。在单体、同源多聚体和异源多聚体上均表现优异 。
2. **AlphaFold 验证:** 将 ProteinMPNN 设计的序列输入 AlphaFold 进行结构预测，发现预测出的结构与目标骨架高度一致。特别是经过噪声训练的模型，其生成的序列在 AlphaFold 预测中表现出更高的置信度（pLDDT） 。
3. **速度:** 处理 100 个残基的蛋白质仅需 1.2 秒，而 Rosetta 需要 4.3 分钟 。

#### B. 湿实验验证 (Experimental Validation)

这是论文最强有力的部分，展示了 ProteinMPNN "拯救" 了许多用传统方法失败的设计。

1. **拯救 AlphaFold "幻觉" (Hallucinations):**
   - **背景：** 使用 AlphaFold 生成的全新蛋白质骨架（Hallucinations），原始序列在通过大肠杆菌表达时通常不溶解（溶解率极低）。
   - **结果：** 使用 ProteinMPNN 对这些骨架重新设计序列后，**76%** 的设计（73/96）可溶表达，且许多表现出极高的热稳定性（耐受 95°C） 。
   - **结构确认：** 解析了一个单体设计的晶体结构，与设计模型高度吻合（RMSD 2.35 Å） 。
2. **蛋白质纳米颗粒 (Protein Nanoparticles):**
   - **挑战：** 设计由多个亚基组成的四面体纳米笼。之前使用 Rosetta 需要大量人工干预和计算。
   - **结果：** ProteinMPNN 自动化设计了 76 个序列，其中 13 个成功组装成预期的纳米颗粒。解析出的晶体结构与设计模型差异极小（1.2 Å RMSD） 。
3. **功能性蛋白设计 (Functional Design - SH3 Binder):**
   - **任务：** 设计一个支架蛋白，使其包含一段富含脯氨酸的肽段，从而结合 Grb2 SH3 结构域。这是一个 Rosetta 失败的难题。
   - **结果：** ProteinMPNN 设计的蛋白在实验中显示出与目标 SH3 结构域的强结合信号，证明了其在设计具体生化功能方面的潜力 。

# ProteinMPNN模型结构剖析

**B**: Batch size（批次大小）

**L**: Sequence length（蛋白质序列长度，填充后的最大长度）

**K**: Neighbors（K近邻的数量，代码中默认为30或32）

**C**: Hidden dimension（隐藏层维度，默认为128）

**V**: Vocab size（词表大小，21种氨基酸+特殊符号）

### 1. 数据特征化 (`featurize` 函数)

这一步将原始的 PDB 数据转换为模型可用的张量。

- **输入**: `batch` (包含蛋白质序列和坐标的字典列表)。
- **核心逻辑**:
  1. 提取 N, CA, C, O 四个原子的坐标。
  2. 处理多链（Chain）情况，构建 `chain_encoding` 和 `mask`。
  3. 进行 Padding（填充）以对齐 Batch 中的长度。
- **关键张量形状**:
  - **`X` (坐标)**: `[B, L, 4, 3]`
    - 4 代表四个原子 (N, Ca, C, O)。
    - 3 代表 (x, y, z) 坐标。
  - **`S` (序列标签)**: `[B, L]`。真实氨基酸的整数索引，用于计算 Loss。
  - **`mask`**: `[B, L]`。标识哪些位置是真实的残基，哪些是 Padding。
  - **`chain_M`**: `[B, L]`。标识哪些位置需要被预测（masked），哪些是已知上下文（visible）。

| **返回变量**           | **形状 (Shape)**    | **数据类型** | **含义**                                  |
| ---------------------- | ------------------- | ------------ | ----------------------------------------- |
| **X**                  | `[B, L_max, 4, 3]`  | float32      | 骨架原子坐标 (N, Ca, C, O)                |
| **S**                  | `[B, L_max]`        | long         | 真实氨基酸序列标签 (用于计算 Loss)        |
| **mask**               | `[B, L_max]`        | float32      | Padding 掩码 (1=真实残基, 0=填充)         |
| **lengths**            | `[B]`               | int32        | 每个样本的真实总长度                      |
| **chain_M**            | `[B, L_max]`        | float32      | **任务掩码**。1=需预测/设计，0=已知上下文 |
| **residue_idx**        | `[B, L_max]`        | long         | 残基位置索引 (含链间跳跃)                 |
| **mask_self**          | `[B, L_max, L_max]` | float32      | 链内/链间交互掩码 (主要用于辅助 Loss)     |
| **chain_encoding_all** | `[B, L_max]`        | long         | 链 ID (1, 2, 3...)，用于区分不同的链      |

**统一化**：将多链蛋白视为一个长的、不连续的单链进行处理。

**相对位置编码**：通过 `residue_idx` 的跳跃，让模型知道哪些残基在序列上是相邻的，哪些是断开的。

**任务定义**：通过 `chain_M` 区分哪些是 input context（如结合蛋白的 Target），哪些是 output design（如 Binder）。

**鲁棒性**：通过 `random.shuffle` 防止模型记住链的输入顺序。

### 2. 特征提取层 (`ProteinFeatures` 类)

这是模型的入口，负责将几何坐标转换为图（Graph）的节点和边特征。

- **逻辑**:
  1. **构建图 (KNN)**: 基于 CA 原子的距离，为每个残基找到最近的 `top_k` 个邻居。
  2. **构建坐标系**: 计算每个残基的局部坐标系（N-Ca, Ca-C 向量叉乘等），并计算虚拟 Cb 原子。
  3. **边特征 (Edge Features)**:
     - **距离 (RBF)**: 计算所有原子对（N-N, Ca-Ca, C-O 等 16种组合）的距离，并通过径向基函数（RBF）转化为向量。
     - **位置编码**: 结合残基索引差（Seq difference）和链索引（Chain index），判断是否同链以及序列距离。
- **关键张量形状**:
  - 输入 `X`: `[B, L, 4, 3]`
  - **`E_idx` (邻居索引)**: `[B, L, K]`。存储每个节点最近的 K 个邻居在 L 维度上的索引。
  - **`E` (边特征)**: `[B, L, K, C]`。
    - 这是图的边特征，包含了几何距离信息和序列位置信息。经过 `self.edge_embedding` 线性层映射到隐藏维度 `C`。

- **最终输出 `E`**: 形状 `[B, L, K, C]`。
- **最终输出 `E_idx`**: 形状 `[B, L, K]`。

`ProteinFeatures` 不生成节点特征（Node Features），它只生成**边特征（Edge Features）**。

### 3. 辅助函数 (Gather Functions)

由于是图神经网络，数据是非结构化的（即邻居不是固定的网格），需要通过索引来聚合信息。

- **`gather_nodes`**:
  - 从所有节点特征 `[B, L, C]` 中，根据邻居索引 `[B, L, K]` 取出邻居节点的特征。
  - 输出: `[B, L, K, C]`。
- **`cat_neighbors_nodes`**:
  - 将**邻居节点的特征**和**当前边特征**拼接，这是消息传递的基础。

### 4. 编码器层 (`EncLayer` 类)

这是模型的核心消息传递模块。ProteinMPNN 的编码器会同时更新**节点（Nodes）**和**边（Edges）**的信息。

- **架构**:
  1. **消息聚合**: 拼接 `h_V` (当前节点), `h_E` (边), `h_V_neighbor` (邻居节点)。
  2. **MLP**: 通过全连接层处理拼接后的特征。
  3. **Update Nodes (`h_V`)**: 对邻居维度 `K` 求和（`torch.sum`），得到聚合后的“消息”，更新当前节点特征。
  4. **Update Edges (`h_E`)**: 利用更新后的节点信息，再次通过 MLP 更新边特征。
- **输入/输出张量**:
  - 输入 `h_V`: `[B, L, C]` (节点特征，初始为0)。
  - 输入 `h_E`: `[B, L, K, C]` (边特征)。
  - 输出 `h_V`: `[B, L, C]` (更新后的节点)。
  - 输出 `h_E`: `[B, L, K, C]` (更新后的边)。

### 5. 解码器层 (`DecLayer` 类)

解码器也是消息传递层，但它是**自回归（Autoregressive）的，且不更新边特征**，只更新节点特征。

- **架构**:
  1. 逻辑与 Encoder 类似，拼接节点和边信息。
  2. **关键区别**: 引入了 `mask_attend`（通常由随机解码顺序生成）。确保节点在解码时，只能看到“过去”已解码的节点，看不到“未来”的节点。
- **输入/输出张量**:
  - 输入 `h_V`: `[B, L, C]` (当前解码状态)。
  - 输入 `h_E`: `[B, L, K, C]` (来自编码器的上下文信息，通常结合了序列Embedding)。
  - 输出 `h_V`: `[B, L, C]`。

### 6. 主模型 (`ProteinMPNN` 类)

将上述部分串联起来。

#### A. 初始化与特征提取

```python
E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
h_V = torch.zeros(...) # 初始节点特征全为0
h_E = self.W_e(E)      # 边特征投影
```

- `E`: `[B, L, K, C]`
- `h_V`: `[B, L, C]`

#### B. 编码器循环 (Encoder Loop)

```python
for layer in self.encoder_layers:
    h_V, h_E = layer(h_V, h_E, E_idx, ...)
```

通过3层 `EncLayer`，让几何信息在图上充分传播。此时模型“理解”了蛋白质的 3D 骨架结构。

#### C. 随机解码顺序 (Random Decoding Order) - **核心创新点**

```python
decoding_order = torch.argsort(...) 
permutation_matrix_reverse = ...
order_mask_backward = ...
```

ProteinMPNN 的一大特点是不按照 N端->C端 的固定顺序解码，而是**随机顺序**。

- 生成一个随机排列。
- 构建 `mask_attend`：形状 `[B, L, K, 1]`。用于在 Decoder 中屏蔽掉“未来”的邻居节点，只允许聚合“过去”（已生成序列）的邻居信息。

#### D. 解码器准备

```python
h_S = self.W_s(S) # 真实序列的 Embedding [B, L, C]
h_ES = cat_neighbors_nodes(h_S, h_E, E_idx) # 将序列信息拼接到边上
```

这里使用了 **Teacher Forcing**：在训练时，我们将真实的序列 `S` 作为输入，但在 Decoder 内部通过 mask 确保预测第 `i` 个氨基酸时，看不到 `i` 及其之后的真实氨基酸。

#### E. 解码器循环 (Decoder Loop)

```python
for layer in self.decoder_layers:
    h_V = layer(h_V, h_ESV, mask)
```

- `h_ESV` 融合了：几何信息(`h_E`) + 已知序列信息(`h_S`)。
- 通过 `mask_bw` (backward mask) 过滤，确保只能看到解码顺序中之前的序列信息。

#### F. 输出

```python
logits = self.W_out(h_V) # [B, L, 21]
log_probs = F.log_softmax(logits, dim=-1)
```

最终输出每个位置上 21 种氨基酸的对数概率。

### 数据流与形状变化

1. **Input**: `coords [B, L, 4, 3]`
2. **Features**: `Edges [B, L, K, C]` (基于几何距离)
3. **Encoder**:
   - Input: `Nodes [B, L, C]` (zeros), `Edges [B, L, K, C]`
   - Output: `Nodes [B, L, C]` (包含全局几何上下文), `Edges [B, L, K, C]` (更新后的边)
4. **Decoder Prep**:
   - `Seq Embed [B, L, C]` (真实序列)
   - `Random Mask [B, L, K]` (定义解码顺序)
5. **Decoder**:
   - Input: `Encoder Nodes`, `Seq Embed + Edges`
   - Output: `Nodes [B, L, C]` (包含序列+结构信息)
6. **Output**: `Logits [B, L, 21]`