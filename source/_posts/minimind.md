---
title: MiniMind复现笔记记录
mathjax: true
date: 2026/01/23 20:46:25
img: https://raw.githubusercontent.com/jingyaogong/minimind/refs/heads/master/images/logo.png
excerpt: Re-implement of MiniMind
---

# Minimind架构

MiniMind 本质上是一个 **Decoder-only Transformer**。它的核心配置逻辑在 `MiniMindConfig` 中。

- **基础架构**：类似于 LLaMA。
- **参数量级**：极其轻量。`hidden_size=512`, `num_hidden_layers=8`，`vocab_size=6400`。这属于“麻雀”级别的模型，适合个人显卡甚至 CPU 快速实验和复现。
- **特殊机制**：
  - **MoE (混合专家模型)**：可选开启。
  - **Weight Tying (权重共享)**：Embedding 层和输出层（LM Head）共享权重，这是小模型常用的压缩手段。

## A. 位置编码：RoPE + YaRN (Rotary Positional Embeddings)

- **基础 RoPE**：使用旋转位置编码，这是目前主流 LLM 的标配，比绝对位置编码（如 BERT）具有更好的外推性。

RoPE 的核心思想是将 token 的 embedding 向量看作复平面上的向量，通过**旋转角度**来表示位置信息。

对于 $d$ 维向量 (比如 hidden_size=512)，RoPE 将其切分成 $d/2$ 个子空间，每个子空间是 2 维的。
对于第 $j$ 组 (其中 $j \in [0, d/2)$)，我们定义不同的频率 $\theta_j$：

$$\theta_j = 10000^{-2j/d}$$

- 
  - **低频分量 ($j$ 很大)**：$\theta_j$ 接近 1，旋转很快。负责捕捉**局部信息**（比如相邻的词）。
  - **高频分量 ($j$ 很小)**：$\theta_j$ 接近 0，旋转很慢，波长很长。负责捕捉**全局长距离**

- **YaRN (Yet another RoPE extensioN)**：
  - 注意代码段 `if end / orig_max > 1.0:` 及其后的逻辑。
  - 这是一个**长文本外推技术**。当推理长度超过训练时的最大长度（`original_max_position_embeddings`）时，它通过对频率进行插值（Ramp function），动态调整 RoPE 的缩放因子。这意味着即使你用较短的序列训练，模型也有能力处理更长的上下文。

**与其让模型去预测未知的长距离（外推），不如通过修改频率，把长距离“压缩”回模型熟悉的短距离范围内（内插），同时还要保护短距离的精度。**
YaRN 不改变位置索引 $m$，而是修改基频 $\theta_j$。

新的频率 $\theta'_j$ 定义为：

$$\theta'_j = \theta_j \cdot (1 - \gamma(r) + \frac{\gamma(r)}{s})$$​

YaRN 的精髓在于它不是“一刀切”地压缩所有维度，而是分三段处理：

$$\gamma(r) = \begin{cases} 0, & \text{if } r < \alpha \quad (\text{高频/局部维度}) \\ 1, & \text{if } r > \beta \quad (\text{低频/全局维度}) \\ \frac{r - \alpha}{\beta - \alpha}, & \text{otherwise} \quad (\text{中频/过渡维度}) \end{cases}$$

除了修改频率，YaRN 还引入了一个温度系数 $\sqrt{t}$ 来修正注意力分数的幅值：

$$\text{Attention}(Q, K) = \text{softmax}(\frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d} \cdot \sqrt{t}})$$

## B. 注意力机制：GQA (Grouped Query Attention)

### MHA: Multi-Head Attention

每个 Query 头都有自己专属的 Key 和 Value 头。
- **优点**：捕捉信息的能力最强，每个头都能独立“看”不同的特征。
- **缺点**：推理时显存占用极大（KV Cache 很大），计算慢。

### MQA: Multi-Query Attention

所有 Query 头，**共用这唯一的一组** Key 和 Value 头。
- **优点**：KV Cache 极小，推理速度飞快。
- **缺点**：效果下降明显，因为所有头都在看“同一份”上下文特征，容易导致训练不稳定。

### GQA: Group Query Attention

将 Query 头分成几组（Group），**每组**内的 Query 头共享一组 Key/Value。

### KV-cache

KV cache 是 Transformer decoder 在自回归推理时，用显存缓存历史 token 的 Key / Value，从而把生成复杂度从 O(T²) 降到 O(T) 的核心机制。

对第$l$层：

$$Q_t^{(l)} = h_t^{(l)} W_Q^{(l)}$$$$K_t^{(l)} = h_t^{(l)}W_K^{(l)}$$
$$V_t^{(l)} = h_t^{(l)}W_V^{(l)}$$

attention 输出：

$$\text{Attn}_t^{(l)} = \text{softmax}\left( \frac{Q_t^{(l)} [K_1^{(l)}, \dots, K_t^{(l)}]^T} {\sqrt{d_h}} \right) [V_1^{(l)}, \dots, V_t^{(l)}]$$

$$\boxed{ \text{KVCache}^{(l)} = \left( \{K_1^{(l)}, \dots, K_{t-1}^{(l)}\}, \{V_1^{(l)}, \dots, V_{t-1}^{(l)}\} \right) }$$

第$t$步只做：

- 计算$ Q_t, K_t, V_t$
  
- append 到 cache
  
- 用 $Q_t$读整个 cache

$$\text{显存占用} = 2 \times \text{Batch} \times \text{Seq\_Len} \times \text{KV\_Heads} \times \text{Head\_Dim} \times \text{Byte}$$
### Flash Attention

在标准 Attention 计算中：

$$\text{Score} = \text{Softmax}(Q K^T)$$

$$\text{Out} = \text{Score} \cdot V$$

这里有一个巨大的中间矩阵：**Attention Score Matrix**，大小是 $N \times N$（序列长度的平方）。
**显存读写速度 (HBM)** 远慢于 **GPU 核心计算速度 (SRAM)**。GPU 大部分时间都在等数据传输。
Flash Attention 的核心逻辑是：**切块 (Tiling)**。

- **不存大矩阵**：它不把完整的 $N \times N$ 矩阵写入显存。
  
- **分块计算**：它把 $Q, K, V$ 切成小块，放入 GPU 核心极快的小缓存 (SRAM) 中。
  
- **即算即丢**：在 SRAM 里算完局部的 Attention，直接更新最终结果，然后丢弃中间值。如果反向传播需要用到，宁可重新算一遍，也不去读写慢速的显存。

## C. 归一化与激活函数 (Norm & Activation)

### Normalization

为什么需要归一化？ 简单的说，深度神经网络层数太多，数据经过一层层矩阵乘法，数值分布会剧烈波动（忽大忽小）。如果不加以约束，梯度就会爆炸或者消失，模型根本训练不起来。 归一化的作用就是：**强行把每一层的数据拉回到一个标准的分布（比如均值为0，方差为1）**，让训练更稳定。
#### BatchNorm
它是在 **Batch（批次）** 维度上做归一化。
- **依赖 Batch Size**：如果 Batch Size 太小（比如小显存训练），BN 估算的统计量就不准，导致效果极差。
  
- **序列变长**：NLP 里的句子有长有短，用 Padding 补齐。BN 很难处理这种变长数据（Padding 里的 0 会污染统计值）。
  
- **RNN/Transformer 特性**：文本是生成的，推理时每次只进来一个 Token，没有 Batch 的概念，BN 在推理时非常别扭。

#### LayerNorm
它是在 **Feature（特征）** 维度上做归一化。
**不管 Batch Size 是多少**，哪怕只有一句话。它只看这句话里的某一个 Token，把这个 Token 的 512 维向量（Hidden Size）拿来算均值和方差，自己归一化自己。

公式：

$$y = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta$$

- $\mu$：均值 (Center)。
  
- $\sigma$：标准差 (Scale)。
  
- **直觉**：把数据平移（减均值）到 0 附近，再缩放（除标准差）到 1 附近。

推理/训练一致性：BS可能不一致，LN完全一致

#### RMSNorm
- 原理：它是 LayerNorm 的简化版。
  
    作者发现，LayerNorm 中“减去均值 $\mu$”这一步其实没啥大用，真正起作用的是“除以标准差”这一步（控制幅值）。于是干脆把减均值去掉了。
    
- 公式：
    $$y = \frac{x}{\text{RMS}(x)} \cdot \gamma$$$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum x_i^2}$$

RMSNorm 去掉了 Bias ($\beta$)，只保留了缩放参数 Weight ($\gamma$)。

### Activation Function

#### Sigmoid/Tanh
- **形状**：S 型曲线。
  
- **问题**：**梯度消失**。当输入很大或很小时，导数趋近于 0，导致深层网络无法训练。

#### ReLu(Rectified Linear Unit)
- **公式**：$f(x) = \max(0, x)$
  
- **形状**：折线。小于 0 的直接砍掉变成 0，大于 0 的保持不变。
  
- **优点**：计算极快，解决了梯度消失。
  
- **缺点**：**Dead ReLU**。如果输入小于 0，梯度直接没了，神经元“死”了。

#### GELU (Gaussian Error Linear Unit) —— BERT/GPT-2 时代

- **直觉**：它是 ReLU 的平滑版本。ReLU 在 0 处有个尖角，GELU 把这个角磨圆了。
  
- **原理**：引入了概率的思想（高斯分布累积分布函数），使得负值不是直接变成 0，而是有一个平滑的过渡。
$$GELU(x)=x\cdot \Phi(x)$$

其中：

$$\Phi(x) = P(Z \le x), \quad Z \sim \mathcal{N}(0,1)$$
| 区域  | 行为               |
| ----- | ------------------ |
| x ≪ 0 | 近似 0，但不是硬 0 |
| x ≈ 0 | 平滑过渡           |
| x ≫ 0 | 近似线性（≈ x）    |
#### SiLU(Sigmoid Linear Unit)/Swish——Llama时代

- **公式**：$f(x) = x \cdot \text{sigmoid}(x)$
  
- **形状**：和 GELU 非常像，长得几乎一样。
  
- **特点**：
  
    - **平滑**：处处可导。
      
    - **非单调**：在 $x$ 为负值的小区间内（约 -2 到 0），它会输出一个微小的负数，而不是像 ReLU 那样直接归零。这意味着它能保留一点点负区间的梯度信息。
    
- **为什么选它？**：Google 的搜索实验发现它比 ReLU 和 GELU 效果略好一点点。对于大模型来说，能好一点点也是好的。

## FFN层

### 传统Transformer

是 Up -> Activation -> Down 的结构：

$$y = \text{Down}(\text{ReLU}(\text{Up}(x)))$$

这里只有两个矩阵：Up 和 Down。

### SwiGLU FFN
```python
self.gate_proj = nn.Linear(...) # 门控层 
self.up_proj = nn.Linear(...) # 信号层 
self.down_proj = nn.Linear(...) # 输出层 
# forward 函数 
return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```
公式为：

$$y = \text{Down}(\text{SiLU}(\text{Gate}(x)) \times \text{Up}(x))$$

什么GLU就看他的激活函数
实践中：**SwiGLU ≈ GeGLU > ReGLU**
SiLU 门控特点：

- 小值 → 抑制
  
- 大值 → 放行
  
- 连续可微

## MoE(Mixture of Experts)实现

### 混合专家结构 (Shared + Routed)

**Routed Experts**：根据输入 Token 的不同，动态选择激活其中的某几个。

**Shared Experts**：无论输入是什么，**总是被激活**。

**设计意图**：共享专家负责捕获通用的、基础的知识，而路由专家负责捕获专门的、细分的知识。这比传统的纯路由 MoE 更稳定。

### 门控机制 (Gating)

在 `MoEGate` 中：

- **Top-K 路由**：使用 `softmax` 计算分数，选出得分最高的 K 个专家 (`num_experts_per_tok`)。
- **权重归一化**：`norm_topk_prob`，确保选出的专家权重之和为 1。

### 负载均衡 (Load Balancing)

为了防止“专家坍塌”（即某些专家一直被选中，而其他的饿死），代码计算了 `aux_loss`（辅助损失）。

它强迫门控网络尽可能均匀地将 Token 分配给不同的专家。

## 训练和推理差异

**Training**：使用 `repeat_interleave`。简单来说，就是把数据复制扩展，并行地喂给所有专家，然后用掩码（Mask）把不属于该专家的计算结果过滤掉。这种方法对 GPU 并行计算友好，但显存开销大。

**Inference**：调用 `moe_infer`。这是一个优化的推理路径。它不对数据进行复制，而是根据索引对 Token 进行排序 (`argsort`)，将属于专家 A 的 Token 聚在一起一次性计算，再把结果写回原位。这大大减少了推理时的计算量。

# Tokenizer

**读数据**：从 JSONL 语料中读取文本。

**造字典**：使用 **BPE (Byte-Pair Encoding)** 算法，统计高频词汇，生成一个只有 6400 个词的词表。

**定规则**：定义“特殊符号”（如 `<|im_start|>`）和“对话模板”（Chat Template），并保存为 HuggingFace 兼容的格式。

- VOCAB_SIZE = 6400

**优点**：Embedding 层和最后的 LM Head 层参数量极小（$6400 \times Hidden\_Dim$），极大降低显存占用。

**缺点**：单个 Token 包含的信息量变少，同样的句子会被切成更多的 Token，推理速度变慢（因为生成的步数变多了）。

- BPE+ByteLevel

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

**BPE (字节对编码)**：这是目前最主流的分词算法（GPT-2/3/4, Llama 都在用）。它通过不断合并出现频率最高的字符对来构建词表。

**ByteLevel**：这是关键。它不直接处理 Unicode 字符（如“中”），而是先把所有文本转成 **UTF-8 字节流**。

例如：“中” -> `0xE4 0xB8 0xAD` (3个字节)。

**优势**：彻底解决了 **OOV (Out of Vocabulary)** 问题。无论遇到多生僻的字，甚至 Emoji，最差的情况就是退化成单个字节，绝对不会出现 `<UNK>`（未知字符）。

- chat_template

```python
"chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' -%}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else -%}\n        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n   {{- '<|im_start|>' + message.role + '\\n' + content }}\n  {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
```

**这是 Jinja2 模板**。它的作用是告诉 HuggingFace 的 `tokenizer.apply_chat_template` 函数：**如何把一个 Python 的对话列表（List of Dicts）转换成模型能读懂的字符串。 Tokenizer 层面就原生支持了**System Prompt**、**多轮对话**甚至**工具调用（Tools）**。**

- 流式解码

```python
# 流式解码（字节缓冲）测试
for tid in input_ids:
    token_cache.append(tid)
    current_decode = tokenizer.decode(token_cache)
    if current_decode and '\ufffd' not in current_decode:
        # ... 打印 ...
        token_cache = []
```

中文在 UTF-8 中通常占 **3个字节**。

当模型生成 Token A (`E6`) 时，你立即 Decode，解码器发现 `E6` 是个残缺的字节，无法显示成汉字，就会显示`` (即代码里的 \ufffd, Replacement Character)。

**代码逻辑**：它维护一个 `token_cache`。如果解码结果里有 `\ufffd`，说明当前的字节流拼不出一个完整的字，那就**先存着不打印**。等到 Token B (`88 91`) 来了，拼成 `E6 88 91`，`\ufffd` 消失了，再打印。

# Pretrain

LLM首先要学习的并非直接与人交流，而是让网络参数中充满知识的墨水，“墨水” 理论上喝的越饱越好，产生大量的对世界的知识积累。 预训练就是让Model先埋头苦学大量基本的知识，例如从Wiki百科、新闻、书籍整理大规模的高质量训练数据。 这个过程是“无监督”的，即人类不需要在过程中做任何“有监督”的校正，而是由模型自己从大量文本中总结规律学习知识点。 模型此阶段目的只有一个：**学会词语接龙**。

## Dataloader定义

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 1. 加载纯文本数据
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 2. 分词 (Tokenization)
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # 3. 构造 Labels (关键点)
        input_ids = encoding.input_ids.squeeze()
        labels = input_ids.clone()
        # 4. Masking Padding
		# 把所有补齐的 Pad Token 对应的 Label 设为 -100
		# PyTorch 的 CrossEntropyLoss 会自动忽略 -100，不计算梯度
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
```

## 核心Epoch

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        #【断点续训】
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 根据当前进度，计算出这一步应该用的学习率。通常是余弦退火 Cosine Decay，强行赋值给优化器
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
		# autocast_ctx 是混合精度的上下文管理器。
        # 在这个范围内，模型的运算（如卷积、矩阵乘法）会自动转为 float16 或 bfloat16 以节省显存并加速。
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # res.loss 是语言模型的主 Loss（预测下一个词准不准）。
            # res.aux_loss 是 MoE 的辅助 Loss（专家有没有负载均衡）
            loss = res.loss + res.aux_loss
            # 梯度累积标准化，平均梯度
            loss = loss / args.accumulation_steps
		# scaler 是 GradScaler（梯度缩放器），专用于 float16 混合精度训练。
        # 为什么要 scale？因为 float16 精度低，Loss 经常很小（比如 0.00001），
        # 算出的梯度可能下溢出变成 0。scaler 先把 Loss 放大（比如乘 65536），
        # 算出放大的梯度，后面更新时再缩放回来。
        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            # 1. Unscale：把之前放大的梯度缩放回正常大小。
            # 必须在 clip_grad_norm 之前做，否则裁剪的阈值就不准了。
            scaler.unscale_(optimizer)
            # 2. 梯度裁剪 (Gradient Clipping)：
            # 限制梯度的最大范数为 args.grad_clip (通常是 1.0)。
            # 防止“梯度爆炸”，这是训练 LLM 稳定性的关键保险丝。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
			# scaler.step 内部会检查梯度是否有 Inf/NaN。如果有，这步就跳过不更新，防止模型崩坏。
            scaler.step(optimizer)
            # 4. 更新缩放因子：根据这步有没有溢出，动态调整下一次的缩放倍数。
            scaler.update()
			# 5. 清空梯度：准备下一轮累积。
            # set_to_none=True 比 =0 稍微快一点点（节省显存操作）。
            optimizer.zero_grad(set_to_none=True)
		##打印日志.....
		
        # 只有主进程（Rank 0）负责保存，避免多个进程同时写文件导致损坏。
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 如果用了 DDP，模型被包了一层 .module
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            # 如果用了 torch.compile，模型被包了一层 _orig_mod
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 1. 保存权重文件 (.pth)：
            # .half()：转成 float16 保存，文件体积减半。
            # .cpu()：搬到 CPU，防止爆显存。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 2. 保存训练状态 (Checkpoint)：
            # 这是一个单独的文件，存了 optimizer、scaler、当前 epoch 和 step。
            # 这是为了万一训练中断，下次能完美恢复现场（Resume）用的。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, res, loss
```

## 主程序

```python
    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    # 每个进程种子不一样，保证每个进程的数据增强（如果有）和 Dropout 行为不完全一样，增加随机性。
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # lm_checkpoint 不仅仅是加载，它会去 ../checkpoints 目录下找，有没有 args.save_weight 前缀的文件。
	# 如果找到了，就会把文件的路径、epoch、step 等信息读出来放到 ckp_data 字典里。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 创建上下文管理器，后面传给 train_epoch 用
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
	# 略
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 在 DDP 模式下，它负责把整个数据集切成 N 份（N=显卡数）。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler：只有 float16 需要。
	# 如果是 bfloat16，enabled=False，它就什么都不做（空转）。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0: # 第一个epoch且存在检查点
            # SkipBatchSampler 是自定义类。
        	# 它的作用是：快速跳过前 1000 个 batch，只生成 index，不读硬盘数据。
        	# 如果不用这个，Dataloader 傻傻地读前 1000 个数据然后丢掉，启动会巨慢无比。
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        else: # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
```

```bash
torchrun --nproc_per_node 1 train_pretrain.py # 1即为单卡训练，可根据硬件情况自行调整 (设置>=2)
# or
python train_pretrain.py
```

把[匠数大模型数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)的中文部分提取出来， 清洗出字符`<512`长度的大约1.6GB的语料直接拼接成预训练数据 `pretrain_hq.jsonl`，hq即为high quality

文件`pretrain_hq.jsonl` 数据格式为

```
{"text": "如何才能摆脱拖延症？ 治愈拖延症并不容易，但以下建议可能有所帮助..."}
```

# SFT

经过预训练，LLM此时已经掌握了大量知识，然而此时它只会无脑地词语接龙，还不会与人聊天。 SFT阶段就需要把半成品LLM施加一个自定义的聊天模板进行微调。 例如模型遇到这样的模板【问题->回答，问题->回答】后不再无脑接龙，而是意识到这是一段完整的对话结束。 称这个过程为指令微调。 在训练时，MiniMind的指令和回答长度被截断在512，是为了节省显存空间。就像学习写作时，会先从短的文章开始，当学会写作200字作文后，800字文章也可以手到擒来。 在需要长度拓展时，只需要准备少量的2k/4k/8k长度对话数据进行进一步微调即可（此时最好配合RoPE-NTK的基准差值）。

## DataLoader定义

```python
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)
    
	#根据模型 tokenizer 自带的 chat template，把多轮对话拼成 prompt
    def create_chat_prompt(self, cs):
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        # 1. 初始化：假设全都不需要学 (-100)
        labels = [-100] * len(input_ids)
        # 2. 扫描 input_ids 序列
        i = 0
        while i < len(input_ids):
            # 3. 发现 Assistant 开始说话的特征序列 (self.bos_id)
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                # 4. 寻找 Assistant 说话结束的地方 (self.eos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 5. 【关键】解除 Mask
            	# 只把 [start, end] 这段区间的 labels 设为 input_ids 的值
            	# 这意味着模型只会在这一段计算 Loss
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

## 核心Epoch

和pretrain差不多，唯一差在SFT把非回答部分labels设置为-100（PyTorch 的 `CrossEntropyLoss` 有一个隐藏机制：**它会自动忽略值为 -100 的 Label**）

所以只会计算答案部分的loss

## 主程序

和pretrain一样

```python
torchrun --nproc_per_node 1 train_full_sft.py
# or
python train_full_sft.py
```

# 知识蒸馏KD

在前面的所有训练步骤中，模型已经完全具备了基本能力，通常可以学成出师了。 而知识蒸馏可以进一步优化模型的性能和效率，所谓知识蒸馏，即学生模型面向教师模型学习。 教师模型通常是经过充分训练的大模型，具有较高的准确性和泛化能力。 学生模型是一个较小的模型，目标是学习教师模型的行为，而不是直接从原始数据中学习。 在SFT学习中，模型的目标是拟合词Token分类硬标签（hard labels），即真实的类别标签（如 0 或 6400）。 在知识蒸馏中，教师模型的softmax概率分布被用作软标签（soft labels）。小模型仅学习软标签，并使用KL-Loss来优化模型的参数。 通俗地说，SFT直接学习老师给的解题答案。而KD过程相当于“打开”老师聪明的大脑，尽可能地模仿老师“大脑”思考问题的神经元状态。 例如，当老师模型计算`1+1=2`这个问题的时候，最后一层神经元a状态为0，神经元b状态为100，神经元c状态为-99... 学生模型通过大量数据，学习教师模型大脑内部的运转规律。这个过程即称之为：知识蒸馏。 知识蒸馏的目的只有一个：让小模型体积更小的同时效果更好。 然而随着LLM诞生和发展，模型蒸馏一词被广泛滥用，从而产生了“白盒/黑盒”知识蒸馏两个派别。 GPT-4这种闭源模型，由于无法获取其内部结构，因此只能面向它所输出的数据学习，这个过程称之为黑盒蒸馏，也是大模型时代最普遍的做法。 黑盒蒸馏与SFT过程完全一致，只不过数据是从大模型的输出收集，因此只需要准备数据并且进一步FT即可。 

损失，一部分跟着硬标签学，一部分跟着老师（软标签）学

- 蒸馏损失 (`distillation_loss`)

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        # 1. 教师的软标签 (Soft Targets)
        # 温度 T (temperature) 越高，概率分布越平滑，包含的“暗知识”越多。
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 2. 学生的 Log 概率
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # 3. 计算 KL 散度 (Kullback-Leibler Divergence)
    # 衡量两个概率分布的差异。我们要让学生的分布尽可能靠近老师。
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    
    # 4. 梯度缩放
    # 为什么要乘 T^2？因为除以 T 后，梯度的幅值会变小 1/T^2，为了保持梯度量级，必须乘回去。
    return (temperature ** 2) * kl
```

- 训练循环的双重前向传播 (`train_epoch`)

- 损失函数$$Loss_{total} = \alpha \cdot Loss_{CE} + (1 - \alpha) \cdot Loss_{Distill}$$​
- Masking 策略的复用，蒸馏训练依然遵循 SFT 的原则：**只蒸馏回答部分**。

- 显存与性能的挑战：虽然 Teacher 不需要存梯度（Gradients）和优化器状态（Optimizer States），但它的 **参数 (Weights)** 和 **中间激活值 (Activations)** 依然占用大量显存。

# LoRA

LoRA是一种高效的参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）方法，旨在通过低秩分解的方式对预训练模型进行微调。 相比于全参数微调（Full Fine-Tuning），LoRA 只需要更新少量的参数。 LoRA 的核心思想是：在模型的权重矩阵中引入低秩分解，仅对低秩部分进行更新，而保持原始预训练权重不变。

- 定义Lora层

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        # rank: 秩。比如 in=512, out=512, rank=8。
        # 原矩阵参数量：512*512 = 26万
        # LoRA参数量：512*8 + 8*512 = 8千。参数量压缩了 32 倍。
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)  # 降维矩阵
        self.B = nn.Linear(rank, out_features, bias=False) # 升维矩阵
        
        # A 矩阵：高斯分布初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # B 矩阵：全 0 初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        # forward = B(A(x))
        return self.B(self.A(x))
```

- 注入Lora

```python
def apply_lora(model, rank=8):
    # 遍历模型的所有层
    for name, module in model.named_modules():
        # 筛选条件：必须是 Linear 层，且是方阵 (in == out)
        # 为什么要限制方阵？
        # 作者这里做了一个简化假设：只对 Attention 的 Q, K, V, O 投影层做 LoRA，
        # 而在 Transformer 中这些层通常是 hidden_size -> hidden_size 的方阵。
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            
            # 1. 创建一个小 LoRA 模块挂在原模块身上
            lora = LoRA(..., rank=rank)
            setattr(module, "lora", lora) # 相当于 module.lora = lora
            
            # 2. 劫持 forward 函数
            original_forward = module.forward # 保存原版 forward

            # 定义一个新的 forward
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                # 结果 = 原版路径(x) + LoRA旁路(x)
                return layer1(x) + layer2(x)
            
            # 3. 覆盖掉原版
            module.forward = forward_with_lora
```

- 保存与加载

```python
def save_lora(model, path):
    # ...
    # 只提取名字里带 'lora' 的参数
    lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
    state_dict.update(lora_state)
    # 保存。生成的文件非常小（几 MB），而不是几 GB。
    torch.save(state_dict, path)
```

- 训练

```python
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True  # LoRA 参数要更新
        lora_params.append(param)
    else:
        param.requires_grad = False # 原模型参数冻结
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

此时【基础模型+LoRA模型】即可获得垂直场景模型增强的能力，相当于为基础模型增加了LoRA外挂，这个过程并不损失基础模型的本身能力。

PS：只要有所需要的数据集，也可以full_sft全参微调（需要进行通用知识的混合配比，否则过拟合领域数据会让模型变傻，损失通用性）

# 推理模型训练（蒸馏推理）

DeepSeek R1论文指出`>3B`的模型经历多次反复的冷启动和RL奖励训练才能获得肉眼可见的推理能力提升。 最快最稳妥最经济的做法，以及最近爆发的各种各样所谓的推理模型几乎都是直接面向数据进行蒸馏训练， 参数太小的模型直接通过冷启动SFT+GRPO几乎不可能获得任何推理效果

做蒸馏需要准备的依然是和SFT阶段同样格式的数据即可

```python
{
  "conversations": [
    {
      "role": "user",
      "content": "你好，我是小芳，很高兴认识你。"
    },
    {
      "role": "assistant",
      "content": "<think>\n你好！我是由中国的个人开发者独立开发的智能助手MiniMind-R1-Lite-Preview，很高兴为您提供服务！\n</think>\n<answer>\n你好！我是由中国的个人开发者独立开发的智能助手MiniMind-R1-Lite-Preview，很高兴为您提供服务！\n</answer>"
    }
  ]
}
```

回复模板是

```python
<think>\n思考过程\n</think>\n
<answer>\n最终回答\n</answer>
```

这在GRPO中通过设置规则奖励函数约束模型符合思考标签和回复标签（在冷启动靠前的阶段奖励值设置应该提高一些）

- 特殊设计

```python
sp_ids = torch.isin(shift_labels.view(-1), torch.tensor(start_of_think_ids + ...))
loss_mask_flat[sp_ids] = 10
```

它将 `<think>`, `</think>`, `<answer>`, `</answer>` 这些关键结构标记的 Loss 权重设为了 **10倍**。

**目的**：强制模型学会何时开始思考、何时结束思考。这是为了在 SFT 阶段更好地规范模型的推理格式，防止模型输出混乱的标签。

**这段代码 (SFT)**：它的目标是 **“预测下一个字” (Next Token Prediction)**。它在学习老师（比如 DeepSeek-R1）是怎么说话的。如果老师说 "因为A所以B"，模型就死记硬背 "因为A所以B"。它并不判断 "B" 对不对，它只在乎由于老师说了 "B"，所以我也要说 "B"。

**真推理 (RL - PPO/GRPO)**：目标是 **“最大化奖励” (Maximize Reward)**。模型尝试输出 "因为A所以C"，发现奖励很低（做错了）；下次尝试 "因为A所以B"，发现奖励很高（做对了）。通过这种试错，模型才真正理解了逻辑链条的有效性。

**格式规范**：那 10 倍的 Loss 权重 (`loss_mask_flat[sp_ids] = 10`) 就是为了强迫小模型学会：“遇到问题先输出 `<think>`，把过程写完，再输出 `<answer>`”。这是一种**思维链（Chain-of-Thought, CoT）的格式注入**。

**引导逻辑**：虽然小模型没有自我探索，但通过模仿大模型（R1）的高质量思维过程，它能学会“拆解问题”的模式。

- *以前的模型*：问题 -> 瞎猜答案。
- *蒸馏后的模型*：问题 -> 模仿老师的步骤 1 -> 模仿步骤 2 -> 得出答案。
- 结果是：虽然是模仿，但因为步骤对了，答案往往也变准了。

# 强化学习后训练

LLM里的强化学习方法可分两类：

1. **基于人类反馈的强化学习 (Reinforcement Learning from Human Feedback, RLHF)**

- 通过**人类**对模型输出的偏好进行评价来训练模型，使其生成更符合人类价值观和偏好的内容。

1. **基于AI反馈的强化学习 (Reinforcement Learning from AI Feedback, RLAIF)**

- 使用**AI模型**（通常是预训练的语言奖励模型）来提供反馈，而不直接依赖人类的人工标注。
- 这里的“AI”也可以是某些规则奖励，例如数学答案/代码解释器...

| 类型  | 裁判 | 优点               | 缺点                 |
| ----- | ---- | ------------------ | -------------------- |
| RLHF  | 人类 | 更贴近真实人类偏好 | 成本高、效率低       |
| RLAIF | 模型 | 自动化、可扩展性强 | 可能偏离人类真实偏好 |

二者本质上是一样的，都是通过**强化学习的方式**，利用某种形式的"**反馈**"来优化模型的行为。

除了**反馈**的来源不同，其他并无任何区别。

## **基于人类反馈的强化学习 (Reinforcement Learning from Human Feedback, RLHF)**

### DPO(Direct Preference Optimization)

直接偏好优化（DPO）算法，损失为：



其中：

- **策略项**: f(rt)=log⁡rw−log⁡rl (对比chosen vs rejected的概率比)
- **优势项**: g(At) = / (通过偏好对比，无需显式计算优势)
- **正则项**: h(KLt) = 隐含在 β 中 (控制偏离参考模型程度)

特别地，

- DPO从PPO带KL约束的目标推导出对偏好对的解析训练目标，直接最大化"chosen优于rejected"的对数几率；无需同步训练Reward/Value模型。DPO只需跑`actor`与`ref`两个模型，显存占用低、收敛稳定、实现简单。
- 训练范式：off‑policy，使用静态偏好数据集，可反复多轮epoch；Ref模型固定（预先缓存输出）。
- DPO的局限在于不做在线探索，更多用于"偏好/安全"的人类价值对齐；对"能不能做对题"的智力能力提升有限（当然这也取决于数据集，大规模收集正反样本并人类评估很困难）。

```python
torchrun --nproc_per_node 1 train_dpo.py
# or
python train_dpo.py
```

## **基于AI反馈的强化学习 (Reinforcement Learning from AI Feedback, RLAIF)**

