---
title: Attention Is All You Need阅读笔记
mathjax: true
date: 2025/3/11 20:46:25
img: https://img0.baidu.com/it/u=3520508,2967101156&fm=253&fmt=auto&app=138&f=JPEG?w=786&h=500
excerpt: transformer论文阅读笔记
---

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/attention-%E8%AE%A1%E7%AE%97%E5%9B%BE.png)

# 之前的工作

RNN & LSTM

1. **长距离依赖建模困难（梯度消失/爆炸）**：随着序列长度的增加，RNN在捕捉远距离元素间依赖关系时效果不佳，训练过程也变得不稳定。尽管LSTM和GRU在一定程度上缓解了这个问题，但**仍然存在长距离依赖建模的瓶颈**。
2. **串行计算的限制**：每一步的计算都严格依赖于前一步的结果，导致模型**无法充分利用并行计算资源**，严重制约了训练和推理的速度。

CNN

1. **局部感受野的局限**：CNN通过滑动卷积核提取局部特征，要建模长距离依赖关系，**通常需要堆叠非常深的网络层**，这不仅显著增加了模型参数量，也大大提升了计算复杂度。
2. **信息流动效率低**：序列起始位置的信息需要经过多层卷积操作才能传递到尾部位置，**信号路径过长可能导致信息衰减或丢失**。

Transformer

1. **全局信息交互**：自注意力机制允许序列中的**任意两个位置直接建立联系并进行信息交互**，极大地提升了信息传递的效率和范围，从根本上解决了长距离依赖问题。
2. **强大的并行计算能力**：自注意力层的计算可以**在序列长度维度上完全并行化**，这充分利用了现代硬件（如GPU/TPU）的并行计算能力，**显著加速了模型的训练和推理过程**。

# 模型架构

输入$(x_1,x_2,...,x_n)$，编码器输出$(z_1,z_2,...,z_n)$，其中$z_t$是$x_t$的嵌入向量，输入到解码器，根据$y_1$到$y_{t-1}$以及$z_t$输出$y_t$

注意力机制的核心有3个重要的值决定：一个是Q，代表查询变量。一个是K，代表应答变量。一个是V，代表值。Q和K之间计算注意力系数，决定最终取用值的多少。

![pic-1](/img/transformer-notes/pic-1.png)

## 编码器

N=6，每一层有两个子层：多头注意力层和前馈神经网络层，每个子层后面还有一个残差网络和层归一化(Add & Norm)，即每个子层的输出是

$LayerNorm(x + Sublayer(x))$

$d_k = 512$

## LayerNorm & BatchNorm

| 归一化类型    | 归一化维度                                | 适用场景                       |
| :------------ | :---------------------------------------- | :----------------------------- |
| **BatchNorm** | 对同一特征通道跨样本（Batch维度）归一化   | CNN等固定输入结构的模型        |
| **LayerNorm** | 对同一样本的所有特征（Channel维度）归一化 | RNN、Transformer等变长序列模型 |

![pic-3](/img/transformer-notes/pic-3.png)

蓝色是batchnorm，黄色是layernorm

BN对每一个**特征**在一个小批量（mini-batch）计算均值和方差，然后对整个小批量进行归一化，推理的时候要记录全局的均值和方差

由于每个seq的长度可能不一样，导致对不同seq的某个feature进行归一化的时候容易出现抖动（特别是batch比较小的时候）

而LN对每一个**样本**在一个小批量（mini-batch）计算均值和方差，然后对整个小批量进行归一化，训练和推理行为一致

都是在每个样本自身里面归一化，所以比较稳定

$$y=\frac{x-E[x]}{Std[x]+\epsilon}*\gamma+\beta$$

![pic-4](/img/transformer-notes/pic-4.png)

## 解码器

解码器的输入包含两个部分：

**编码器输出** 编码器输出的是英文序列里每个token的embedding。它的维度为512。经过多层编码器的自注意力机制，每个token的embedding都已经根据上下文，计算出恰当的语义信息。它们将作为解码器输出的重要参考。

**已经翻译出来的token序列** Transformer的编码器可以一次性输入完整的英文token序列，但在模型进行实际翻译时，需要解码器逐个生成对应的中文token序列。`<bos>`作为解码器的初始输入，代表序列开始。

对于Transformer模型在推理时，解码器确实如上述过程所述，是逐个生成中文token的。但是在训练时，因为我们已经知道英文对应的中文token序列，所以我们可以通过一种叫做**带掩码的多头注意力机制**（Masked Multi-Head Attention，MMHA）来实现并行化训练。

N=6，在输入添加一个掩码多头注意力，和encoder有两个一样的子层

此外多了一个掩码多头注意力层

## 注意力机制

通过Q，K，V，计算两个词的相似度

Transformer采用自注意力机制

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/QKV-%E7%9F%A9%E9%98%B5%E8%A1%A8%E7%A4%BA.jpg)

## Scaled Dot-Product Attention

Q，K的维度都是$d_k$，V的维度是$d_v$

Q，K做内积，再除以$\sqrt{d_k}$，做一层`softmax`（对每一行，dim=1）就是V的权重

$Attention(Q,K,V) = softmax( \frac{QK^T}{\sqrt{d_k}}  )V$

为了防止dk过大或者过小，使得softmax的值趋于0或者1导致softmax的梯度很小减慢训练速度，所以处理$\sqrt{d_k}$

![pic-5](/img/transformer-notes/pic-5.png)

## 多头自注意力机制

h=8

**多头相当于把原始信息 Source 放入了多个子空间中，也就是捕捉了多个信息，对于使用 multi-head（多头） attention 的简单回答就是，多头保证了 attention 可以注意到不同子空间的信息，捕捉到更加丰富的特征信息**。

也就是先对QKV进行投影到一个新维度，进行h次注意力计算，把h个结果拼接起来，通过Linear投影回原来的维度

例如$z_i \in R^{2×3}$，拼接起来就是$R^{2×24}$，再内积$W^O \in R^{24×4}$，最终得到$Z \in R^{2×4}$，通过这样操作，可学习的参数就会大大增加

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/multi-head-%E6%8B%BC%E6%8E%A5.jpg)

投影参数矩阵

$W^Q_i \in R^{d_{model}×d_k}$ 

$W^K_i \in R^{d_{model}×d_k}$ 

$W^V_i \in R^{d_{model}×d_v}$ 

$W_O \in R^{ {hd_v}×d_{model} }$

$d_k = d_v = d_{model}/h = 64$

计算消耗和一次注意力差不多，但是能学到更多的信息。

## masked多头注意力

在计算的时候，只考虑前面出现过的，所以他的输入是output embedding

计算出来的结果作为Q和input进行交叉注意力

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/mask-attention-map-softmax.jpg)

## Encoder-Decoder注意力层

KV来自编码器，Q来自解码器

## 前馈神经网络

由两个线性回归和一个ReLU的全连接层

$FFN(x) = ReLU(xW1 + b1)W2 + b2$

内层的维度2048

512->(W1)2048->(W2)512

（先由attention计算相似度拿到感兴趣的信息，语义空间在MLP隐藏层里面转化为2048维）

## 嵌入层和Softmax

在两个嵌入层的矩阵参数选择一样的，然后再乘以$\sqrt{d_{model}}$

（可能由于L2正则化权重值很小，下面还要和位置编码相加，保证两个向量的scale差不多，所以乘）

## 位置编码

**由于 Attention 值的计算最终会被加权求和，也就是说两者最终计算的 Attention 值都是一样的，进而也就表明了 Attention 丢掉了 X1的序列顺序信息。**

Attention自己是没有包含时序的信息的
所以要有位置编码

$w_i = \frac{1}{10000^{\frac{2i}{d_{model}}}}$

$PE(pos,2i) = sin(pos/10000^{2i/d_{model}})$

$PE(pos,2i + 1) = cos(pos/10000^{2i/d_{model}})$

PE都在[-1,1]且$PE_{pos_k}$是$PE_{pos}$​的线性组合

位置编码的低维度，$w_i$大，用波长短的sin函数，这样值的变化快。高维度用波长长的sin函数，这样值的变化慢。

**某个单词的位置信息是其他单词位置信息的线性组合，这种线性组合就意味着位置向量中蕴含了相对位置信息。**

$X_{final\_embedding}=Embedding+PositionalEmbedding$

## 性能

| Layer Type     | Complexity per Layer | Sequential Operations | Maximum Path Length |
| -------------- | -------------------- | --------------------- | ------------------- |
| Self-Attention | O(n2 · d)            | O(1)                  | O(1)                |
| Recurrent      | O(n · d2)            | O(n)                  | O(n)                |
| Convolutional  | O(k · n · d2)        | O(1)                  | O(logk(n)           |

# Code

```python
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # embedding特征大小
        self.h = h  # 头的个数
        # 确保d_model可以被h整除
        assert d_model % h == 0, "d_model 不能被 h整除"

        self.d_k = d_model // h  # 每个头特征大小
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # 获取d_k的值。
        d_k = query.shape[-1]
        # Q乘以K的转置，除以根号下d_k。
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # 给mask为0的位置填入一个很大的负值，这样在进行softmax，注意力就为0。
            attention_scores.masked_fill_(mask == 0, -1e9)
        # 进行softmax，归一化。得到注意力权重
        # (batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # 注意力权重乘以V，得到更新后的embedding。
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # 通过3个全连接层，获取Q、K、V矩阵
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # 对多头进行拆分
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # 计算注意力
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # 多个头合并
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # 乘以输出层
        return self.w_o(x)
```

Transformer里的tensor将batch size放在第一个维度，因为Transformer里可以同时对所有token进行处理，并不需要按照序列顺序依次处理。而在RNN里将seq_len放在第一个维度，是因为RNN里是按照序列顺序处理数据，seq_len放在第一个维度会方便一些。

Attention计算时，可以传入一个mask矩阵，mask矩阵用0标记了哪些位置不参与注意力计算。比如对于`<pad>` token就不必参与注意力计算。对于mask标记了0的位置，在注意力logits值计算完成后，给赋值一个很大的负值，这样在进行softmax后，对于这个位置的注意力就为0。相当于不参加注意力计算。

```python
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        # 可学习权重
        self.alpha = nn.Parameter(torch.ones(features))
        # 可学习偏差
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # 保留维度来进行广播
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # eps 是为了防止除0设置的很小的值
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
```

```python
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # 创建一个空的tensor
        pe = torch.zeros(seq_len, d_model)  # (seq_len, d_model)
        # 创建一个位置向量
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  
        # 计算分母
        div_term = torch.pow(10000.0, -torch.arange(0, d_model, 2, dtype=torch.float) / d_model)  # (d_model / 2)
        # 偶数位调用sin
        pe[:, 0::2] = torch.sin(position * div_term) 
        # 奇数为调用cos
        pe[:, 1::2] = torch.cos(position * div_term) 
        # 增加batch维度
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        # 注册位置编码为一个buffer，这个tensor不会参与训练，但是会随同模型一起被保存或者迁移到GPU。
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) 
        return self.dropout(x)
```

```python
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        # 定义多头自注意力模块
        self.self_attention_block = self_attention_block
        # 定义全连接模块
        self.feed_forward_block = feed_forward_block
        # 定义两个Add & Norm模块
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # 第一个残差连接，跳过多头注意力模块
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # 第二个残差连接，跳过全连接模块
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
```

```python
class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # 交叉注意力模块的Q矩阵来自Decoder，K,V矩阵来自Encoder的输出
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output,encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
# src只mask padding，tgt mask之后的词
```

