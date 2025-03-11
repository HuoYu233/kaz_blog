---
title: Attention Is All You Need阅读笔记
mathjax: true
date: 2025/3/11 20:46:25
img: https://img0.baidu.com/it/u=3520508,2967101156&fm=253&fmt=auto&app=138&f=JPEG?w=786&h=500
excerpt: transformer论文阅读笔记
---

# 之前的工作

RNN 存在长期依赖问题（隐含信息一直传递到后面会消失），且无法并行

LSTM 通过遗忘门，输入门，输出门解决长期依赖问题，但是还是无法并行

Transformer仅仅依赖注意力机制（Attention），不需要考虑两个词离得多远，且能实现并行

# 模型架构

输入$(x_1,x_2,...,x_n)$，编码器输出$(z_1,z_2,...,z_n)$，输入到解码器输出$(y_1,y_2,...,y_m)$

每一步都是自回归的（编码的时候可以看到一整个句子，但是在解码的时候只能看到已经生成好的句子，叫做自回归，它将时间序列的当前值表示为过去若干个值的线性组合)

![pic-1](/img/transformer-notes/pic-1.png)

## 编码器

N=6，每一层有两个子层：多头注意力层和前馈神经网络层，每个子层后面还有一个残差网络和层归一化(Add & Norm)

## 解码器

N=6，在输入添加一个掩码多头注意力

## 注意力机制

通过Q，K，V，计算两个词的相似度

## Scaled Dot-Product Attention

Q，K的维度都是$d_k$，V的维度是$d_v$

Q，K做内积，再除以$\sqrt{d_k}$，做一层`softmax`就是V的权重

$Attention(Q,K,V) = softmax( \frac{QK^T}{\sqrt{d_k}}  )V$

为了防止softmax的梯度很小减慢训练速度，所以处理$\sqrt{d_k}$

![pic-2](/img/transformer-notes/pic-2.png)

## 多头自注意力机制

h=8，八头自注意力

TODO

## 其他

在Encoder-Decoder注意力层

KV来自编码器，Q来自解码器

掩码多头注意力时，点积设置为-∞

## 前馈神经网络

由两个线性回归和一个ReLU的全连接层

$FFN(x) = ReLU(xW1 + b1)W2 + b2$

内层的维度2048

（先由attention计算相似度拿到感兴趣的信息，语义空间在MLP隐藏层里面转化为2048维）

## 嵌入层和Softmax

在两个嵌入层的矩阵参数选择一样的，然后再乘以$\sqrt{d_model}$

（可能由于L2正则化权重值很小，下面还要和位置编码相加，保证两个向量的scale差不多，所以乘）

## 位置编码

Attention自己是没有包含时序的信息的
所以要有位置编码

$PE(pos,2i) = sin(pos/10000^{2i/d_{model}})$

$PE(pos,2i + 1) = cos(pos/10000^{2i/d_{model}})$

PE都在[-1,1]且$PE_{pos_k}$是$PE_{pos}$的线性组合
