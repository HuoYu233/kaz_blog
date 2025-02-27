---
title: 基于Transformer的催化反应产率预测
mathjax: true
date: 2025/2/25 20:46:25
img: https://datawhaler.feishu.cn/87dd39a9-09e1-45f5-9e68-fd28666f6058
excerpt: RT
---
# 任务概述

构建一个能够准确预测碳氮成键反应产率的预测模型。  

通过对反应中所包含的反应底物、添加剂、溶剂以及产物进行合理的特征化，运用机器学习模型或者深度学习模型拟合预测反应的产率。

或者利用训练集数据对开源大语言模型进行微调以预测反应的产率。

训练集中包含19999条反应数据，测试集中包含3539条反应数据。约85%与12%。每条训练数据包含 rxnid, Reactant1, Reactant2 , Product , Additive , Solvent , Yield字段。其中 Reactant1 , Reactant2 , Product , Additive , Solvent 字段中为对应物质的SMILES字符串，Yield字段为目标字段，是经过归一化的浮点数。

**评价指标**

实验真实结果与预测结果$R^2$决定系数来进行评测:

![img](https://datawhaler.feishu.cn/space/api/box/stream/download/asynccode/?code=OGQ0MzBhZTc0NmY4YTc3NTIxZmQwZmEyYjM3NDhmOWNfZE1MSkE2WmRPV01nSnhJM2tERGRlVEVucTgyMkxjOEpfVG9rZW46REVoUmJmeDAyb1lTNXR4ZE9ROGNyT0RhbmFoXzE3NDA0NjQyOTU6MTc0MDQ2Nzg5NV9WNA)

# baseline

1. **导入库**：首先，代码导入了需要用到的库，包括 `pandas`（用于数据处理和分析），`scikit-learn`（机器学习库），`rdkit`（化学信息工具）。
2. **读取数据**：代码通过使用 `pd.read_csv` 函数从文件中读取训练集和测试集数据。
3. **使用Morgan分子指纹建模SMILES**：

   \- 这个过程需要调用rdkit的相关模块。然后将Reactant1,Reactant2,Product,Additive,Solvent字段的向量拼接到一起，组成一个更长的向量。

1. **使用随机森林预测结果**：

   \- 这里直接调用`sklearn`的`RandomForestRegressor`模块实例化一个随机森林模型，并对`n_estimators`等重要参数进行指定。最后使用model.fit(x, y)训练模型。模型保存在本地`'./random_forest_model.pkl'`。

1. **加载模型进行预测，并将保存结果文件到本地：**

   ` pkl`文件直接使用`pickle.load()`加载，然后使用`model.predict(x)`进行预测。

## SMILES

SMILES,全称是Simplified Molecular Input Line Entry System，是一种将化学分子用ASCII字符表示的方法，是化学信息学领域非常重要的工具。

SMILES将化学分子中涉及的原子、键、电荷等信息，用对应的ASCII字符表示；环、侧链等化学结构信息，用特定的书写规范表达。以此，几乎所有的分子都可以用特定的SMILES表示，且SMILES的表示还算比较直观。

在SMILES中，原子由他们的化学符号表示，=表示双键、#表示三键、[]里面的内容表示侧基或者特殊原子（例如[Cu+2]表示带电+2电荷的Cu离子）。通过SMLIES，就可以把分子表示为序列类型的数据了。

（注：SMILES有自己的局限性：例如选择不同的起始原子，写出来的SMILES不同；它无法表示空间信息。）

由于Reactant1,Reactant2,Product,Additive,Solvent都是由SMILES表示。所以，可以使用rdkit工具直接提取SMILES的分子指纹（向量），作为特征。

## Morgan fingerprint
位向量（bit vector）形式的特征，即由0,1组成的向量。

分子指纹是一个具有固定长度的位向量（即由0，1组成），其中，每个为1的值表示这个分子具有某些特定的化学结构。

通常，分子指纹的维度都是上千的，也即记录了上千个子结构是否出现在分子中。

## RDKit

RDkit会将分子读取为RDkit中专属的rdkit.Chem.rdchem.Mol对象，并以Mol对象为基础，可以对分子进行转化为各种表达形式，例如SMILES

RDkit是化学信息学中主要的工具，是开源的。网址：http://www.rdkit.org，支持WIN\MAC\Linux，可以被python、Java、C调用。几乎所有的与化学信息学相关的内容都可以在上面找到。

## 结果

baseline的$R^2 = 0.0745336043830066$，约0.08

#RNN建模

RNN（Recurrent Neural Network）是处理序列数据的一把好手。RNN的网络每层除了会有自己的输出以外，还会输出一个隐向量到下一层。

![img](https://datawhaler.feishu.cn/space/api/box/stream/download/asynccode/?code=NGJhNWM3ZWY1MmYzNGQ0NTNjMzVlMDg3MzhhNDhhM2NfRDc0NFh6dmFvSFdiT2tyNGZJRk91U0xzcklUYU52WjBfVG9rZW46Q0JsMmJzZUxQb1EyTVp4emtCTmNYRmlrbmNkXzE3NDA0NzAxNTU6MTc0MDQ3Mzc1NV9WNA)

其中，每一层相当于做了一次线性变换：

$$h_n = \sigma(W_{hh}h_{n-1} + W_{hx}x_n + b_n)$$

每层的输出：$$ y_n = Softmax(Vh_n + c)$$

通过隐向量的不断传递，序列后面的部分就通过“阅读”隐向量，获取前面序列的信息，从而提升学习能力。

但是RNN也有缺点：如果序列太长，那么两个相距比较远的字符之间的联系需要通过多个隐藏向量。

同时，RNN需要一层一层地传递，所以并行能力差，同时也比较容易出现梯度消失或梯度爆炸问题。
