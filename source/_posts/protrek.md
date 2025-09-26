---
title: 【文献阅读】ProTrek
mathjax: true
date: 2025/9/26 20:46:25
img: https://cbirt.net/wp-content/uploads/2024/06/ProTrek.webp
excerpt: 基于三模态对比学习的蛋白质嵌入模型
---
![Fig. 1: Illustration of ProTrek. a, ProTrek architecture and tri-modal contrast learning. b, Cross-modal and uni-modal retrieval. ProTrek supports nine searching tasks. c, After the tri-modal contrast learning, the protein sequence encoder encodes almost universal representation of proteins, which can be fine-tuned to predict diverse downstream tasks, such as protein fitness and stability prediction. d, Using ProTrek’s natural language capabilities to decode the protein universe. Each cluster represents proteins with close embedding distances. Over 99% of protein entries in UniProt remain unreviewed, as shown in the top right.](./img/protrek/fig1.png)

![Fig. 2: ProTrek performance on protein search and representation tasks. a, Top chart: Search protein functional descriptions using sequences/structures. Bottom chart: Search protein sequences/structures using textual descriptions. x-axis: Specific protein function categories left of the dashed line; aggregated categories (residue-, protein-, all-level) right of it. “Global retrieval” indicates a search across the entire database, not within individual categories. The y-axis is MAP (mean average precision), a commonly used ranking metric for searching tasks. b, ProTrek employs “zinc ion binding” as the query term, while Foldseek utilizes P13243 as a query template, which is the protein with the most hits. In the testing set, 220 proteins share similar functional annotations with P13243. Foldseek identified 18 true hits, whereas ProTrek discovered 198 true hits. The TM-score results in the right subfigure reveal that proteins with similar functions can exhibit diverse structures. Conversely, proteins with similar structures (e.g., A9L2CB) may encode different functions. c, Searching proteins with similar functions using protein sequence/structure as input. TP (true-positive): Matches sharing ≥1 GO term. FP: Matches sharing no GO terms. d, Comparing alignment speed (CPU time) for 100 query proteins on UniRef50 with 50 million candidate proteins, utilizing 24 CPU cores. e, Evaluating the protein representation ability of the ProTrek AA sequence encoder.](./img/protrek/fig2.png)

## Expectation

- Text-guided Protein Design

Fengyuan Dai, Yuliang Fan, Jin Su, Chentong Wang, Chenchen Han, Xibin Zhou, Jianming Liu, Hui Qian, Shunzhi Wang, Anping Zeng, et al. Toward de novo protein design from natural language. bioRxiv, pages 2024–08, 2024.

Shengchao Liu, Yutao Zhu, Jiarui Lu, Zhao Xu, Weili Nie, Anthony Gitter, Chaowei Xiao, Jian Tang, Hongyu Guo, and Anima Anandkumar. A text-guided protein design framework. arXiv preprint arXiv:2302.04611, 2023.

- Advanced Protein ChatGPT

Chao Wang, Hehe Fan, Ruijie Quan, and Yi Yang. Protchatgpt: Towards understanding proteins with large language models. arXiv preprint arXiv:2402.09649, 2024.

## Method

todo
