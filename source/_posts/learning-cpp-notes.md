---
title: 从0开始的Cpp学习笔记
mathjax: true
date: 20235/4/26 20:46:25
img: https://i2.hdslb.com/bfs/archive/6487769f0e49718a24c293df29bd840be0ca2e9c.png
excerpt: RT
---
windows: vs开发桌面c++，安装略

# C++是如何工作的？

源代码（.cpp和.h文件）-> 预处理（处理#开头指令，包括头文件展开，宏替换，条件编译等，生成.i或.ii文件） -> 编译（将预处理文件转化成机器可读的目标文件，先变成汇编文件.s，再变成目标文件.o或.obj） -> 链接（合并目标文件和库，生成可执行文件）

## 编译器如何工作？

