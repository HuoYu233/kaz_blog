---
title: Algorithm-Skills
mathjax: true
date: 2023/10/22 20:46:25
img: https://cdn.acwing.com/media/activity/surface/QQ%E5%9B%BE%E7%89%8720231022233411.jpg
excerpt: 算法竞赛中的一些技巧
---

一般竞赛时间限制在1s或2s，所以时间复杂度尽量控制在$10^7 - 10^8$

下面给出在不同数据范围下，代码的时间复杂度和算法如何选择：

1. $n \le 30$，dfs+剪枝、状态压缩dp
2. $n \le 100$，$O(n^3)$，Floyd、dp、高斯消元
3. $n \le 1000$，$O(n^2)$，dp、二分、朴素Dijsktra、朴素Prim、Bellman-Ford
4. $n \le 10000$，$O(n \sqrt n)$，块状链表、分块、莫队
5. $n \le 10^5$，$O(nlogn)$，sort、线段树、树状数组、set/map、heap、拓扑排序、堆优化Dijkstra、堆优化Prim、Kruskal、spfa、二分、CDQ分治、整体二分、后缀数组
6. $n \le 10^6$，$O(n)$，单调队列、hash、双指针、BFS、并查集、kmp、AC自动机；常数比较小的$O(nlogn)$，sort、树状数组、heap、dijkstra、prim
7. $n \le 10^7$，$O(n)$，双指针扫描、Kmp、AC自动机、线性筛素数
8. $n \le 10^9$，$O(\sqrt n)$，判断质数
9. $n \le 10^{18}$，$O(nlogn)$，最大公约数、快速幂、数位dp
10. $n \le 10^{1000}$，$O((logn)^2)$，高精度加减乘除

一些常见数据类型的大小

1. long long 内的最大阶乘 20!
2. int 内的最大阶乘 12!
3. int => $2^{31}$，$2*10^9$
4. long long => $2^{63}$，$9*10^{18}$
5. float => 38位
6. double => 308位

memset常赋值：-1，0，0x3f，-0x3f

无穷大：0x3f3f3f3f

`cout`相关：

- 设置场宽: `left(right)<<setw()`
- 设置精度:`fixed<<setprecision()`
- 此时要导入头文件`#include <iomanip>`

`cin`相关：

- 读入整行:`cin.getline(c,N,'\n')` c表示目标char数组，N表示长度，'\n’表示结束符

结构体小于号重载

```cpp
struct s{
    int a;
    string b;
    bool operator< (const s &ss) const{
        return a < ss.a
	}
}
```

`string`相关：

```cpp
string s = "I love China";
s.substr(start,len); //取子串
char c = s.c_str(); //转成char数组
strstr(s.c_str(),"love") //kmp，返回出现以后的子串，这里返回"love China"
s.find("China") //kmp,返回第一次出现的下标，不存在则返回s.npos
//与int
string s = to_string(i);
int i = stoi(s);
```