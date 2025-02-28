---
title: Algorithm-DP
mathjax: true
date: 2023/09/13 20:46:25
img: https://cdn.acwing.com/media/activity/surface/QQ%E5%9B%BE%E7%89%8720231022233411.jpg
excerpt: 算法竞赛中的动态规划与背包问题
---

### 动态规划

- 状态表示

  $f[i,j]$

  - 集合
  - 属性（Max，Min，Cnt）

- 状态计算

  - 集合的划分

### 线性DP

**[数字三角形](https://www.luogu.com.cn/problem/P1216)**

$O(n^2)$

$f[i,j]$表示到坐标为$[i,j]$的路径的和最大值

$f[1][1] = a[1][1]$

$f[i][j] = max(f[i-1][j-1]+a[i][j], f[i-1][j]+a[i][j])$

**[最长上升子序列](https://www.luogu.com.cn/problem/B3637)**

**朴素**：$O(n^2)$

$f[i]$表示以第$i$个数结尾的序列中上升子序列长度的最大值

遍历$a_i$所有可能的前一个数$a_j$($a_j<a_i$且$0 \le j \le i-1$)

$f[i] = max(f[j]+1,f[i]),j \in [0,i-1]$

如果要保存最长序列：$g[i]$保存从哪一步$j$转移过来

代码：https://www.luogu.com.cn/record/124595657

**优化**：$O(nlogn)$

用一个q数组储存长度为i的序列的结尾数字的最小值

可以证明$q_i>q_{i-1}>...>q_2>q_1$，即数组严格单调递增

对于$a_i$，二分找到最大的$q_k<=a_i$，$f[i] = k+1$，更新$q_k = a_i$

代码：https://www.luogu.com.cn/record/133704642

**[最长公共子序列](https://www.luogu.com.cn/problem/P1439)**

**朴素**：$O(n^2)$

$f[i][j]$表示所有在第一个序列的前i个字母中出现，且在第二个序列的前j个字母中出现的子序列的最大值

集合划分依据：是否选择$a[i],b[j]$

分为四个集合：选择$a[i],b[j]$ ; 选择$a[i]$ 不选择$b[j]$ ; 不选择$a[j]$选择$b[j]$ ; 都不选择$a[i],b[j]$

分别表示为 $f[i-1][j-1] , f[i,j-1] , f[i-1][j] , f[i-1][j-1]+1$

其中第二三种情况**包含**上面对应的集合（由于是求Max，所以有重复不影响结果）

且第二三种集合也包含第一个集合，所以只要对后三种集合求最大值即可

$f[i,j] = max(f[i-1,j],f[i,j-1])$

当$a[i]==b[j]$时,$f[i,j] = max(f[i,j],f[i-1,j-1]+1)$

**优化**：$O(nlogn)$

**[编辑距离](https://www.luogu.com.cn/problem/P2758)**

$f[i,j]$所有将$a[1-i]$变成$b[1-j]$的操作方式的最小步数

区间划分，①删除最后一个数、②增加最后一个数、③修改最后一个数

① $f[i-1,j]+1$

②$f[i,j-1]+1$

③$f[i-1,j-1]+1$ （如果$a[i]==b[j]$则不需要加一，即不需要进行修改操作）

### 区间DP

**[石子合并](https://www.acwing.com/problem/content/284/)**

$f[i,j]$表示将第$i$堆石子到第$j$堆石子合并成一堆石子的方式的代价最小值/最大值

$O(n^3)$

```cpp
for(int len=2;len<=n;len++){
    for(int i=1;i+len-1<=n;i++){
        int l = i,r = i+len-1;
        f_max[l][r] = -1e8,f_min[l][r] = 1e8;
        for(int k=l;k<r;k++){
            f_max[l][r] = max(f_max[l][r],f_max[l][k]+f_max[k+1][r]+s[r]-s[l-1]);
            f_min[l][r] = min(f_min[l][r],f_min[l][k]+f_min[k+1][r]+s[r]-s[l-1]);
        }
    }
}
```

### 计数类DP

### 数位统计DP

**[计数问题](https://www.acwing.com/problem/content/340/)**

设$n=abcdefg$，枚举第$i$位是 $x \in [0,9]$

举例$x=1,i=4$的情况：

设数字为$xxx1yyy$

- 当$abc>xxx,xxx \in [000,abc-1], y \in [000,999]$，则共有$abc * 1000$个

- 当$abc<xxx$，则共有0个

- 当

  $abc=xxx$

  - 当$d<1$，无解
  - 当$d=1$，$yyy \in [000,efg]$,则有$efg+1$种
  - 当$d>1$，$yyy \in [000,999]$,有1000种

**当x=0时，注意前导0，即对于第一种情况，$xxx \in [001,abc-1]$**，即有$(abc-1)*1000$情况

**[圆形数字](https://www.acwing.com/problem/content/341/)**

### 状态压缩DP

**[蒙德里安的梦想](https://www.acwing.com/problem/content/293/)**

$f[i][j]$表示第i列，上一列横着摆的数量j,其中j是一个二进制数。

**[最短Hamilton路径](https://www.acwing.com/problem/content/93/)**

$f[i][j]$表示从0号点走到j号点，走过的所有点是i的所有路径(二进制数i表示某个点是否已经走过了)的最小路径长度

### 树形DP

**[没有上司的舞会](https://www.acwing.com/problem/content/287/)**

$f[u][0]$表示所有以u为根的子树中选择，并且不选u这个点的方案的最大值

$f[u][1]$表示所有以u为根的子树中选择，并且选u这个点的方案的最大值

设点u的子节点$s_1,s_2,s_3....s_i$

$f[u][0] = \sum_{1}^{i}max(f[s_i][0],f[s_i][1])$

$f[u][1] = \sum_{1}^{i}f[s_i][0]$

找出根节点，递归求最大值即可

### 记忆化搜索

**[滑雪](https://www.luogu.com.cn/problem/P1434)**

用$s[i][j]$表示从(i,j)点出发能走的最长距离。

每次搜索一次记忆一次即可。

举例

```none
3 3 
1 1 3
2 3 4
1 1 1
```

先去找(1,1)的最长距离，很明显为1

接着找(1,2)的最长距离，很明显为1

接着找(1,3)的最长距离，为2((1,3)->(1,2))

然后找(2,1)的最长距离，为2((2,1)->(1,1))

然后是(2,2)的最长距离，如果没有记忆化，那么搜索过程为：(2,2)->(2,1)->(1,1)

但是（2,1）之前已经搜过了，再去搜就是浪费时间，之前搜索已经知道(2,1)的值为2，那么搜索过程就是缩短为：(2,2)->(2,1),即为3

### 背包问题

给定$n$个物品和容量$v$的背包，每个物品都有体积$v_i$和价值$w_i$，求当$\sum_{i=1}^{n} v_i \le v$时最大的$w$是多少

#### 01背包问题

每个物品只能用0/1次

$f[i,j] = max(f[i-1,j],f[i-1,j-v_i]+w_i)$

[01背包问题](https://www.acwing.com/problem/content/2/)

[采药](https://www.luogu.com.cn/problem/P1048)

#### 完全背包问题

物品可以无限次使用

$f[i,j] = Max(f[i-1,j-v_i \times k]+w[i] \times k)$

$k \subseteq [0,\frac{j}{v_i}]$

即$f[i,j] = Max(f[i-1,j],f[i-1,j-v_i]+w_i,f[i-1,j-2v_i]+2w_i,....,f[i-1][j-kv_i]+kw_i)$

$f[i,j-v_i] = Max(f[i-1][j-v_i],f[i-1][j-2v_i]+w_i,...,f[i-1][j-kv_i]+(k-1)w_i)$

$f[i][j]$的后$k$项等于$f[i][j-v_i]+w_i$

得

$f[i,j] = Max(f[i-1,j],f[i,j-v_i]+w_i)$

[完全背包问题](https://www.acwing.com/problem/content/3/)

#### 多重背包物品

每个物品的个数不一致

朴素做法

$f[i,j] = Max(f[i-1,j],f[i-1,j-v_i]+w_i,f[i-1,j-2v_i]+2w_i,....,f[i-1][j-kv_i]+kw_i)$

$k \subseteq [0,s_i]$

三重循环即可

[多重背包问题1](https://www.acwing.com/problem/content/4/)

**优化：二进制优化**

```cpp
for(int i=1;i<=n;i++){
    int a,b,s;
    cin>>a>>b>>s;
    //v w s;
    int k = 1;
    while(k<=s){
        cnt++;
        v[cnt] = a*k;
        w[cnt] = b*k;
        s-=k;
        k*=2;
    }
    if(s>0){
        cnt++;
        v[cnt] = a*s;
        w[cnt] = b*s;
    }
}
```

对物品进行二进制分组，组数为$cnt$，转化为01背包问题求解

```cpp
n = cnt;
for(int i=1;i<=n;i++){
    for(int j=0;j<=m;j++){
        f[i][j] = f[i-1][j];
        if(j>=v[i]) f[i][j] = max(f[i][j],f[i-1][j-v[i]]+w[i]);
    }
}
cout<<f[n][m]<<endl;
```

[多重背包问题2](https://www.acwing.com/problem/content/5/)

#### 分组背包问题

有$N$组，每一组只能选其中一种物品

$f[i][j] = Max(f[i-1,j],f[i-1,j-v_{i,k}]+w_{i,k})$

[分组背包问题](https://www.acwing.com/problem/content/9/)



