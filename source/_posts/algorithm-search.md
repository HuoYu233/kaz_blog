---
title: Algorithm-Search
mathjax: true
date: 2023/08/03 20:46:25
img: https://cdn.acwing.com/media/activity/surface/QQ%E5%9B%BE%E7%89%8720231022233411.jpg
excerpt: 算法竞赛中的搜索问题
---

### DFS与BFS

- 深度优先搜索（DFS）

用$Stack$递归，空间$O(h)$，不具有最短性

- 宽度优先搜索（BFS）

用$Queue$，空间$O(2^h)$，“最短路”

**回溯、剪枝**

在矩阵中4个方向遍历

```cpp
int dx[] = {1,0,-1,0},y = {0,1,0,-1};
```

防止走相反的方向导致搜索回溯

```cpp
if(i ^ 2 == d) continue;
```

8个方向遍历

```cpp
int dx[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
int dy[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
```

防止走相反的方向导致搜索回溯

```cpp
if(i ^ 4 == d) continue;
```

### 树和图的存储

树是特殊的无环连通图

**有向图$a \to b$**

- 邻接矩阵 $g[a][b]$
- 邻接表，用链表储存点$i$可以到达的点

```cpp
// 对于每个点k，开一个单链表，存储k所有可以走到的点。h[k]存储这个单链表的头结点
int h[N], e[N], ne[N], idx;

// 添加一条边a->b
void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

// 初始化
idx = 0;
memset(h, -1, sizeof h);
```

#### 树和图的遍历

时间复杂度$O(n+m)$，n表示点数，m表示边数

- 深度优先遍历

```cpp
int dfs(int u)
{
    st[u] = true; // st[u] 表示点u已经被遍历过

    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (!st[j]) dfs(j);
    }
}
```

- 宽度优先遍历

```cpp
queue<int> q;
st[1] = true; // 表示1号点已经被遍历过
q.push(1);

while (q.size())
{
    int t = q.front();
    q.pop();

    for (int i = h[t]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (!st[j])
        {
            st[j] = true; // 表示点j已经被遍历过
            q.push(j);
        }
    }
}
```

#### 拓扑排序

时间复杂度$O(n+m)$，n表示点数，m表示边数

有向无环图一定可以拓扑排序，序列可能不唯一

**入度、出度**：有多少条边指向自己/从自己这里指出去

1. 将入度为0的点入队
2. 宽搜，枚举队头的所有出边$t \to j$，删掉$t \to j$，$t$的出度减一

```cpp
bool topsort()
{
    int hh = 0, tt = -1;

    // d[i] 存储点i的入度
    for (int i = 1; i <= n; i ++ )
        if (!d[i])
            q[ ++ tt] = i;

    while (hh <= tt)
    {
        int t = q[hh ++ ];

        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            d[j]--;
            if (d[j] == 0)
                q[ ++ tt] = j;
        }
    }

    // 如果所有点都入队了，说明存在拓扑序列；否则不存在拓扑序列。
    return tt == n - 1;
}
```

**一个有向无环图至少有一个入度为0的点**