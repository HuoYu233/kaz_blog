---
title: Algorithm-Basic
mathjax: true
date: 2023/4/27 20:46:25
img: https://cdn.acwing.com/media/activity/surface/QQ%E5%9B%BE%E7%89%8720231022233411.jpg
excerpt: 算法竞赛基础算法
---

# 排序

## 快速排序

分治思想,时间复杂度$O(nlogn)-O(n^2) $

期望时间复杂度$O(nlogn)$

1. 数组中找一个值$x$作为分界点（可以是$arr\left [ l \right ]$ ,$arr\left [ r \right ]$,$arr\left [ \frac{l+r}{2} \right ]$ 等等…）
2. 调整区间，使得左边的区间所有数$\le$x，右边区间所有数$>$x
   - 定义两个指针分别在左右边界
   - $i$不断右移，直到遇到$arr[i]$ $>x$，就停下
   - $j$不断左移，直到遇到$arr[j]\le x$，就停下
   - 交换$arr[i]$与$arr[j]$
3. 递归处理左右区间

**模版**

```java
public static void quick_sort(int q[], int l, int r){
    if (l >= r) return;
    int i = l - 1, j = r + 1, x = q[l + r >> 1];
    while (i < j){
        do i ++ ; while (q[i] < x);
        do j -- ; while (q[j] > x);
        if (i < j) {
            int t = q[i];
            q[i] = q[j];
            q[j] = t;           
        }
    }
    quick_sort(q, l, j);
    quick_sort(q, j + 1, r);
}
```

## 归并排序

分治思想,$O(nlogn)$

1. 确定分界点 $mid = \frac{l+r}{2}$
2. 递归排序$left$和$right$
3. 归并：合二为一
   - 双指针指向$left$和$right$的第一个元素
   - 创建一个空数组$res$存放结果
   - 指针比较，如果$left[i]<right[j]$，则把$left[i]$放入$res$，$i$向后移动一位，继续比较
   - 如果$left[i]=right[j]$，则把$left[i]$放入$res$，以维持稳定

**模版**

```java
public static void merge_sort(int q[], int l, int r){
    if (l >= r) return;
    int mid = l + r >> 1;

    merge_sort(q, l, mid);
    merge_sort(q, mid + 1, r);

    int k = 0, i = l, j = mid + 1;

    while (i <= mid && j <= r)
        if (q[i] < q[j]) tmp[k ++ ] = q[i ++ ];
    else tmp[k ++ ] = q[j ++ ];

    while (i <= mid) tmp[k ++ ] = q[i ++ ];
    while (j <= r) tmp[k ++ ] = q[j ++ ];

    for (i = l, j = 0; i <= r; i ++, j ++ ) q[i] = tmp[j];
}
```

# 二分

## 整数二分

**提示信息**

- 题目保证有解
- 单调性
- 求最大值的最小化

**思路**

对于区间$[l,r]$，其中一部分满足条件$check(x)=true$，另一部分不满足

- 对于寻找不满足区间的边界

$mid = \frac{l+r+1}{2}$

若$check(mid)=true$ 则说明边界值在$[mid,r]$
更新语句为$l = mid$

若$check(mid)=false$ 则说明边界值在$[l,mid-1]$
更新语句为$r = mid-1$

- 对于寻找满足区间的边界

$mid = \frac{l+r}{2}$

若$check(mid)=true$ 则说明边界值在$[l,mid]$
更新语句为$r = mid$

若$check(mid)=false$ 则说明边界值在$[mid+1,r]$
更新语句为$l=mid+1$

**模版**

```java
public static boolean check(int x) {/* ... */} // 检查x是否满足某种性质

// 区间[l, r]被划分成[l, mid]和[mid + 1, r]时使用：
public static int bsearch_1(int l, int r)
{
    while (l < r)
    {
        int mid = l + r >> 1;
        if (check(mid)) r = mid;    // check()判断mid是否满足性质
        else l = mid + 1;
    }
    return l;
}
// 区间[l, r]被划分成[l, mid - 1]和[mid, r]时使用：
public static int bsearch_2(int l, int r)
{
    while (l < r)
    {
        int mid = l + r + 1 >> 1;
        if (check(mid)) l = mid;
        else r = mid - 1;
    }
    return l;
}
```

！！如果$l=mid$ ，最开始的$mid$就要补上$+1$

！！`check`函数中记得取等于号

## 浮点数二分

```java
public static boolean check(double x) {/* ... */} // 检查x是否满足某种性质

double bsearch_3(double l, double r)
{
    final double eps = 1e-6;   
    // eps 表示精度，取决于题目对精度的要求
    //比需要保留的位数多2
    while (r - l > eps)
    {
        double mid = (l + r) / 2;
        if (check(mid)) r = mid;
        else l = mid;
    }
    return l;
}
```

**精度比需要保留的位数多-2次方**

可以把$while$循环直接换成`for`100次

# 高精度

$C++$模版

高精度加法

```cpp
// C = A + B, A >= 0, B >= 0
vector<int> add(vector<int> &A, vector<int> &B)
{
    if (A.size() < B.size()) return add(B, A);

    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size(); i ++ )
    {
        t += A[i];
        if (i < B.size()) t += B[i];
        C.push_back(t % 10);
        t /= 10;
    }

    if (t) C.push_back(t);
    return C;
}
```

## 高精度减法

```cpp
// C = A - B, 满足A >= B, A >= 0, B >= 0
vector<int> sub(vector<int> &A, vector<int> &B)
{
    vector<int> C;
    for (int i = 0, t = 0; i < A.size(); i ++ )
    {
        t = A[i] - t;
        if (i < B.size()) t -= B[i];
        C.push_back((t + 10) % 10);
        if (t < 0) t = 1;
        else t = 0;
    }

    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
```

## 高精度乘低精度

```cpp
// C = A * b, A >= 0, b >= 0
vector<int> mul(vector<int> &A, int b)
{
    vector<int> C;

    int t = 0;
    for (int i = 0; i < A.size() || t; i ++ )
    {
        if (i < A.size()) t += A[i] * b;
        C.push_back(t % 10);
        t /= 10;
    }

    while (C.size() > 1 && C.back() == 0) C.pop_back();

    return C;
}
```

## 高精度除以低精度

```cpp
// A / b = C ... r, A >= 0, b > 0
vector<int> div(vector<int> &A, int b, int &r)
{
    vector<int> C;
    r = 0;
    for (int i = A.size() - 1; i >= 0; i -- )
    {
        r = r * 10 + A[i];
        C.push_back(r / b);
        r %= b;
    }
    reverse(C.begin(), C.end());
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
```

## 高精度乘以高精度

```cpp
cin>>a1>>b1;
int lena=strlen(a1);
int lenb=strlen(b1);
for(i=1;i<=lena;i++)a[i]=a1[lena-i]-'0';
for(i=1;i<=lenb;i++)b[i]=b1[lenb-i]-'0';
for(i=1;i<=lenb;i++)
    for(j=1;j<=lena;j++)
        c[i+j-1]+=a[j]*b[i];
for(i=1;i<lena+lenb;i++)
    if(c[i]>9)
    {
        c[i+1]+=c[i]/10;
        c[i]%=10;
    }
len=lena+lenb;
while(c[len]==0&&len>1)len--;
```

# 前缀和与差分

一对逆运算

## 一维前缀和

设有一列数据${a}_1,{a}_2,...,{a}_{n-1},{a}_n$

定义${S}_i=a_1+a_2+...+a_i$

一般下标从1开始，$S_0=0$

$S_i$的初始化: $S_i = S_{i-1}+a_i$

**作用**

快速地求出原数组中一段区间数的和

对于区间$[l,r]$

$\sum_{i=l}^{r}a_i = S_r-S_{l-1}$

## 二维前缀和

对于二维数组（矩阵）$\begin{pmatrix} a_{11}& a_{12} & ... & a_{1j}\\ a_{21}& a_{22} & ... & a_{2j} \\ ...& ... & ... & ...\\ a_{i1}& a_{i2} & ... & a_{ij} \end{pmatrix}$

$S_{ij}$代表$a_{ij}$左上角的所有元素和

- 对于点$(i,j)$，其二维前缀和$S_{ij}$的初始化

  $S_{ij}=S_{i-1,j}+S_{i,j-1}-S_{i-1,j-1}+a_{i,j}$

- 设点$(x_1,y_1)$在$(x_2,y_2)$的左上角，则两点围成的矩形中所有元素和
  $S=S_{x_2,y_2}-S_{x_2,y_1-1}-S_{x_1-1,y_2}+S_{x_1-1,y_1-1}$

## 一维差分

对一列数据$a_1,a_2,a_3,...,a_i$

构造$b_1,b_2,b_3,...,b_i$使得$a_i=b_1+b_2+...+b_i$

即$a$为$b$的前缀和，$b$就是$a$的差分

$\left\{\begin{matrix} b_1=a_1\\ b_2=a_2-a_1\\ b_3=a_3-a_2\\ ......\\ b_n=a_n-a_{n-1} \end{matrix}\right.$

**作用**

若要把$a_1,a_2,a_3,...,a_i$中$[l,r]$区间的$a$加$c$

只需要使$b_l+=c,b_{r+1}-=c$

**模版**

```java
import java.util.Scanner;

public class Diff {
    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);
        // 给出n数组大小和k增加次数
        int n = scanner.nextInt();
        int k = scanner.nextInt();

        // 搭建数组
        int[] arr = new int[n+1];
        int[] brr = new int[n+1];

        // 为arr赋值
        for (int i = 1; i < n+1; i++) {
            arr[i] = scanner.nextInt();
        }

        // 为brr赋值
        for (int i = 1; i < n+1; i++){
            brr[i] = arr[i] - arr[i-1];
        }

        while (k-- > 0){
            // 我们为arr的[l,r]区间加上c
            int l = scanner.nextInt();
            int r = scanner.nextInt();
            int c = scanner.nextInt();

            brr[l] += c;
            brr[r+1] -= c;
        }

        // 计算输出结果即可（这里输出的需要是由b累计出来的a）
        // 也可以使用注释代码，最后输出arr即可
        for (int i = 1; i < n+1; i++) {
            brr[i] += brr[i-1];
            //arr[i] = brr[i]+arr[i-1];
        }

        // 最后输出结果
        for (int i = 1; i < n+1; i++) {
            System.out.println(brr[i]);
        }

    }
}
```

## 二维差分

原矩阵$a_{ij}$,差分矩阵$b_{ij}$

$b_{x1,y1}+=c$

$b_{x2+1,y1}-=c$

$b_{x1,y2+1}-=c$

$b_{x2+1,y2+1}+=c$

# 其他

## 双指针算法

- 两个序列，两个指针
- **一个序列，两个指针**

**结构**

```cpp
for(int i=0,j=0;i<n;i++){
    while(j<i && check(i,j)) j++;
    //每道题具体的逻辑
}
```

**核心思想**

复杂度由$O(n^2)$优化到$O(n)$

先想出朴素做法，寻找i与j之间的关系，是否有单调性，进行双指针优化

[A-B数对](https://www.luogu.com.cn/problem/P1102)

## 位运算

计算机以二进制表示数据，以表示电路中的正反。在二进制下，一个位只有0和1。逢二进一位。

计算机中存储数据，以字节为单位，一个字节有8个位，即可以表示-128~127范围的数字。

### 基础运算

> 与

用符号&表示，运算规律是：真真为真，真假为假，假假为假（一假即假）

```bash
1&1 //1
1&0 //0
0&0 //0
```

> 或

用符号|表示，运算规律是：真真为真，真假为真，假假为假（一真即真）

```bash
1|1 //1
1|0 //1
0|0 //0
```

> 非

运算符为~，取反的逻辑，运算规律：二进制位若为1，取反后为0。若为0，取反后为1

```bash
~1 //11111110
```

> 左移

将二进制数向左移位操作，高位溢出则丢弃，低位补0

```bash
a=11;
a<<1;
移位前：0000 1011
移位后：0001 0110（十进制值为22）
```

对一个数左移1位就是乘以2，左移n位就是乘以2的n次方（而左移运算比乘法快得多）

> 右移

右移位运算中，无符号数和有符号数的运算并不相同。对于无符号数，右移之后高位补0；对于有符号数，符号位一起移动，正数高位补0，负数高位补1

```text
无符号数
a=16;
a>>3;
移位前：0001 0000
移位后：0000 0010（十进制值为2）
有符号数（正数）
b=32;
a>>3;
移位前：0010 0000
移位后：0000 0100（十进制值位4）
有符号数（负数）
b=-32;
b>>3;
移位前：1010 0000
移位后：1000 0100（十进制值为-4）
```

- n的二进制表示中第k位数字是几

（k从个位开始算0,1,2…）

1. 先把第k位移到最后一位`n>>k`

2. 看个位是几 `x&1`

   **n>>k&1**

- lowbit(x)

  树状数组基本操作，返回x的最后一位1

  **x&(-x)**

  原理：`-x=(~x+1)`

## 区间合并

- 按照区间左端点排序

- 判断下一个区间与当前区间的关系

  - 相交

    - 更新右端点为两个区间的$max$

  - 不相交

    - 将当前区间更新为不相交的这个区间

    **C++模版**

```cpp
// 将所有存在交集的区间合并
void merge(vector<PII> &segs)
{
    vector<PII> res;

    sort(segs.begin(), segs.end());

    int st = -2e9, ed = -2e9;
    for (auto seg : segs)
        if (ed < seg.first)
        {
            if (st != -2e9) res.push_back({st, ed});
            st = seg.first, ed = seg.second;
        }
        else ed = max(ed, seg.second);

    if (st != -2e9) res.push_back({st, ed});

    segs = res;
}
```