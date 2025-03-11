---
title: Algorithm-Math
mathjax: true
date: 2023/09/07 20:46:25
img: https://cdn.acwing.com/media/activity/surface/QQ%E5%9B%BE%E7%89%8720231022233411.jpg
excerpt: 算法竞赛中的数学问题
---

### 质数

对于**大于一**的整数，如果只包含一和本身这两个约数，它就是质数（也叫素数）

#### 试除法

$O(\sqrt n)$

```cpp
bool is_prime(int x)
{
    if (x < 2) return false;
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
            return false;
    return true;
}
```

#### 试除法分解质因数

```cpp
N = p1^c1 * p2^c2 * ... *pk^ck
```

从小到大枚举每一个数

得到每一对$(p,c)$

```cpp
void divide(int x)
{
    // n中只有最多一个大于sqrt(n)的质因子
    // 枚举到sqrt(t)，最后一个特殊处理 O(sqrt(n))
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
        {
            int c = 0;
            while (x % i == 0) x /= i, c ++ ;
            cout << i << ' ' << c << endl;
        }
    if (x > 1) cout << x << ' ' << 1 << endl;
    cout << endl;
}
```

#### 朴素筛法求素数

枚举每一个数，如果它没有被筛，则加入质数集合，并且把它的所有倍数都筛掉

优化：埃氏筛法，只需要把质数的倍数筛掉

$O(nloglogn)$

质数定理：1-n中有$\frac{n}{ln_{}{n}}$个质数

```cpp
int primes[N], cnt;     // primes[]存储所有素数
bool st[N];         // st[x]存储x是否被筛掉

void get_primes(int n)
{
    for (int i = 2; i <= n; i ++ )
    {
        if (st[i]) continue;
        primes[cnt ++ ] = i;
        for (int j = i + i; j <= n; j += i)
            st[j] = true;
    }
}
```

#### 线性筛法求素数

$n$只会被他的最小质因子筛掉

$O(n)$

```cpp
int primes[N], cnt;     // primes[]存储所有素数
bool st[N];         // st[x]存储x是否被筛掉

void get_primes(int n)
{
    for (int i = 2; i <= n; i ++ )
    {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= n / i; j ++ )
        {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break; //此时primes[j]是i的最小质因子
        }
    }
}
```

### 约数

#### 试除法求所有约数

$O(\sqrt{n})$

```cpp
vector<int> get_divisors(int x)
{
    vector<int> res;
    for (int i = 1; i <= x / i; i ++ )
        if (x % i == 0)
        {
            res.push_back(i);
            if (i != x / i) res.push_back(x / i); //防止相同的数被push进去两倍，例如4*4=16
        }
    sort(res.begin(), res.end());
    return res;
}
```

#### 约数个数和约数之和

如果 N = p1^c1 * p2^c2 * … *pk^ck
约数个数： (c1 + 1) * (c2 + 1) * … * (ck + 1)
约数之和： (p1^0 + p1^1 + … + p1^c1) * … * (pk^0 + pk^1 + … + pk^ck)

#### 欧几里得算法求最大公约数

```cpp
int gcd(int a, int b)
{
    return b ? gcd(b, a % b) : a;
}
```

可以使用库函数`__gcd(int a, int b)`，此外最小公倍数=$\frac{a  b}{gcd(a,b)}$

### 欧拉函数

$\phi(n)$：1-n中与n互质的数的个数

$\phi(n) = n*(1-\frac{1}{p_1})*(1-\frac{1}{p_2})*...*(1-\frac{1}{p_n})$

```cpp
int phi(int x)
{
    int res = x;
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
        {
            res = res / i * (i - 1);
            while (x % i == 0) x /= i;
        }
    if (x > 1) res = res / x * (x - 1);

    return res;
}
```

#### 筛法求欧拉函数

```cpp
int primes[N], cnt;     // primes[]存储所有素数
int euler[N];           // 存储每个数的欧拉函数
bool st[N];         // st[x]存储x是否被筛掉


void get_eulers(int n)
{
    euler[1] = 1;
    for (int i = 2; i <= n; i ++ )
    {
        if (!st[i])
        {
            primes[cnt ++ ] = i;
            euler[i] = i - 1;
        }
        for (int j = 0; primes[j] <= n / i; j ++ )
        {
            int t = primes[j] * i;
            st[t] = true;
            if (i % primes[j] == 0)
            {
                euler[t] = euler[i] * primes[j];
                break;
            }
            euler[t] = euler[i] * (primes[j] - 1);
        }
    }
}
```

#### 欧拉定理

若$a$与$n$互质，则

$a^{\phi{(n)}} \equiv 1 (mod \ n)$

### 快速幂

在$O(logk)$时间内求出求出$a^k mod p$

```cpp
int qmi(int m, int k, int p)
{
    int res = 1 % p, t = m;
    while (k)
    {
        if (k&1) res = res * t % p;
        t = t * t % p;
        k >>= 1;
    }
    return res;
}
```

### 扩展欧几里得算法

裴蜀定理：对于正整数$a,b$，一定存在整数$x,y$，使得

$ax+by = gcd(a,b)$

```cpp
// 求x, y，使得ax + by = gcd(a, b)
int exgcd(int a, int b, int &x, int &y)
{
    if (!b)
    {
        x = 1; y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= (a/b) * x;
    return d;
}
```

### 中国剩余定理

给定一些两两互质的数$m_1,m_2,m_3,m_k$，求解线性同余方程组

$x \equiv a_1 (mod \ m_1)$

$x \equiv a_2 (mod \ m_2)$

$...$

$x \equiv a_k (mod \ m_k)$

$M = m_1 * m_2*...*m_k$

$M_i = \frac{M}{m_i}$

$M_i^{-1}$表示$M_i$模$m_i$的逆

$x = a_1M_1M_1^{-1}+a_2M_1M_2^{-1}+...+a_kM_1M_k^{-1}$

### 高斯消元

在$O(n^3)$内求解线性方程组

```cpp
// a[N][N]是增广矩阵
int gauss()
{
    int c, r;
    for (c = 0, r = 0; c < n; c ++ )
    {
        int t = r;
        for (int i = r; i < n; i ++ )   // 找到绝对值最大的行
            if (fabs(a[i][c]) > fabs(a[t][c]))
                t = i;
		//eps精度 1e-6
        if (fabs(a[t][c]) < eps) continue;

        for (int i = c; i <= n; i ++ ) swap(a[t][i], a[r][i]);      // 将绝对值最大的行换到最顶端
        for (int i = n; i >= c; i -- ) a[r][i] /= a[r][c];      // 将当前行的首位变成1
        for (int i = r + 1; i < n; i ++ )       // 用当前行将下面所有的列消成0
            if (fabs(a[i][c]) > eps)
                for (int j = n; j >= c; j -- )
                    a[i][j] -= a[r][j] * a[i][c];

        r ++ ;
    }

    if (r < n)
    {
        for (int i = r; i < n; i ++ )
            if (fabs(a[i][n]) > eps)
                return 2; // 无解
        return 1; // 有无穷多组解
    }
`
    for (int i = n - 1; i >= 0; i -- )
        for (int j = i + 1; j < n; j ++ )
            //储存答案
            a[i][n] -= a[i][j] * a[j][n];

    return 0; // 有唯一解
}
```

### 组合数

- $1 \le b \le a \le 2000$ 递推 $N^2$
- $1 \le b \le a \le 10^5$ 预处理 $NlogN$
- $1 \le b \le a \le 10^{18}, 1 \le p \le 10^5$ 卢卡斯定理Lucas
- 

组合数$C_{n}^{m}=\frac{n!}{m!(n-m)!} $

#### 朴素求法

```cpp
LL C(int a,int b){
    LL res = 1;
    for(int i=a,j=1;j<=b;i--,j++){
        res = res*i/j;
    }
    return res;
}
```

#### 递推法求组合数

$C_{a}^{b} = C_{a-1}^{b} + C_{a-1}^{b-1}$

```cpp
// c[a][b] 表示从a个苹果中选b个的方案数
for (int i = 0; i < N; i ++ )
    for (int j = 0; j <= i; j ++ )
        if (!j) c[i][j] = 1;
        else c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % mod;
```

#### 通过预处理逆元的方式求组合数

```cpp
首先预处理出所有阶乘取模的余数fact[N]，以及所有阶乘取模的逆元infact[N]
如果取模的数是质数，可以用费马小定理求逆元
int qmi(int a, int k, int p)    // 快速幂模板
{
    int res = 1;
    while (k)
    {
        if (k & 1) res = (LL)res * a % p;
        a = (LL)a * a % p;
        k >>= 1;
    }
    return res;
}

// 预处理阶乘的余数和阶乘逆元的余数
fact[0] = infact[0] = 1;
for (int i = 1; i < N; i ++ )
{
    fact[i] = (LL)fact[i - 1] * i % mod;
    infact[i] = (LL)infact[i - 1] * qmi(i, mod - 2, mod) % mod;
}
```

#### Lucas定理

```cpp
若p是质数，则对于任意整数 1 <= m <= n，有：
    C(n, m) = C(n % p, m % p) * C(n / p, m / p) (mod p)

int qmi(int a, int k, int p)  // 快速幂模板
{
    int res = 1 % p;
    while (k)
    {
        if (k & 1) res = (LL)res * a % p;
        a = (LL)a * a % p;
        k >>= 1;
    }
    return res;
}

int C(int a, int b, int p)  // 通过定理求组合数C(a, b)
{
    if (a < b) return 0;

    LL x = 1, y = 1;  // x是分子，y是分母
    for (int i = a, j = 1; j <= b; i --, j ++ )
    {
        x = (LL)x * i % p;
        y = (LL) y * j % p;
    }

    return x * (LL)qmi(y, p - 2, p) % p;
}

int lucas(LL a, LL b, int p)
{
    if (a < p && b < p) return C(a, b, p);
    return (LL)C(a % p, b % p, p) * lucas(a / p, b / p, p) % p;
}
```

#### 分解质因数法求组合数

```cpp
当我们需要求出组合数的真实值，而非对某个数的余数时，分解质因数的方式比较好用：
    1. 筛法求出范围内的所有质数
    2. 通过 C(a, b) = a! / b! / (a - b)! 这个公式求出每个质因子的次数。 n! 中p的次数是 n / p + n / p^2 + n / p^3 + ...
    3. 用高精度乘法将所有质因子相乘

int primes[N], cnt;     // 存储所有质数
int sum[N];     // 存储每个质数的次数
bool st[N];     // 存储每个数是否已被筛掉


void get_primes(int n)      // 线性筛法求素数
{
    for (int i = 2; i <= n; i ++ )
    {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= n / i; j ++ )
        {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
}


int get(int n, int p)       // 求n！中的次数
{
    int res = 0;
    while (n)
    {
        res += n / p;
        n /= p;
    }
    return res;
}


vector<int> mul(vector<int> a, int b)       // 高精度乘低精度模板
{
    vector<int> c;
    int t = 0;
    for (int i = 0; i < a.size(); i ++ )
    {
        t += a[i] * b;
        c.push_back(t % 10);
        t /= 10;
    }

    while (t)
    {
        c.push_back(t % 10);
        t /= 10;
    }

    return c;
}

get_primes(a);  // 预处理范围内的所有质数

for (int i = 0; i < cnt; i ++ )     // 求每个质因数的次数
{
    int p = primes[i];
    sum[i] = get(a, p) - get(b, p) - get(a - b, p);
}

vector<int> res;
res.push_back(1);

for (int i = 0; i < cnt; i ++ )     // 用高精度乘法将所有质因子相乘
    for (int j = 0; j < sum[i]; j ++ )
        res = mul(res, primes[i]);
```

### 卡特兰数

给定n个0和n个1，它们按照某种顺序排成长度为2n的序列，满足任意前缀中0的个数都不少于1的个数的序列的数量为： Cat(n) = C(2n, n) / (n + 1)