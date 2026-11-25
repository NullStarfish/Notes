
 **Einstein summation convention（爱因斯坦求和约定）** 

> 用“下标名字”描述 tensor 的维度；某个下标如果出现在输入里、但不出现在输出里，就对它求和。

最经典的是矩阵乘法。

假设：

```
A.shape == (m, k)
B.shape == (k, n)
```

普通写法：

```
C = A @ B
```

用 einsum 写：

```
C = np.einsum('mk,kn->mn', A, B)
```

这里：

```
A: m k
B: k n
--------
C: m   n
```

`k` 在输入中出现，但输出 `mn` 里没有，所以：

$$ C_{mn} = \sum_k A_{mk} B_{kn} $$

这就是矩阵乘法。

你可以把它看成一种非常通用的“tensor index language”。

比如点积：

```
np.einsum('i,i->', a, b)
```

对应：

$$ \sum_i a_i b_i $$

输出没有任何 axis，所以得到 scalar。

逐元素乘法：

```
np.einsum('ij,ij->ij', A, B)
```

这里 `i,j` 都保留，所以不求和，就是 elementwise multiply。

转置：

```
np.einsum('ij->ji', A)
```

甚至不需要乘法，单纯改变 axis 顺序。

求 trace：

```
np.einsum('ii->', A)
```

因为两个 index 都叫 `i`，只取对角线，然后输出里又没有 `i`，所以求和：

$$ \sum_i A_{ii} $$

einsum 真正厉害的地方是高维 tensor。

比如：

```
Q.shape = (batch, heads, query, dim)
K.shape = (batch, heads, key,   dim)
```

attention 里的：

```
scores = einsum(
    Q, K,
    'b h q d, b h k d -> b h q k'
)
```

严格对应：

$$ S_{bhqk} = \sum_d Q_{bhqd} K_{bhkd} $$

这里 `d` 消失，所以沿 `d` 求和；`b,h,q,k` 保留。

这其实比 `matmul` 更一般。

你可以把 `einsum` 理解成：

```
rearrange + elementwise multiply + reduce
```

的一种统一描述语言。

甚至很多 einsum：

```
'b h q d, b h k d -> b h q k'
```

可以脑补成：

```
先把两个 tensor broadcast 到共同 shape
(b, h, q, k, d)

然后 elementwise multiply

最后 reduce d
```

这和你刚才学 einops 时的 axis 思维是非常一致的。

另外，`einops.einsum` 和 NumPy/PyTorch 的区别主要是：einops 允许使用完整 axis 名。

NumPy：

```
np.einsum('bhqd,bhkd->bhqk', Q, K)
```

einops：

```
einsum(Q, K, 'batch head query dim, batch head key dim -> batch head query key')
```

后者可读性强很多。

如果只记一句话：

> **einsum 就是在写 tensor 的下标公式：输出里没出现的下标，就被 sum 掉。**

这个规则基本能解释 80% 的 einsum。