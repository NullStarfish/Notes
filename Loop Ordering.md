
Loop Ordering 决定了复用潜力如何实现。 即使 Tile Shape 完全相同，不同的循环顺序也可能导致完全不同的访存量。


GEMM 中的三类数据复用
#### 复用 A

固定 i、k，变化 j。同一个 A Tile 为多个输出列贡献结果。

#### 复用 B

固定 k、j，变化 i。同一个 B Tile 为多个输出行贡献结果。

#### 复用 C：固定 i、j，变化 k

$$C_{ij}\leftarrow C_{ij}+A_{ik}B_{kj} $$

C Tile 作为 Partial Sum，在不同 K 迭代之间不断累加，而不必每次写回 Global Memory。

|固定的数据|需要遍历的维度|复用机会|
|---|---|---|
|A(i,k)|j|复用 A|
|B(k,j)|i|复用 B|
|C(i,j)|k|复用 Partial Sum|

## 对比Loop Ordering
假设每个 Tile 都已经定义好，Global Memory 存放 A、B、C；
#### A，B，C各自只有一个local tile sram

设：

$$ I=\frac{M}{B_M},\quad J=\frac{N}{B_N},\quad L=\frac{K}{B_K} $$

我们暂时只统计 Global Memory 与 Local Memory 之间的传输，不统计 Matrix Engine 内部的寄存器读写。


### 1. Output Stationary

```python
for i in range(I):
    for j in range(J):
        C_tile = zeros()

        for k in range(L):
            A_tile = load_A(i, k)
            B_tile = load_B(k, j)
            C_tile += matmul(A_tile, B_tile)

        store_C(i, j, C_tile)
```

A和B需要不断读取，在K上iteration
C不动 （随i, j store）

传输次数：
$$ \begin{aligned} Q_A &= IJL\\ Q_B &= IJL\\ Q_C^{read} &=0\\ Q_C^{write}&=IJ \end{aligned} $$

### 2. A stationary

```python
for i in range(I):
    for k in range(L):
        A_tile = load_A(i, k)

        for j in range(J):
            B_tile = load_B(k, j)
            partial = load_partial_C(i, j)
            partial += matmul(A_tile, B_tile)
            store_partial_C(i, j, partial)
```


$$ \begin{aligned} Q_A &= IL\\ Q_B &= IJL\\ Q_C^{read}&=IJ(L-1)\\ Q_C^{write}&=IJL \end{aligned} $$



### 3. B stationary
```python
for j in range(J):
    for k in range(L):
        B_tile = load_B(k, j)

        for i in range(I):
            A_tile = load_A(i, k)
            partial = load_partial_C(i, j)
            partial += matmul(A_tile, B_tile)
            store_partial_C(i, j)
```

$$ \begin{aligned} Q_A &= IJL\\ Q_B &= JL\\ Q_C^{read}&=IJ(L-1)\\ Q_C^{write}&=IJL \end{aligned} $$






很明显，由于C通常size比A/B要大的多，因此Output Stationary依旧是主流


但是，通常而言NPU中的sram很多：
```python
for i in range(I):
    C_tiles = [zeros() for j in range(J)]

    for k in range(L):
        A_tile = load_A(i, k)

        for j in range(J):
            B_tile = load_B(k, j)
            C_tiles[j] += matmul(A_tile, B_tile)

    for j in range(J):
        store_C(i, j, C_tiles[j])
```

我们可以保留一个J循环的C tile。但是这样我们需要J倍的accumulator消耗




### 完整版

```python
for i0 in range(0, I, P):
    for j0 in range(0, J, Q):

        # 当前输出分组的 P×Q 个 C Tile
        C = zeros(P, Q, BM, BN)

        for k in range(L):
            A = load_A_tiles(i0, P, k)
            B = load_B_tiles(j0, Q, k)

            for p in range(P):
                for q in range(Q):
                    C[p,q] += matmul(A[p], B[q])

        store_C_tiles(i0, j0, C)
```

这是一个很大的A，B，C
我们的硬件资源让我们驻留P \* Q个 C tile。
这有点像有些架构中的L1 sram，用来装更大的tile
此时，A也扩大P倍，B扩大Q倍

容量约束：
$$ \boxed{ S= 2PB_MB_K+ 2QB_KB_N+ 4PQB_MB_N } $$

每轮 K 迭代：

$$ \text{Compute}=2PQB_MB_NB_K $$

A/B 的输入字节数：

$$\text{Bytes}=s_A PB_MB_K+s_B QB_KB_N $$

如果输入均为 FP16：

$$ AI=\frac{PQB_MB_N}{PB_M+QB_N} $$


这一结果十分重要，因为它说明 通过跨 Tile 的复用，Arithmetic Intensity 可以在不改变单个硬件 Matrix Tile Shape 的情况下继续提高。