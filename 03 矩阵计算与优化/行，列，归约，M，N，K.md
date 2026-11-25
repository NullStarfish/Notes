你这里刚好理解反了！不过这主要是因为**数学中的矩阵维度表示习惯，与我们日常描述矩形尺寸的习惯不同**。

在线性代数、编程以及 GEMM 中，矩阵形状统一使用：

$$ \boxed{\text{Rows}\times\text{Columns}=\text{行数}\times\text{列数}} $$

也就是**先高度，后宽度**，而不是我们平常说的“长 × 宽”。

例如，一个 $3\times 4$ 的矩阵：

因此对于 GEMM：

$$ \underbrace{A_{M\times K}}_{\text{M 行，K 列}} \quad \underbrace{B_{K\times N}}_{\text{K 行，N 列}} = \underbrace{C_{M\times N}}_{\text{M 行，N 列}} $$

具体到 Tiling：

| 矩阵  | Tile 形状         | 行数    | 列数    |
| --- | --------------- | ----- | ----- |
| A   | $B_M\times B_K$ | $B_M$ | $B_K$ |
| B   | $B_K\times B_N$ | $B_K$ | $B_N$ |
| C   | $B_M\times B_N$ | $B_M$ | $B_N$ |

其实你可以直接从矩阵乘法的定义记住这个关系：

$$ C_{ij}=\sum_{k=0}^{K-1} A_{ik}B_{kj} $$

这里有三个指标：

- \(i\)：A 和 C 的行索引，对应 \(M\)。
    
- \(j\)：B 和 C 的列索引，对应 \(N\)。
    
- \(k\)：A 的列索引、B 的行索引，对应归约维度 \(K\)。
    

**所以** \(K\) **并不是固定的“行”或者“列”，而是矩阵乘法的 Reduction Dimension。**

这也是为什么我们讨论 GEMM Tiling 时，更倾向于将三个维度理解为：

$$ \boxed{ B_M=\text{输出行方向},\quad B_N=\text{输出列方向},\quad B_K=\text{归约方向} } $$

这样以后不论是研究 Loop Ordering、Arithmetic Intensity，还是将 \(M,N,K\) 映射到 NPU 的空间阵列上，都不容易混淆。