
>[!question]
>Tiling是什么？为什么会有Tiling的分析？

我们在设计npu的时候，肯定会设计一些local的sram来加速访存，存放一些local中间变量：

很明显，无限容量的local sram能够直接消灭globalmemory
但是我们并没有，因此我们必须回答：
__*如何在有限资源下，达到最高的计算效率，减少访存次数*__

>[!note]
>Tiling中没有对计算主体的建模，不含matrix engine或者compute core
>我们仅仅分析Arithmetic Intensity: 负责关注BM, BN, BK，关注的是理想情况：
>single core，single load-store unit时候的访存-计算比分析






运算的proof:
[[Tiling Proof]]


## AI

我们先固定一个 \(b\times b\) 的输出 Tile，并且每次处理 K 维度上的 \(b\) 个元素。

于是：

$$ C_t^{b\times b} \mathrel{+}= A_t^{b\times b}B_t^{b\times b} $$

一次 Tile Matrix Multiply

一次计算：

- 加载 A Tile：\(b^2\) 个元素；
    
- 加载 B Tile：\(b^2\) 个元素；
    
- 计算：\(2b^3\) FLOPs。
    
假设A和B的byte size = s
因此，如果暂时只考虑 A、B 从 Global Memory 到 Local Memory 的流量：

$$ \text{Arithmetic Intensity} =\frac{2b^3}{2b^2s} =\boxed{\frac{b}{s}\text{ FLOPs/Byte}} $$


>[!note]
>所以在其他条件相同、且 Tile 内部数据确实能够复用的情况下，更大的 Tile 往往能获得更高的 Arithmetic Intensity。
>但是我们不能无限扩大 Tile，因为 Local Memory 的容量有限。



variables:
`M, N, K, TM, TN, TK`

# 讨论TM, TN, TK的选取
## 给定有限的 Local Memory，如何选择 GEMM Tile Size？
$$ C_{B_M\times B_N} \mathrel{+}= A_{B_M\times B_K}B_{B_K\times B_N} $$


$$ F=2B_MB_NB_K $$
(总共BM\*BN元素，每个都需要2BK)

如果 A、B 都采用 FP16，且仅考虑每次 K 迭代从 Global Memory 读取 A、B 的字节数：

$$ Q=2B_MB_K+2B_KB_N $$

于是：

$$ \boxed{AI=\frac{B_MB_N}{B_M+B_N}} $$


### 通常C一定是方形：
假设我们暂时固定 \(B_K\)，同时 A、B 必须放在 Local Memory 中。

A、B 占用的存储量：

$$ S_{AB}=2B_K(B_M+B_N) $$

假设可用于 A/B 的空间固定，也就是：

$$ B_M+B_N=P $$

此时 Arithmetic Intensity 为：

$$ AI=\frac{B_MB_N}{P} $$

根据均值不等式：

$$ B_MB_N\leq\left(\frac{P}{2}\right)^2 $$

当且仅当

$$ \boxed{B_M=B_N=\frac P2} $$


## 考虑C时：
现在假设：

- A、B：FP16，每元素 2 Bytes
    
- C Accumulator：FP32，每元素 4 Bytes
    
- 本地可用总容量：64 KiB
    
- 不考虑 Double Buffering、Padding 以及额外硬件保留区
    

那么必须满足：

$$ \boxed{ 2B_MB_K+2B_KB_N+4B_MB_N\leq65536 } $$

这里出现了一个明显的设计冲突：

- 扩大 $B_M,B_N$，可以提高 A/B 数据复用。
    
- 但是扩大 $B_M,B_N$，会以乘积形式增加 C Accumulator 的容量需求。
    
- 扩大 $B_K$，会增加 A/B Working Set，却不直接增加当前模型中的 Arithmetic Intensity。

（这里working set指的是tile数据占据的内存空间）
减小BK，能让上面这个不等式约束更容易满足



这听起来似乎说明应该让 $$B_K$$ 尽可能小。但实际 NPU 还需要考虑矩阵指令的最小 K 粒度、每次 Load/Compute 的启动开销、K 循环迭代次数及硬件效率。


1）K 循环开销

整个 GEMM 需要：

$$ L=\left\lceil\frac{K}{B_K}\right\rceil $$

次 K 迭代。

\(B_K\) 越小，迭代次数越多，可能增加 DMA 启动、指令发射、同步与地址生成的开销。

（2）Matrix Engine 的硬件粒度

假如矩阵单元一次要求消费 K 维度上至少 32 个元素，那么 \(B_K=8\) 可能需要 Padding 或无法有效利用硬件。

（3）Pipeline 效率

较小的 \(B_K\) 让每个 K Tile 的计算任务更短，可能难以摊薄每次数据搬运和调度的固定开销；较大的 \(B_K\)又可能增加 Buffer 占用并降低灵活性。

（4）独立的 Accumulator 存储资源

实际架构可能满足：

$$ s_A B_MB_K+s_B B_KB_N\leq S_{\mathrm{SRAM}} $$

$$ s_C B_MB_N\leq S_{\mathrm{ACC}} $$

这和将 A/B/C 放在同一个容量约束中的优化问题不同。
