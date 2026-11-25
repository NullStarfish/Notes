# Tiling：从数据复用到多级分块

> [!question] Tiling 是什么？为什么要分析它？
> NPU 用有限的 Local SRAM 保存数据和中间结果。我们希望在数据离开这一层之前，尽量多地使用它，从而减少更外层的重复读取与写回。Tiling 要回答：**一次保留哪些数据、保留多少、在释放前完成哪些计算？**

大容量本地存储可以减少外层访存，但不能无条件消除输入、输出的必要传输。现实中容量有限，因此需要把计算切成能容纳的工作集，并安排数据复用。

本文依次讨论 GEMM 的维度、单 Tile 算术强度、驻留策略、多级与阵列级 Tile，最后把它们统一到尺寸选择。每次乘加按 2 FLOPs 计；AI 指 Arithmetic Intensity，单位为 FLOPs/Byte。块矩阵乘法为什么成立，见 [[Tiling Proof]]。

## 1. 先把 M、N、K 与 Tile 维度说清楚

$$
C_{M\times N}=A_{M\times K}B_{K\times N},\qquad
C_{ij}=\sum_k A_{ik}B_{kj}.
$$

> [!question] 我曾把 A 的 $B_M$ 当成列数、$B_K$ 当成行数，因为矩形一般说“长 × 宽”。这里应该怎么理解？

矩阵形状的约定是 **行数 × 列数**。$M$ 对应 A、C 的行方向，$N$ 对应 B、C 的列方向；$K$ 对应 A 的列与 B 的行，是被求和的归约方向。

| 数据 | 完整形状 | Tile 形状 | 行数 | 列数 |
| --- | --- | --- | --- | --- |
| A | $M\times K$ | $B_M\times B_K$ | $B_M$ | $B_K$ |
| B | $K\times N$ | $B_K\times B_N$ | $B_K$ | $B_N$ |
| C | $M\times N$ | $B_M\times B_N$ | $B_M$ | $B_N$ |

例如 $B_M=4,B_K=8,B_N=4$，就是一个 $4\times8$ 的 A 块乘一个 $8\times4$ 的 B 块，更新一个 $4\times4$ 的 C 块。$B_K$ 不是固定的“行”或“列”，而是两侧共同的归约长度。

## 2. 单个 Tile 怎样提高 Arithmetic Intensity？

### 2.1 数学上的复用机会与实际的数据复用

朴素 GEMM 可以写成：

```python
for i in range(M):
    for j in range(N):
        acc = 0
        for k in range(K):
            acc += A[i, k] * B[k, j]
        C[i, j] = acc
```

如果每次使用 A/B 都直接从 Global Memory（GM）读取，而没有 cache 帮助，A 的同一个元素会为不同输出列反复读取，B 的同一个元素会为不同输出行反复读取。

Tiling 把多个输出一起算，使一次载入的 A/B 能服务多个乘加。**数学上会被多次使用，不代表实际只搬运一次；数据要被保留或共享，复用才会落实。**

### 2.2 从方块开始

固定一个 $b\times b$ 输出块，每次处理 K 方向的 $b$ 个元素：

$$
C_t^{b\times b}\mathrel{+}=A_t^{b\times b}B_t^{b\times b}.
$$

读取 A/B 各 $b^2$ 个元素，完成 $2b^3$ FLOPs。若输入每元素 $s$ 字节，且只统计 A/B 从 GM 到 Local 的流量：

$$
AI_{AB}=\frac{2b^3}{2b^2s}=\boxed{\frac bs}.
$$

在输入确实被复用的前提下，计算量按立方增长，输入量按平方增长，所以更大的块通常提高这一输入 AI；代价是更大的本地工作集。

### 2.3 推广到一般 Tile

$$
C_{B_M\times B_N}\mathrel{+}=A_{B_M\times B_K}B_{B_K\times B_N}.
$$

$B_MB_N$ 个输出各执行 $B_K$ 次乘加，因此：

$$
F=2B_MB_NB_K,\qquad
Q_{AB}=s_AB_MB_K+s_BB_KB_N.
$$

$$
AI_{AB}=\frac{2B_MB_N}{s_AB_M+s_BB_N}.
$$

A/B 均为 FP16 时：

$$
\boxed{AI_{AB}=\frac{B_MB_N}{B_M+B_N}}.
$$

直观上，每个 A 元素服务 $B_N$ 个输出列，每个 B 元素服务 $B_M$ 个输出行。$B_K$ 同时线性增加计算量和输入量，因而在比值中抵消。

> [!important] 这个公式的范围
> 它只统计一次 Tile 计算的 A/B 输入，不包含 C 的初始读取、最终写回与中间部分和搬运，也未计入不同输出 Tile 之间的输入复用。因此它是一个明确模型下的输入 AI，还不能代表完整 GEMM 的全部流量。

## 3. Stationary：从单 Tile AI 走向真实流量

### 3.1 哪些数据可以保留？

把 $C_{ij}=\sum_k A_{ik}B_{kj}$ 中的索引理解为 Tile 索引，就得到三类复用：

| 固定的数据 | 改变的索引 | 复用方向 | 驻留策略 |
| --- | --- | --- | --- |
| $A_{ik}$ | $j$ | 同一输入服务不同输出列组 | A stationary |
| $B_{kj}$ | $i$ | 同一输入服务不同输出行组 | B stationary |
| $C_{ij}$ | $k$ | 部分和在归约过程中持续累加 | Output / C stationary |

当 A 是 activation、B 是 weight 时，A/B stationary 分别对应这里的 input/weight stationary；通用 GEMM 中使用 A/B 的名字更明确。

例如 C 被划分为四块，同一轮 K 计算为：

$$
\begin{bmatrix}C_{00}&C_{01}\\C_{10}&C_{11}\end{bmatrix}
\mathrel{+}=
\begin{bmatrix}A_0B_0&A_0B_1\\A_1B_0&A_1B_1\end{bmatrix}.
$$

$A_0$ 可以服务 $C_{00}$ 和 $C_{01}$，$B_0$ 可以服务 $C_{00}$ 和 $C_{10}$；每块 C 又需要跨 K 累加。保留哪块数据、同时容纳多少块，会决定这些复用能否同时实现。

Stationary 必须指定层级。**C 驻留在 Local SRAM，可以减少 GM–Local 的部分和往返，但计算单元仍可能在 SRAM 与寄存器之间读写 C。**它不等于所有部分和始终停留在 PE 寄存器中。

### 3.2 Output stationary：先算完一个输出块

```python
for i in range(I):
    for j in range(J):
        C_tile = zeros(BM, BN)
        for k in range(L):
            A_tile = load_A(i, k)
            B_tile = load_B(k, j)
            C_tile += matmul(A_tile, B_tile)
        store_C(i, j, C_tile)
```

固定一个 C Tile，做完全部 K 归约再写回。假设 $C=AB$，输入每个 K 子块各加载一次，C 无须读入旧值，则这一个输出块的完整 AI 为：

$$
\boxed{AI_{\mathrm{OS}}=
\frac{2B_MB_NK}
{s_AB_MK+s_BKB_N+s_{\mathrm{out}}B_MB_N}}.
$$

C 的写回由完整的 K 归约摊销，因此这里是 $K$，不是一次迭代的 $B_K$。只有在 C 流量相对较小时，FP16 输入的完整 AI 才接近 $B_MB_N/(B_M+B_N)$。

例：计算 $C_{64\times64}=A_{64\times1024}B_{1024\times64}$，$B_K=32$，A/B 为 FP16，C 为 FP32。32 轮共读取 A/B 256 KiB，最后写回 C 16 KiB，总流量 272 KiB。计算量为 $2\times64\times64\times1024$，完整 AI 约为 30.12，而只统计输入时为 32。

若语义是 $C\leftarrow AB+C$，应再加一次读取初始 C 的流量；若 C 每轮都溢出本地，则还需加入部分和往返。

### 3.3 换成 A/B stationary，省下的输入会不会被 C 流量抵消？

以 A stationary 为例，交换循环，让一块 A 跨多个输出列组复用：

```python
for i in range(I):
    for k in range(L):
        A_tile = load_A(i, k)
        for j in range(J):
            B_tile = load_B(k, j)
            partial = zeros(BM, BN) if k == 0 else load_partial_C(i, j)
            partial += matmul(A_tile, B_tile)
            store_partial_C(i, j, partial)
```

A 的读取次数降低了，但不同 K 轮之间，那些 C Tile 的部分和放在哪里？若本地放不下而写回 GM，下一轮还需要重新读入。这是本例的存储选择，不是 A stationary 的必然要求。

为了定量比较，假设尺寸整除：

$$
I=M/B_M,\qquad J=N/B_N,\qquad L=K/B_K.
$$

只保留指定驻留块与当前计算所需工作块，没有额外 cache 或外层复用；统计 GM–Local 流量。B stationary 与 A stationary 对称，循环为 $j\to k\to i$。

| 策略 | A 读取次数 | B 读取次数 | C 读取次数 | C 写入次数 |
| --- | --- | --- | --- | --- |
| C stationary：$i\to j\to k$ | $IJL$ | $IJL$ | 0 | $IJ$ |
| A stationary：$i\to k\to j$ | $IL$ | $IJL$ | $IJ(L-1)$ | $IJL$ |
| B stationary：$j\to k\to i$ | $IJL$ | $JL$ | $IJ(L-1)$ | $IJL$ |

例如 A 总共只有 $IL$ 块，驻留时每块读一次；C 有 $IJ$ 块，第一轮初始化不读，其后 $L-1$ 轮各读一次，但每轮都写，所以读 $IJ(L-1)$ 次、写 $IJL$ 次。

表中是 **Tile 次数**。令每块字节数为 $a=s_AB_MB_K$、$b=s_BB_KB_N$、$c=s_CB_MB_N$，先假设中间 C 与最终 C 同宽，则：

$$
\begin{aligned}
Q_{\mathrm{OS}}&=IJL(a+b)+IJc,\\
Q_{\mathrm{AS}}&=ILa+IJLb+IJ(2L-1)c,\\
Q_{\mathrm{BS}}&=IJLa+JLb+IJ(2L-1)c.
\end{aligned}
$$

三者计算量相同，都是 $2MNK$，各自的完整 AI 为计算量除以对应总字节数。若中间部分和宽度为 $s_{\mathrm{acc}}$，最终输出为 $s_{\mathrm{out}}$，非驻留方案每块 C 的流量应写成：

$$
\bigl[2(L-1)s_{\mathrm{acc}}+s_{\mathrm{out}}\bigr]B_MB_N.
$$

### 3.4 一个完整的数值对比

取 $I=J=2,L=4$，$B_M=B_N=B_K=32$，A/B 为 FP16，所有 C 传输均为 FP32。A/B 每块 2 KiB，C 每块 4 KiB：

| 策略 | A 读取 | B 读取 | C 读取 | C 写入 | 总流量 | 完整 AI |
| --- | --- | --- | --- | --- | --- | --- |
| C stationary | 32 KiB | 32 KiB | 0 | 16 KiB | 80 KiB | 12.8 |
| A stationary | 16 KiB | 32 KiB | 48 KiB | 64 KiB | 160 KiB | 6.4 |
| B stationary | 32 KiB | 16 KiB | 48 KiB | 64 KiB | 160 KiB | 6.4 |

AI 单位为 FLOPs/Byte。A stationary 虽然省了 16 KiB 的 A 读取，却增加了 96 KiB 的 C 往返。**只优化一种数据的复用，不保证总流量减少。**

### 3.5 为什么不能同时保留 C、复用 A/B？

可以，代价是更大的工作集。例如固定一个输出行组，同时保留两个 C Tile：

```python
C_tiles = [zeros(BM, BN), zeros(BM, BN)]
for k in range(L):
    A_tile = load_A(i, k)
    for j in range(2):
        B_tile = load_B(k, j)
        C_tiles[j] += matmul(A_tile, B_tile)
# 全部 K 结束后，再写回两个 C Tile
```

上例中 A 读取从 16 次降到 8 次，B 仍为 16 次，C 仍只写 4 次。总流量变成 64 KiB，完整 AI 为 16；但每个输出行组要同时保留 8 KiB 的 C，而不是 4 KiB。

推广到 $R\times T$ 个输出 Tile 的复用区域，每轮加载 $R$ 个 A 块、$T$ 个 B 块：

$$
F_{\mathrm{group}}=2RTB_MB_NB_K,
$$

$$
Q_{AB,\mathrm{group}}=s_ARB_MB_K+s_BT B_KB_N.
$$

FP16 输入时：

$$
AI_{AB,\mathrm{group}}=\frac{RTB_MB_N}{RB_M+TB_N}.
$$

同时驻留的 C 占 $s_{\mathrm{acc}}RTB_MB_N$ 字节。固定小块为 $32\times32\times32$ 时：

| 输出 Tile 分组 | A/B 输入 AI | C 累加容量 |
| --- | --- | --- |
| $1\times1$ | 16 | 4 KiB |
| $1\times2$ | $21.33$ | 8 KiB |
| $2\times2$ | 32 | 16 KiB |
| $4\times4$ | 64 | 64 KiB |

这些容量只包含 C，A/B 还需额外空间。**硬件小块不变，也可以通过扩大软件复用区域提高外层 AI。**这自然引出多级 tiling。

## 4. 多级 Tiling：引入 PE/Core 之后，原分析还成立吗？

> [!question] 最初选 $B_M,B_N,B_K$ 时没有考虑计算主体。一个全 NPU 的大 Tile，还要分给 core、私有 SRAM 和 matrix engine；是不是每层还要讨论 AI？固定执行粒度会不会改变它？

是的。基础 AI 推导可以先不建模计算主体，但落到硬件后需要指定每层的存储、驻留与共享范围。相同的比值方法可以逐层使用，分母却未必相同。

```text
Global Memory
    ↓ 外层数据块
Shared SRAM / 上层缓冲
    ↓ 分发、复制或共享
Core Local SRAM
    ↓ 更小的操作数块与部分和
Matrix Instruction / 计算单元
```

| 层次 | 尺寸或参数 | 作用 |
| --- | --- | --- |
| 指令 / micro-tile | $m_0,n_0,k_0$ | 硬件一次能处理的基本形状 |
| 单核 Tile | $B_M,B_N,B_K$ | 核心本地的一组计算与工作集 |
| 阵列级输出区域 | $H_M=P_MB_M,\ H_N=P_NB_N$ | 多核心共同覆盖的输出区域 |
| 上层搬运块 | 如 $K_L$ | 可以拆成多个单核 $B_K$ 子块 |

因此不能把 GM–Shared 的 AI 直接当成 Shared–Local 或 Local–寄存器的 AI。外层数据可以只读一次，内层却可能存在多个副本或重复读取。

### 4.1 先理解阵列级块乘法，不预设内部时序

这里的 PE/Core 表示能独立处理矩阵子块的计算位置，可以是 tile core，不一定是单个标量 MAC。设阵列有 $P_M$ 行、$P_N$ 列，核心 $(p,q)$ 负责一个 $B_M\times B_N$ 输出块，K 维分批累加。

以 $2\times3$ 阵列为例，每核 $B_M=B_N=4,B_K=8$。每轮处理：

$$
A_{8\times8}B_{8\times12}\longrightarrow C_{8\times12}.
$$

把 A 分成两个 $4\times8$ 行块 $A_0,A_1$，B 分成三个 $8\times4$ 列块 $B_0,B_1,B_2$：

|  | B₀ | B₁ | B₂ |
| --- | --- | --- | --- |
| A₀ | 核心(0,0)：$C_{00}\mathrel{+}=A_0B_0$ | 核心(0,1)：$C_{01}\mathrel{+}=A_0B_1$ | 核心(0,2)：$C_{02}\mathrel{+}=A_0B_2$ |
| A₁ | 核心(1,0)：$C_{10}\mathrel{+}=A_1B_0$ | 核心(1,1)：$C_{11}\mathrel{+}=A_1B_1$ | 核心(1,2)：$C_{12}\mathrel{+}=A_1B_2$ |

这张表首先是块矩阵乘法：同一行核心消费相同 A 块，同一列核心消费相同 B 块。它只规定数据需求与输出归属，不规定操作数逐周期如何到达。

### 4.2 “为什么只加载 $P_MB_MB_K$？不是所有 A 都要加载吗？”

> [!question] 我在讨论中看到 A 的加载量是 $P_MB_MB_K$，并说它被 $P_N$ 列核心共同使用。我困惑的是：肯定所有元素都要加载，为什么这样算？

所有**不同**的 A 元素确实都要加载。这里节省的是为不同核心重复读取同一元素的次数。

在上面的例子中：

- 整个 A 块有 $8\times8=64$ 个不同元素，由两个 A 行块组成。
- 每个核心需要 $4\times8=32$ 个 A 元素；6 个核心独立从 GM 读取，就产生 $6\times32=192$ 次元素读取。
- 同一行的三个核心需要同一块 A。若只从 GM 读取一次，再在片上分发，则源端只读 $2\times32=64$ 个 A 元素。

所以一般情况下，阵列这一轮所需的不同 A、B 元素分别是：

$$
P_MB_MB_K,\qquad P_NB_KB_N.
$$

这些元素可以通过共享缓冲、广播或其他分发机制服务多个核心。这里讨论的是复用关系；具体怎样实现阵列的数据传递，留待阵列计算部分展开。

### 4.3 到底在哪一层统计“加载量”？

继续上例，即使 GM 只读取 64 个 A 元素，若每个核心私有 SRAM 各放一份，所有核心合计仍接收 192 个 A 元素：

| 统计位置 | A 元素量 |
| --- | --- |
| GM 读取 / 阵列共享源端 | 64 |
| 6 个核心目的端接收量之和 | 192 |
| 不同 A 元素数量 | 64 |

这三个数量不矛盾。共享源端的读取量、目的端副本总量、网络中各跳累计传输量，是不同指标。Stationary 或广播带来的节省，只适用于它实际减少传输的那一级。

### 4.4 阵列级输入 AI

假设 A 沿核心行共享、B 沿核心列共享，源端不为每个核心重复读取。一次 K 轮：

$$
F_{\mathrm{array}}=2P_MP_NB_MB_NB_K,
$$

$$
Q_{AB,\mathrm{source}}=s_AP_MB_MB_K+s_BP_NB_KB_N.
$$

FP16 输入时：

$$
\boxed{AI_{AB,\mathrm{array}}
=\frac{P_MP_NB_MB_N}{P_MB_M+P_NB_N}
=\frac{H_MH_N}{H_M+H_N}}.
$$

它与上一节 $R\times T$ 分组的公式一致：同一组复用可以由单核分时完成，也可以映射到多个核心。源端流量由复用范围决定，不能仅凭“增加了核心数”就断定 AI 上升。

在 $2\times3$、$4\times8\times4$ 的例子中，A/B 共享读取为 $(64+96)\times2=320$ Bytes，独立读取为 $(192+192)\times2=768$ Bytes；两者都完成 1536 FLOPs，源端输入 AI 分别为 4.8、2。

### 4.5 完整 GEMM 的 $Q_A,Q_B,Q_C$ 从哪里来？

> [!question] 讨论中的 $Q_A=MNKs_A/(P_NB_N)$、$Q_B=MNKs_B/(P_MB_M)$、$Q_C=MNs_{\mathrm{out}}$ 是怎么推出来的？

假设 $C=AB$；每个 $H_M\times H_N$ 输出区域跨完整 K 驻留；组内输入共享；不同输出区域之间不再复用 A/B；尺寸都能整除，不计边界填充。

沿 M、N 方向分别有 $M/H_M$、$N/H_N$ 组。一个输出区域需要读取 $H_M\times K$ 的 A、$K\times H_N$ 的 B，因此：

$$
\begin{aligned}
Q_A&=\frac{M}{H_M}\frac{N}{H_N}(s_AH_MK)
=\boxed{\frac{MNKs_A}{P_NB_N}},\\
Q_B&=\frac{M}{H_M}\frac{N}{H_N}(s_BKH_N)
=\boxed{\frac{MNKs_B}{P_MB_M}}.
\end{aligned}
$$

也可以按重复读取次数理解：**完整 A 有 $MK$ 个元素，为每个输出列组读一次，共 $N/H_N$ 次；完整 B 有 $KN$ 个元素，为每个输出行组读一次，共 $M/H_M$ 次。**

例如 $M=16,N=24,K=16$，阵列区域仍为 $H_M=8,H_N=12$。共有 $2\times2$ 个区域，每个区域读取 128 个 A、192 个 B；总计 A 512 个、B 768 个元素。也就是完整 A（256 个）读两次、完整 B（384 个）读两次。

Output stationary 保证每个输出元素最后写一次：

$$
Q_C=MN s_{\mathrm{out}}.
$$

所以：

$$
\boxed{AI_{\mathrm{outer}}
=\frac{2MNK}{Q_A+Q_B+Q_C}
=\frac{2K}{Ks_A/H_N+Ks_B/H_M+s_{\mathrm{out}}}}.
$$

$B_K$ 再次抵消，是因为全部 K 子块最终都要读取，不是 K 方向的数据可以省略。若改变跨区域复用、C 语义或部分和驻留方式，流量公式也要重新统计。

## 5. 现在怎样选择 $B_M,B_N,B_K$？

AI 说明复用收益，存储容量与硬件粒度决定哪些方案可行。先前“不建模计算主体”的分析由此变成候选 Tile 的约束分析。

### 5.1 什么时候输出 Tile 倾向于方形？

先固定 $B_K$，A/B 都为 FP16，仅限制它们的本地占用：

$$
S_{AB}=2B_K(B_M+B_N).
$$

若 $B_M+B_N=D$ 固定，则：

$$
AI_{AB}=\frac{B_MB_N}{D},\qquad
B_MB_N\le(D/2)^2.
$$

因此连续的对称模型下，$B_M=B_N=D/2$ 最优。两侧均衡，让 A、B 的复用更均衡；这还没考虑 C、阵列形状和指令限制，不能推成“C 一定是方形”。

### 5.2 C 容量使输出复用与 K 工作集互相制约

FP16 输入、FP32 累加，A/B/C 共享 64 KiB，本节先不考虑双缓冲、padding 与额外保留区：

$$
\boxed{2B_MB_K+2B_KB_N+4B_MB_N\le65536}.
$$

增大 $B_M,B_N$ 提高输入复用，但 C 占用按乘积增长；增大 $B_K$ 则增加 A/B 工作集，却不直接增加上述输入 AI。

固定 $B_K=32$：

| $B_M\times B_N\times B_K$ | A/B/C 总容量 | A/B 输入 AI | 64 KiB 能否容纳 |
| --- | --- | --- | --- |
| $32\times32\times32$ | 8 KiB | 16 | 能 |
| $64\times64\times32$ | 24 KiB | 32 | 能 |
| $96\times96\times32$ | 48 KiB | 48 | 能 |
| $128\times128\times32$ | 80 KiB | 64 | 不能 |

这里方形输出边长为 $b$，FP16 输入 AI 为 $b/2$；容量中还包含 FP32 的 C 累加空间。

### 5.3 有缓冲份数时，先算 $B_K$ 的可行上限

设 A/B 缓冲份数为 $d_A,d_B$，C 为单份累加缓冲，则：

$$
\boxed{d_As_AB_MB_K+d_Bs_BB_KB_N+s_{\mathrm{acc}}B_MB_N
\le S_{\mathrm{local}}}.
$$

固定输出块，得到：

$$
B_K\le\frac{S_{\mathrm{local}}-s_{\mathrm{acc}}B_MB_N}
{d_As_AB_M+d_Bs_BB_N}.
$$

例如 64 KiB、FP16 A/B 双缓冲、FP32 C 单缓冲：

- $B_M=B_N=64$ 时：$512B_K+16384\le65536$，故 $B_K\le96$。
- $B_M=B_N=96$ 时：$768B_K+36864\le65536$，故 $B_K\le37\tfrac13$。

若合法 K 粒度为 16，对齐后的最大 $B_K$ 分别为 96、32。增大输出块挤占了可用于 K 工作集的容量。

双缓冲在这里仅作为容量份数计入。实际还需扣除栈、对齐和布局填充等空间。若 accumulator 独立，则分别约束：

$$
d_As_AB_MB_K+d_Bs_BB_KB_N\le S_{\mathrm{SRAM}},
\qquad s_{\mathrm{acc}}B_MB_N\le S_{\mathrm{ACC}}.
$$

### 5.4 Matrix Engine 固定粒度怎样适配？

> [!question] 一个软件 Tile 比矩阵指令大，继续细分会不会破坏原来的 AI？

不一定。若一次指令支持 $m_0\times k_0$ 乘 $k_0\times n_0$，常见的规则分块要求：

$$
B_M=u m_0,\qquad B_N=v n_0,\qquad B_K=w k_0,
\quad u,v,w\in\mathbb{Z}_{>0}.
$$

例如指令块为 $16\times16\times16$，软件块为 $64\times64\times32$，需要 $4\times4\times2=32$ 个指令块更新。若输入在本地保存并复用，细分并不要求从 GM 再读 32 份；但 Local–寄存器流量取决于这些小块的实际执行顺序与驻留。

不能整除时需要 padding、mask 或尾部路径。例如 $B_K=24$ 映射到只能完整处理 16 长度的归约块时，若采用补零方案，实际执行长度可能为 32。应区分有用的 $2MNK$ 工作量与填充产生的无效运算，并根据实际传输方式统计字节数。

上层搬运块也不必等于单核 K 块。例如上层缓存 $K_L=224$，单核每次处理 $B_K=56$，可以先搬入上层，再在本地拆成四段；只要上层块未被提前释放，就不必为这四段反复读取同一份 GM 数据。这里的数值用于说明分层关系，不代表任意指令集都支持这一粒度。

### 5.5 那是不是 $B_K$ 越小越好？

仅看理想输入 AI，缩小 $B_K$ 不直接降低该比值，还能释放容量给更大的输出块。但实际候选仍受到以下因素约束：

- **最小粒度**：若硬件要求 $k_0=32$，$B_K=8$ 可能需要填充或特殊路径。
- **迭代次数**：K 循环次数为 $\lceil K/B_K\rceil$；块越小，地址生成、任务准备与同步次数可能越多。
- **驻留与缓冲**：更大的 $B_K$ 增加输入容量，更小的块则改变数据生命周期及缓冲组织。

因此，“公式中消掉 $B_K$”只能说明该流量模型的比值对它不敏感，不能确定唯一最佳值。执行效率与时间模型属于进一步的分析。

### 5.6 阵列加入后，应该让哪一级方形？

阵列输入 AI 为 $H_MH_N/(H_M+H_N)$。若固定阵列输出面积 $H_MH_N$，连续、无边界和粒度限制时，应使 $H_M+H_N$ 最小，即：

$$
\boxed{H_M=H_N\quad\Longleftrightarrow\quad P_MB_M=P_NB_N}.
$$

例如 $P_M=2,P_N=4$ 时，单核取 $B_M=2B_N$，阵列整体才覆盖方形。单核方形与阵列方形不是同一个条件。

扩大 $B_N$ 是在单核内扩大 A 的复用；扩大 $P_N$ 是让 A 服务更多核心。它们在外层流量分母 $P_NB_N$ 中呈现相似效果，但前者消耗更多每核存储，后者需要更多计算位置与分发能力。B 的复用对 $B_M,P_M$ 同理。

本节固定的是阵列输出面积，5.1 固定的是 A/B 空间与 $B_K$，不能把不同约束下的方形结论混用。实际需结合每核容量、指令粒度、问题尺寸、阵列形状与输入共享能力共同筛选。

## 6. 把 Tiling 分析串起来

一次完整分析可以按以下顺序进行：

1. 写清 $M,N,K$、数据宽度及 $C=AB$ 或 $C\leftarrow AB+C$ 的语义。
2. 指定存储边界，明确统计输入、部分和与输出中的哪些流量。
3. 选择驻留与复用区域；用循环和数据生命周期检查 A/B/C 到底搬运几次。
4. 引入指令块、单核块、阵列块和上层搬运块，区分源端共享与目的端副本。
5. 用容量、缓冲份数、对齐和尾部损失筛选合法尺寸，再比较有用计算量与实际字节数。

Tiling 决定复用区域和工作集，循环顺序与驻留决定复用能否实现，空间映射决定哪些计算位置共享数据。它们共同决定各级 AI；只有这些条件，还不足以判定唯一的性能最优 Tile。

本文讨论到数据复用、通信量、容量和粒度。阵列内部如何计算、操作数如何在时间上对齐，以及计算与访存的时间模型，留待后续阵列计算部分。

相关笔记：[[Tiling Proof]]、[[Loop Ordering]]、[[Matrix Layout]]、[[Matrix Engine]]、[[Broadcast Core Array 与 Systolic Array]]。

整理依据：[原始知识网络讨论](https://chatgpt.com/c/6ad71a38-abc4-83ea-baaa-b37d3f6585e2)、[分层 Tiling 与数据驻留讨论](https://chatgpt.com/share/6af6a1df-fb0c-83ea-8d69-18d14cf01fa9)、[PE/Core 阵列与 Tile 选择讨论](https://chatgpt.com/share/6af6a245-6df0-83ea-a202-7461dd404968)。
