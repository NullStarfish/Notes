```python
for i in range(M):
    for j in range(N):
        acc = 0
        for k in range(K):
            acc += A[i, k] * B[k, j]
        C[i, j] = acc
```
>[!note]
>- 总计算量约为 $2MNK$ FLOPs（一次乘加记为 2 FLOPs）。
> - 读取 A、B 的元素总次数为 $2MNK$。
> - C 的写入次数为 $MN$。

### __A 和 B 中的元素被反复读取了很多次。__


 这是因为我们完全假设，没有local sram来存放A和B
[[NPU Memory]]
 假如有这样的buffer存在，可以很大程度减少读取次数


显然：__把A和B全部存起来是最快的方式__

但是我们需要权衡 __sram size成本__ 和 __compute速度__
这就是[[tiling]]

同时，我们也并不一定要按照i， j，k的循环方式来进行循环
例如，把j放最内侧：我们可以最大程度利用A
这就是[[loop ordering]]


- A ，B，C所在的内存层级是什么
- A，B，C的layout是什么？如何被访问？
- A和B的元素怎么参与运算？



