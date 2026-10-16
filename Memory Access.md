[PCIe固态与DDR对比](https://chatgpt.com/c/6acdb243-28f0-83e9-b6b4-3eae350a0133)

| 符号  | 概念                  | 单位       |
| --- | ------------------- | -------- |
| $L$ | Latency，单次操作延迟      | s        |
| $B$ | Bandwidth，数据吞吐量     | B/s      |
| $S$ | Request Size，请求大小   | B        |
| $I$ | IOPS，每秒完成的 I/O 数    | ops/s    |
| $N$ | Concurrency，平均在途请求数 | requests |


## IOPS 与 Bandwidth 的关系

$$ \boxed{B=I\times S} $$

这是在每个请求都传输相同有效数据量 \(S\) 时成立的吞吐量关系。



## Little's Law：并发为什么能隐藏延迟


$$ \boxed{N=I\times L} $$


# Bandwidth-Delay Product

把两个公式结合起来：

$$ B=I S,\qquad N=I L $$

得到：

$$ \boxed{N=\frac{BL}{S}} $$

进一步：

$$ \boxed{NS=BL} $$

右边的 \(BL\) 就是 Bandwidth-Delay Product（BDP，带宽时延积）。

它告诉我们，为了在给定延迟下维持某个目标带宽，需要有多少数据保持在途。