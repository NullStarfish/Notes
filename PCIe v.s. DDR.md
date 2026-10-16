一个有意思的对比：[PCIe固态与DDR对比](https://chatgpt.com/c/6acdb243-28f0-83e9-b6b4-3eae350a0133)

|设备|理论/典型顺序带宽|
|---|---|
|DDR-400，单通道|3.2 GB/s|
|DDR2-800，单通道|6.4 GB/s|
|DDR3-1600，单通道|12.8 GB/s|
|DDR4-3200，单通道|25.6 GB/s|
|DDR5-6400，单通道|51.2 GB/s|
|高端 PCIe 5.0 ×4 SSD|约 14–15 GB/s|

|指标|DRAM|NVMe SSD|
|---|---|---|
|随机读取延迟|约 50–100 ns|通常几十 μs|
|最小访问粒度|Cache Line（通常 64B）|通常以 4KiB 块处理|
|高带宽实现|多通道、Bank 并行|NAND 并行、多个未完成 I/O|
|CPU 访问|Load/Store|NVMe 命令、DMA|

观察到pcie带宽和DRAM接近。
SSD 的高带宽主要建立在 ___大块传输___ 和 ___高并发___ 的基础上，而 DRAM 可以在较小访问粒度下提供低得多的延迟。


举个例子：

### Dependent random access:

```C
Node* p = head;
while (p != nullptr) {
    p = p->next;
}
```

区别于Random access和Sequential Acess

dependent random access特意增加了dependency让access变得真正顺序

$$ \text{Concurrency} \approx \text{Throughput} \times \text{Latency} $$

假设一块 SSD 的单次 I/O 平均延迟为 100 μs，想实现 200 万 IOPS：

$$ 2\times10^6 \times 100\times10^{-6}=200 $$

也就是说，需要大约 200 个同时未完成的 I/O 请求 来隐藏延迟。