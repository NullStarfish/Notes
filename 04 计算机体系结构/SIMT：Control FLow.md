
## Control Flow



> [!quote] 
> In SIMT, all threads in the warp are executing the same kernel code, but each thread may follow different branches through the code.
> [Source]([1.2. Programming Model — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html))

>[!quote]
>All threads in the warp execute the same instruction simultaneously. If some threads within a warp follow a control flow branch in execution while others do not, the threads which do not follow the branch will be masked off while the threads which follow the branch are executed. For example, if a conditional is only true for half the threads in a warp, the other half of the warp would be masked off while the active threads execute those instructions.

>[!quote]
>When different threads in a warp follow different code paths, this is sometimes called ==warp divergence==. It follows that utilization of the GPU is maximized when threads within a warp follow the same control flow path.

```
if (condition) {
    A();
} else {
    B();
}
C();
```
>[!note]
>- **整个 warp 的线程都选择 A**：可以通过跳转跳过 B，没必要把 B 的指令逐条发射一遍。
>- **一部分选择 A，另一部分选择 B**：两条路径都需要执行。执行某条路径时，用 mask 指定参与的线程；路径之间仍然可以通过跳转转移执行位置。
>- 
>当为情况二时，和CPU有本质区别：util下降



![active-warp-lanes.png (5299×2475)](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/active-warp-lanes.png)


