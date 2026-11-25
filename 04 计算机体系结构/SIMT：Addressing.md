![figure](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-thread-blocks.png)


>Grid of Thread Blocks. Each arrow represents a thread (the number of arrows is not representative of actual number of threads

```cpp
int main()
{
    ...
    dim3 grid(16,16);
    dim3 block(8,8);
    MatAdd<<<grid, block>>>(A, B, C);
    ...
}
```


> [!quote]
> - `threadIdx` gives the index of a thread within its thread block. Each thread in a thread block will have a different index. 
>- `blockDim` gives the dimensions of the thread block, which was specified in the execution configuration of the kernel launch.
>- `blockIdx` gives the index of a thread block within the grid. Each thread block will have a different index.    
> - `gridDim` gives the dimensions of the grid, which was specified in the execution configuration when the kernel was launched.

grid和block都是dim3类型。可以用threadIdx, blockIdx来获得index

```cpp
__global__ void vecAdd(float* A, float* B, float* C)
{
   // calculate which element this thread is responsible for computing
   int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

   // Perform computation
   C[workIndex] = A[workIndex] + B[workIndex];
}

int main()
{
    ...
    // A, B, and C are vectors of 1024 elements
    vecAdd<<<4, 256>>>(A, B, C);
    ...
}
```