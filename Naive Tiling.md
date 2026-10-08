```cpp
// C[M, N] = A[M, K] @ B[K, N]

void gemm_naive_tiling(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int BM, int BN, int BK
) {
    for (int m0 = 0; m0 < M; m0 += BM) {
        for (int n0 = 0; n0 < N; n0 += BN) {

            // 一个 C tile: [BM, BN]
            for (int i = m0; i < min(m0 + BM, M); ++i) {
                for (int j = n0; j < min(n0 + BN, N); ++j) {
                    C[i * N + j] = 0.0f;
                }
            }

            // 沿 reduction dimension K 分块
            for (int k0 = 0; k0 < K; k0 += BK) {

                // A tile: [BM, BK]
                // B tile: [BK, BN]
                // C tile: [BM, BN]

                for (int i = m0; i < min(m0 + BM, M); ++i) {
                    for (int j = n0; j < min(n0 + BN, N); ++j) {

                        float acc = C[i * N + j];

                        for (int k = k0; k < min(k0 + BK, K); ++k) {
                            acc += A[i * K + k] * B[k * N + j];
                        }

                        C[i * N + j] = acc;
                    }
                }
            }
        }
    }
}


```

```
for each C tile:

    C_tile = 0

    for each K tile:

        A_tile = A[m0:m0+BM, k0:k0+BK]
        B_tile = B[k0:k0+BK, n0:n0+BN]

        C_tile += A_tile @ B_tile
```