
###  tiling 真正利用的是矩阵乘法的 distributivity

核心公式其实是：

$$ A \begin{bmatrix} B_1&B_2 \end{bmatrix} = \begin{bmatrix} AB_1&AB_2 \end{bmatrix} $$

以及：

$$ \begin{bmatrix} A_1\\A_2 \end{bmatrix} B = \begin{bmatrix} A_1B\\A_2B \end{bmatrix} $$

还有：

$$ \begin{bmatrix} A_1&A_2 \end{bmatrix} \begin{bmatrix} B_1\\B_2 \end{bmatrix} = A_1B_1+A_2B_2 $$

第三个公式尤其重要，因为它就是 **K tiling**：

$$ \boxed{ AB=A_1B_1+A_2B_2+\cdots } $$

### block matrix multiplication 和普通 matrix multiplication 完全一致

比如：

$$ A= \begin{bmatrix} A_{00}&A_{01}\\ A_{10}&A_{11} \end{bmatrix}, \qquad B= \begin{bmatrix} B_{00}&B_{01}\\ B_{10}&B_{11} \end{bmatrix} $$

那么：

$$ AB= \begin{bmatrix} A_{00}B_{00}+A_{01}B_{10} & A_{00}B_{01}+A_{01}B_{11} \\ A_{10}B_{00}+A_{11}B_{10} & A_{10}B_{01}+A_{11}B_{11} \end{bmatrix} $$

你可以发现这和普通：

$$ c_{ij}=\sum_k a_{ik}b_{kj} $$

完全一样。
