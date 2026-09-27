---
aliases: [Linear Regression, 线性回归]
---

属于 [[supervised learning]]。假设 hypothesis 对参数是线性的，学习的是参数 $\theta$。

## 模型与符号

$m$ 是样本数，$n$ 是原始特征数。引入 $x_0=1$ 表示截距，令 $d=n+1$，$x,\theta\in\mathbb R^d$ 都是列向量。

$$
h_\theta(x)=\theta_0+\theta_1x_1+\cdots+\theta_nx_n=\theta^Tx.
$$

$x^{(i)}$ 是第 $i$ 个样本，$x_j^{(i)}$ 是它的第 $j$ 个特征；括号上标不是乘方。

## 损失函数与求导

采用平方误差之和的一半作为目标函数：

$$
J(\theta)=\frac12\sum_{i=1}^m\left(h_\theta(x^{(i)})-y^{(i)}\right)^2.
$$

令误差 $e^{(i)}=h_\theta(x^{(i)})-y^{(i)}$。对参数求导时，训练数据 $x,y$ 是常量。由链式法则：

$$
\begin{aligned}
\frac{\partial J}{\partial\theta_j}
&=\frac12\sum_{i=1}^m 2e^{(i)}\frac{\partial e^{(i)}}{\partial\theta_j}\\
&=\sum_{i=1}^m e^{(i)}\frac{\partial}{\partial\theta_j}
\left(\sum_{k=0}^n\theta_kx_k^{(i)}-y^{(i)}\right)\\
&=\boxed{\sum_{i=1}^m\left(h_\theta(x^{(i)})-y^{(i)}\right)x_j^{(i)}}.
\end{aligned}
$$

关键是 $\partial h_\theta(x^{(i)})/\partial\theta_j=x_j^{(i)}$。截距也适用，因为 $x_0^{(i)}=1$。

## 用 gradient descent 学习参数

算法原理见 [[gradient descent]]。把上面的偏导代入，得到 batch gradient descent：

$$
\boxed{\theta_j^{(t+1)}=\theta_j^{(t)}-\alpha\sum_{i=1}^m
\left(h_{\theta^{(t)}}(x^{(i)})-y^{(i)}\right)x_j^{(i)}}
\qquad(j=0,\ldots,n).
$$

所有分量使用同一份旧参数计算，并同时更新。也可以写成加上 $\alpha\sum_i(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}$，两者完全相同。

单个样本的 LMS / SGD 更新是：

$$
\theta\leftarrow\theta-\alpha\left(\theta^Tx^{(i)}-y^{(i)}\right)x^{(i)}.
$$

> [!note] 不要混用求和与平均
> 若定义 $J_{\mathrm{avg}}=\frac1{2m}\sum_i(e^{(i)})^2$，梯度和更新项都要多乘 $1/m$。最优解不变，但相同步长下更新幅度不同。要保持同一次更新，需令 $\alpha_{\mathrm{avg}}=m\alpha_{\mathrm{sum}}$。

## 矩阵形式：把所有样本一起计算

将每个样本的转置作为一行组成 $X\in\mathbb R^{m\times d}$，标签组成列向量 $y\in\mathbb R^m$。因此 $X\theta$ 是所有预测，$X\theta-y$ 是残差向量。

$$
J(\theta)=\frac12\|X\theta-y\|_2^2,
\qquad\boxed{\nabla_\theta J=X^T(X\theta-y)}.
$$

$$
\boxed{\theta^{(t+1)}=\theta^{(t)}-\alpha X^T(X\theta^{(t)}-y)}.
$$

矩阵推导也可以直接展开：

$$
J=\frac12\theta^TX^TX\theta-\theta^TX^Ty+\frac12y^Ty.
$$

因为 $X^TX$ 对称，用 $\nabla_\theta(\theta^TA\theta)=(A+A^T)\theta$ 得：

$$
\nabla_\theta J=X^TX\theta-X^Ty.
$$

令梯度为零得到正规方程 $X^TX\theta=X^Ty$。若 $X$ 的列线性独立，$X^TX$ 可逆，唯一最优参数为 $(X^TX)^{-1}X^Ty$；否则不能使用该逆矩阵表达式。

## 手算一次更新

两个样本：原始特征分别为 $1,2$，标签分别为 $2,3$。加入截距后：

$$
X=\begin{bmatrix}1&1\\1&2\end{bmatrix},\quad
y=\begin{bmatrix}2\\3\end{bmatrix},\quad
\theta^{(0)}=\begin{bmatrix}0\\0\end{bmatrix}.
$$

$$
e=\begin{bmatrix}-2\\-3\end{bmatrix},\quad
X^Te=\begin{bmatrix}-5\\-8\end{bmatrix}.
$$

取 $\alpha=0.1$，得到 $\theta^{(1)}=(0.5,0.8)^T$，预测为 $(1.3,2.1)^T$。损失从 $6.5$ 降到 $0.65$。
