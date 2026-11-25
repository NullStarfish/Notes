---
aliases: [Gradient Descent, 梯度下降]
---

## Gradient
$$
f:\mathbb R^n\to\mathbb R,\qquad
\nabla f(x)=\begin{bmatrix}\partial f/\partial x_1\\\vdots\\\partial f/\partial x_n\end{bmatrix}.
$$

## Update
$$
\boxed{x_{k+1}=x_k-\alpha_k\nabla f(x_k)},\qquad\alpha_k>0.
$$
$\alpha_k$：step size；所有分量同时更新。

## Steepest descent direction
Euclidean norm，$g=\nabla f(x)\ne0$。由 [[chain rule#Directional derivative|chain rule]] 与 [[Cauchy-Schwarz inequality]]：

$$
D_uf(x)=g^Tu\ge-\|g\|_2,\qquad\|u\|_2=1.
$$

$$
\boxed{\underset{\|u\|_2=1}{\arg\min}\;D_uf(x)=-\frac{g}{\|g\|_2}}.
$$

$$
f(x-\alpha g)=f(x)-\alpha\|g\|_2^2+o(\alpha),\qquad\alpha\to0^+.
$$

## Quadratic objective
$$
f(x)=\frac12x^TAx-b^Tx,\qquad A=A^T\succ0.
$$

$$
\nabla f(x)=Ax-b,\qquad x^*=A^{-1}b.
$$

$$
x_{k+1}-x^*=(I-\alpha A)(x_k-x^*).
$$

$$
0<\alpha<\frac{2}{\lambda_{\max}(A)}\quad\Rightarrow\quad x_k\to x^*.
$$

Stationary point $\nabla f(x)=0$ 不一定是 minimum。

Related: [[Gauss-Newton method]].