---
aliases: [Gauss–Newton method, Gauss Newton, 高斯牛顿法]
---

## Nonlinear least squares
$$
\min_{x\in\mathbb R^n}F(x)=\frac12\|r(x)\|_2^2,\qquad r:\mathbb R^n\to\mathbb R^m.
$$

Residual $r$；Jacobian $J$：
$$
J(x)=\left[\frac{\partial r_i}{\partial x_j}\right]_{m\times n},\qquad
\nabla F(x)=J(x)^Tr(x).
$$

## Linearization → Update

$$
r(x_k+p)\approx r_k+J_kp,\qquad r_k=r(x_k),\ J_k=J(x_k).
$$

$$
p_k\in\underset{p}{\arg\min}\;\frac12\|r_k+J_kp\|_2^2
\quad\Rightarrow\quad
\boxed{J_k^TJ_kp_k=-J_k^Tr_k}.
$$

$$
\boxed{x_{k+1}=x_k+p_k}.
$$

$\operatorname{rank}(J_k)=n$：

$$
p_k=-(J_k^TJ_k)^{-1}J_k^Tr_k.
$$

Rank deficient：可取 minimum-norm solution $p_k=-J_k^+r_k$（$J_k^+$：pseudoinverse）。

## Hessian approximation

$r_i\in C^2$：

$$
\nabla^2F(x)=J^TJ+\sum_{i=1}^m r_i(x)\nabla^2r_i(x)
\approx J^TJ.
$$

忽略 residual-weighted second-order term；不保证 global convergence。

## Comparison

$$
\begin{aligned}
\text{Gradient descent:}\quad&p_k=-\alpha_kJ_k^Tr_k,\\
\text{Gauss–Newton:}\quad&J_k^TJ_kp_k=-J_k^Tr_k.
\end{aligned}
$$

Related: [[gradient descent]].

Reference: [Gauss–Newton method — UCLA](https://www.seas.ucla.edu/~vandenbe/236C/lectures/gn.pdf).
