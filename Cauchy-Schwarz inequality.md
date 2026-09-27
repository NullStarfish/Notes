---
aliases: [Cauchy–Schwarz inequality, Cauchy Schwarz, 柯西施瓦茨不等式]
---

## Inner product form

$$
\boxed{|\langle u,v\rangle|^2\le\langle u,u\rangle\langle v,v\rangle}
\quad\Longleftrightarrow\quad
|\langle u,v\rangle|\le\|u\|\,\|v\|.
$$

Equality $\Longleftrightarrow$ $u,v$ linearly dependent。

## Euclidean form

$$
u,v\in\mathbb R^n,\qquad
\boxed{\left(\sum_{i=1}^n u_iv_i\right)^2\le
\left(\sum_{i=1}^n u_i^2\right)\left(\sum_{i=1}^n v_i^2\right)}.
$$

## Proof (real case)

$v\ne0$：

$$
0\le\left\|u-\frac{u^Tv}{\|v\|_2^2}v\right\|_2^2
=\|u\|_2^2-\frac{(u^Tv)^2}{\|v\|_2^2}.
$$

$v=0$：两边均为 $0$。

Application: [[gradient descent#Steepest descent direction]].
