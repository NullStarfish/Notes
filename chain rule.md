---
aliases: [Chain Rule, 链式法则]
---

各层函数在相应点 differentiable。

## Scalar composition

$$
y=f(z),\quad z=g(t)
\quad\Rightarrow\quad
\boxed{\frac{dy}{dt}=f'(g(t))g'(t)}.
$$

$$
\frac{d}{dt}\sin(t^2)=\cos(t^2)\cdot2t.
$$

## Along a curve

$$
f:\mathbb R^n\to\mathbb R,\qquad z:\mathbb R\to\mathbb R^n.
$$

$$
\boxed{\frac{d}{dt}f(z(t))
=\sum_{i=1}^n\frac{\partial f}{\partial z_i}(z(t))\frac{dz_i}{dt}
=\nabla f(z(t))^Tz'(t)}.
$$

## Jacobian form

$$
x\in\mathbb R^n\xrightarrow{\ g\ }z\in\mathbb R^m
\xrightarrow{\ f\ }y\in\mathbb R^p.
$$

$$
(J_g)_{ij}=\frac{\partial g_i}{\partial x_j},\qquad
(J_f)_{ki}=\frac{\partial f_k}{\partial z_i}.
$$

$$
\frac{\partial(f\circ g)_k}{\partial x_j}(x)
=\sum_{i=1}^m\frac{\partial f_k}{\partial z_i}(g(x))
\frac{\partial g_i}{\partial x_j}(x).
$$

$$
\boxed{J_{f\circ g}(x)=J_f(g(x))J_g(x)},\qquad
(p\times n)=(p\times m)(m\times n).
$$

## Gradient form

Scalar output，gradient 为 column vector：

$$
\boxed{\nabla_x(f\circ g)(x)=J_g(x)^T\nabla_zf(g(x))}.
$$

$$
z=Ax-b,\quad f(z)=\frac12z^Tz
\quad\Rightarrow\quad
\nabla_xf(Ax-b)=A^T(Ax-b).
$$

## Directional derivative

固定 $x,u\in\mathbb R^n$，令 $z(t)=x+tu$：

$$
\begin{aligned}
D_uf(x)
&=\lim_{t\to0}\frac{f(x+tu)-f(x)}{t}\\
&=\left.\frac{d}{dt}f(z(t))\right|_{t=0}\\
&=\nabla f(x)^T\underbrace{z'(0)}_{u}
=\boxed{\nabla f(x)^Tu}.
\end{aligned}
$$

此等式不要求 $\|u\|_2=1$；比较 steepest descent direction 时才固定单位长度，见 [[gradient descent]]。

## Explicit dependence

$$
y(t)=F(t,z(t))
\quad\Rightarrow\quad
\boxed{\frac{dy}{dt}
=\frac{\partial F}{\partial t}(t,z(t))
+\nabla_zF(t,z(t))^Tz'(t)}.
$$

$\partial/\partial t$ 固定 $z$；$d/dt$ 包含所有路径的变化。
