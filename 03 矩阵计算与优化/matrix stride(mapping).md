a mapping of :
logical position (r, c) to physical location



$$\boxed{offset = r\cdot stride_r + c\cdot stride_c}$$
actually:
Row-major:
$$
stride_r = Cols
$$
$$
stride_c = 1
$$

Col-major:
$$
	stride_r = 1  
	;stride_c = Rows
$$




$$addr(r,c) = ptr+(r\cdot ld+c)$$



我们可以有一些not valid的废数据：
例如：
```
a00, a01, a02, a03, not care, not care
a10, a11, a12, a13, not care, not care

```

我们可以设置$stride_r = 6, stride_c = 1$


>[!question]
>AI chip的Memory Controller实际上可以用stride + fractal参数来描述！这样可以省去握手

