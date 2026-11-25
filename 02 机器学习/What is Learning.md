Learning 最核心的就是Generalization泛化
无关于数据集类型。我们学习的是一种model，针对某些已知的数据集进行维护


通常而言，学习有三种形式：


1. [[supervised learning]]
2. [[unsupervised learning]]
3. [[RL]]



- supervised: dataset + label。我们有若干数量的$(x_i, y_i)$希望预测对于任意x的y是什么
- unsupervised: no label。最常见的就是clustering。但是不知道他实际意味着什么。例如最大似然，给了一堆猫的图片，我们维护自己原有的假设模型（例如高斯分布）使得数据集的概率最大。然后我们可以对任意一个输入数据预测一个p
- RL：非固定的dataset，我们有feedback



注意：学习可能会有一个先验的假设model：例如
supervised:可能假设是一个linear model or quadric model or fourier。
unsupervised: 可能有最大似然假设，有gaussian layout的假设




