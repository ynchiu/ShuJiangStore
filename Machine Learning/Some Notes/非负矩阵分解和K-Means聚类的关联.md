---
title: 非负矩阵分解和K-Means聚类的关联
tags: 非负矩阵分解,K-Means聚类
---

# 前言
很多文章中，都会把正交非负矩阵分解写成诗k-Means聚类的形式，如 [[1]]中Chris Ding的著作。虽然Chris Ding教授对它们两者之间的关系论述得已经非常清晰，但是仍然有不少文章对他们两者关系表示得比较含糊。在此，本Post对它们两者的进行比较详细的论述。

# 谈K-Means聚类
对于给定  ` !$ X \in R^{n \times m} $`，其中`!$X = [\mathbf{x}_1,\mathbf{x}_2,...,\mathbf{x}_m]$`代表有`!$m$`个样本，并且这`!$m$`个样本中，一共有`!$K$`类，每一类的样本均值`!$\mu_i$`。K-Means算法会不断地更新每一类的均值，使得第`!$i$`类的样本点和他们均值`!$\mu_i$`的平方差最小。直观点理解就是，同一类的样本都是相聚在一起的，并且使得他们的方差比较小。用公式表达如下：
```mathjax!
$$\min \sum_{i=1}^{K} \sum_{\mathbf{x}_j \in C_i} z_{ij} (\mathbf{x}_j - \mu_i )^2 = || \mathbf{X} - \mathbf{MZ}||^2 $$
```
其中：
`!$z_{ij} = 1$`，当且仅当`!$x_j \in C_i$`时，因此我们这里构成的矩阵`!$\mathbf{Z}$`为一个**正交**的矩阵，而`!$\mathbf{M}$`的每一列则为每一类的聚类中心点向量。

# 谈非负矩阵分解
对于一个非负矩阵`!$\mathbf{X}$`，其中`!$X = [\mathbf{x}_1,\mathbf{x}_2,...,\mathbf{x}_m]$`代表有`!$m$`个样本。对非负矩阵`!$\mathbf{X}$`进行分解可得：
```mathjax!
$\mathbf{x} = ||\mathbf{X} - \mathbf{ZH`}||^2$
```

[1]: http://ranger.uta.edu/~chqding/papers/NMF-SDM2005.pdf