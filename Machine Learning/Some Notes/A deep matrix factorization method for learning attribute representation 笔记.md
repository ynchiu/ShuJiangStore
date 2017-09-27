---
title: A deep matrix factorization method for learning attribute representation
tags: Deep Learning, Matrix Factorization, Representation Learning
grammar_cjkRuby: true
---

# 相关文献索引
- ==相关知识文献 #670080==：
相类似的多层矩阵分解的工作：
1. A hierarchical unsupervised growing neural network for clustering algorithms for document datasets;
2. Hierarchical clustering algorithms for document datasets.

# 对文章的疑问
1. 为什么要多层？多层对于聚类效果而言有什么理论的解释或者可视化的方法吗？



# 论点
- As Semi-NMF has a close relation to k-means clustering, Deep Semi-NMF also has a clustering interpretation according to the different laten attributes of our dataset, as demonstrated in Figure 2.   
解释为何Deep Semi-NMF拥有多个不同属性的直觉感受？

- 还有一个观点：对于k-means中，`!$\mathbf{H}$`应该是一个正交矩阵，这些正交矩阵可以看作是线性变换的基底，然后`!$\mathbf{Z}$`看作是不同类别的均值向量，但是为什么这里的`!$\mathbf{H}$`没有做成是正交的？  
文章认为，如果矩阵`!$\mathbf{H}$`不是正交的， 那么可以看作是一个软k-means聚类问题（soft clustering method）。
==但是 #4b0dfe==，我们不可否认的是，正交约束能够使学习得到的特征具有不相关性，即具有更好的降维和聚类效果。




- 降维效果的好坏能够衡量出来呢 ？也就是说如何可视化高纬的数据？  
这里拟使用t-SNE的方法
# 理论推理

# 读后问题
1. 如果要对深度非负矩阵分解进行正交约束，那么实质上也就等同于了k-Means聚类问题，那么相比聚类深度非负矩阵分解能够有的优越性是？
