---
title: 机器学习中的Inductive Bias 
tags: Inductive Bias,Prior,机器学习
grammar_cjkRuby: true
---
test 


机器学习中的Inductive Bias介绍 
机器学习算法中，假设学习器在预测中逼近正确的结果，其中包括在训练中未出现的样本。既然是未知的状况，结果可以是任意的结果，若没有其他假设，这任务就无法解决。这种关于目标函数的必要假设就称为*归纳偏置*。

归纳偏差有点像我们的先验(prior)，但是有点不同的是归纳偏差在学习中是不会被更新的，但是先验在学习中会不断地被更新。

Algorithm | Inductive Bias
---|---
Linear Regression | The relationship between the attributes x and the output y is linear. The goal is to minimize the sum of squared errors.
Single-Unit Perceptron | Each input votes independently toward the final classification (interactions between inputs are not possible).
Neural Networks with Backpropagation | Smooth interpolation between data points.
K-Nearest Neighbors | The classification of an instance x will be most similar to the classification of other instances that are nearby in Euclidean distance.
Support Vector Machines | Distinct classes tend to be separated by wide margins.
Naive Bayes | Each input depends only on the output class or label; the inputs are independent from each other.