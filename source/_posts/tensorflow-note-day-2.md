---
title: tensorflow 學習筆記 Day2 Get started
date: 2016-12-17 20:53:10
tags:
- deeplearning
- tensorflow
---

工欲善其事，必先利其器．
因此我們要先來找一個好用的平台來學習 tensorflow．而在 python 資料分析上面的首選一定是 jupyter notebook．雖然從頭建起使用環境是滿快的，但是最近剛好看到一個線上的 jupyter notebooks 平台，馬上來試用看看．

<!--more-->

## Microsoft Azure Notebooks
[Microsoft Azure Notebooks](https://notebooks.azure.com/) 是 M\$ 最新開出來的平台，不過在這裡不能稱呼他為 M\$ 因為這是完全免費的！可以在上面開 notebook 來使用 jupyter ，大家馬上趕快去試試看，不知道可以免費到多久呢．

## Check version


```python
import sys
sys.version
```


    '2.7.11 |Anaconda custom (64-bit)| (default, Jun 15 2016, 15:21:30) \n[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]'

azure notebooks 的套件適用 Anaconda 安裝的，不過其中並沒有包含 tensorflow


```python
sys.version_info
```


    sys.version_info(major=2, minor=7, micro=11, releaselevel='final', serial=0)

其中我使用的 python 的版本為 **2.7.11**

## Install tensorflow by pip


```python
!pip install tensorflow
```


## tensorflow with linear regression
這裡照個 tensorflow 的 tutorial 來做一個簡單的 linear regression demo．


```python
import tensorflow as tf
import numpy as np

# 用 numpy 亂數產生 100 個點，並且
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.) 
# 等等 tensorflow 幫我們慢慢地找出 fitting 的權重值

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
        plt.plot(x_data, y_data, 'ro', label='Original data')
        plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label='Fitted line')
        plt.legend()
        plt.show()

# Learns best fit is W: [0.1], b: [0.3]
```

以下就是每 20 round 印出的 W 還有 b，可以發現越來越接近原本的設定值

```
(0, array([-0.69138807], dtype=float32), array([ 0.36239833], dtype=float32))
(20, array([-0.28371689], dtype=float32), array([ 0.53784662], dtype=float32))
(40, array([-0.1455344], dtype=float32), array([ 0.45219433], dtype=float32))
(60, array([-0.0571136], dtype=float32), array([ 0.39738676], dtype=float32))
(80, array([-0.00053454], dtype=float32), array([ 0.36231628], dtype=float32))
(100, array([ 0.03566952], dtype=float32), array([ 0.33987522], dtype=float32))
(120, array([ 0.05883594], dtype=float32), array([ 0.32551554], dtype=float32))
(140, array([ 0.07365976], dtype=float32), array([ 0.31632701], dtype=float32))
(160, array([ 0.08314531], dtype=float32), array([ 0.31044737], dtype=float32))
(180, array([ 0.08921498], dtype=float32), array([ 0.30668509], dtype=float32))
(200, array([ 0.09309884], dtype=float32), array([ 0.30427769], dtype=float32))
```

而這個是每 20 round 繪出的 gif，可以它會慢慢地 fit 到原始的資料．

![](http://i.imgur.com/tNGrsy6.gif)