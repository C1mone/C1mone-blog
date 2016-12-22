---
title: Tensorflow Day 7 : 卷積神經網路實現 Convolutional Neural Network
date: 2016-12-22 23:12:52
tags:
- tensorflow
- deeplearning
- ithome鐵人
---

## 今日目標
* 建立卷積神經網路 (convolutional neural network)
* 用 MNIST 來訓練 CNN
* 評估模型

今天翻譯 tensorflow Guides 中的[Deep MNIST for Experts](https://www.tensorflow.org/tutorials/mnist/pros/) 如下 : 

<!--more-->

## 建立多層的卷積神經網路

前一個模型我們得到 92% 的準確率是非常不及格的．在這裡我們將建立一個更為適當且複雜的模型．也就是一個小型的 **卷積神經網路** (convolutional neural network)．在 MNIST 的準確度會提升到 99.2% 或許不是最好但也相當不錯的成績．

## 權重初始化

在建立模型之前我們需要建立一系列的權重 (weights) 還有偏移量 (biases)．為了避免 權重 (weight) 過於對稱還有 0 梯度 (gradient) 我們必須加入一小部分的為正的噪音 (noise)．因為我們使用的是 [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))，因此特別適合在偏移量 (bias) 初始化為為正的數值來避免 **'神經元死去 (dead neurons)'**．這裡建立了兩個函數來避免每次使用他的時候都重新建立程式碼．


```python
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
```

## 卷積還有池化 (Convolution and Pooling)

(在這裡實在不知道怎麼翻譯，因此用阿陸仔的名詞來稱呼)

Tensorflow 同樣給我們很大的彈性來做卷積還有池化這兩個動作．如何處理邊界? 我們的 stride 大小要設多少? 在這個範例中，我們會一直使用 vanilla 的版本．我們的卷積過程中的參數 `stride` 會是 1 而 `padded` 則是 0．也因此輸入還有輸出都會是同樣的大小 (size)．而我們的 polling 則是用 2X2 的傳統 polling 來做．為了讓我們的程式更加簡潔，我們同樣把這樣的操作抽象成函數．


```python
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
```

## 第一個卷積層

我們現在可以來實現第一個卷積層．他會先有一個卷積接著一個 max polling 來完成．這個卷積會從 5x5 的 patch 算出 32 個特徵．他的權重 tensor 的形狀是 [5, 5, 1, 32]．頭兩個維度是 patch 的大小，下一個則是輸入的 channels，最後一個則是輸出的 channels．同樣的在輸出也會有偏移量向量 (bias vector)．


```python
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
```

現在要來把輸入 `x` 來導入我們剛剛建立的第一個卷積層，那必須先把 `x` 轉換成一個 4d 的 tensor，其中第二個和第三個維度對應到了圖片的寬度和高度，而最後一個則對應到了顏色的 channel 數 (這裡因為是灰階的所以 channel 為 1)


```python
x_image = tf.reshape(x, [-1, 28, 28, 1])
```

我們接下來把 `x_image` 還有權重 tensor 輸入剛剛定義的卷積函數，再來加上偏移值 (bias) 後輸入 **ReLU** 函數，最後經過 max pooling． `max_pool_2x2` 函數會把圖片的大小縮成 14x14．


```python
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```

## 第二個卷積層

為了建立比較深度的神經網路，我們把許多層疊在一起．第二層會從 5x5 的 patch 中取出 64 個特徵．


```python
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
```

## 密集的全連接層

現在讓我們想像圖片的大小已經被縮成了 7x7 的大小，我們加入一個 1024 的全連接層來把前面的全部輸出輸入全連接層．其中包含了先把 pooling 的輸出展開後乘上一個權重矩陣再加上一個偏移量向量，最後輸入 ReLU．


```python
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
```

## Dropout

為了減少 overfitting，在輸出層之前我們會加入 dropout 層．首先我們建立了一個站位子 (`placeholder`) 來代表神經元在 dropout 過程中不變的機率．這可以讓我們決定要在訓練的時候打開 dropout，而在測試的時候關閉 dropout． Tensorflow 的 `tf.nn.dropout` 除了會遮蔽神經元的輸出以外也會自動對輸入值做 scaling，所以我們在使用的時候可以不用考慮 scale．

註: 對於這種比較小的卷積網路，有沒有 dropout 對於成果不會有太大影響．dropout 是一個非常好的方法來減低 overfitting，但僅限於比較大的神經網路．


```python
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

## 輸出層

最後我們加上像之前 softmax regression 一樣的層．


```python
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
```

## 訓練以及評估模型

那我們的模型表現的如何呢? 這裡我們會用之前 `Softmax` 範例的大部分程式碼來訓練以及評估這個模型．

不過有幾點不同的是:

* 我們會把 gradient descent 最佳化演算法換成更為精密的 ADAM 最佳化演算法．
* 我們會在 `feed_dict` 參數之中加入 `keep_prob` 這個參數來控制 dropout 的機率．
* 我們會在每一百個回合的訓練中印出紀錄

現在你可以來執行這段程式，但它需要 20,000 回合的訓練，可能會需要比較久的時間 (大概半小時)，當然如果你的處理器比較好的話也可能會比較快．


```python
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```

```
test accuracy 99.18%
```

最後的測試準確度大約為 99.2%

我們學到了用 Tensorflow 是可以很快速而且簡單的建置，訓練，以及評估一個較為精密的深度學習模型．

## 心得

今天依照 Tensorflow 官網實現了一個卷積神經網路來處理 MNIST 的問題，並且把官網中的說明翻譯成中文．其中比較困難的點在於對 CNN 只知道大略性的概念，對於細節不太熟悉，因此先看了一下這篇 [CNN 介紹](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/)(非常推薦這篇，寫得非常好!)，還有照著 Tensorflow 官網的程式碼以及說明做．
從兩個地方學到的點就是知道了 CNN 可以分為卷積層和全連結層這兩個不同的層．而通常會是數個卷積層後面接一個全連結層，例如這裡就是兩個卷積層再接上一個全連結層．而卷積層又可以分成 `convolution` 以及 `max_pooling`，之後還會再接一個 ReLU (activation 函數)．但對於這兩個層處理的參數像是 `stride` `padding` 是如何影響到 convolution 則還不是太清楚，我想這就會是明天或後天的課題來了解卷積層中發生了什麼事讓這個準確度高出這麼多!

今天文章的 [ipython 連結](https://github.com/c1mone/Tensorflow-101/blob/master/notebooks/Ch1.2_MNIST_Convolutional_Network.ipynb)