---
title: Day6 訓練和評估 MNIST Softmax 模型
date: 2016-12-21 23:40:05
tags:
- tensorflow
- deeplearning
- ithome鐵人
---

## 今日目標
- 了解如何在 tensorflow 中訓練模型
- 了解如何在 tensorflow 中評估模型的好壞

翻譯的 tutorial 如下

## 模型訓練

為了要訓練的我們的模型，我們必須先定義一下怎樣的模型才是好的模型．事實上在機器學習中，通常是定義一個模型怎樣算是不好的．我們把這個定義稱作成本 (cost) 或是損失 (loss)．它代表我們的模型和預期的結果間的差距．我們會嘗試要最小化這些成本，當這些成本或損失越低的時候，就代表著我們的模型越好．

有一個非常常見而且很棒的成本函數稱作 `cross-entropy`．它原先產生於通訊理論中的通訊壓縮編碼，但從博弈到機器學習等領域都有著很重要的地位．它的定義如下:

![http://ithelp.ithome.com.tw/upload/images/20161221/20103494WDaGWulA96.png](http://ithelp.ithome.com.tw/upload/images/20161221/20103494WDaGWulA96.png)

`y` 是預測的機率分佈，而 `y'` 是真實的機率分佈 (one-hot 數字向量)．概略地來說 `cross-entropy` 用來量測我們的預測和真實之間的差距．更多的探討 `cross-entropy` 有點超出這裡這份的指引的範圍，但很推薦你好好地理解[它](http://colah.github.io/posts/2015-09-Visual-Information/)．

為了實現 `cross-entropy` 我們必須先加入一個新的佔位子 (placeholder) 來放置正確的答案．


```python
y_ = tf.placeholder(tf.float32, [None, 10])
```

然後我們可以來實現 cross-entropy 函數 ![http://ithelp.ithome.com.tw/upload/images/20161221/20103494deMkJWm5Ca.png](http://ithelp.ithome.com.tw/upload/images/20161221/20103494deMkJWm5Ca.png)


```python
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
```

首先 `tf.log` 會先對每個 `y` 的元素取 log．接下來我們把每個 `y_` 中的元素乘上 `tf.log(y)` 中對應的元素．接下來使用 `tf.reduce_sum` 把第二個維度的元素加總起來，(reduction_indices=[1]，這個參數)．最後 `tf.reduce_mean` 計算出這一輪的平均值．

(在程式碼裡面我們並沒有直接使用這段程式碼，因為它是 numerically unstable．取而代之的我們使用 `tf.nn.softmax_cross_entropy_with_logits`．並把 `tf.matmul(x, W) + b` 當作函數輸入．在你自己的程式裡面請考慮使用 `tf.nn(sparse_)softmax_cross_entropy_with_logits`．

好的，現在我們已經知道我們要我們的模型做什麼了，而且 Tensorflow 也已經知道整個模型的計算流程圖了，現在就讓 Tensorflow 來幫你訓練模型吧．它可以自動的計算[反向傳遞](http://colah.github.io/posts/2015-08-Backprop/) (backpropagation algorithm)並且調整參數來讓成本 (lost) 最小化．當然的你可以自己選擇要使用哪一個調整參數的最佳化演算法．


```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

在這個例子中，我們要 Tensorflow 使用梯度下降法 [gradient descent algorithm](https://en.wikipedia.org/wiki/Gradient_descent) 來最小化 `cross_entropy`，而它的學習速率 (learning rate) 是 0.5．梯度下降法 (Gradient descent) 是一個簡單的學習方法，Tensorflow 會把每個參數往最小化 cost 的方向調整．不過 Tensorflow 同時也提供了許多[最佳化](https://www.tensorflow.org/api_docs/python/train#optimizers)的演算法：而且只要調整一行的程式碼就可以使用這些演算法了．

實際上 Tensorflow 在這裡做的事情是在你所定義的計算圖用一系列後台的計算來實現反向傳遞以及梯度下降法．最後它給你的只是一個單一簡單的函數，當運行的時候，他就會利用梯度下降法來訓練你的模型參數，不斷地減低 cost．

現在我們已經設置好我們的模型了，但在執行之前還有最後一件事情是我們要先來初始化我們所建立的變數．注意一下這時候只是定義而已還沒有真正的執行．


```python
init = tf.global_variables_initializer()
```

我們現在可以利用 `Session` 來初始化我們的參數以及啟動我們的模型了．


```python
sess = tf.Session()
sess.run(init)
```

開始訓練模型！我們會執行 1000 次的訓練


```python
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

在每一次 loop 中我們會從訓練數據中隨機抓取一批 100 筆數據，然後把這些數據去替換掉之前我們設定的站位子 (`placeholder`)來進行訓練．

使用一小部分的隨機數據稱作隨機訓練 (stochastic training)，更精確地說是隨機梯度下降．理想上我們希望用所有的數據來訓練，這樣會有更好的訓練結果，但這樣需要很大的計算消耗．所以每一次使用不同的訓練子集，這樣做可以有一樣的效果但是比較少的計算消耗．

## 評估我們的模型

我們的模型表現的如何呢？
讓我們看看我們預測的數字是否正確．`tf.argmax` 是一個特別有用的函數，它可以讓我們找到在某一維的 tensor 中找到最大的數值的索引值 (index)．例如 `tf.argmax(y, 1)` 代表著模型對於每一筆輸入認為最有可能的數字，`tf.argmax(y_, 1)` 則是代表著正確的數字．我們可以使用 `tf.equal` 來確認我們的預測是否正確．



```python
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
```

這列出了一系列的布林值．為了來看看有多少比重的預測是正確的，我們把布林值轉化成福點數然後取平均值．例如 `[True, False, True, True]` 會變成 `[1, 0, 1, 1]`平均值是 `0.75`．


```python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

終於我們可以來印出我們的測試資料執行出來的準度了．


```python
print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))
```

    0.9185


出來的結果大概是 92%

這樣的結果算是好的嗎？其實是非常差的．這是因為我們用的是非常簡單的模型．如果做一些小調整，可以得到 97% 的精準度．而最好的模型可以達到 99.7%！(更多的資訊可以看一下[一些結果](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)．

重要的是我們從這個模型學到了什麼，如果你覺得這樣的結果很令人沮喪的話看一下接下來的[教材](https://www.tensorflow.org/tutorials/mnist/pros/index)吧！學習用 Tensorflow 來建立更好的模型！

## 心得

今天把訓練模型以及評估模型的兩個部分實作以及讀了一遍並且翻譯成中文．複習了一下 cross-entropy 的概念，還有如何執行 tensorflow 的模型．覺得特別棒的一點就是 tensorflow 把許多的函數 (像這裡的 gradient desent) 都實現成只要一行指令就可以使用的函數，這真的非常的方便，不然要像當初修課的時後一樣手刻到死真的太不人道了．這中間覺得比較困惑的地方在於提到 cross_entropy 會有 numerical unstable 的問題不太清楚到底是為什麼．而自己覺得學到比較多的點是在 placeholder 的應用還有熟悉了 tensorflow 的計算圖是如何執行的．
除此之外這裡還可以展示一下 softmax 出來的機率分配結果，還記得第一個資料點對應到的數字是 7

```python
print(sess.run(y[0,:], feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))
```

印出來的結果就是
```
[  7.95254346e-06   2.58060628e-10   2.88528317e-05   1.33952976e-03   4.69748301e-07   8.23298251e-06   1.33806644e-09   9.98502851e-01   4.31090166e-06   1.07761858e-04]
```

很明顯地看到 index = 6 也就是數字 7 的值 (9.98502851e-01) 是最大的！

