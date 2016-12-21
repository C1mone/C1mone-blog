---
title: Day5 實作 MNIST Softmax 模型
date: 2016-12-21 23:56:02
tags:
- tensorflow
- deeplearning
- ithome鐵人
---

## 目標
- 實作 Softmax Regressions
- 了解 tensorflow computation graph

今天翻譯的部分是 [tutorials](https://www.tensorflow.org/tutorials/mnist/beginners/) 中的 **Implementing the Regression** 段落．

<!--more-->

## 實現 Regression

我們通常會用 Numpy 這類的套件來在 Python 中更有效率地處理像是矩陣相乘這樣的數值運算，而它會把這些計算移到 Python 外面並且使用別種的程式語言以及更有效率的實現方法來完成計算．很不幸的是這樣的方法當把結果移回 Python 的時候會有 overhead 的情形．特別是程式執行在 GPUs 或者分散式系統的時候，移動資料的成本會變得非常的高．

Tensorflow 同樣的也把這些計算移到 Python 外，但是它用了一些方法來避免 overhead．它先讓我們先敘述一個交互操作的圖，然後再把所有交互計算的過程移到 Python 外面，而不是只是在 Python 外面執行單一個昂貴的操作．(這樣的方式可以在一些機器學習套件中看到)

要開始執行 tensorflow 之前先讓我們 import 它．


```python
import tensorflow as tf
```

我們用操作符號變數來描述這一些交互操作單元．讓我們來建立一個範例:


```python
x = tf.placeholder(tf.float32, [None, 784])
```

`x` 不是一個特定的數值．他是一個佔位子 (`placeholder`)，是一個先要求 Tensorflow 預先保留的數值，在真正計算的時候才把數值輸入進去．我們想要可以輸入任意數量的 MNIST 圖片，每一張圖都會先轉化成 784 維的向量．用 2-D 的浮點數 tensor 來表現它．它的形狀是 `[None, 784]`．(這裡的 `None` 意味著它第一個維度可以是任一長度的．)

我們的模型同時需要權重還有偏移值．可以把它們看作一個額外的輸入，但是 Tensorflow有定義更好的方法來表示他們: `Variable`．一個 `Variable` 就是一個在 Tensorflow 交互操作圖中可以被變更的 tensor．他可以在計算中被取用和變更．在機器學習中，通常模型參數就會用 `Variables` 來表示．


```python
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```

我們經由給予 `tf.Variable` 初始值的方式來建立 Variable: 在這個例子中我們同時建立了 `W` 和 `b` 這兩個都為 0 的 tensor．因為 `W` 和 `b` 都會在學習過程中被學習出來，所以我們並不需要特別考慮他的初始值是多少．

值得注意的是 `W` 的形狀為 [784, 10] 因為我們想要把一個 784 維的向量經由矩陣相乘後產生一個 10 維的證據 (evidence）向量來表示不同的數字．b 則是一個長度為 10 的向量，然後我們可以把他加入最後的輸出中．

我們現在可以來實現我們的模型了．它僅僅需要一行來定義它!


```python
y = tf.nn.softmax(tf.matmul(x, W) + b)
```

首先我們把 `x` 乘上 `W` 這裡的表示方式就是 `tf.matmul(x, W)` 對應了之前的矩陣相乘的數學式 **Wx**，`x` 是一個代表著多個輸入的 2D tensor．我們接下來把它加上 `b`，最後把它輸入 `tf.nn.softmax`．

到這裡，我們用了幾行程式來設定變數，然後再一行就建立起我們自己的模型了．Tensorflow 不僅僅可以讓 `softmax regression` 變得如此簡單，像是其他的數值運算，機器學習模型，甚至是物理學的模擬都可以利用 Tensorflow 來描述．而只要這些模型被定義後，它就可以在任何的設備上使用，例如你的電腦的 CPU 和 GPU，甚至是手機呢！

## 心得
今天最重要的學習收穫就是了解了為什麼之前大家提到 tensorflow 的時候都會說跟什麼圖什麼圖有關，現在才知道原來這是計算圖的意思呀！為了把完整的計算都移出 python 外面，所以用計算圖的方法來先定義流程，定義完以後才一起執行！而 placeholder 則是另外一個以前沒有的觀念，這是為了讓在建立模型的時候先預留下輸入的空間，到最後真的執行的時候再插入輸入數值．
我想之後每天發文的形式就會是我會先翻譯一段 tutorials 然後再加上自己的學習心得，這樣的內容也會比較完整!繼續加油摟!

