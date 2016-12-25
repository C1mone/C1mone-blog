---
title: tensorflow-note-day-10
date: 2016-12-25 13:44:31
tags:
- tensorflow
- deeplearning
- ithome鐵人
---

## 今日目標
* 觀察第二個卷積層輸出
* 全連結層以及 dropout 用意
* 深度是啥米

<!--more-->

## 第二卷積層輸出

前一篇中我們主要觀察了第一個卷積層的輸出以及內部結構．那我們今天要來觀察的就是第二個卷積層的作用．

還記得前一層中最後的結果是 14 x 14 x 32 的輸出，而這輸出就是要在把它餵入第二個卷積層．第二個卷積層構造跟前一個幾乎是一模一樣

* 5 x 5 的過濾器但是會產生 64 個輸出
* MaxPooling 再一次，因此會做 downsampling 一次，使得輸出為 7 x 7
* 一樣有 ReLU

好的那讓我們看一下各個位置的輸出:


```python
# 印出第二層的 weights
plot_conv_weights(W_conv2, 64)
```


![http://ithelp.ithome.com.tw/upload/images/20161225/20103494gw95tGpcSY.png](http://ithelp.ithome.com.tw/upload/images/20161225/20103494gw95tGpcSY.png)



```python
# 印出第二層經過過濾器的結果
plot_conv_layer(conv2d(h_pool1, W_conv2), mnist.test.images[0], 64)
```


![http://ithelp.ithome.com.tw/upload/images/20161225/20103494FZXZUe4a2N.png](http://ithelp.ithome.com.tw/upload/images/20161225/20103494FZXZUe4a2N.png)



```python
# 印出第二層經過 ReLU 的結果
plot_conv_layer(tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2), mnist.test.images[0], 64)
```


![http://ithelp.ithome.com.tw/upload/images/20161225/201034947uafnbmFjZ.png](http://ithelp.ithome.com.tw/upload/images/20161225/201034947uafnbmFjZ.png)



```python
# 印出第二層經過 MaxPooling 的結果
plot_conv_layer(max_pool_2x2(tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)), mnist.test.images[0], 64)
```


![http://ithelp.ithome.com.tw/upload/images/20161225/201034940Fyolc6j71.png](http://ithelp.ithome.com.tw/upload/images/20161225/201034940Fyolc6j71.png)


## 好的我實在不知道它學出了什麼特徵了！
## 但可以從白點的位置看到，第二層相比第一層會針對更小的特徵起反應．

## 全連結層 Fully Connected Layer
為什麼最後需要這一個全連結層呢？
可以想像的是，前面的層學出了很多的 **特徵** ，而這些特徵的 **組合** 可以幫助我們分辨現在這個影像輸入是哪一個數字！然後加入全連結層是一個非常簡便的方法來把這些特徵集合組合在一起然後從中做分類．

## Dropout
而在全連結層之後會接一個 dropout 函數，而它是為了來避免 overfitting 的神器，通常在訓練過程會使用（這裡的 p = 0.5）意思就是會這些神經元會隨機的被關掉，要這樣的做的原因是避免神經網路在訓練的時候防止特徵之間有合作的關係．
隨機的關掉一些節點後，原本的神經網路就被逼迫著從剩下不完整的網路來學習，而不是每次都透過特定神經元的特徵來分類．
通俗的講法就是，我們要百般刁難這個網路，讓它在各種很艱困的情形下學習，**不經一番寒徹骨，哪得梅花撲鼻香**（怎麼越來越覺得訓練網路好像在做軍事訓練一樣．．．）



# '深度'學習
看完了以上 CNN 的結構以後，那為什麼這個方法叫做**深度**學習呢？

因為呀就是大家發現如果**越多層**效果會越好!!

試了一下把第二個卷積層拿掉以後，準確率就掉到了 98.88%．

而在知名的圖片比賽 ImageNet，更可以看到這幾年贏的模型，可說是越來越深呢．．．

![](https://pic2.zhimg.com/v2-a2e264580fd9856daccf20eb15c32571_b.jpg)
(原圖連結：https://www.zhihu.com/question/43370067）

## 今日心得
今天從第二層的輸出看到了和第一層不同辨識特徵的差別，然後了解了為什麼後面還要接一個全連接層和 dropout 的原因．今天文字比較少的原因是我想要把它多加幾層結果就...