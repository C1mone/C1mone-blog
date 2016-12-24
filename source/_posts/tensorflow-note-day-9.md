---
title: Tensorflow Day9 卷積神經網路 (CNN) 分析 (2) - Filter, ReLU, MaxPolling
date: 2016-12-24 20:49:34
tags:
- deeplearning
- tensorflow
- ithome鐵人
---

## 今日目標

* 了解過濾器 (Filter) 運作方式
* 了解 ReLU 激活函數 (Activation) 運作方式
* 了解最大池化器 MaxPooling 運作方式

## 過濾器 (Filter)

從昨天的結果中我們看到過濾器中似乎會對有同樣特徵的輸入產生反應，例如過濾器如果有紅色橫線的存在，那對應輸出圖片的紅色橫線就會特別明顯．那這段過程中過濾器和輸入圖片是如何的互動關係呢？

答案是 : **Convolution！**

避免掉了數學上的複雜我們直接用下面的 gif 來解釋，假設有一個 4 x 4 x 1 的輸入圖片，過濾器是 3 x 3 x 1，然後給定參數 stride = 1，可以看到過濾器開始從左上角往右以及往下和紅色的區域做點積 (每個對應元素相乘，最後全部相加)，得到的結果就是 2．再來我們會注意到紅色區域往右移了一格，而一格就是 stride 的值所對應的，同樣的當紅色區域到達最右邊後他會從左邊往下一格開始依序做點積，而最後的成果就是輸出矩陣了．

我們可以依此類推，如果 stride = 2 那就是紅色區域一次會移動兩格的意思．

![](http://imgur.com/8XHiO5I.gif)



但這時侯回頭看一下 conv2d 的定義會看到還有一個參數值是 padding．

```pthon
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')
```

padding 為 SAME 的意思是就是要讓輸入和輸出的大小是一樣的，因為從上面的例子看到經過 convolution 以後 size 變小了，那如果要一樣的話，我們做的事情就是把輸入一開始補上 0 如下圖，那再去跑一次 convolution 就會得到一樣大小的圖片了．

![](http://imgur.com/qch3liz.jpg)

## ReLU 激活函數

在這裡我們會把上面的輸出都通過一個函數如下，其實就是 `x < 0` 的時候全部為 0，而 `x > 0` 則為 x．要做激活函數的原因是因為要模擬出非線性函數，簡單想像就是加入激活函數可以讓神經網路學到更多奇奇怪怪的東西．

而之前小弟在學校訓練原始 nn 的時候好像是用 `sigmoid` or `tanh`，大概查了一下是因為 ReLU 更簡單 ，在深度的網路裡面計算更快．(結果小於零就認定你辨識不出來，就把你關掉！譬如說一個過濾器是來辨識眼睛，若是發現圖片完全沒有眼睛，則輸出以後就會小於零，那經過 ReLU 以後，後面的神經網路就不會知道有這個特徵了，因為這個特徵被關掉了．)．

![](http://cs231n.github.io/assets/nn1/relu.jpeg)

如下圖，就是經過 ReLU 後的結果，只剩下紅色大於零的輸出．

![](http://imgur.com/89OTxR2.jpg)

## 最大池化器 MaxPooling

好的，convolutional layer 的最後一個地方就是 max pooling 了，話說我一直不太喜歡池化器這個翻譯...因為好饒口啊．

記得前面有提到說 CNN 有個目標就是想要把參數降低，而我們學習是為了學習特徵，因此不一定要從高清（？中學習，可以把它變得沒那麼清楚一點．這裡做的事情就是 `downsampling`！那要怎麼樣既 down sampling 又保留特徵呢？以下就是一個示範：

首先我們用一個 2x2 的矩陣來掃過輸入 (stride = 2)，然後在每個紅色區域裡都找那個區域裡最大值當作輸出，就這樣就完成了 down sampling 了！

![](http://imgur.com/MxuEsSo.gif)

以下就是經過 max-pooling 的結果了！

![](http://imgur.com/403IOhD.jpg)

## 心得

呼，今天把關於 convolutional layer 我學到的東西盡量簡化描述出來，知道了 convolution ReLU 以及 maxpooling 在做些什麼事情，以及輸出是什麼．但是記得這只是第一層的結果而已，明天要來看看第二層的結果會是如何呢？

在這段自我學習的過程中，覺得最困難的事就是從數學還有敘述中理解 cnn 在做些什麼事，而最後終於大概理解以後心裡對於創造出這一切的人真的是萬分佩服．．．LeCun 和其他人實在太神啦．

如果今天以及一系列我的筆記，大家有什麼問題或是寫錯的地方，再麻煩各位大大來跟我提醒一下，感謝！祝大家聖誕快樂！

### 學習資源連結

[超猛教學 CS231n](http://cs231n.github.io/convolutional-networks/)

[Convolution 究極理解](http://colah.github.io/posts/2014-07-Understanding-Convolutions/)