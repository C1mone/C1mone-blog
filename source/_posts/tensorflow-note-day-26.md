---
title: Tensorflow Day26 LSTM 
date: 2017-01-11 16:07:55
tags:
- tensorflow
- deeplearning
- ithome鐵人
---

## 今日目標

- 了解 LSTM 內部結構

<!--more-->

## 介紹

之前提到了 LSTM 可以有效的解決 gradient vanishing 的問題，那到底其中的結構有什麼魔法呢？讓我們從以下的內部結構圖來看一下．

### Cell State

LSTM 的第一個定義是 cell 裡面會存著狀態 (state) : C(t) ，而 C(t) 會跟上個時間點 C(t-1) 有關，當然在時間點 t 會對其做些修改，然後再繼續傳遞給下一個時間點 t+1．

![lstm_inner1](https://github.com/c1mone/Tensorflow-101/blob/master/notebooks/images/11_Char_RNN/lstm_inner1.jpg?raw=true)



### Forget Gate

第二個特性是其中會有一個 forget gate 來決定上一個時刻的 C(t-1) 中的內容哪些要保留哪些要留下來，而這是由 H(t-1) 以及 X(t) 經過一個 sigmoid 所決定的，其輸出的內容是一個 0 到 1 的數，0 代表完全捨棄；1 代表完全留下．而這個結果會乘上 C(t-1) 來決定過去的狀態有哪些要被留下，以及留下的程度．



$$ \mathbf{f_{t}} = \sigma(\mathbf{W_{xf}} \mathbf{X_{t}} + \mathbf{W_{hf}H_{t-1}} + \mathbf{b_{f}}) $$



![lstm_inner2](https://github.com/c1mone/Tensorflow-101/blob/master/notebooks/images/11_Char_RNN/lstm_inner2.jpg?raw=true)



### Input Gate

接著要決定的事情是在 cell 狀態裡面要加入哪些新的資訊，其中新的資訊由 X(t) 以及 H(t-1) 經過 tanh 產生，而資訊的強弱程度則由 X(t) 以及 H(t-1) 經過 sigmoid 來決定．

$$ \mathbf{i_{t}}=\sigma(\mathbf{W_{xi}X_{t}} + \mathbf{W_{hi}X_{i}} + \mathbf{b_{i}})*tanh(\mathbf{W_{xc}X_{t}} + \mathbf{W_{hc}X_{i}} + \mathbf{b_{c}})$$

最後新的 cell 狀態 C(t) 會由舊狀態 (C(t-1)) 和其遺忘的程度 (f(t)) 加上新訊息 (i(t)) 所決定．

$$\mathbf{C_{t}} = \mathbf{C_{t-1} * f_{t}} + \mathbf{i_t} $$



![lstm_inner3](https://github.com/c1mone/Tensorflow-101/blob/master/notebooks/images/11_Char_RNN/lstm_inner3.jpg?raw=true)

### Output Gate

最後還需要來決定輸出 H(t)，其會由 cell 的狀態加上一些操作來決定．首先它會把 C(t) 乘上一個 tanh 來把它的數值轉換到 -1 以及 1 之間，之後乘上 H(t-1) 以及 X(t) 經過 sigmoid 的值，以此決定輸出數值的強弱程度．到此就完成了一個標準的 LSTM．

$$\mathbf{H_{t}} = tanh(\mathbf{C_{t}}) * \sigma(\mathbf{W_{xo}X_{t} + W_{ho}H_{t-1}} + b_{o})$$

![lstm_inner4](https://github.com/c1mone/Tensorflow-101/blob/master/notebooks/images/11_Char_RNN/lstm_inner4.jpg?raw=true)



## 小結

經由圖解以及數學式了解了一個基本的 LSTM 內部基本結構，

### 問題

嘗試使用基本的 tensorflow op 來實現此 LSTM 結構．

## 學習資源連結

- [Colah LSTM blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)