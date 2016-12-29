---
title: Tensorflow day12 儲存以及載入模型參數
date: 2016-12-27 22:02:07
tags:
- tensorflow
- deeplearning
- ithome鐵人

---

## 今日目標

- 了解如何儲存訓練好的模型參數
- 儲存模型範例
- 載入模型範例

<!--more-->

[Github Ipython Notebook 完整連結](https://github.com/c1mone/Tensorflow-101/blob/master/notebooks/3_Save_Restore_Model.ipynb)


還記得當我們在定義模型的時候，會使用 [Variables](https://www.tensorflow.org/api_docs/python/state_ops) 來儲存模型參數，而在訓練的時候則會一次又一次的更新它．這些參數在這些過程中都是儲存在記憶體裡．不過我們仍然需要在訓練過後把訓練完成的參數儲存在硬碟裡面，以便之後來使用或者對模型對進一步分析．

在 Tensorflow 裡面最簡單的方法是使用 `tf.train.Saver` 這個物件 (object) 來儲存模型，在初始化這個物件的時候，裡面包含了計算圖 (graph) 中對於變量 (variables) 的許多種操作方法 (ops) ．而物件中又提供了許多方法 (method) 來執行這些操作方法 (ops)．它可以幫助我們簡單的把變量 (variables) 儲存成檔案，以提供之後使用．以下先附上兩個 class 的 api 連結，可以在裡面看到更為完整的說明．

* [tf.Variable](https://www.tensorflow.org/api_docs/python/state_ops/variables#Variable) class
* [tf.train.Saver](https://www.tensorflow.org/api_docs/python/state_ops/saving_and_restoring_variables#Saver) class

## 儲存模型

### Checkpoint 檔案

變量 (Variables) 是被以 binary 的方式儲存成一個 checkpoint 檔 (.ckpt)，簡單的說它儲存了變量 (variable) 的名字和對應的張量 (tensor) 數值．

當你建立了一個 Saver 物件以後，是可以選擇每個 variable 要用什麼名字儲存在 checkpoint 檔裡的．預設上則是會使用 [Variable.name](https://www.tensorflow.org/api_docs/python/state_ops/variables#Variable.name) 的值．

為了要來了解到底儲存了什麼 variable 在 checkpoint 檔裡，你可以使用 [inpect_checkpoint](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/inspect_checkpoint.py) 中的 `print_tensors_in_checkpoint_file` 函數來檢測．

### 儲存變量 (Variables)

以下是一個利用 `tf.train.Saver()` 來建立物件來儲存變量 (variables) 的範例．


```python
import tensorflow as tf
import os
os.makedirs("/tmp/model")
os.makedirs("/tmp/model-subset")
```


```python
# 建立一些變數以及對應的名字．

v1 = tf.Variable([0.1, 0.1], name="v1")
v2 = tf.Variable([0.2, 0.2], name="v2")

# 建立所有 variables 的初始化 ops
init_op = tf.global_variables_initializer()

# 建立 saver 物件
saver = tf.train.Saver()

with tf.Session() as sess:
    
    # 執行初始化
    sess.run(init_op)
    
    #重新指定 v2 的值
    ops = tf.assign(v2, [0.3, 0.3])
    sess.run(ops)
    
    print sess.run(tf.global_variables())
    # ... 中間略去許多模型定義以及訓練，例如可以是 MNIST 的定義以及訓練
    
    save_path = saver.save(sess, "/tmp/model/model.ckpt") # 儲存模型到 /tmp/model.ckpt
```

```shell
[array([ 0.1,  0.1], dtype=float32), array([ 0.30000001,  0.30000001], dtype=float32)]
```

```python
# 使用 inspect_checkpoint.py tool 來印出 ckpt 檔
!python /usr/local/lib/python2.7/dist-packages/tensorflow/python/tools/inspect_checkpoint.py --file_name=/tmp/model/model.ckpt
```

```shell
v1 (DT_FLOAT) [2]
v2 (DT_FLOAT) [2]
```



### 儲存特定變量 (Variables)

如果我們沒有指定特定的參數給 `tf.train.Saver()`，它會自動把計算圖裡的所有變量都儲存進去．

但有時候在儲存的時候我們會想要指定不一樣的名字，例如建立了一個變量叫做 "weights" 但是在儲存的時候我們會希望他的名字叫做 "params"

也有些時候我們只想儲存部分的變量．例如訓練了一個 5 層的神經網路，但是我們只需要前 4 層的變量．

我們可以很簡單地在 `Python dictionary` 裡指定變量以及對應的名字來給 `tf.train.Saver()` 當參數，就可以完成上面的目標了．範例如下

註: 
- 你可以建立許多個 `saver` 物件，並且在這些物件之間，儲存或者載入部分的模型變量．同樣的變量會在不同的 `saver` 物件中存在，而且只有在執行 `restore()` 的時候才會被改變．
- 如果你只在一個 session 裡載入了一個模型中一部分的變量，你必須要先用 `tf.global_variables_initializer()` 的方法先把其他的沒有載入的變量先初始化起來．


```python
tf.reset_default_graph()
v1 = tf.Variable([0.1, 0.1], name="v1")
v2 = tf.Variable([0.4, 0.4], name="v2")
init_ops = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_ops)
    saver = tf.train.Saver({"my_v2": v2})
    save_path = saver.save(sess, "/tmp/model-subset/model.ckpt")
```


```python
# 使用 inspect_checkpoint.py tool 來印出 ckpt 檔
!python /usr/local/lib/python2.7/dist-packages/tensorflow/python/tools/inspect_checkpoint.py --file_name=/tmp/model-subset/model.ckpt
```

    my_v2 (DT_FLOAT) [2]

## 載入模型

### 1. 預先定義好模型
第一種方法來載入模型的時候，你必須先定義好原本的計算圖 (模型)，例如這裡我們先定義了 `v1` 和 `v2` 然後用 `sess.restore` 的方法載入之前被更改過的值．
同樣的 Saver 物件也是可以拿來載入變量 (variables)．值得注意的是當要從檔案中載入變量 (variables) 的時候是不用先把它初始化的．

```python
tf.reset_default_graph()

v1 = tf.Variable(tf.constant(0.1, shape = [2]), name="v1")
v2 = tf.Variable(tf.constant(0.2, shape = [2]), name="v2")

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "/tmp/model/model.ckpt")
    print("Model restored.")
    print("all values %s" % sess.run(tf.global_variables()))
    print("v2 value : %s" % sess.run(v2))
```

    Model restored.
    all values [array([ 0.1,  0.1], dtype=float32), array([ 0.30000001,  0.30000001], dtype=float32)]
    v2 value : [ 0.30000001  0.30000001]


### 2. 載入 meta

另一種方法則是在 `0.11.0RC1` 之後可以用載入 `meta` 的方式，就不用預先定義計算圖了．但是需要從重新拿取每一個參數的名字，以下就是範例．


```python
tf.reset_default_graph()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('/tmp/model-subset/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('/tmp/model-subset/'))
    sess.run(tf.global_variables_initializer())
    all_vars = tf.trainable_variables()
    for v in all_vars:
        print("%s with value %s" % (v.name, sess.run(v)))
```

    v1:0 with value [ 0.1  0.1]
    v2:0 with value [ 0.40000001  0.40000001]

## 今日心得

今天學習到了如何儲存以及載入訓練好的模型參數．其中比較困難的點是因為 ipython 的每一個步驟可能會不小心執行很多次，而這之中如果有重複建立計算圖的情形就會出現錯誤 (上一個步驟定義了變量，下一個步驟定義了同樣名稱的變量)．這需要在每個單獨步驟前面加入 `tf.reset_default_graph()` 來確保每次都是重新使用新的 `graph`．

不過因為如此讓我對 tensorflow 建立計算圖的流程更為清楚

1. 建立計算圖
2. 建立 session
3. 執行 ops (訓練或是讀取 tensor 數值)

## 學習資源連結

- [[tensorflow 官方文件] variables](https://www.tensorflow.org/how_tos/variables/)
- [[stackoverflow] How to restore model](http://stackoverflow.com/questions/33759623/tensorflow-how-to-restore-a-previously-saved-model-python)



