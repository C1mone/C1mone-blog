---
title: FP In Scala week2 assignment
date: 2016-11-21 10:37:31
tags: 
- scala
- coursera
- functional programming
---

# Assignment：Functional Sets

week2 的作業花了我許多的時間在想解法呀！要從以前的觀念體系到函數的體系不是想像中簡單的事情，不過最大的錯誤在於沒有把題目好好看清楚，所以在一些地方都卡住了，以下篇幅就紀錄我的思考歷程．
<!--more-->

## forall 

在實現 `forall` 這個函數的時候，題目就給了充足的 hint：（嗚嗚我不該沒仔細看題目直接看程式的．．．對不起 odersky）**在 bound 之內把每一個 integer 都 iterate 一遍，測試看看這個 integer 有沒有在 set 裡面** ，所以應該可以寫出如下的程式碼：

```scala
def forall(s: Set, p: Int => Boolean): Boolean = {
  def iter(a: Int): Boolean = {
    if (s(a)) p(a) && iter(a - 1) //如果存在 s 內，並且符合 p 的話就繼續搜尋下一個 integer；如果不符合 p 的話就會回傳 false
    else if (a < -bound) true
    else iter(a - 1)
  }
  iter(bound)
}
```

不過後來偷看了一下網路上其他人的解答，覺得我第一個判斷式有點醜改寫成以下會比較好看

```scala
if (s(a) && !p(a)) false
```

## exists

一樣題目也有給了 hint 來利用 forall 來實現 exists，不過 forall 和 exists 兩個定義是不一樣的．

* forall：這個 set 中所有的 element 都符合條件 p


* exists：這個 set 中至少有一個 element 符合條件 p

因此我們要把 forall 做小加工，關鍵在**如果 forall 都符合 !p ，那他的反面就是至少有一個 符合 p !**

不會是像我寫的第一版 `forall(s, p(x))`．．．正確請看以下

```scala
  /**
   * Returns whether there exists a bounded integer within `s`
   * that satisfies `p`.
   */
    def exists(s: Set, p: Int => Boolean): Boolean = {
      !forall(s, {x: Int => !p(x)})
    }
  
```

## map

map 應該是這次裡面最難的，偶腦子卡住了實在想不出來，就偷 google 一下大家是怎麼寫的，才發現原來要用 exists 來解．感謝友站連結教學（跪

[東學西毒部落格](http://fu-sheng-wang.blogspot.tw/2016/10/scala-8.html)

假設 `s = {1,2,3}, f = {x => 2*x}`, 則被 mapped 過的 set 應該要是 `{2,4,6}`

`map(s, f)(2) == map(s, f)(4) == map(s,f)(6) == true`

但我一開始的還是想錯了寫成了第一個版本如下

```scala

def map(s: Set, f: Int => Int): Set = {
  x: Int => exists(s, {y: Int => y == f(x)})
}
```

正確的話應該是讓外面進來的 x 測試符不符合裡面的 f(y)

```scala
def map(s: Set, f: Int => Int): Set = {
  x: Int => exists(s, {y: Int => x == f(y)})
}
```

祝大家學習愉快
