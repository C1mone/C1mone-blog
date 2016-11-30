---
title: FP In Scala week3 note(1):class & binary tree
date: 2016-11-22 23:38:48
tags:
- scala
- coursera
- functional programming
---

# class & binary tree example

## abstract class and class

這裡介紹了 abstract class 還有普通的 class 並且用這些定義來實作二元樹，其中的一個練習就是要用 function 的方法來實作兩個樹的 union ．
<!--more-->

```scala
abstract class IntSets {
  def contains(x: Int): Boolean
  def incl(x: Int): IntSets
  def union(other: IntSets): IntSets
}

class Empty extends IntSets {
  def incl(x: Int): IntSets = new NonEmpty(x, new Empty, new Empty)
  def contains(x: Int): Boolean = false
  override def toString:String = { "." }
  def union(other: IntSets) = other
}

class NonEmpty(elem: Int, left: IntSets, right: IntSets) extends IntSets {
  def contains(x: Int): Boolean = {
    if (x < elem) left.contains(x)
    else if (x > elem) right.contains(x)
    else true
  }
  def incl(x: Int): IntSets = {
    if (x < elem) new NonEmpty(elem, left.incl(x), right)
    else if (x > elem) new NonEmpty(elem, left, right.incl(x))
    else this
  }
  override def toString:String = "{" + left.toString + elem + right.toString + "}"
  def union(other: IntSets) = {
    ((left union right) union other) incl elem
//    right union (left union (other incl elem))
  }
}
```

不得不說這個 union 的方法真的很難想，想了半天根本想不出來 QQ 只好直接看解答．看完解答真的是覺得太神了，完全想不到可以這樣解，然後想了幾個 case 仔細的 trace 下去都是沒問題的，不過其中樹的結構在跑演算法的過程中會被重新建立，沒有那麼的直覺．

另外一個方法則寫在底下的 comment 中，如果用簡單的語意來表達的話，第一個方法是把原先的樹從樹葉的部分開始和新的數結合，而第二個方法則是從樹根的地方開始，歡迎大家去 trace 看看（scala worksheet 非常適合來 trace）XD

## Dynamic Binding

在這段課程裡的意思就是 `Empty` 和 `NonEmpty` 都共同實作同一個 contains method ，但實際上會被呼叫的 method 內容是不一樣的，像兩個 method 都有實作 contains 不過內容卻不一樣．

## Object and Higher Order Function

最後有兩個問題如下

* Objects are in term of higher-order functions?
* Higher-order functions in term of objects?

第二個問題似乎比較容易回答，但第一個問題我現在想到的例子就是 week2 中用 pure function 來實現 set 的做法．除此之外在 google 上也找到一個 [部落格連結](http://lukasatkinson.de/2015/emerging-objects/) 裡面展示了如何用 function 來實作 objects．
