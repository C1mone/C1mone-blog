---
title: FP In Scala week3 note(2):Class & Polymorphism
date: 2016-11-23 15:15:54
tags:
- scala
- coursera
- functional programming
---

# How Classes Are Organized

## Traits

Traits 就像是 Java 的 interface 只是可以擁有 **field** 以及 **concret method**

## Hierachy

以下是 Scala 中各個 class 的繼承關係，有幾點記得比較清楚的是

* 所有的 class 都擁有 subclass Nothing
* 而 Int 參數不能被指派 null ： `val x: Int = null` 會報錯

![scala hierachy](http://imgur.com/VSoN27U.jpg)

# Polymorphism

## Polymorphism （多型）

多型在 Scala 裡面有兩種實現方法

1. subtype：一個 class 可以使用它子 class 的實體就是多型，例如上一個 note 中我們指定 `val x: string = null` 就是一種多型化的例子
2. generic：一個 function 或是 class 可以接受不同種 type 的參數，也就是 type parameter

以下就是 Scala 中 List 的實現

```scala
trait List[T]{
  def isEmpty: Boolean
  def head: T
  def tail: List[T]
}

class Cons[T](val head: T, val tail: List[T] ) extends List[T]{
  def isEmpty: Boolean = false
}

class Nil[T] extends List[T]{
  def isEmpty: Boolean = true

  def head: Nothing = throw new NoSuchElementException("nil head")

  def tail: Nothing = throw new NoSuchElementException("nil tail")
}
```

其中 Cons 中使用的參數就會自動實現 trait 中的 head 以及 tail function，同意於下列程式

```scala
class Cons[T](val _head: T, val _tail: List[T]) extends List[T]{
  head = _head
  tail = _tail
}
```

課堂的練習要我們實現一個 nth function ，這不會很難實現．

```scala
val list = new Cons(1, new Cons(2, new Cons(3, new Nil)))

def singleton[T](elem: T) = new Cons[T](elem, new Nil[T])

def nth[T](n: Int, list: List[T]): T = {
  if (list.isEmpty) throw new IndexOutOfBoundsException("")
  else if (n == 0) list.head
  else nth(n - 1, list.tail)
}
nth(-1, list)
```

重複一下這裡使用 T 的 function, class, trait 都是一種 generic 的表現，而 Nil class 定義中的 `def tail: Nothing` 則是 subtype

## Type Erasure

定義：在 evaluate 程式的時候會把 type 去除
