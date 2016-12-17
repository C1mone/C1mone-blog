---
title: 在 github pages 上使用 custom domain  
date: 2016-12-17 14:30:08
tags:
- githubpages
- customdomain
- gandi
---

**github pages** 在之前有提到是一個很方便讓工程師來寫部落格的好方法，但是它的 domain 被限制在 **username.github.io**，如果我們想要使用自己的 domain 呢？這當然是可以的！github 就有一篇 [指引](https://help.github.com/articles/using-a-custom-domain-with-github-pages/) 在說明這如何完成這件工作．而我自己之前一直想要買個 domain 加在 blog 上面，但最近都滿（ㄓㄨㄤ）忙，所以就一直拖．但今天終於來把它搞定啦．這中間找了許多方法一一介紹流程在底下．

<!--more-->

## 申請自定義 domain 

我是在 [Gandi](https://www.gandi.net) 上面申請的，最近 `.com.tw` 這個 domain 在特價，每年只要 88 塊台幣，實在是有夠俗的，如果想要的話趕快趁這個時候申請吧！在簡單申請完 domain 之後 ![](http://imgur.com/Cw4JnGk.jpg)

右下角就會有一個 zone files 的編輯選項，給他大力地點進去就會看到

![](http://imgur.com/ZmjhX5G.jpg)

可以看到這是 gandi 的初始設定檔，在這裡我們需要做兩件事情

1. 把 type A 的 設定綁到 GitHub server 上

   GitHub 的 ip 是 192.30.252.153, 192.30.252.154 所以要像下圖一樣增加兩個 record

   ![](http://imgur.com/dZWpXgY.jpg)

2. 設定自己的 CNAME 

   在我的計畫裡面是想要設定 `blog.c1mone.com.tw` 這個 sub-domain 到 GitHub pages 去，因此我需要額外再加入以下的設定．

   ![](http://imgur.com/5HtZHMC.jpg)

   最後的成果如下，其實一些不相關的設定應該要把它刪掉．

   ![](http://imgur.com/M528vGI.jpg)

## 加入 CNAME 到 Hexo

當在 gandi 上面設定完以後還有一個步驟要做要在 github pages 裡面加入 CNAME ，據 GitHub 的說明文件顯示這樣會加速轉頁面的速度．而因為我是用 travis build 生成頁面的，因此需要安裝一個 plugin 來在產生 pages 的時候自動產生 CNAME 如下．

```shell
yarn add hexo-generator-cname
```

而他自動產生的方法可能就是從 _config.yml 裡面抓 url 出來

最後就 ok 啦！輸入 `blog.c1mone.com.tw` 就會轉到 GitHub pages 了．

```shell
dig example.com +nostats +nocomments +nocmd
; <<>> DiG 9.8.3-P1 <<>> blog.c1mone.com.tw +nostats +nocomments +nocmd
;; global options: +cmd
;blog.c1mone.com.tw.		IN	A
blog.c1mone.com.tw.	10800	IN	CNAME	c1mone.github.io.
c1mone.github.io.	3600	IN	CNAME	github.map.fastly.net.
```



### 相關連結

[Trouble Shooting custom domain](https://help.github.com/articles/troubleshooting-custom-domains/)

[設定 custom domain](https://rck.ms/jekyll-github-pages-custom-domain-gandi-https-ssl-cloudflare/)

[設定 CNAME](http://blog.dj1020.net/Hexo-%E7%99%BC%E4%BD%88%E5%88%B0-GitHub-%E7%9A%84%E6%96%B9%E5%BC%8F%EF%BC%8C%E8%A8%AD-CNAME/)