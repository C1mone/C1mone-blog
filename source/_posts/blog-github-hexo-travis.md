---
title: 使用 travis-ci 自動寫 hexo blog 到 github-pages
date: 2016-11-30 16:40:43
tags:
- Travis
- CI
- CD
- hexo
- github
- blog
---

使用 github pages, hexo 以及漂亮簡單的 theme Next 就可以很快地寫文章成部落格．但是每次在寫文章的時候，還要自己先編譯成靜態網頁以後部署上去，原本的文章卻沒有被 git 記錄起來，這讓我一直覺得很阿雜，剛好今天早上有點空就來研究一下自動化的方法．
<!--more-->

## 目標流程！

這裡先介紹一下最後會完成的流程讓大家先有一個感覺

1. 在 local 寫廢文
2. commit 到遠端的 master pages
3. travis 發現 master pages 有異動就把他抓下來自動產生靜態網頁！
4. travis 把產生完成的靜態網頁 push 到遠端的 gh-pages branch 
5. 登登登！就可以完成部落格文章更新了！

## 建立 hexo 以及使用 Next theme

這段大家可以參考 [Hexo 官網](https://hexo.io) [Next 官網](http://theme-next.iissnan.com) ，就可以先在本地建立起網站！

```shell
npm install hexo-cli -g
hexo init blog
cd blog
git clone https://github.com/iissnan/hexo-theme-next themes/next
rm themes/next/.gitignore
rm -r themes/next/.git
```

這裡把 .gitignore 還有 .git 刪掉的原因是要把 Next 的 code 還有自己的設定都 commit 進入 git，也可以使用 gitmodule 的做法，但是這樣就會只能使用 default 的設定．

## 申請 Github **Personal Access Tokens**

因為要讓 travis 有權限來使用使用者的 github，搜尋網路上的做法大都是是申請 github personal access token 來限定使用範圍（總不能給他完整的 ssh key 吧 XD）．大家到 github 的 settings 裡面就可以看到申請頁面如下，特別注意這裡一定要把 token 記下來，因為下一頁以後他就會消失摟．

![Github Personal Acces Tokens](http://imgur.com/m15up3A.jpg)

## 加密 Github Personal Access Tokens

當拿到 token 以後，有這個 token 的 service 就可以對你的 repository 做存取，而且是全部的 repository 喔！所以這個東西一定要謹慎使用．這裡我是使用 [Encrypted Variables](https://docs.travis-ci.com/user/environment-variables/#Encrypted-Variables) 把它加密以後寫在 travis 的設定檔 .travis.yml 裡面，讓最後 travis 有權限在最後 push code 到 gh-pages branch

```shell
# 安裝 travis 命令列
gem install travis

# 加密 Personal Access Token
# XXX 就是你的 Github Peronsal Access Token
travis encrypt -r C1mone/C1mone-blog GITHUB_TOKEN=XXX
# 這裡會得到 secure: "..." 一長串東西，要複製下來

travis login --github-token GITHUB_TOKEN
travis whoami
# 測試一下 token 有沒有效
```

## Travis 設定

接下來就是到 travis 頁面把 github repository 做一個連接摟！

![travis](http://imgur.com/qtMVx7F.jpg)

## Travis yml 檔

最後最重要的步驟就是在 repository 裡面寫一個 `.travis.yml` 檔案了，這裡獻醜附上我的 yml 檔案如下，當然也會解說一下．

* cache：這裡把 node_modules 做 cache 可以加快建構的速度
* before_script：我們會先把 gh-pages 的內容先 clone 下來到 public ，這麼做的原因是讓之前的 commit 紀錄不要消失．
* script：讓 hexo 產生靜態網頁檔案到 public 資料夾去．
* after_success：最後一段就是把 public 中的內容經由 github access token 的方式 commit 到遠端的 gh-pages 去．
* branches：這裡是只監聽 master branch 的錯誤．
* env： secure 指的是上面經由 travis cli 所產生出來的加密一長串 token．

```yaml
language: node_js
node_js: stable

cache:
    directories:
        - node_modules

before_script:
    - git clone --branch gh-pages https://github.com/C1mone/C1mone-blog.git public

script:
    - yarn install
    - yarn hexo generate

after_success:
    - cd public
    - git config user.name "C1mone"
    - git config user.email "c1mone.tsai@gmail.com"
    - git add --all .
    - git commit -m "Travis CI Auto Builder Update"
    - git push --quiet "https://$GITHUB_TOKEN@$GITHUB_REF" gh-pages:gh-pages

branches:
    only:
        - master
env:
    global:
        - GITHUB_REF: github.com/C1mone/C1mone-blog.git
        - secure: 

```

## 結論

最後就是小弟我現在寫 blog 可以直接 commit 完就不用管他啦，然後沒有裝 node 的環境也可以寫，只要有 git 和 vim （誤？大家趕快來試用看看吧．