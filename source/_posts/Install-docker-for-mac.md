---
title: Upgrade docker for mac and run redmine
date: 2016-11-25 12:29:32
tags:
- docker
- redmine
---

## docker upgrade

最近看了一些文章發現自己的 docker 已經好久沒有升級了（將近快一年），不過因為後來工作都比較少在用 docker 的關係，一直都是把只要原有的 redmine container 起得來就不管他了！但是心血來潮想說來跟隨一下新潮的 docker ，因此做了一次升級的動作．
<!--more-->

不看還好，一看原來 docker 早就已經在 mac 上面有一個新的版本稱作 **docker for mac** 使用了 **[xhyve](https://github.com/mist64/xhyve/)** 的一個技術來虛擬化 docker 在 mac 上．這句話有點文鄒鄒的，但最簡單的意思就是現在在 mac 上就不用再用麻煩的 docker-machine (boot2docker) 可以直接在命令列下面執行 docker 了，不然有時候 docker machine 怪怪的還要重開就一整個麻煩．但是在安裝 docker for mac 之前還是有一些步驟要做來把 docker-machine 的東西刪乾淨．

## Uninstall docker & boot2docker

這裡我參考了以下的網址來完整的移除舊有的 docker．

[How to Fully Uninstall the Official Docker OS X Installation](https://therealmarv.com/how-to-fully-uninstall-the-offical-docker-os-x-installation/)

移除 default machine ，如果你有建立多個記得全部刪掉．

```shell
docker-machine rm -f default
```

停止和刪除 boot2docker 

```shell
boot2docker stop
boot2docker delete
```

移除 boot2docker & docker app

```shell
sudo rm -rf /Applications/boot2dockersudo
rm -rf /Applications/Docker
```

移除 /usr/local 底下的一堆指令

```shell
sudo rm -f /usr/local/bin/docker
rm -f /usr/local/bin/boot2docker 
rm -f /usr/local/bin/docker-machine 
rm -r /usr/local/bin/docker-machine-driver* 
rm -f /usr/local/bin/docker-compose
```

移除 docker package

```shell
sudo pkgutil --forget io.docker.pkg.docker
pkgutil --forget io.docker.pkg.dockercompose
pkgutil --forget io.docker.pkg.dockermachine
pkgutil --forget io.boot2dockeriso.pkg.boot2dockeriso
```

移除 boot2docker images

```shell
sudo rm -rf /usr/local/share/boot2dockerr
m -rf ~/.boot2docker
```

移除 docker ssh key

`rm ~/.ssh/id_boot2docker*`

移除 `/private` 資料夾中的 boot2docker 檔案

```shell
sudo rm -f /private/var/db/receipts/io.boot2docker.*
sudo rm -f /private/var/db/receipts/io.boot2dockeriso.*
```

移除 docker 設定資料夾

```shell
rm -rf ~/.docker
```

## Install Docker for Mac

跟著[安裝步驟](https://docs.docker.com/docker-for-mac/)應該就 ok 了，沒有遇到什麼問題

## Run Redmine docker

使用 docker-compose 的方法就可以快速建立一個 redmine docker 了！超快！

```shell
wget https://raw.githubusercontent.com/sameersbn/docker-redmine/master/docker-compose.yml
docker-compose up
```

收工示意圖

![](http://imgur.com/FY3Xmap.jpg)