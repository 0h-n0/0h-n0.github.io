---
layout: single
title: "Google Adsenseのads.txtをGithub pagesでホストされているjekyllのサイトへデプロイする方法"
excerpt: how to deploy your ads.txt on your site
categories:
  - TechBlog
tags:
  - Jekyll
  - Google AdSense
toc: true
toc_sticky: true
---


Google AdSenseの許可がおり、自分のサイトに広告を表示したいとき、`ads.txt`をドメイン直下に配置しなければなりません。。参考のリンク[^adstxt]にも記載していますが、`ads.txt`の配置は以下のような目的に基づきます。

> [引用] ads.txt を使用すれば、購入者が偽の広告枠を判別できるため、サイト運営者様としても偽の広告枠に収益が流れるのを阻止し、その分収益を増やすことができるというメリットがあります。
{: .prompt-info}

Github Pageを使いjekyllでサイトをホストするとStaticなファイルをドメイン直下に置く方法がわかりづらいと思います。私は以下の方法で用いました。

1. repository-name/assets/ads.txtを配置し、内容を以下のように書き換える。

```
---
permalink: /ads.txt
---

google.com, pub-????????????????, DIRECT, ????????????????
```
{: file="repository-name/assetsads.txt"}

お困りの皆様、ぜひとお試しください。

---
[^adstxt]: https://support.google.com/adsense/answer/7532444?hl=ja