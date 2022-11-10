---
layout: post
title: "書籍『SolidityとEthereumによる実践スマートコントラクト開発 ―Truffle Suiteを用いた開発の基礎からデプロイまで』の環境設定補足"
description: blockchain for machine learning engineers
categories: [TechBlog]
tags: [Blockchain, Solidity, Ethereum]
---

オライリーの書籍『[SolidityとEthereumによる実践スマートコントラクト開発 ―Truffle Suiteを用いた開発の基礎からデプロイまで](https://amzn.to/3EkYG57)』を読みつつ、詰まったところや一部書き換えが必要なところをまとめました。この内容は本書の第三章の部分の補足となります。

# 環境構築(for WLS, Linux(Ubuntu), mac)

--- 
## Parityのインストール(OpenEthereum)
イーサリウムのクライアント環境であるParityをインストールします。Parityはrustで実装されており、他のクライアントと比べると高速らしいです。ただ、本に記載している通りにone-Linerでインストールしようとすると、Urlが死んでいるため地道にインストールしていきます。Openethererumはparityを引き継いだオープンソースとなります。parityのインストールとほぼ同四の手順でインストールできます。

### そもそもの環境

ここからの説明はLinux環境を想定しており、基本的な開発ツールが入っていることを想定しています。以降はaptが使える環境を想定していますが、環境に応じて`Xcode`や`yum`に置き換えてください。

```shell
$ # Debian-based Linux distribution, including Ubuntu.
$ sudo apt install build-essential
$ # 必要な開発系の追加ライブラリのインストール
$ sudo apt install cmake clang llvm
$ sudo apt install librocksdb-dev
```

### rustupのインストール
rust言語をビルドするために、rustupをインストールします。以下の手順に従ってインストールしてください。

```shell
$ curl https://sh.rustup.rs -sSf | sh
$ # confirm
$ rustup --version
$ echo '. "$HOME/.cargo/env"' >> ~/.bashrc
$ source ~/.bashrc
```
### parity(OpenEthereum)のソースからのインストール

githubからソースファイルをダウンロードしビルドします。ビルドした実行ファイルにパスを通すことで、コマンドライン上で使用できるようにします。

```shell
# download Parity Ethereum code
$ git clone https://github.com/paritytech/parity-ethereum
$ cd parity-ethereum
$ git checkout stable
# download OpenEthereum code
$ git clone https://github.com/openethereum/openethereum
$ cd openethereum
$ git checkout stable
```

最新のrust versionだとエラーが出るので、各ライブラリごとにRustのバージョンを変更します。

```shell
$ # それぞれのディレクトリに移動後下記を実行します。
$ rustup override set 1.51.0 # for Parity
$ rustup override set nightly # for Parity
```

versionが変更されているか確認後、ビルドを実施してください。

```shell
$ cargo --version 
cargo 1.51.0 (43b129a20 2021-03-16) # for parity
cargo 1.67.0-nightly (9286a1beb 2022-11-04) # for openethereum
$ cargo build --release --features final
```

ビルドが出来ていれば、下記のコマンドが実行できるはずです。使いやすいように適切にパスに設定してください。

```shell
$ ./target/release/parity
$ ./target/release/openethereum
```

openethereumの場合、DBのアップデートが必要なようです。下記コマンドを実行してください。

```shell
$ PARITY_PATH=~/.local/share/io.parity.ethereum
$ cd /tmp/
$ git clone https://github.com/openethereum/3.1-db-upgrade-tool.git
$ cd 3.1-db-upgrade-tool
$ cargo run "$PARITY_PATH/chains/ethereum/db/906a34e69aec8c0d/overlayrecent"
$ # Archive node run with this
$ cargo run "$PARITY_PATH/chains/ethereum/db/906a34e69aec8c0d/archive"
$ cd ~
$ rm -Rf /tmp/3.1-db-upgrade-tool
```

参考URL

* https://github.com/openethereum/3.1-db-upgrade-tool


> #### Parityの場合に想定されるエラー
> 以下のようなエラーが出る場合は、rustのバージョンが原因です。
> ```shell
> error[E0061]: this function takes 1 argument but 2 arguments were supplied
>     --> /home/hogehoge/.cargo/registry/src/github.com-1ecc6299db9ec823/logos-derive-0.7.7/src/lib.rs:55:20
> ~~~~~~
> ```
> 以下のコマンドでrustのバージョンを下げてください。
> ```shell
> $ rustup override set 1.51.0
> ```
> * 参考
>   * https://github.com/maciejhirsz/logos/issues/224
>   * https://github.com/openethereum/openethereum/issues/442
{: .prompt-warning }
> #### OpenEthereumの場合に想定されるエラー
> 以下のようなエラーが出る場合は、rustのバージョンが原因です。
> ```shell
> error: edition 2021 is unstable and only available with -Z unstable-options.
>   error: could not compile `oe-rpc-common`    
> ```
> 以下のコマンドでrustのバージョンをnightlyにします。
> ```shell
> $ rustup override set nightly
> ```
{: .prompt-warning }


参考
* [Github/parity-ethereum](https://github.com/openethereum/parity-ethereum)
* [Github/OpenEthereum](https://github.com/openethereum/openethereum)

---

## MetaMaskのインストール

[metamask.io](https://metamask.io/)にアクセスしてChrome拡張をインストールしてください。

---
## NodeJSのインストール(NVMでインストール)

好きなところにインストールするのが良いと思いますが、/usr/binや/binにインストールするのは、私は好きではありません。なので以下の方法でインストールします。

```shell
$ curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.2/install.sh | bash
```

下記のコマンドで正常にインストールできたことを確認します。

```shell
$ node -v
v18.9.0
```

参考
* https://github.com/nvm-sh/nvm

---
## Truffleのインストール

```shell
$ npm install -g truffle
$ npm install -g truffle@5.1.31 # 本ではバージョン指定していますが、no supportのエラーがでます。
```

> エラーが出るときは、nodeバージョンを下げるか、truffleのバージョンを上げてください。
{: .prompt-warning }

下記のコマンドで正常にインストールできたことを確認します。

```shell
$ truffle -v
Truffle v5.6.5 (core: 5.6.5)
Ganache v7.5.0
Solidity v0.5.16 (solc-js)
Node v18.9.0
Web3.js v1.7.4
```
---
## GanacheのGUIのインストール

[ganache](https://trufflesuite.com/ganache/)にアクセスして各自のOSにあったバージョンをフルバージョンでインストールしてください。