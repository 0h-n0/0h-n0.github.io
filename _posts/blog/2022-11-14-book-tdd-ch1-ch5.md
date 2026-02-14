---
layout: single
title: "書籍『テスト駆動開発』を読みつつ、Pythonで書き直してみる。"
excerpt: book Test Driven Development
categories:
  - TechBlog
tags:
  - TDD
toc: true
toc_sticky: true
---

　オライリーの書籍『[テスト駆動開発](https://amzn.to/3hyTQIF)』を読みつつ、詰まったところや一部書き換えが必要なところをまとめました。そして一部のサンプルをpythonベースで書き換えています。詳しい内容は是非とも書籍を購入して確認してください。


## 前置き

　この本を手に取っているということは、ある程度のエンジニア歴があると仮定します。このブログはMLE/DS向けに記載しているため、書籍ではJavaのサンプルコードが記載されていますが、Pythonで書き換えていきます(書籍の第二部はサンプルコードはPythonです)。書籍に沿ってTDDの話を進めるとともに、Pythonでのテスト方法を紹介していきます。こからソースコードは一緒になりますが、どのようにテストを実行するかを分けて書いていきます。

1. [外部ライブラリに頼らない一番シンプルな方法](#外部ライブラリに頼らない一番シンプルな方法)
1. [poetryを用いて、組み込みの`Unittest`を使ってテストをする方法](#poetryを用いて組み込みのunittestを使ってテストをする方法)
1. [poetryを用いて、pythonのテストフレームワークの`pytest`を使ってテストをする方法](#poetryを用いてpythonのテストフレームワークのpytestを使ってテストをする方法)


## TDDとは

>　TDDとは、『テスト作成➡テスト失敗➡コード作成➡テスト実行➡リファクタリング』の手順で実行しサイクルを回しながら開発する手法です。
{: .prompt-info}

```mermaid
flowchart LR
    A([1. テスト作成])
    B([2. テスト失敗])
    C([3. コード作成])
    D([4. テスト実行])        
    E([5. リファクタリング])        
    A --> B --> C --> D --> E --> A
```


　この手法はアジャイル開発と非常に相性が良く、アジャイル開発をするならば必須だと考えれています。アジャイル開発はソフトウェア開発のアジリティを上げるだけではなく、顧客の欲求や満足を最大限満たしつつ、いち早く顧客へ届ける開発手法です。_日々変化していく顧客の欲求に対して、素早く対応するためにはコードの品質を高める_必要がありました。TDDは、コードの品質を高めつつ開発スピードを落とさない開発手法の一つとなります。繰り返しになりますが、詳細は書籍を参照してください。

## 外国通貨プログラムの実装

まずは外国通貨プログラムの持つべき機能をTODOリストにまとめます。書籍の通り。

> 小さいステップで!!!
{: .prompt-info}

```mermaid
flowchart LR
    A([1. <bf>テスト作成</bf><br>小さいテストを一つ書く])
    B([2. テスト失敗<br>全てのテストを実行し<br>1つ失敗することを確認する])
    C([3. コード作成<br>小さい変更を行う])
    D([4. テスト実行<br>再びテストを実行し<br>すべて成功することを確認する])        
    E([5. リファクタリング<br>リファクタリングを行い<br>重複を除去する])            
    A --> B --> C --> D --> E --> A
```

満たすべき要求を洗い出します。どれから手を付けるかは、実装しやすい順番で作成していけばよい。

**テストTODOリスト**
- [ ] $5 + 10 CHF = $10（レートが2:1の場合）
- [ ] $5 * 2 = $10
- [ ] amountをprivateにする
- [ ] Dollarの副作用どうする？
- [ ] Moneyの丸め処理どうする？

### 外部ライブラリに頼らない一番シンプルな方法

　書籍のソースコードをpythonに書き換えつつ、組み込みのUnittestでテストを作成します。

#### 準備

 ディレクトリの作成とソースファイルを作成します。pythonでのテストファイルは慣例的に`test_hogehoge.py`とします。

```shell
$ mkdir -p tdd-book-normal/src
$ toch tdd-book-normal/src/test_money.py
$ toch tdd-book-normal/src/dollar.py
```


それぞれのファイルの中身は以下のようになります。(テスト作成フェーズ)

```mermaid
flowchart LR
    A([1. テスト作成])
    B([2. テスト失敗])
    C([3. コード作成])
    D([4. テスト実行])        
    E([5. リファクタリング])        
    style A fill:#488707,stroke:#fee6f0,stroke-width:4px
    A --> B --> C --> D --> E --> A
```

```py
import unittest

import dollar


class TestMoney(unittest.TestCase):
    """Test Money
    """
    def test_multiplication(self):
        """ここにテストの説明を書きます。test_hogehogeのmethodがテストの対象になります。
        """
        five = dollar_01.Dollar(5)
        five.times(2)
        self.assertEqual(10, five.amount)


if __name__ == '__main__':
    unittest.main()
```
{: file="tdd-book-normal/src/test_money.py"}

```py
class Dollar:
    pass
```
{: file="tdd-book-normal/src/dollar.py"}


上記のファイルを作成後以下のコマンドでテストを実行します。すると想像通りエラーが出ます。（テスト失敗）

```mermaid
flowchart LR
    A([1. テスト作成])
    B([2. テスト失敗])
    C([3. コード作成])
    D([4. テスト実行])        
    E([5. リファクタリング])        
    style B fill:#488707,stroke:#fee6f0,stroke-width:4px
    A --> B --> C --> D --> E --> A
```

```shell
$ cd tdd-book-normal/src/
$ python test_money.py 
E
======================================================================
ERROR: test_multiplication (__main__.TestMoney)
test_hogehogeのmethodをテストの対象とします。
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/hogehoge/tdd-book-normal/src/test_money.py", line 12, in test_multiplication
    five = dollar.Dollar(5)
TypeError: Dollar() takes no arguments

----------------------------------------------------------------------
Ran 1 test in 0.000s

FAILED (errors=1)
```

下記のようにテストにパスするようにDollar.pyを書き換えます。（コード作成）

```mermaid
flowchart LR
    A([1. テスト作成])
    B([2. テスト失敗])
    C([3. コード作成])
    D([4. テスト実行])        
    E([5. リファクタリング])        
    style C fill:#488707,stroke:#fee6f0,stroke-width:4px
    A --> B --> C --> D --> E --> A
```

```py
class Dollar:
    amount = 10
    def __init__(self, amount: int):
        pass

    def times(self, multipler: int):
        pass
```
{: file="tdd-book-normal/src/dollar.py"}



もう一度テスト実行します。(テスト実行)

```mermaid
flowchart LR
    A([1. テスト作成])
    B([2. テスト失敗])
    C([3. コード作成])
    D([4. テスト実行])        
    E([5. リファクタリング])        
    style D fill:#488707,stroke:#fee6f0,stroke-width:4px
    A --> B --> C --> D --> E --> A
```


```shell
$ python test_money.py -v
test_multiplication (__main__.TestMoney)
test_hogehogeのmethodをテストの対象とします。 ... ok

----------------------------------------------------------------------
Ran 1 test in 0.000s

OK
```


依存性と重複を取り除くために、プロダクトコードをリファクタリングする。(リファクタリング)

```mermaid
flowchart LR
    A([1. テスト作成])
    B([2. テスト失敗])
    C([3. コード作成])
    D([4. テスト実行])        
    E([5. リファクタリング])        
    style E fill:#488707,stroke:#fee6f0,stroke-width:4px
    A --> B --> C --> D --> E --> A
```


```py
class Dollar:
    def __init__(self, amount: int):
        self.amount = amount

    def times(self, multipler: int):
        self.amount *= multipler
```
{: file="tdd-book-normal/src/dollar.py"}


> **依存性と重複の除去**<br>
> **重複**：テストコードとプロダクトコードの間に存在する重複を取り除く。この書籍では、`10`というハードコードした数字がテストコードとプロダクトコードの両方に存在していることなどが該当する。<br>
> **依存性**：依存性の片方のコードを変更したときに、もう片方も変更しないといけない場合、依存性が強い状態である。上記の例に照らし合わせると、テストコードでハードコードした`10`を`5`に変えた場合、プロダクトコードも変更しなければならない。<br>
> :star: 次のテストを実行する前に、**依存性と重複を除去**することによって、次のテストを成功させる確率を最大化することが出来る。
{: .prompt-info}

---
### poetryを用いて、組み込みの`Unittest`を使ってテストをする方法

[公式ドキュメント](https://python-poetry.org/docs/)を参考にしてpoetryをインストールしてください。インストール後、下記のコマンドを実行してください。

```shell
$ poetry new tdd_book_poetry
```

実行後、ディレクトリ構成は以下のようになっています。

```shell
$ tree .
.
├── README.md
├── pyproject.toml
├── tdd_book_poetry
│   └── __init__.py
└── tests
    └── __init__.py
```

上で作成した`tdd-book-normal/src/test_money.py`{: .filepath}と`tdd-book-normal/src/dollar.py`{: .filepath}をpoetryで作成したディレクトリに適切にコピーします。するとディレクトリの構成内容は以下のようになります。


```shell
$ tree .
.
├── README.md
├── pyproject.toml
├── tdd_book_poetry
│   ├── __init__.py
│   └── dollar.py # 追加
└── tests
    ├── __init__.py
    └── test_money.py # 追加
```    

さらに`tdd_book_poetry/tests/test_money.py`{: .filepath}を以下のように変更します。

```py
import unittest

import tdd_book_poetry.dollar


class TestMoney(unittest.TestCase):
    """Test Money
    """
# continue    
```
{: file="tdd_book_poetry/tests/test_money.py"}


変更後、以下のコマンドでテストが可能です。

```shell
$ poetry run python -m unittest discover -v
test_multiplication (tests.test_money.TestMoney)
test_hogehogeのmethodをテストの対象とします。 ... ok

----------------------------------------------------------------------
Ran 1 test in 0.000s

OK
```

---

### poetryを用いて、pythonのテストフレームワークの`pytest`を使ってテストをする方法

上記の例では、pythonの組み込みにテストフレームワークである`unittest`を使っていましたが、もっと便利な`pytest`を使って書き換えてみましょう。poetryで新しいプロジェクトを作成し、`pytest`をインストールしましょう。

```shell
$ poetry new tdd_book_poetry_pytestCreated package tdd_book_poetry_pytest in tdd_book_poetry_pytest
$ cd tdd_book_poetry_pytest
$ poetry add pytest --group test
```

先ほどと同様に必要なファイル(`dollar.py`と`test_money.py`)はコピーしてください。そして、`test_money.py`の`import`の後を適切に書き換えてください。

```shell
$ tree .
.
├── README.md
├── pyproject.toml
├── tdd_book_poetry
│   ├── __init__.py
│   └── dollar.py # 追加
└── tests
    ├── __init__.py
    └── test_money.py # 追加
```    

`pytest`を使うと`unittest`よりさらに簡単にテストを書くことが出来ます。

```py
import tdd_book_poetry_pytest.dollar

def test_multiplication():
    """ここにテストの説明を書きます。test_hogehogeのmethodがテストの対象になります。
    """
    five = dollar_01.Dollar(5)
    five.times(2)
    assert 10 == five.amount
```
{: file="tdd_book_poetry_pytest/tests/test_money.py"}

そして、`pytest`を実行します。

```shell
$ poetry run pytest
==================================== test session starts ====================================
platform linux -- Python 3.10.4, pytest-7.2.0, pluggy-1.0.0
rootdir: /home/hogehoge/tdd_book_poetry_pytest
plugins: mock-3.9.0, cov-4.0.0
collected 1 item                                                                            

tests/test_money.py .                                                                 [100%]

===================================== 1 passed in 0.01s =====================================
```

## 最後に

３つの方法を使って、pythonの実施方法を学びました。プログラムが大きくなるほどテストが複雑になります。また、依存ライブラリーなど環境面を再現できなければ適切なテストを行うことすらできません。そのような観点から、最後に紹介した`poetry`+`pytest`が一番やりやすいと思います。本当は、これに加えて`github-action`や`gitlab-ci`を説明する予定でしたが、心が折れたので気が向いたら本記事を更新したいと思います。最後までご愛読頂き、誠に感謝します。ありがとうございました！