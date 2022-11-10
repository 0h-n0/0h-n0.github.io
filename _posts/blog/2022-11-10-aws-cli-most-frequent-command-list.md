---
layout: post
title: "AWS CLIでよく使う基本的なコマンド(逆引き)"
description: AWS Command Line interface fundamental command list
categories: [TechBlog]
tags: [Machine Learning Engineer, Data Scientist, AWS, Cloud]
---

 AWS CLIでよく使うコマンドなどを逆引きでまとめました。この記事は随時更新予定です。

## 逆引きコマンドチートシート

||command|
|--:|:--|
|[登録しているAWSプロファイルリストの一覧](#登録しているawsプロファイルリストの一覧)|`aws configure list-profiles`|
|[IAMユーザーやロール、それに紐づくクレデンシャルを知りたいとき](#iamユーザーやロールそれに紐づくクレデンシャルを知りたいとき)|`aws sts get-caller-identify`|
|[IAMロールを作るとき](#iamロールを作るとき)|`aws iam create-role`|
|[現在作成しているIAMロールの確認](#現在作成しているIAMロールの確認)|`aws iam list-roles`|
|[IAMロールを削除する](#IAMロールを削除する)|`aws iam delete-role`|
|[作成したポリシーファイルが間違っていないかの確認方法](#作成したポリシーファイルが間違っていないかの確認方法)|`aws accessanalyzer validate-policy`|



## 認証情報やアカウント切り替えなどの基礎コマンド

### 登録しているAWSプロファイルリストの一覧

```bash
$ aws configure list-profiles
default
hogehoge
higehige
```

### コマンド毎でのプロファイルの切り替え

```bash
$ aws sts get-caller-identify --profile hogehoge
{
    "UserId": "UserIDForHogehoge",
    "Account": "AcountHogehoge"
    "Arn": "arn:aws:iam::AcountHogehoge:user/hogehoge"
}
```

### IAMユーザーやロール、それに紐づくクレデンシャルを知りたいとき

```bash
$ aws sts get-caller-identify
{
    "UserId": "UserID",
    "Account": "Acount"
    "Arn": "arn:aws:iam::AcountHogehoge:user/username"
}
$ # key{UserId, Account, Arn}を指定することで情報の一部を取得可能
$ aws sts get-caller-identify --query Arn
"arn:aws:iam::AcountHogehoge:user/username"
$ # アウトプットのスタイルも指定可能
$ aws sts get-caller-identity --query Arn --output text
arn:aws:iam::AcountHogehoge:user/username
```

* [公式ドキュメント：aws sts get-caller-identify](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/sts/get-caller-identity.html)


### 作成したポリシーファイルが間違っていないかの確認方法
下記のように、`policy-template.json`を作成する。

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::AcountHogehoge:user/hogehoge"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

下記のCLIコマンドで確認することで、間違っている箇所があればエラーがでる。指示にしたがって修正する。

```shell
$ aws accessanalyzer validate-policy --policy-document file://policy-template.json --policy-type IDENTITY_POLICY
```

* [公式ドキュメント：aws accessanalyzer validate-policy](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/accessanalyzer/validate-policy.html)


---

## `aws iam `の後によく使うコマンド

まずは公式ドキュメントを参考にしましょう。
* [AWS Identity and Access Management (IAM)](https://aws.amazon.com/jp/iam/)
* [IAMのベストプラクティス](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)


### IAMロールを作るとき

下記のように、`policy-template.json`を作成する。
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::AcountHogehoge:user/hogehoge"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

上記のコマンドでIAMロールを作成する。jsonファイルにエラーがあるとコマンドを実行することができない。エラーの出力がわかりにくいため、エラーが出る場合は[作成したポリシーファイルが間違っていないかの確認方法](#作成したポリシーファイルが間違っていないかの確認方法)で確認する方が良い。
```shell
$ aws iam create-role --role-name HOGEHOGERole --assume-role-policy-document file://policy-template.json
```

### 現在作成しているIAMロールの確認

以下のコマンドで出力することが出来ます。

```shell
$ aws iam list-roles
```

上記のコマンドでは出力が冗長になり過ぎるので、以下のオプションで適切にフィルタリングするとよい。

```shell
$ aws iam list-roles --query 'Roles[*].RoleName'
```
grep するのもおススメ

```shell
$ aws iam list-roles --query 'Roles[*].RoleName' | grep AWS
```


* [公式ドキュメント：AWS CLI 出力をフィルタリングする](https://docs.aws.amazon.com/ja_jp/cli/latest/userguide/cli-usage-filter.html)


### IAMロールを削除する

```shell
$ aws iam delete-roles --role-name RoleName
```
もちろん、上記の方法でクエリも使用できます。公式のドキュメントでは最近のroleの使用状況も確認しております。適切な手順で間違えないように削除しましょう。

* [公式ドキュメント：ロールまたはインスタンスプロファイルの削除](https://docs.aws.amazon.com/ja_jp/IAM/latest/UserGuide/id_roles_manage_delete.html)


