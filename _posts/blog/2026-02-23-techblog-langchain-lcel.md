---
layout: post
title: "LangChain公式解説: LCEL（LangChain Expression Language）の設計思想とRunnable API"
description: "LangChain公式ブログのLCEL発表記事を解説し、宣言的パイプライン構築の設計思想を深掘りする"
categories: [blog, tech_blog]
tags: [LangChain, LCEL, Runnable, pipeline, Python, LLM]
date: 2026-02-23 10:00:00 +0900
source_type: tech_blog
source_domain: blog.langchain.com
source_url: https://blog.langchain.com/langchain-expression-language/
zenn_article: a5be5c172a5a99
zenn_url: https://zenn.dev/0h_n0/articles/a5be5c172a5a99
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [LangChain Expression Language - LangChain Blog](https://blog.langchain.com/langchain-expression-language/) の解説記事です。

## ブログ概要（Summary）

2023年8月にLangChain公式ブログで発表されたLCEL（LangChain Expression Language）は、LLMアプリケーションのパイプラインを宣言的に構築するための表現言語である。SQLAlchemy Expression Languageにインスパイアされた設計思想で、Pythonのパイプ演算子（`|`）を使ってプロンプト→モデル→出力パーサーの処理フローを1行で記述できる。公式ブログによると、LCELは「プロトタイプからプロダクションまでコード変更なしで移行できる」ことを設計目標としており、batch/async/streamingの3つの実行モードを追加コードなしで提供する。

この記事は [Zenn記事: LangChain LCEL実践ガイド：LLMチェーンのレイテンシを50%削減する最適化手法](https://zenn.dev/0h_n0/articles/a5be5c172a5a99) の深掘りです。Zenn記事ではLCELの実践的な使い方を解説していますが、本記事ではLCEL設計の背景にある技術的思想と、公式ブログで言及された設計判断を深掘りします。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://blog.langchain.com/langchain-expression-language/](https://blog.langchain.com/langchain-expression-language/)
- **組織**: LangChain, Inc.
- **発表日**: 2023年8月14日（2023年11月のv0.1.0で正式採用）

## 技術的背景（Technical Background）

LCELが生まれた背景には、LangChainの初期バージョンにおけるチェーン構築の課題があった。公式ブログでは、従来の`SequentialChain`について「amazingly usableとは言えなかった」と率直に認めている。具体的な問題点は以下の通りである：

1. **入出力の型管理**: `SequentialChain`は辞書ベースの入出力を使用しており、キー名の不一致がランタイムエラーを引き起こしていた
2. **実行モードの制限**: 同期実行のみで、ストリーミングや非同期実行には個別の実装が必要だった
3. **可観測性の欠如**: チェーン内部の処理フローを追跡するにはCallbackの手動設定が必要だった

LCELはこれらの課題を「テキストをユニバーサルインターフェースとする」宣言的言語として解決する。公式ブログでは「LLMアプリケーションにおけるチェーンはデータパイプラインに類似しており、同様のオーケストレーション（バッチ処理、並列化、フォールバック）が必要」と述べている。

## Runnableインターフェースの設計（Architecture）

### Runnable Protocol

LCELの中核となるのは`Runnable`インターフェースである。すべてのLCELコンポーネント（プロンプト、モデル、パーサー等）はこのインターフェースを実装する。

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, AsyncIterator, Iterator

Input = TypeVar("Input")
Output = TypeVar("Output")

class Runnable(ABC, Generic[Input, Output]):
    """LCELの基本インターフェース

    すべてのLCELコンポーネントはこのインターフェースを実装する。
    invoke/batch/streamの3モードとそれぞれの非同期版を標準提供。
    """

    @abstractmethod
    def invoke(self, input: Input) -> Output:
        """単一入力の同期実行"""
        ...

    async def ainvoke(self, input: Input) -> Output:
        """単一入力の非同期実行"""
        return self.invoke(input)

    def batch(self, inputs: list[Input], max_concurrency: int = 0) -> list[Output]:
        """複数入力のバッチ実行（内部最適化あり）"""
        ...

    def stream(self, input: Input) -> Iterator[Output]:
        """ストリーミング実行（逐次出力）"""
        yield self.invoke(input)

    async def astream(self, input: Input) -> AsyncIterator[Output]:
        """非同期ストリーミング"""
        yield await self.ainvoke(input)
```

公式ブログによると、このインターフェース設計には3つの意図がある：

1. **統一的な実行モデル**: invoke/batch/streamの3モードをすべてのコンポーネントに標準提供することで、実行方法の切り替えが設定変更だけで完了する
2. **合成可能性**: `|` 演算子で任意のRunnableを結合できるため、小さなコンポーネントを組み合わせて複雑なパイプラインを構築できる
3. **LangSmith統合**: Runnableインターフェースに準拠したコンポーネントはLangSmithでの自動トレースに対応し、Callback管理が不要になる

### パイプ演算子の内部実装

`|` 演算子はPythonの`__or__`マジックメソッドで実装されている。`a | b`は内部的に`RunnableSequence(a, b)`を返す。

```python
class Runnable:
    def __or__(self, other: "Runnable") -> "RunnableSequence":
        """パイプ演算子: a | b → RunnableSequence(a, b)"""
        return RunnableSequence(first=self, last=other)

class RunnableSequence(Runnable):
    """逐次実行チェーン

    first.invoke(input) の出力を last.invoke() の入力に渡す。
    ストリーミング時は最終ステージのみストリーミング対象。
    """

    def __init__(self, first: Runnable, last: Runnable):
        self.first = first
        self.last = last

    def invoke(self, input):
        intermediate = self.first.invoke(input)
        return self.last.invoke(intermediate)

    def stream(self, input):
        intermediate = self.first.invoke(input)
        yield from self.last.stream(intermediate)
```

この設計により、`prompt | model | parser`は`RunnableSequence(RunnableSequence(prompt, model), parser)`に展開される。

### RunnableParallel: 並列実行の設計

`RunnableParallel`は辞書形式で複数のRunnableを定義し、同一入力に対して並列実行する。公式ブログでは「steps can be executed in parallel (such as fetching documents from multiple retrievers)」と述べており、マルチリトリーバーの並列検索が主要なユースケースとして想定されている。

```python
from langchain_core.runnables import RunnableParallel
import concurrent.futures

class RunnableParallel(Runnable):
    """並列実行Runnable

    辞書のvalue部分の各Runnableを並列実行し、
    結果を同じキー名の辞書として返す。
    """

    def __init__(self, steps: dict[str, Runnable]):
        self.steps = steps

    def invoke(self, input):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                key: executor.submit(runnable.invoke, input)
                for key, runnable in self.steps.items()
            }
            return {
                key: future.result()
                for key, future in futures.items()
            }
```

## バッチ処理の内部最適化

公式ブログでは、batchメソッドについて「takes in a list of inputs. If optimizations can be done internally (like literally batching calls to LLM providers) those are done」と述べている。これは単なる逐次実行の繰り返しではなく、LLMプロバイダーのバッチAPIを直接利用する最適化が内部で行われることを意味する。

```python
# 内部最適化のイメージ（実際の実装は langchain-core に準拠）
class ChatOpenAI(Runnable):
    def batch(self, inputs: list[dict], max_concurrency: int = 5) -> list[str]:
        """バッチ実行 - APIプロバイダーのバッチ機能を活用

        max_concurrencyでAPIレートリミットに対応。
        OpenAI Tier 1: 500 RPM制限 → max_concurrency=5程度が安全。
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrency
        ) as executor:
            futures = [executor.submit(self.invoke, inp) for inp in inputs]
            return [f.result() for f in futures]
```

## ストリーミングの実装パターン

公式ブログによると、LCELチェーンでのストリーミングは「get the best possible time-to-first-token」を実現する。チェーンの最終ステージがLLMの場合、中間ステージ（プロンプトテンプレート等）は即座に実行され、LLMからのトークン生成が開始されると同時にストリーミングが始まる。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("{query}")
    | ChatOpenAI(model="gpt-4o", streaming=True)
    | StrOutputParser()
)

# ストリーミング実行: TTFTを最小化
for chunk in chain.stream({"query": "Pythonの型ヒントについて"}):
    print(chunk, end="", flush=True)
```

**設計上の注意点**:
- `StrOutputParser`はストリーミング対応（チャンクをそのまま通過させる）
- `JsonOutputParser`はバッファリングを行うため、完全なJSONが生成されるまで出力がブロックされる
- `RunnableParallel`内のストリーミングは各ブランチが独立してストリーミングされる

## LangSmithとの統合

公式ブログの重要な主張の一つは「LCEL chains integrate seamlessly with LangSmith」であり、これがLCEL採用の大きな動機となっている。従来のカスタムチェーンではCallbackHandlerの手動設定が必要だったが、LCELコンポーネントはRunnableインターフェースに準拠することで自動的にLangSmithのトレース対象となる。

**トレースで確認できる情報**:
- 各ステージの入出力
- 実行時間（レイテンシ分析）
- トークン使用量（コスト計算）
- エラー発生箇所と例外詳細

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $50-150 | Lambda + Bedrock + DynamoDB |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $300-800 | Lambda + ECS Fargate + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,000-5,000 | EKS + Karpenter + EC2 Spot |

**LCELパイプラインのデプロイ**:

LCELチェーンはFastAPI + LangServeの組み合わせでAPIとしてデプロイするのが公式推奨パターンである。

```python
# app.py - LangServe によるLCELチェーンのデプロイ
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="LCEL Pipeline API")

chain = (
    ChatPromptTemplate.from_template("{query}")
    | ChatOpenAI(model="gpt-4o", temperature=0.0)
    | StrOutputParser()
)

add_routes(app, chain, path="/chat")
```

**コスト試算の注意事項**:
- 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

```hcl
resource "aws_iam_role" "lambda_lcel" {
  name = "lcel-pipeline-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_lambda_function" "lcel_handler" {
  filename      = "lambda.zip"
  function_name = "lcel-pipeline-handler"
  role          = aws_iam_role.lambda_lcel.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 60
  memory_size   = 1024

  environment {
    variables = {
      OPENAI_API_KEY   = data.aws_secretsmanager_secret_version.openai.secret_string
      LANGSMITH_API_KEY = data.aws_secretsmanager_secret_version.langsmith.secret_string
    }
  }
}

resource "aws_secretsmanager_secret" "openai" {
  name = "lcel-openai-key"
}
```

### コスト最適化チェックリスト

- [ ] LCELの`max_concurrency`をAPIレートリミットに合わせて設定
- [ ] `RunnableParallel`で独立処理を並列化（レイテンシ40-60%削減）
- [ ] `with_fallbacks`で高コストモデル→低コストモデルのフォールバック設定
- [ ] `batch()`でバルク処理をバッチ化（APIコール数削減）
- [ ] LangSmithでトークン使用量を監視（コスト異常検知）
- [ ] Bedrock Batch API活用で50%割引
- [ ] Prompt Caching有効化で30-90%削減

## パフォーマンス最適化（Performance）

公式ブログおよびLangChainドキュメントで言及されているパフォーマンス特性：

- **TTFT（Time-to-First-Token）**: LCELストリーミングにより、LLMプロバイダーの生のトークン生成速度と同等のTTFTを実現
- **並列化効果**: `RunnableParallel`で独立処理を並列化した場合、逐次実行比で最短ブランチの実行時間に収束
- **バッチ最適化**: APIプロバイダーのバッチ機能を内部で活用し、APIコールのオーバーヘッドを削減

**ボトルネック特定方法**: LangSmithのトレースUIで各ステージのレイテンシを可視化し、最もレイテンシの大きいステージを特定する。

## 運用での学び（Production Lessons）

公式ブログおよびコミュニティのフィードバックから得られた運用上の知見：

1. **チェーンの長さ制限**: 5段階以上のチェーンはデバッグが困難になる。中間変数に分割してLangSmithでトレースを推奨
2. **エラー伝播**: チェーン内のいずれかのステージで例外が発生すると、後続のステージはすべてスキップされる。`with_fallbacks`でクリティカルなステージを保護する
3. **メモリ使用量**: 大量バッチ処理時は`max_concurrency`で同時実行数を制限。無制限の並列実行はOOMの原因となる

## 学術研究との関連（Academic Connection）

LCELの設計思想は、以下の学術的概念と関連している：

- **関数合成（Function Composition）**: パイプ演算子による合成は、圏論のモルフィズム合成に対応する。LCELのRunnable型は入出力型のペアで特徴付けられ、`|`演算子は型安全な合成を提供する
- **Dataflow Programming**: LCELのDAGベースのパイプラインは、データフロープログラミングの概念を採用しており、各ノードが独立して実行可能な設計となっている
- **AOP (CIDR 2025)**: 自動的なLLMパイプラインオーケストレーションの研究では、LCELと同様のDAG構造を用いた並列実行最適化が報告されている

## まとめと実践への示唆

LCELは、LLMアプリケーション開発における「プロトタイプからプロダクションまでコード変更なし」という設計目標を、Runnableインターフェースの標準化によって実現した。2026年時点では、LangChain 1.0のリリースにより、線形パイプラインにはLCEL、複雑なエージェントにはLangGraphという明確な使い分けが確立されている。公式ブログの設計思想を理解することで、LCELの機能を最大限に活用したパイプライン設計が可能になる。

## 参考文献

- **Blog URL**: [https://blog.langchain.com/langchain-expression-language/](https://blog.langchain.com/langchain-expression-language/)
- **LangChain Docs**: [https://python.langchain.com/docs/concepts/lcel/](https://python.langchain.com/docs/concepts/lcel/)
- **LangChain v1 Migration Guide**: [https://docs.langchain.com/oss/python/migrate/langchain-v1](https://docs.langchain.com/oss/python/migrate/langchain-v1)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/a5be5c172a5a99](https://zenn.dev/0h_n0/articles/a5be5c172a5a99)
