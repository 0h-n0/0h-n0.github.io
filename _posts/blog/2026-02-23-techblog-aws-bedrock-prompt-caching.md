---
layout: post
title: "AWS公式解説: Amazon Bedrockプロンプトキャッシュの技術仕様とConverse API実装"
description: "Amazon Bedrockのプロンプトキャッシュ機能を詳細解説。Converse API・InvokeModel双方のキャッシュチェックポイント実装、5分/1時間TTL、対応モデル一覧、コスト最適化戦略を網羅"
categories: [blog, tech_blog]
tags: [AWS, Bedrock, prompt-caching, KV-cache, Claude, LLM, cost-optimization]
date: 2026-02-23 12:00:00 +0900
source_type: tech_blog
source_domain: aws.amazon.com
source_url: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
zenn_article: 555a4e799660de
zenn_url: https://zenn.dev/0h_n0/articles/555a4e799660de
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [AWS公式ドキュメント: Prompt caching for faster model inference](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html) および [AWS What's New: Amazon Bedrock 1-hour prompt caching](https://aws.amazon.com/about-aws/whats-new/2026/01/amazon-bedrock-one-hour-duration-prompt-caching/) の解説記事です。

## ブログ概要（Summary）

Amazon Bedrockのプロンプトキャッシュは、リクエスト間で共通するプロンプトプレフィックスのKV計算結果をサーバー側で保存・再利用する機能である。公式ドキュメントによると、キャッシュ読取はベース入力トークン価格の割引料金で処理され、**レイテンシ最大85%改善・コスト最大90%削減**が報告されている。Converse APIとInvokeModel APIの双方で利用可能であり、Claude（Opus/Sonnet/Haiku各世代）とAmazon Nova（Micro/Lite/Pro/Premier）の計15モデル以上で対応している。2026年1月にはClaude Opus 4.5、Sonnet 4.5、Haiku 4.5で1時間TTLが追加され、長時間のエージェントワークフローにも対応可能となった。

この記事は [Zenn記事: LangGraph×Claude Sonnet 4.6のプロンプトキャッシュ最適化でAgentic RAGコスト90%削減](https://zenn.dev/0h_n0/articles/555a4e799660de) の深掘りです。

## 情報源

- **種別**: クラウドプロバイダ公式ドキュメント / What's New
- **URL**: [https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- **組織**: Amazon Web Services
- **更新日**: 2026年1月（1時間TTL対応の告知含む）

## 技術的背景（Technical Background）

LLMの推論ではプロンプトの入力トークンに対してTransformerの各レイヤーでKV（Key-Value）テンソルを計算する。マルチターン会話やRAGアプリケーションでは、システムプロンプト・ツール定義・検索結果が各リクエストで繰り返し送信されるため、同一内容のKV計算が冗長に発生する。

Amazon Bedrockのプロンプトキャッシュは、この冗長計算を排除する機能である。ユーザーがリクエスト内にキャッシュチェックポイントを配置すると、チェックポイントまでのプレフィックスのKV状態がサーバー側に保存される。次回同一プレフィックスが送信された場合、保存済みKVを再利用してKV計算をスキップする。

Anthropic APIのプロンプトキャッシュとは以下の点で異なる：

- **API形式**: BedrockはConverse API（AWS統一形式）とInvokeModel API（モデル固有形式）の2種類のキャッシュインターフェースを提供
- **モデル横断**: Claude系だけでなくAmazon Novaモデルでもキャッシュが利用可能
- **キャッシュチェックポイント構文**: Converse APIでは`cachePoint`オブジェクト、InvokeModel APIでは`cache_control`（Claude）または`cachePoint`（Nova）を使用

## 対応モデルとキャッシュ仕様

AWS公式ドキュメントに記載されている対応モデル一覧（2026年2月時点）：

**Claude系モデル:**

| モデル | Model ID | 最小トークン/CP | 最大CP数 | TTL | キャッシュ対象 |
|--------|----------|-----------------|---------|-----|--------------|
| Claude Opus 4.5 | anthropic.claude-opus-4-5-* | 4,096 | 4 | 5分, 1時間 | system, messages, tools |
| Claude Sonnet 4.5 | anthropic.claude-sonnet-4-5-* | 1,024 | 4 | 5分, 1時間 | system, messages, tools |
| Claude Haiku 4.5 | anthropic.claude-haiku-4-5-* | 4,096 | 4 | 5分, 1時間 | system, messages, tools |
| Claude Opus 4.1 | anthropic.claude-opus-4-1-* | 1,024 | 4 | 5分 | system, messages, tools |
| Claude Sonnet 4 | anthropic.claude-sonnet-4-* | 1,024 | 4 | 5分 | system, messages, tools |
| Claude 3.7 Sonnet | anthropic.claude-3-7-sonnet-* | 1,024 | 4 | 5分 | system, messages, tools |
| Claude 3.5 Haiku | anthropic.claude-3-5-haiku-* | 2,048 | 4 | 5分 | system, messages, tools |

**Amazon Novaモデル:**

| モデル | 最小トークン/CP | 最大キャッシュトークン | TTL | キャッシュ対象 |
|--------|-----------------|---------------------|-----|--------------|
| Nova Micro/Lite/Pro/Premier | 1,000 | 20,000 | 5分 | system, messages |

ここで、CP = Cache Checkpoint（キャッシュチェックポイント）である。

**重要な違い**: Claude系は`tools`フィールドもキャッシュ対象だが、Novaモデルは`system`と`messages`のみ。エージェント型アプリケーションでツール定義をキャッシュする場合はClaude系を選択する必要がある。

## 実装アーキテクチャ（Architecture）

### Converse APIでのキャッシュ実装

Converse APIはAWS統一のマルチターン会話インターフェースである。キャッシュチェックポイントは`cachePoint`オブジェクトとして配置する。

```python
import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def converse_with_cache(
    model_id: str,
    system_prompt: str,
    tools: list[dict],
    messages: list[dict],
    ttl: str = "5m",
) -> dict:
    """Converse APIでプロンプトキャッシュを使用したリクエスト

    Args:
        model_id: Bedrockモデル ID
        system_prompt: システムプロンプト
        tools: ツール定義リスト
        messages: 会話メッセージリスト
        ttl: キャッシュTTL ("5m" or "1h")

    Returns:
        Converse APIレスポンス
    """
    # システムプロンプト + キャッシュチェックポイント
    system = [
        {"text": system_prompt},
        {"cachePoint": {"type": "default", "ttl": ttl}},
    ]

    # ツール定義 + キャッシュチェックポイント
    tool_config = {
        "tools": [
            *[{"toolSpec": t} for t in tools],
            {"cachePoint": {"type": "default", "ttl": ttl}},
        ]
    }

    response = bedrock.converse(
        modelId=model_id,
        system=system,
        toolConfig=tool_config,
        messages=messages,
    )
    return response
```

### InvokeModel APIでのキャッシュ実装（Claude向け）

InvokeModel APIでは、Anthropic APIと同じ`cache_control`構文をBedrock経由で使用する：

```python
def invoke_model_with_cache(
    model_id: str,
    system_prompt: str,
    rag_context: str,
    user_query: str,
    ttl: str = "5m",
) -> dict:
    """InvokeModel APIでClaude向けキャッシュを使用

    Args:
        model_id: Claude Bedrock Model ID
        system_prompt: システムプロンプト
        rag_context: RAG検索結果テキスト
        user_query: ユーザークエリ
        ttl: キャッシュTTL ("5m" or "1h")

    Returns:
        InvokeModel APIレスポンス
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral", "ttl": ttl},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": rag_context,
                        "cache_control": {"type": "ephemeral", "ttl": ttl},
                    },
                    {"type": "text", "text": user_query},
                ],
            }
        ],
    }

    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
    )
    return json.loads(response["body"].read())
```

### キャッシュメトリクスの取得

Converse APIのレスポンスには以下のキャッシュメトリクスが含まれる：

```python
def extract_cache_metrics(response: dict) -> dict:
    """レスポンスからキャッシュメトリクスを抽出

    Args:
        response: Converse APIレスポンス

    Returns:
        キャッシュメトリクス辞書
    """
    usage = response.get("usage", {})
    return {
        "cache_read_input_tokens": usage.get("cacheReadInputTokens", 0),
        "cache_write_input_tokens": usage.get("cacheWriteInputTokens", 0),
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "cache_hit_rate": (
            usage.get("cacheReadInputTokens", 0)
            / max(usage.get("inputTokens", 1)
                  + usage.get("cacheReadInputTokens", 1), 1)
        ),
    }
```

## キャッシュチェックポイントの設計戦略

### 簡易キャッシュ管理（推奨）

AWS公式ドキュメントでは、Claude向けの「簡易キャッシュ管理」を推奨している。静的コンテンツの末尾に1つのキャッシュチェックポイントを配置するだけで、自動的に最適なプレフィックスマッチングが行われる：

```
[ツール定義] → [システムプロンプト] → [cachePoint] → [動的メッセージ]
```

この方式では、システムがチェックポイントから最大20ブロック前まで遡って最長一致するプレフィックスを探索する。マルチターン会話で会話が成長しても、自動的にキャッシュヒット範囲が拡大する。

### 明示的マルチチェックポイント（高度な制御）

変更頻度が異なるコンテンツを個別にキャッシュする場合、最大4つのチェックポイントを配置する：

```
[ツール定義][CP1] → [システムプロンプト][CP2] → [RAGコンテキスト][CP3] → [会話履歴][CP4]
```

ツール定義やシステムプロンプトは高いキャッシュヒット率を維持でき、RAGコンテキストはクエリごとに変動するが同一チャンクが再利用されるケースでヒットする。

### TTL選択の指針

5分TTLと1時間TTLの使い分けについて、AWS公式ドキュメントは以下の指針を示している：

**5分TTL（デフォルト）**:
- 高頻度アクセス（5分以内に再利用されるプロンプト）
- チャットボットの通常会話
- バッチ処理の連続リクエスト

**1時間TTL（2026年1月〜、Claude Opus 4.5/Sonnet 4.5/Haiku 4.5のみ）**:
- エージェントワークフロー（ツール呼び出し・RAG検索で5分を超える処理）
- 低頻度ユーザー対話（ユーザーの応答間隔が5分〜1時間）
- バッチAPI処理の共通プレフィックス

**TTL混在時の制約**: 1つのリクエスト内で1時間と5分のTTLを混在させる場合、長いTTL（1時間）のチェックポイントを短いTTL（5分）のチェックポイントより前に配置する必要がある。逆順にするとAPI 400エラーが返る。

## パフォーマンス最適化（Performance）

### キャッシュ効果の理論的分析

プロンプト全体のトークン数を $N$、キャッシュされるプレフィックスのトークン数を $N_{\text{cached}}$ とすると、キャッシュヒット時のKV計算削減率は：

$$
\text{Reduction} = \frac{N_{\text{cached}}}{N}
$$

10ターンのマルチターン会話（各ターン1,000トークン）でシステムプロンプト2,000トークンの場合：

$$
N = 2000 + 10 \times 1000 = 12000, \quad N_{\text{cached}} = 11000
$$

$$
\text{Reduction} = \frac{11000}{12000} \approx 91.7\%
$$

### コスト試算

Claude Sonnet 4.5（Bedrock経由）でRAGアプリケーションを運用する場合の試算：

**前提条件**:
- 1日1,000リクエスト
- システムプロンプト: 2,000トークン（静的）
- ツール定義: 3,000トークン（静的）
- RAGコンテキスト: 5,000トークン（50%再利用）
- ユーザークエリ: 200トークン（動的）

**キャッシュなし**: $1{,}000 \times 10{,}200 \times \$3.00/\text{MTok} = \$30.60/\text{日}$

**キャッシュあり（5分TTL、80%ヒット率想定）**:
- キャッシュ書込（20%）: $200 \times 10{,}000 \times \$3.75/\text{MTok} = \$7.50$
- キャッシュ読取（80%）: $800 \times 10{,}000 \times \$0.30/\text{MTok} = \$2.40$
- 非キャッシュ部分: $1{,}000 \times 200 \times \$3.00/\text{MTok} = \$0.60$
- **合計**: $\$10.50/\text{日}$（**約65.7%削減**）

AWS公式ドキュメントでは、レイテンシ最大85%改善・コスト最大90%削減と報告されているが、これは100Kトークンの書籍対話など大規模プレフィックスの理想的条件での値である。

## 運用での学び（Production Lessons）

### 最小トークン閾値への対応

モデルによって最小キャッシュ可能トークン数が異なる（1,024〜4,096トークン）。短いシステムプロンプト単体ではキャッシュ閾値に達しない場合がある。対策として、ツール定義とシステムプロンプトを合わせて閾値を超えるよう設計する。

### キャッシュヒット率のモニタリング

本番環境ではキャッシュヒット率の継続的監視が重要である：

```python
import logging
from typing import Any

logger = logging.getLogger(__name__)

def log_cache_metrics(response: dict[str, Any], session_id: str) -> None:
    """キャッシュメトリクスを構造化ログに記録

    Args:
        response: Bedrock APIレスポンス
        session_id: セッション識別子
    """
    usage = response.get("usage", {})
    cache_read = usage.get("cacheReadInputTokens", 0)
    cache_write = usage.get("cacheWriteInputTokens", 0)
    total_input = usage.get("inputTokens", 0) + cache_read

    logger.info(
        "bedrock_cache_metrics",
        extra={
            "event": "bedrock_cache_metrics",
            "session_id": session_id,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "uncached_tokens": usage.get("inputTokens", 0),
            "cache_hit_rate": cache_read / max(total_input, 1),
        },
    )
```

### Cross-Region Inferenceとの互換性

AWS公式ドキュメントでは、プロンプトキャッシュがCross-Region Inference（クロスリージョン推論）と互換であることが記載されている。ただし、高負荷時にはキャッシュ書込が増加する可能性があるため、コスト監視が重要である。

### キャッシュチェックポイントの安定性

キャッシュヒットにはプレフィックスの100%完全一致が必要である。以下の変更はキャッシュを無効化する：

- ツール定義の追加・削除・変更
- システムプロンプトの文言変更
- メッセージ内容の変更（キャッシュチェックポイントより前の部分）
- TTL値の変更

LangGraphエージェントでは、ツール定義を動的に変更するとキャッシュが毎回無効化される。Anthropicの設計指針と同様に、全ツールを常にリクエストに含めてプレフィックスを安定させることが推奨される。

## 学術研究との関連（Academic Connection）

Amazon Bedrockのプロンプトキャッシュは、以下の学術研究の商用実装と位置づけられる：

- **CachedAttention (arXiv:2407.01928)**: マルチターン会話のKVキャッシュをCPUにオフロード・再利用する手法。Bedrockのキャッシュ機能はこの概念をマネージドサービスとして提供
- **Don't Break the Cache (arXiv:2601.06007)**: プロンプトキャッシュヒット率を最大化するための設計原則。BedrockのConverse APIにおける簡易キャッシュ管理はこの論文のStatic-before-Dynamic原則と一致
- **RAGCache (arXiv:2404.14294)**: RAGシステム向けのチャンク単位KVキャッシュ。Bedrockのプレフィックスベースキャッシュとは異なるアプローチだが、RAGコンテキストのキャッシュという目的は共通

## Anthropic API直接利用との比較

| 項目 | Anthropic API直接 | Amazon Bedrock |
|------|-------------------|----------------|
| キャッシュ方式 | 自動/明示的ブレークポイント | cachePoint / cache_control |
| 最大チェックポイント | 4 | 4 |
| TTL | 5分 / 1時間 | 5分 / 1時間（一部モデル） |
| Novaモデル対応 | なし | あり |
| Converse API | なし | あり（統一インターフェース） |
| VPC Endpoint | なし | あり（PrivateLink対応） |
| IAM統合 | なし | あり（ロールベースアクセス制御） |
| レート制限 | 組織単位 | アカウント・モデル単位 |
| キャッシュヒット時のレート制限消費 | あり | なし（公式ドキュメント記載） |

Bedrockの特筆すべき利点として、**キャッシュヒット時にレート制限が消費されない**点が挙げられる。高トラフィック環境ではキャッシュヒット率の向上がレート制限の実質的な拡大にもつながる。

## まとめと実践への示唆

Amazon Bedrockのプロンプトキャッシュは、AWSエコシステム内でLLMアプリケーションの推論コストとレイテンシを大幅に削減する機能である。Converse APIの統一インターフェースにより、Claude・Nova間でのモデル切替時もキャッシュ実装を共通化できる点が実務上の利点である。

実践上の要点は以下に集約される：
1. **静的コンテンツ（ツール定義・システムプロンプト）を先頭に配置**し、キャッシュチェックポイントで区切る
2. **5分TTLと1時間TTLを使い分け**、エージェントワークフローでは1時間TTL対応モデル（Claude Opus 4.5/Sonnet 4.5/Haiku 4.5）を選択する
3. **キャッシュメトリクス（cacheReadInputTokens / cacheWriteInputTokens）を監視**し、ヒット率80%以上を目標とする
4. **ツール定義を動的に変更しない**設計でプレフィックスの安定性を確保する

## 参考文献

- **AWS Docs**: [https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- **AWS What's New (1-hour TTL)**: [https://aws.amazon.com/about-aws/whats-new/2026/01/amazon-bedrock-one-hour-duration-prompt-caching/](https://aws.amazon.com/about-aws/whats-new/2026/01/amazon-bedrock-one-hour-duration-prompt-caching/)
- **AWS Blog**: [https://aws.amazon.com/blogs/machine-learning/effectively-use-prompt-caching-on-amazon-bedrock/](https://aws.amazon.com/blogs/machine-learning/effectively-use-prompt-caching-on-amazon-bedrock/)
- **Bedrock Pricing**: [https://aws.amazon.com/bedrock/pricing/](https://aws.amazon.com/bedrock/pricing/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/555a4e799660de](https://zenn.dev/0h_n0/articles/555a4e799660de)
