---
layout: post
title: "OpenAI解説: Responses API・Conversations API・Compaction APIの設計思想と実装パターン"
description: "Assistants API廃止に伴うResponses/Conversations/Compaction APIの技術的設計と移行パターンを解説"
categories: [blog, tech_blog]
tags: [OpenAI, API-design, chatbot, context-management, openai]
date: 2026-07-18 11:00:00 +0900
source_type: tech_blog
source_domain: developers.openai.com
source_url: https://developers.openai.com/blog/openai-for-developers-2025
zenn_article: c94e21f061ebbb
zenn_url: https://zenn.dev/0h_n0/articles/c94e21f061ebbb
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [OpenAI for Developers in 2025](https://developers.openai.com/blog/openai-for-developers-2025) および関連する公式ドキュメント（[Conversation State](https://developers.openai.com/api/docs/guides/conversation-state)、[Compaction](https://developers.openai.com/api/docs/guides/compaction)）の解説記事です。

## ブログ概要（Summary）

OpenAIは2025年3月にResponses APIを発表し、同年8月26日にAssistants API（ベータ）の廃止を通告した。廃止期限は2026年8月26日である。Responses APIはChat Completionsのシンプルさ とAssistants APIのツール使用能力を統合し、Conversations APIで永続的な会話状態管理、Compaction APIでコンテキスト圧縮を提供する。この記事では、これら3つのAPIの設計思想、技術的詳細、および移行パターンを解説する。

この記事は [Zenn記事: Assistants API Thread移行実践：肥大化対策からConversations API再設計まで](https://zenn.dev/0h_n0/articles/c94e21f061ebbb) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（OpenAI公式）
- **URL**: [https://developers.openai.com/blog/openai-for-developers-2025](https://developers.openai.com/blog/openai-for-developers-2025)
- **組織**: OpenAI
- **発表日**: 2025年

## 技術的背景（Technical Background）

### Assistants APIの設計上の課題

Assistants APIは2023年11月のDevDayで発表され、以下の設計上の課題を抱えていた。

1. **非同期ポーリングモデル**: Run作成後、完了まで`create_and_poll`でポーリングする必要があり、レイテンシとコードの複雑性が増大
2. **Thread状態の不透明性**: Thread内のメッセージ数・トークン数を把握しにくく、コスト予測が困難
3. **Truncation Strategyの限界**: `auto`モードはブラックボックス、`last_messages`は単純な直近N件切り詰めのみ
4. **100,000メッセージ制限**: 大規模だが有限であり、長期運用でのデータ管理に課題

### Responses APIの設計原則

OpenAIの公式ドキュメントによると、Responses APIは以下の設計原則で再設計された。

- **同期的レスポンス**: リクエスト→レスポンスの直接的なフローで、ポーリング不要
- **Items統合**: メッセージだけでなくツールコール・ツール出力・Compaction Itemを含む型付き出力
- **状態管理の選択制**: stateless（毎回全コンテキストを送信）、`previous_response_id`チェーン、Conversations APIの3つから選択

## 実装アーキテクチャ（Architecture）

### 3つの状態管理パターン

```mermaid
graph TD
    subgraph Pattern1[パターン1: Stateless]
        A1[クライアント] -->|全メッセージ配列| B1[/responses]
        B1 -->|output items| A1
    end

    subgraph Pattern2[パターン2: Response Chaining]
        A2[クライアント] -->|previous_response_id + 新メッセージ| B2[/responses]
        B2 -->|response_id + output| A2
    end

    subgraph Pattern3[パターン3: Conversations API]
        A3[クライアント] -->|conversation_id + input| B3[/responses]
        B3 -->|output items → Conversation格納| C3[Conversation]
        C3 -.->|永続化| D3[OpenAIストレージ]
    end
```

各パターンの特性比較は以下の通りである。

| 項目 | Stateless | Response Chaining | Conversations API |
|------|-----------|-------------------|-------------------|
| 状態管理 | クライアント側 | OpenAI側（30日TTL） | OpenAI側（無期限） |
| コンテキスト制御 | 完全制御 | 限定的 | Compactionで制御 |
| データ保持 | なし | 30日（`store: false`で無効化可） | 無期限 |
| ユースケース | バッチ処理、プライバシー重視 | 短期会話、プロトタイプ | 長期顧客対応 |
| ZDR互換性 | `store: false`で対応 | `store: false`で対応 | 対応（Items永続化） |

### Conversations APIの内部構造

公式ドキュメントによると、Conversations APIは以下の特性を持つ。

**Itemsの構成**: ConversationはMessagesだけでなく、以下のアイテムを格納する。

```python
from openai import OpenAI

client = OpenAI()

conversation = client.conversations.create(
    metadata={"user_id": "usr_123", "session_type": "support"},
    items=[
        {
            "type": "message",
            "role": "system",
            "content": "あなたはカスタマーサポートエージェントです。",
        },
        {
            "type": "message",
            "role": "user",
            "content": "注文のステータスを教えてください。",
        },
    ],
)
```

**アイテム追加制限**: 公式ドキュメントによれば、一度に追加可能なアイテムは最大20件である。大量メッセージのThread移行時にはバッチ処理が必要となる。

**TTLの非適用**: Response単体は30日TTLが適用されるが、Conversationに紐づいたアイテムにはTTLが適用されない。OpenAIの公式ドキュメントには「Conversation objects and items in them are not subject to the 30 day TTL」と明記されている。

### Compaction APIの技術設計

Compaction APIは2つのモードで提供される。

**サーバーサイドCompaction**: `context_management`パラメータで`compact_threshold`を設定すると、レンダリング後のトークン数が閾値を超えた時点で自動的にCompactionが実行される。

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": "プロジェクトの進捗を報告します..."}],
    conversation={"id": "conv_abc123"},
    context_management=[
        {
            "type": "compaction",
            "compact_threshold": 50_000,
        }
    ],
)
```

**スタンドアロンCompaction**: `/responses/compact`エンドポイントを直接呼び出し、完全にstatelessで圧縮を実行する。

```python
from openai import OpenAI

client = OpenAI()

compacted = client.responses.compact(
    model="gpt-4.1",
    input=conversation_items,
)
```

公式ドキュメントには「OpenAIの最新モデルは、事前の会話状態を分析し、暗号化されたトークン効率の高い表現として重要な状態を保持するCompaction Itemを生成するよう訓練されている」と記載されている。

**Compaction Itemの特性**:

- **暗号化**: 内容を直接読み取ることはできない
- **不可分**: 公式ドキュメントによれば「出力をプルーニングしてはならない。返されたウィンドウが次のコンテキストウィンドウの正規の表現」とされている
- **トークン効率**: 元の会話履歴よりも少ないトークンで同等の文脈を保持

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

Conversations API + Compaction APIを用いた会話システムのAWS構成を示す。

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $60-180 | Lambda + OpenAI API + DynamoDB |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $400-1,000 | ECS Fargate + OpenAI API + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,500-6,000 | EKS + OpenAI API + RDS PostgreSQL |

**コスト試算の注意事項**: 上記は2026年7月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です。OpenAI APIコスト（GPT-4.1: $2.00/1M入力トークン、$8.00/1M出力トークン）は別途必要です。Compaction APIの使用によりOpenAI API側のコストは30-60%削減が見込めます。最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください。

### Terraformインフラコード

**Small構成 (Serverless): Lambda + DynamoDB（会話ID管理）**

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "conversations-api-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["ap-northeast-1a", "ap-northeast-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]

  enable_nat_gateway   = false
  enable_dns_hostnames = true
}

resource "aws_iam_role" "lambda_conversations" {
  name = "lambda-conversations-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_lambda_function" "conversation_handler" {
  filename      = "lambda.zip"
  function_name = "conversation-api-handler"
  role          = aws_iam_role.lambda_conversations.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 60
  memory_size   = 512

  environment {
    variables = {
      OPENAI_API_KEY_SECRET = aws_secretsmanager_secret.openai_key.arn
      DYNAMODB_TABLE        = aws_dynamodb_table.conversation_mapping.name
      COMPACT_THRESHOLD     = "50000"
    }
  }
}

resource "aws_dynamodb_table" "conversation_mapping" {
  name         = "user-conversation-mapping"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "user_id"

  attribute {
    name = "user_id"
    type = "S"
  }
}

resource "aws_secretsmanager_secret" "openai_key" {
  name = "openai-api-key"
}

resource "aws_cloudwatch_metric_alarm" "api_cost_monitor" {
  alarm_name          = "openai-api-cost-spike"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = 3600
  statistic           = "Sum"
  threshold           = 120000
  alarm_description   = "OpenAI API呼び出し時間異常（コスト急増の可能性）"

  dimensions = {
    FunctionName = aws_lambda_function.conversation_handler.function_name
  }
}
```

### 運用・監視設定

```sql
-- CloudWatch Logs Insights: Compaction発動頻度の監視
fields @timestamp, conversation_id, compaction_triggered, token_count_before, token_count_after
| filter compaction_triggered = true
| stats count(*) as compaction_count,
        avg(token_count_before) as avg_before,
        avg(token_count_after) as avg_after
  by bin(1h)

-- 会話ごとのトークン使用量推移
fields @timestamp, conversation_id, prompt_tokens, completion_tokens
| stats sum(prompt_tokens + completion_tokens) as total_tokens by conversation_id
| sort total_tokens desc
| limit 20
```

### コスト最適化チェックリスト

- [ ] ~100 req/日 → Lambda + OpenAI API (Serverless) - $60-180/月
- [ ] ~1000 req/日 → ECS Fargate + OpenAI API (Hybrid) - $400-1,000/月
- [ ] 10000+ req/日 → EKS + OpenAI API (Container) - $2,500-6,000/月
- [ ] Compaction API: `compact_threshold`をモデルとコスト目標に合わせて設定
- [ ] Response Caching: `store: false`でプライバシー要件に対応
- [ ] OpenAI Batch API: 非リアルタイム処理で50%割引
- [ ] モデル選択: 簡易タスクにGPT-4.1-mini（$0.40/1M入力）、複雑タスクにGPT-4.1（$2.00/1M入力）
- [ ] `max_output_tokens`設定で過剰生成防止
- [ ] AWS Budgets: 月額予算設定（80%で警告）
- [ ] CloudWatch: Compaction発動頻度とトークン使用量の監視
- [ ] Cost Anomaly Detection: 自動異常検知
- [ ] 日次コストレポート: SNS/Slack通知
- [ ] DynamoDB: TTL設定で古い会話マッピングを自動削除
- [ ] Lambda: メモリとタイムアウトの最適化
- [ ] Secrets Manager: OpenAI APIキーの安全な管理
- [ ] EC2 Spot Instances優先（EKS構成時）
- [ ] Reserved Instances購入検討（1年コミットで72%削減）
- [ ] タグ戦略: 環境別でコスト可視化
- [ ] 未使用リソース削除: Trusted Advisor活用
- [ ] 開発環境は夜間停止（Auto Start/Stop）

## パフォーマンス最適化（Performance）

**Responses API vs Assistants APIのレイテンシ比較**:

公式ドキュメントの情報に基づくと、Responses APIの同期モデルはAssistants APIの非同期ポーリングモデルに比べてレイテンシが削減される。ポーリング間隔（通常1-5秒）が不要になるためである。

**Compactionのレイテンシ影響**:

サーバーサイドCompactionが発動した場合、圧縮処理のための追加レイテンシが発生する。`compact_threshold`を適切に設定し、Compactionの発動頻度を制御することが重要である。

- 閾値が低すぎる場合: 頻繁なCompactionで平均レイテンシが増加
- 閾値が高すぎる場合: コンテキストが大きくなりすぎ、推論レイテンシが増加

## 運用での学び（Production Lessons）

**移行の複雑度**: OpenAIコミュニティの報告によれば、単純なチャットボットの移行は1-4エンジニア週だが、マルチテナント本番システム（Thread履歴、カスタムAssistant設定、複雑なツールオーケストレーション）では数ヶ月の作業が必要とされている。

**よくある移行問題**:

| 問題 | 原因 | 対処法 |
|------|------|-------|
| 過去文脈の喪失 | メッセージ移行漏れ | `order="asc"`で全件取得、ページネーション確認 |
| Compaction後の品質低下 | `compact_threshold`が低すぎる | 段階的に閾値を引き上げ、A/Bテスト |
| ファイル参照エラー | Assistants APIのファイルIDが無効 | Files APIで再アップロード |
| レート制限 | 大量Thread一括移行 | `asyncio.Semaphore`で並行数制御 |

**`tool_resources`の移行**: Assistants APIの`tool_resources`（Code Interpreter、File Searchのベクトルストア）はThread・Assistantレベルで設定されており、Conversations APIではResponses APIのツール定義として再構成が必要である。特にFile Searchのベクトルストアは、Vector Store APIで再作成する必要がある。

## 学術研究との関連（Academic Connection）

Conversations APIとCompaction APIの設計は、以下の学術研究と接点がある。

- **MemGPT** (Packer et al., 2023): OS仮想メモリ着想のメモリ管理。Conversations APIのItems格納は、MemGPTのRecall Storageに対応する
- **ACON** (Kang et al., 2025): コンテキスト圧縮最適化。Compaction APIの自動圧縮は、ACONの履歴圧縮$C_{\text{hist}}$に対応する
- **Verbatim Chunks** (An, 2025): 生テキスト保持vs構造化抽出の比較。Compaction APIが暗号化圧縮を採用した理由を示唆する

## まとめと実践への示唆

OpenAIのResponses API + Conversations API + Compaction APIは、Assistants APIのThread管理が抱えていたトークンコスト累積問題に対する包括的な解決策を提供する。Compaction APIによるサーバーサイド圧縮は、開発者が圧縮ロジックを実装する負担を軽減しつつ、長期会話のコスト制御を可能にする。2026年8月26日の廃止期限に向けて、段階的な移行計画の策定が推奨される。

## 参考文献

- **Blog URL**: [https://developers.openai.com/blog/openai-for-developers-2025](https://developers.openai.com/blog/openai-for-developers-2025)
- **Conversation State Guide**: [https://developers.openai.com/api/docs/guides/conversation-state](https://developers.openai.com/api/docs/guides/conversation-state)
- **Compaction Guide**: [https://developers.openai.com/api/docs/guides/compaction](https://developers.openai.com/api/docs/guides/compaction)
- **Migration Guide**: [https://developers.openai.com/api/docs/assistants/migration](https://developers.openai.com/api/docs/assistants/migration)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/c94e21f061ebbb](https://zenn.dev/0h_n0/articles/c94e21f061ebbb)
