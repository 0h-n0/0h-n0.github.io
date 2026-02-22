---
layout: post
title: "Google解説: Gemini 3 API主要アップデート — thinking_level・media_resolution・Thought Signaturesの技術詳細"
description: "Gemini 3のAPI新機能（thinking_level推論深度制御、media_resolutionトークン最適化、Thought Signatures推論チェーン維持）を技術的に解説"
categories: [blog, tech_blog]
tags: [Gemini, multimodal, API, thinking, Google, LLM]
date: 2026-02-22 10:00:00 +0900
source_type: tech_blog
source_domain: developers.googleblog.com
source_url: https://developers.googleblog.com/new-gemini-api-updates-for-gemini-3/
zenn_article: df5295d69a456f
zenn_url: https://zenn.dev/0h_n0/articles/df5295d69a456f
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Google Developers Blog: New Gemini API updates for Gemini 3](https://developers.googleblog.com/new-gemini-api-updates-for-gemini-3/) の解説記事です。

## ブログ概要（Summary）

Google Developers Blogが公開したGemini 3 APIの主要アップデート記事では、推論深度を制御する`thinking_level`パラメータ、画像・動画・ドキュメント入力のトークン使用量を調整する`media_resolution`パラメータ、そしてマルチターン会話で推論チェーンを維持する`Thought Signatures`の3つの新機能が紹介されている。これらはGemini 3ファミリー（3 Flash, 3 Pro, 3.1 Pro）で利用可能であり、開発者がコスト・レイテンシ・精度のトレードオフを細粒度で制御するための仕組みとして設計されている。

この記事は [Zenn記事: Gemini 3.1 Pro マルチモーダルAPI実践ガイド：画像・音声・動画をPythonで統合処理する](https://zenn.dev/0h_n0/articles/df5295d69a456f) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://developers.googleblog.com/new-gemini-api-updates-for-gemini-3/](https://developers.googleblog.com/new-gemini-api-updates-for-gemini-3/)
- **組織**: Google Developers
- **発表日**: 2026年2月

## 技術的背景（Technical Background）

マルチモーダルLLMのAPI設計において、従来は`temperature`や`top_p`といった生成パラメータでモデルの出力品質を制御するのが一般的であった。しかし、Gemini 3ファミリーでは「推論深度」という新しい制御軸が導入された。これは、OpenAI o1やClaude 3.5 Sonnetの Extended Thinking に類似するアプローチであるが、Gemini 3ではAPI側でレベルを明示的に指定する点が特徴的である。

従来のGemini 2.x系では、推論の深さはモデル側で自動判断されており、開発者が直接コントロールする手段は限られていた。`temperature`の調整は生成のランダム性を制御するものであり、推論の「深さ」とは本質的に異なる。Gemini 3の`thinking_level`は、この課題に対するGoogleの回答といえる。

ブログによれば、Gemini 3では「デフォルト温度が1.0で最適化」されており、従来のように`temperature=0.2`などの低温設定を行うと、推論タスクでループや品質低下が発生する場合がある。この設計変更は、推論制御を`temperature`から`thinking_level`に移行させるという意図を反映している。

## 実装アーキテクチャ（Architecture）

### thinking_level パラメータの設計

`thinking_level`は3段階（HIGH / MEDIUM / LOW）で推論深度を制御する。ブログでは「Gemini 3 treats these levels as relative guidelines for reasoning rather than strict token guarantees（Gemini 3はこれらのレベルを厳密なトークン保証ではなく、推論の相対的なガイドラインとして扱う）」と記述されている。

この設計は、推論トークン数の「保証」ではなく「ヒント」として機能することを意味する。モデルはタスクの実際の複雑さに応じて推論トークン数を動的に調整する可能性がある。

```python
from google import genai
from google.genai import types

client = genai.Client()

# HIGH: 複雑な多段階推論（コード脆弱性分析、戦略的意思決定）
response_high = client.models.generate_content(
    model="gemini-3.1-pro-preview",
    contents="このPythonコードのセキュリティ脆弱性を網羅的に分析してください",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
    ),
)

# LOW: 単純なタスク（分類、抽出、フォーマット変換）
response_low = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="このテキストのカテゴリを{tech, business, science}から選択",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
    ),
)
```

**推論深度とコストの関係**:

| thinking_level | 想定ユースケース | 推論トークン消費 | レイテンシ |
|---------------|----------------|----------------|-----------|
| HIGH | コード脆弱性分析、複雑な推論チェーン | 多い（数千〜数万トークン） | 高い |
| MEDIUM | 要約、翻訳、一般的なQA | 中程度 | 中程度 |
| LOW | 分類、抽出、構造化データ変換 | 少ない | 低い |

### media_resolution パラメータの設計

画像・動画・ドキュメント入力のトークン使用量を3段階（HIGH / MEDIUM / LOW）で制御する。これはマルチモーダル入力のコスト最適化において重要な機能である。

```python
# 高解像度: OCR、設計図の詳細分析
config_high = types.GenerateContentConfig(
    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
)

# 低解像度: 全体的なシーン把握、サムネイル分類
config_low = types.GenerateContentConfig(
    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
)
```

**トークン消費量の比較**:

| 解像度 | 画像あたりのトークン | 動画（1秒あたり） | 想定コスト比率 |
|--------|-------------------|------------------|--------------|
| HIGH | 最大約2,000 | 約300 | 1.0x（基準） |
| MEDIUM | 約770 | 約200 | 0.4x |
| LOW | 約258 | 約100 | 0.13x |

LOW設定を使えば、HIGHと比較して約87%のトークン削減が可能である。ただし、細かいテキスト認識や微小オブジェクトの検出精度は低下する。

### Thought Signatures

Thought Signaturesは、マルチターン会話においてモデルの内部推論過程を暗号化して保持する仕組みである。具体的には以下の特性を持つ：

1. **推論チェーンの維持**: 複数ターンにわたるやり取りで、前回の推論プロセスをコンテキストとして再利用できる
2. **暗号化された表現**: 内部推論は暗号化されており、開発者が直接内容を確認することはできない
3. **厳密な検証**: 関数呼び出し（Function Calling）と画像生成では、Thought Signaturesの検証が厳密に適用される。署名が不正または欠落している場合、400エラー（Bad Request）が返される

この仕組みにより、エージェント型ワークフローにおいて推論の一貫性を保証できる。ただし、暗号化された推論内容のデバッグは困難であり、開発者は入力と出力のペアから間接的に推論品質を評価する必要がある。

```python
# マルチターン会話でのThought Signatures活用例
chat = client.chats.create(model="gemini-3.1-pro-preview")

# 1ターン目: 推論開始
response1 = chat.send_message(
    "Pythonでマージソートを実装してください",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
    ),
)

# 2ターン目: 前回の推論を引き継いで最適化を議論
# Thought Signaturesにより推論チェーンが維持される
response2 = chat.send_message(
    "このマージソートのメモリ使用量を最適化する方法を提案してください",
)
```

## パフォーマンス最適化（Performance）

### thinking_levelによるコスト削減効果

ブログの記述に基づくと、タスク複雑度に応じて`thinking_level`を適切に選択することで、APIコストを大幅に削減できる。

**コスト試算例（Gemini 3.1 Pro, $2.00/Mトークン入力 + $12.00/Mトークン出力）**:

| シナリオ | thinking_level | 推定出力トークン | 月間1万リクエスト時コスト |
|---------|---------------|----------------|----------------------|
| 文書分類 | LOW | 約50トークン | 約$6 |
| 記事要約 | MEDIUM | 約500トークン | 約$60 |
| コード分析 | HIGH | 約5,000トークン | 約$600 |

LOWとHIGHの間で約100倍のコスト差が生じる可能性がある。ブログではthinking_levelが「strict token guarantees」ではないと説明されているため、実際のトークン消費はタスクの内容に依存する。

### media_resolutionによる動画処理コスト削減

10分の動画を処理する場合の試算：

```
HIGH: 300トークン/秒 × 600秒 = 180,000トークン → $0.36
MEDIUM: 200トークン/秒 × 600秒 = 120,000トークン → $0.24
LOW: 100トークン/秒 × 600秒 = 60,000トークン → $0.12
```

LOWを選択することで、HIGHと比較して約67%のコスト削減が可能である。

## 運用での学び（Production Lessons）

### Grounding with Google Searchの価格変更

ブログによると、Grounding with Google Searchの価格体系が従量課金制（1,000クエリあたり$14）に変更された。これは、大量のグラウンディングクエリを発行するアプリケーションにおいてコスト予測を可能にする変更である。

従来の価格体系からの変更点として、構造化出力（Structured Output）との組み合わせが可能になった点も重要である。これにより、検索結果をJSON形式で取得し、パイプラインの後続処理に直接渡すことが容易になる。

### エラーハンドリングの注意点

Thought Signaturesに関して、関数呼び出し（Function Calling）時にはSignatureの検証が厳密に行われる。署名の不一致は400エラーとなるため、マルチターン会話の状態管理には注意が必要である。特に以下のケースでエラーが発生しやすい：

1. **会話履歴の不完全な引き渡し**: 前ターンのレスポンスを省略してメッセージを送信した場合
2. **異なるモデル間での会話継続**: Gemini 3 FlashからGemini 3.1 Proに切り替えた場合
3. **タイムアウト後の再送**: セッションタイムアウト後に古い会話コンテキストを使用した場合

## 学術研究との関連（Academic Connection）

Gemini 3のthinking_level設計は、LLMにおける「System 1 / System 2」思考フレームワーク（Kahneman, 2011）を実装に落とし込んだものと解釈できる。

System 1（直感的・高速）は`thinking_level="LOW"`に、System 2（分析的・低速）は`thinking_level="HIGH"`に対応する。この二重プロセス理論のAPI化は、OpenAI o1のチェーン・オブ・ソート推論やAnthropicのExtended Thinkingと同様のアプローチであるが、Gemini 3ではユーザーが明示的にレベルを指定できる点で異なる。

また、`media_resolution`によるトークン消費量制御は、Vision Transformer（ViT）における画像パッチサイズの選択と概念的に関連する。低解像度設定は大きなパッチサイズに、高解像度設定は小さなパッチサイズに相当し、処理する情報量とコストのトレードオフを制御している。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

Gemini 3 APIを本番環境で運用する際のAWS構成を、トラフィック量別に示す。

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|-------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $80-200 | Lambda + API Gateway + DynamoDB |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $400-1,000 | Lambda + ECS Fargate + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,500-6,000 | EKS + Karpenter + EC2 Spot |

**Small構成の詳細** (月額$80-200):
- **Lambda**: 1GB RAM, 60秒タイムアウト ($30/月)
- **Gemini API**: thinking_level=LOW中心、$2/Mトークン ($100/月)
- **DynamoDB**: On-Demand, プロンプトキャッシュ ($10/月)
- **API Gateway**: REST API ($10/月)
- **CloudWatch**: 基本監視 ($5/月)

**コスト削減テクニック**:
- thinking_level適切選択で推論コスト最大90%削減
- media_resolution=LOW選択でマルチモーダルコスト87%削減
- DynamoDBでプロンプトキャッシュ（同一プロンプトの再処理防止）
- Gemini 3 Flash ($0.50/M) と 3.1 Pro ($2.00/M) のルーティング

**コスト試算の注意事項**:
- 上記は2026年2月時点のGoogle Gemini API料金およびAWS ap-northeast-1（東京）リージョン料金に基づく概算値です
- Gemini API料金はGoogleの料金改定により変動する可能性があります
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) および [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing) で確認してください

### Terraformインフラコード

**Small構成 (Serverless): Lambda + API Gateway + DynamoDB**

```hcl
# --- IAMロール（Gemini API呼び出し用） ---
resource "aws_iam_role" "lambda_gemini" {
  name = "lambda-gemini-api-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_permissions" {
  role = aws_iam_role.lambda_gemini.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:Query"]
        Resource = aws_dynamodb_table.prompt_cache.arn
      },
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = aws_secretsmanager_secret.gemini_api_key.arn
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# --- Secrets Manager（APIキー管理） ---
resource "aws_secretsmanager_secret" "gemini_api_key" {
  name = "gemini-api-key"
}

# --- Lambda関数 ---
resource "aws_lambda_function" "gemini_handler" {
  filename      = "lambda.zip"
  function_name = "gemini-api-handler"
  role          = aws_iam_role.lambda_gemini.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 60
  memory_size   = 1024

  environment {
    variables = {
      GEMINI_SECRET_ARN  = aws_secretsmanager_secret.gemini_api_key.arn
      DYNAMODB_TABLE     = aws_dynamodb_table.prompt_cache.name
      DEFAULT_MODEL      = "gemini-3-flash-preview"
      DEFAULT_THINKING   = "MEDIUM"
    }
  }
}

# --- DynamoDB（プロンプトキャッシュ） ---
resource "aws_dynamodb_table" "prompt_cache" {
  name         = "gemini-prompt-cache"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "prompt_hash"

  attribute {
    name = "prompt_hash"
    type = "S"
  }

  ttl {
    attribute_name = "expire_at"
    enabled        = true
  }
}

# --- CloudWatch アラーム ---
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "gemini-lambda-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "Lambda エラー率異常"
  dimensions = {
    FunctionName = aws_lambda_function.gemini_handler.function_name
  }
}
```

### 運用・監視設定

**CloudWatch Logs Insights クエリ**:

```sql
-- Gemini APIレスポンスタイム分析
fields @timestamp, thinking_level, response_time_ms, token_count
| stats avg(response_time_ms) as avg_latency,
        pct(response_time_ms, 95) as p95,
        sum(token_count) as total_tokens
  by thinking_level, bin(1h)

-- thinking_level別コスト異常検知
fields @timestamp, thinking_level, output_tokens
| filter thinking_level = "HIGH"
| stats sum(output_tokens) as high_tokens by bin(1h)
| filter high_tokens > 500000
```

**コスト監視アラーム（Python）**:

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Gemini API呼び出し回数の急増アラート
cloudwatch.put_metric_alarm(
    AlarmName='gemini-api-call-spike',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='GeminiAPICalls',
    Namespace='Custom/GeminiAPI',
    Period=3600,
    Statistic='Sum',
    Threshold=1000,
    AlarmDescription='Gemini API呼び出し数が1時間あたり1000回を超過'
)
```

### コスト最適化チェックリスト

**thinking_level最適化**:
- [ ] 分類・抽出タスクは`LOW`に設定
- [ ] 要約・翻訳タスクは`MEDIUM`に設定
- [ ] 複雑な推論のみ`HIGH`に設定
- [ ] デフォルト値を`MEDIUM`に設定し、必要時のみ変更

**media_resolution最適化**:
- [ ] 全体把握のみの場合は`LOW`に設定
- [ ] 一般的な画像分析は`MEDIUM`（デフォルト）
- [ ] OCR・細字認識のみ`HIGH`に設定
- [ ] 動画の全体要約は`LOW`で十分

**モデルルーティング**:
- [ ] 単純タスクはGemini 3 Flash ($0.50/M)
- [ ] 中程度タスクはGemini 3 Pro ($2.00/M)
- [ ] 複雑推論のみGemini 3.1 Pro ($2.00/M)
- [ ] ルーティングロジックの自動化

**監視・アラート**:
- [ ] thinking_level別のトークン使用量監視
- [ ] media_resolution別のコスト追跡
- [ ] Thought Signatures検証エラーの監視
- [ ] 月間API予算アラート設定

**リソース管理**:
- [ ] DynamoDBキャッシュのTTL設定（24時間推奨）
- [ ] Lambda同時実行数の制限
- [ ] CloudWatch Logsの保持期間設定
- [ ] 不要なAPIキーのローテーション

## まとめと実践への示唆

Google Developers Blogで紹介されたGemini 3 APIの主要アップデートは、マルチモーダルLLMのAPI設計における重要な方向性を示している。`thinking_level`による推論深度の明示的制御、`media_resolution`によるマルチモーダル入力のコスト最適化、`Thought Signatures`によるマルチターン推論の一貫性保証という3つの機能は、いずれもプロダクション運用における実践的な課題に対するソリューションである。

特に`thinking_level`は、従来の`temperature`パラメータでは実現できなかった「推論の深さ」の制御を可能にしており、タスク複雑度に応じた適切なレベル選択により、コストとレイテンシの大幅な最適化が期待できる。

## 参考文献

- **Blog URL**: [https://developers.googleblog.com/new-gemini-api-updates-for-gemini-3/](https://developers.googleblog.com/new-gemini-api-updates-for-gemini-3/)
- **Gemini API Docs**: [https://ai.google.dev/gemini-api/docs/gemini-3](https://ai.google.dev/gemini-api/docs/gemini-3)
- **Gemini API Pricing**: [https://ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/df5295d69a456f](https://zenn.dev/0h_n0/articles/df5295d69a456f)
