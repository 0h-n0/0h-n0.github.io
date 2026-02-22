---
layout: post
title: "Care Accessが Amazon Bedrock プロンプトキャッシュで86%コスト削減を達成した事例解説"
description: "医療系企業Care Accessが電子カルテ処理にBedrock Prompt Cachingを適用し、86%のコスト削減と66%の処理速度向上を達成した実装事例"
categories: [blog, tech_blog]
tags: [aws, bedrock, prompt-caching, rag, cost-optimization, healthcare, python, anthropic]
date: 2026-02-22 09:00:00 +0900
source_type: tech_blog
source_domain: aws.amazon.com
source_url: https://aws.amazon.com/blogs/machine-learning/how-care-access-achieved-86-data-processing-cost-reductions-and-66-faster-data-processing-with-amazon-bedrock-prompt-caching/
zenn_article: d027acf4081b9d
zenn_url: https://zenn.dev/0h_n0/articles/d027acf4081b9d
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

Care Accessは医療スクリーニング参加者の健康リソース（臨床試験を含む）のマッチングを行う企業である。AWSとの協力により、Amazon Bedrockを使った医療記録レビューソリューションを6週間で立ち上げた。電子カルテ（EHR）の処理にBedrock Prompt Cachingを適用した結果、データ処理コストを86%削減し、処理速度を66%向上させたと報告されている。

この記事は [Zenn記事: Bedrock AgentCore×1時間キャッシュで社内RAGコスト90%削減](https://zenn.dev/0h_n0/articles/d027acf4081b9d) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（AWS Machine Learning Blog）
- **URL**: [https://aws.amazon.com/blogs/machine-learning/how-care-access-achieved-86-data-processing-cost-reductions-and-66-faster-data-processing-with-amazon-bedrock-prompt-caching/](https://aws.amazon.com/blogs/machine-learning/how-care-access-achieved-86-data-processing-cost-reductions-and-66-faster-data-processing-with-amazon-bedrock-prompt-caching/)
- **組織**: AWS / Care Access
- **発表日**: 2025年11月20日

## 技術的背景（Technical Background）

### 課題: 電子カルテの大量処理

医療分野では、患者1人あたりの電子カルテ（EHR: Electronic Health Record）が数千〜数万トークンに達する。Care Accessのユースケースでは、同一の医療記録に対して複数の分析質問（臨床試験適格性チェック、疾患スクリーニング等）を実行する必要がある。

従来のアプローチでは、各質問ごとに医療記録全体をLLMに送信していた。これは以下の問題を引き起こす。

1. **コストの非効率性**: 同一の医療記録を質問の数だけ繰り返し入力トークンとして課金される
2. **処理速度の低下**: 大量トークンの処理が毎回発生し、Time-to-First-Token（TTFT）が長い
3. **スケーラビリティの制約**: 記録取り込みがスパイクした際にコストが予測不能に増大

### Prompt Cachingによる解決

Bedrock Prompt Cachingは、静的なコンテキスト（医療記録本文）をキャッシュし、動的な部分（分析質問）のみを各リクエストで送信する仕組みを提供する。Bedrockにはキャッシュ有効化の最小トークン数として1,024トークンの閾値が設定されており、Care Accessの医療記録はこの閾値を超えるため、自動的にキャッシュが有効化される。

### Zenn記事との関連

Zenn記事ではBedrock AgentCoreの1時間TTLキャッシュを社内RAGに適用するアーキテクチャを解説している。Care Accessの事例は、同様のキャッシュ戦略を医療記録処理パイプラインに適用した実プロダクション事例として位置づけられる。Zenn記事が解説するConverse APIの`cachePoint`配置戦略が、Care Accessのアーキテクチャでも中核的な役割を果たしている。

## 実装アーキテクチャ（Architecture）

### システム構成

Care Accessの医療記録処理パイプラインは、以下の構成で設計されている。

```
EHR取り込み
    ↓
前処理（トークン数チェック ≥ 1,024）
    ↓ 閾値以上
Bedrock Converse API（cachePoint付き）
    ↓
┌─────────────────────────────┐
│  キャッシュ層（医療記録本文） │ ← 静的プレフィックス
│  ・患者基本情報               │
│  ・検査結果                  │
│  ・診断履歴                  │
│  ・処方記録                  │
└─────────────────────────────┘
    ↓
動的質問（質問ごとに変化）
    ↓
回答生成 → 構造化出力
```

**キャッシュ戦略のポイント**: 医療記録の本文をキャッシュプレフィックスとして配置し、分析質問を動的部分として後ろに追加する。同一患者の記録に対して複数の質問を投げる場合、2回目以降のリクエストではキャッシュ読み取り料金（通常の入力トークン料金の10%）が適用される。

### API実装パターン

Zenn記事のConverse API実装と同様の構造が採用されている。

```python
import boto3
from typing import Any

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def analyze_ehr_with_cache(
    ehr_content: str,
    questions: list[str],
    model_id: str = "anthropic.claude-sonnet-4-20250514-v1:0",
) -> list[dict[str, Any]]:
    """医療記録に対して複数の分析質問をキャッシュ付きで実行

    Args:
        ehr_content: 電子カルテの全文テキスト
        questions: 分析質問のリスト
        model_id: Bedrockモデル ID

    Returns:
        各質問に対する回答リスト
    """
    results = []
    for question in questions:
        response = bedrock.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": ehr_content,
                            "cachePoint": {"type": "default"},
                        },
                        {
                            "text": f"上記の医療記録について以下の質問に回答してください:\n{question}",
                        },
                    ],
                }
            ],
        )
        usage = response.get("usage", {})
        results.append({
            "question": question,
            "answer": response["output"]["message"]["content"][0]["text"],
            "cache_read_tokens": usage.get("cacheReadInputTokens", 0),
            "cache_write_tokens": usage.get("cacheWriteInputTokens", 0),
        })
    return results
```

### スケーリング戦略

ブログによると、Care Accessでは記録取り込みのスパイクが予測よりも早期に発生した。Prompt Cachingの導入により、最小限のコード変更でコスト増大を抑制できたと報告されている。

Care Access社のリーダーシップは「記録取り込みが予測よりも早くスパイクした際、Bedrockのプロンプトキャッシュ機能により最小限の技術的変更でコストを管理できた」と述べている。

## パフォーマンス最適化（Performance）

### 実測結果

ブログで報告されている主要メトリクスは以下の通りである。

| メトリクス | Before（キャッシュなし） | After（キャッシュあり） | 改善率 |
|-----------|---------------------|---------------------|--------|
| データ処理コスト | 基準 | 基準の14% | **86%削減** |
| データ処理速度 | 基準 | 基準の34% | **66%高速化** |

### コスト削減の構造

コスト削減は以下の料金体系から生じている（Claude Sonnet 4基準）。

$$
\text{Cost}_{\text{cached}} = N_{\text{write}} \times P_{\text{write}} + (M - 1) \times N_{\text{read}} \times P_{\text{read}} + M \times N_{\text{dynamic}} \times P_{\text{input}}
$$

ここで、
- $N_{\text{write}}$: キャッシュ書き込みトークン数（初回のみ）
- $P_{\text{write}}$: キャッシュ書き込み単価（入力トークン料金の125%）
- $N_{\text{read}}$: キャッシュ読み取りトークン数（2回目以降）
- $P_{\text{read}}$: キャッシュ読み取り単価（入力トークン料金の10%）
- $N_{\text{dynamic}}$: 動的部分（質問）のトークン数
- $M$: 同一記録に対する質問数

キャッシュなしのコストは $M \times (N_{\text{static}} + N_{\text{dynamic}}) \times P_{\text{input}}$ であるため、$M$ が大きいほどキャッシュの効果が増大する。

### 処理速度向上の要因

66%の処理速度向上は、キャッシュされたKVキャッシュの再利用により、プレフィル（入力トークンの処理）段階をスキップできることに起因する。50,000トークンの医療記録の場合、プレフィル処理が2回目以降はほぼ不要となり、Time-to-First-Token（TTFT）が大幅に短縮される。

## 運用での学び（Production Lessons）

### スパイク対応

Care Accessの事例で注目すべきは、「記録取り込みが予測よりも早くスパイクした」という実運用上のイベントである。Prompt Cachingの導入により、コード変更を最小限に抑えながらコスト増大を管理できた点は、以下の教訓を示している。

1. **コスト予測可能性**: キャッシュにより、スパイク時のコスト増加がリクエスト数に比例するのではなく、ユニーク記録数に比例する構造に変わる
2. **最小侵襲な導入**: 既存のConverse API呼び出しに`cachePoint`フィールドを追加するだけで有効化できる
3. **6週間での本番稼働**: AWSとの協力により、ソリューション全体を6週間で構築・デプロイした点は、Bedrockのマネージドサービスとしての成熟度を示している

### 最小トークン閾値の活用

Bedrockのプロンプトキャッシュには1,024トークンの最小閾値が設定されている。Care Accessのパイプラインでは、前処理段階でトークン数チェックを実施し、閾値以上の記録のみキャッシュを有効化する設計としている。短い記録（閾値未満）はキャッシュなしで処理される。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

Care Accessの事例を参考に、医療記録処理パイプラインをAWSで構築する場合の推奨構成を示す。コスト試算は2026年2月時点のap-northeast-1リージョン概算値である。

| 構成 | トラフィック | 月額コスト目安 | 主要サービス |
|------|------------|-------------|------------|
| Small | ~100記録/日 | $80-200 | Lambda + Bedrock + S3 |
| Medium | ~1,000記録/日 | $500-1,500 | ECS Fargate + Bedrock + DynamoDB |
| Large | 10,000+記録/日 | $3,000-8,000 | EKS + Bedrock + SQS + S3 |

**コスト削減テクニック**:
- Prompt Caching有効化で入力トークンコスト86%削減（Care Access実績値）
- SQSキューイングによるバースト制御でBedrock呼び出しの平準化
- S3 Intelligent-Tieringで医療記録ストレージコスト最適化
- Bedrock Batch APIによる非リアルタイム処理で50%削減

### Terraformインフラコード

**Small構成（Serverless — 医療記録処理パイプライン）**:

```hcl
# EHR処理パイプライン - Lambda + Bedrock + S3
resource "aws_s3_bucket" "ehr_records" {
  bucket = "care-ehr-records-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ehr_sse" {
  bucket = aws_s3_bucket.ehr_records.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.ehr.arn
    }
  }
}

resource "aws_kms_key" "ehr" {
  description             = "EHR records encryption key"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}

resource "aws_lambda_function" "ehr_analyzer" {
  function_name = "ehr-cached-analyzer"
  runtime       = "python3.12"
  handler       = "handler.lambda_handler"
  memory_size   = 1024
  timeout       = 120

  environment {
    variables = {
      MODEL_ID       = "anthropic.claude-sonnet-4-20250514-v1:0"
      CACHE_MIN_TOKENS = "1024"
      S3_BUCKET      = aws_s3_bucket.ehr_records.id
    }
  }

  role = aws_iam_role.ehr_lambda.arn
}

resource "aws_iam_role" "ehr_lambda" {
  name = "ehr-analyzer-lambda"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "ehr_policy" {
  name = "ehr-analyzer-policy"
  role = aws_iam_role.ehr_lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = ["arn:aws:bedrock:*::foundation-model/anthropic.*"]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = ["${aws_s3_bucket.ehr_records.arn}/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["kms:Decrypt", "kms:GenerateDataKey"]
        Resource = [aws_kms_key.ehr.arn]
      }
    ]
  })
}
```

### 運用・監視設定

**CloudWatch Logs Insights — キャッシュヒット率分析**:

```
fields @timestamp, cache_read_tokens, cache_write_tokens, total_input_tokens
| stats sum(cache_read_tokens) as total_cache_reads,
        sum(cache_write_tokens) as total_cache_writes,
        sum(total_input_tokens) as total_inputs
| eval cache_hit_rate = total_cache_reads / (total_cache_reads + total_cache_writes) * 100
```

**コスト異常検知アラーム（Python）**:

```python
import boto3

cloudwatch = boto3.client("cloudwatch")

cloudwatch.put_metric_alarm(
    AlarmName="EHR-Processing-Cost-Spike",
    MetricName="BedrockInputTokens",
    Namespace="EHRAnalyzer",
    Statistic="Sum",
    Period=3600,
    EvaluationPeriods=2,
    Threshold=10_000_000,
    ComparisonOperator="GreaterThanThreshold",
    AlarmActions=["arn:aws:sns:ap-northeast-1:123456789:ops-alerts"],
    AlarmDescription="1時間で1000万入力トークン超過（キャッシュミス疑い）",
)
```

### コスト最適化チェックリスト

**アーキテクチャ選択**:
- [ ] 記録処理量に応じた構成選定（Serverless / Hybrid / Container）
- [ ] バッチ処理とリアルタイム処理の分離

**リソース最適化**:
- [ ] Lambda: メモリ1024MB推奨（Bedrock API呼び出しのレイテンシ最適化）
- [ ] S3: Intelligent-Tiering有効化
- [ ] ECS/EKS: Spot Instances優先
- [ ] Reserved Instances: 安定ワークロード向け
- [ ] Savings Plans検討

**LLMコスト削減**:
- [ ] Prompt Caching有効化（`cachePoint`配置）
- [ ] トークン数閾値チェック（≥1,024で自動有効化）
- [ ] 同一記録への複数質問バッチ化
- [ ] Batch API活用（非リアルタイム処理）
- [ ] max_tokens制限設定

**監視・アラート**:
- [ ] AWS Budgets設定
- [ ] キャッシュヒット率モニタリング
- [ ] CloudWatch入力トークン数アラーム
- [ ] Cost Anomaly Detection有効化
- [ ] 日次コストレポート

**リソース管理**:
- [ ] 処理済み記録のS3ライフサイクル設定
- [ ] CloudWatch Logs保持期間設定
- [ ] タグ戦略（PatientBatch, Environment, CostCenter）
- [ ] 開発環境の夜間停止
- [ ] 未使用IAMロール・ポリシーの定期削除

## 学術研究との関連（Academic Connection）

Care Accessの事例は、プロンプトキャッシュに関する以下の研究を実プロダクションで検証した事例として位置づけられる。

- **Prompt Cache (Gim et al., 2024, arXiv 2311.04934)**: KVキャッシュの位置独立な再利用を理論的に提案。Care AccessはBedrockの実装を通じてこの概念を医療分野で活用
- **FrugalGPT (Chen et al., 2023, arXiv 2305.05176)**: LLMコスト最適化の体系的整理。Care Accessのキャッシュ戦略は、FrugalGPTが示す「Prompt Adaptation」カテゴリの実装例

## まとめと実践への示唆

Care Accessの事例は、Bedrock Prompt Cachingの実プロダクション適用において86%のコスト削減と66%の処理速度向上を達成した具体例である。特に「同一の大規模ドキュメントに対して複数の分析質問を実行する」パターンでキャッシュが有効に機能することを示している。Zenn記事で解説する社内RAGシステムも同様のパターンに該当し、1時間TTLの活用によりさらなる効果が期待される。ソリューション全体を6週間で構築した点は、Bedrockのマネージドサービスとしての導入容易性を示している。

## 参考文献

- **Blog URL**: [https://aws.amazon.com/blogs/machine-learning/how-care-access-achieved-86-data-processing-cost-reductions-and-66-faster-data-processing-with-amazon-bedrock-prompt-caching/](https://aws.amazon.com/blogs/machine-learning/how-care-access-achieved-86-data-processing-cost-reductions-and-66-faster-data-processing-with-amazon-bedrock-prompt-caching/)
- **Related Papers**: [arXiv 2311.04934](https://arxiv.org/abs/2311.04934)（Prompt Cache）
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/d027acf4081b9d](https://zenn.dev/0h_n0/articles/d027acf4081b9d)
