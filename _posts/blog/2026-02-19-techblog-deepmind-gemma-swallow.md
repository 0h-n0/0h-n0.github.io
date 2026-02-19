---
layout: post
title: "Gemma-2-Llama Swallow: 科学大学×AISが構築した日本語特化LLMの技術詳細"
description: "Gemma 2ベースの継続事前学習により70Bクラスに匹敵する日本語性能を27Bパラメータで実現したSwallowプロジェクトの技術解説"
categories: [blog, tech_blog]
tags: [Gemma, Swallow, Japanese-LLM, continual-pretraining, llm, ai]
date: 2026-02-19 09:00:00 +0900
source_type: tech_blog
source_domain: deepmind.google
source_url: https://deepmind.google/models/gemma/gemmaverse/gemma-2-llama-swallow/
zenn_article: 3a4f2089113d8e
zenn_url: https://zenn.dev/0h_n0/articles/3a4f2089113d8e
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

Google DeepMindのGemmaverseページおよびSwallow LLMプロジェクトの公式サイトでは、科学大学（旧東京工業大学）岡崎研究室・横田研究室と産業技術総合研究所（AIST）が共同開発したGemma-2-Llama Swallowの技術詳細が公開されている。本モデルはGemma 2をベースに日本語データで継続事前学習を行い、2B/9B/27Bの3サイズを提供する。特筆すべきは、27Bパラメータで70Bクラスモデル（Llama 3.1 Swallow 70B）に匹敵する日本語性能を達成した点であり、これはGemma 2のトークナイザが元来日本語文字・語彙を豊富に含んでいたことと、高品質な日本語学習データの構築によるものである。

この記事は [Zenn記事: 2026年2月版 日本語LLM選定ガイド：ベンチマーク・料金・用途別に徹底比較](https://zenn.dev/0h_n0/articles/3a4f2089113d8e) の深掘りです。

## 情報源

- **種別**: 企業テックブログ / プロジェクトサイト
- **URL**: [https://deepmind.google/models/gemma/gemmaverse/gemma-2-llama-swallow/](https://deepmind.google/models/gemma/gemmaverse/gemma-2-llama-swallow/)
- **補助情報**: [https://swallow-llm.github.io/gemma2-llama-swallow.en.html](https://swallow-llm.github.io/gemma2-llama-swallow.en.html)
- **組織**: 科学大学（Institute of Science Tokyo）岡崎研究室・横田研究室、AIST
- **発表日**: 2025年5月19日（v0.1リリース）

## 技術的背景（Technical Background）

### 日本語LLMの課題

海外発の大規模言語モデルは、学習データにおける日本語の比率が低く、日本語タスクでの性能が相対的に劣る。この課題に対し、従来のアプローチは以下の2つに大別される：

1. **トークナイザ拡張 + 継続事前学習**: 日本語トークンを語彙に追加し、日本語テキストで追加学習（Llama Swallowシリーズ）
2. **ゼロからの学習**: 日本語を含む多言語データで最初から事前学習（PLaMo, Tanuki等）

Gemma-2-Llama Swallowは新たな第3のアプローチを提示した。Gemma 2のトークナイザが既に多数の日本語文字・語彙を含んでいたため、**トークナイザ修正を一切行わず**に継続事前学習のみで日本語性能を大幅に向上させることに成功した。

### 学術研究との関連

SwallowプロジェクトはこれまでにもLlama 2 Swallow、Llama 3/3.1 Swallowなど複数の日本語特化モデルを公開してきた。Gemma 2の選択は、ベースモデルの日本語適性が高いことによる戦略的判断であり、継続事前学習のコスト効率を大幅に改善した。岡崎直昭教授（科学大学）を中心とするチームは、「主権AI（Sovereign AI）」の文脈で日本語処理能力の自律的確保を目指している。

## 実装アーキテクチャ（Architecture）

### モデルサイズと構成

| モデル | パラメータ数 | ベースモデル | 変種 |
|--------|------------|------------|------|
| Gemma-2-Llama Swallow 2B | ~2.6B | Gemma 2 2B | PT / IT |
| Gemma-2-Llama Swallow 9B | ~9B | Gemma 2 9B | PT / IT |
| Gemma-2-Llama Swallow 27B | ~27B | Gemma 2 27B | PT / IT |

- **PT（Pre-trained）**: 継続事前学習のみ
- **IT（Instruction-tuned）**: PT上にさらに指示調整

### なぜ "Llama" が名前に含まれるのか

重要な注意点として、Gemma-2-Llama Swallowの「Llama」はアーキテクチャの変換を意味しない。学習データの合成にLlama 3.3 70B Instructを使用したため、ライセンス上Llama 3.3の派生モデルとして扱われる。具体的には、コーディングデータ（Swallow Code）の合成にLlama 3.3 70Bが使われたことが理由である。

### 継続事前学習の設計

#### 学習データ構成

```
┌─────────────────────────────────────────┐
│        継続事前学習コーパス               │
├─────────────────────────────────────────┤
│ 英語データ                               │
│ ├── Cosmopedia                          │
│ ├── DCLM-baseline-1.0                   │
│ ├── FineMath-4+                         │
│ └── English Wikipedia                   │
├─────────────────────────────────────────┤
│ 日本語データ                             │
│ ├── Swallow Corpus V2 (上位10%)          │
│ │   └── 教育品質分類器でフィルタリング     │
│ ├── 日本語Wikipedia                      │
│ ├── Laboro ParaCorpus                   │
│ └── 日本語QA合成テキスト                  │
├─────────────────────────────────────────┤
│ コードデータ                             │
│ └── Swallow Code (Stack v2サブセット)     │
│     └── Llama 3.3 70Bで品質フィルタリング │
└─────────────────────────────────────────┘
```

特に注目すべきは**Swallow Corpus Version 2**の品質フィルタリングである。教育品質分類器を用いてコーパス全体の上位10%のみを選択することで、学習効率を大幅に向上させている。

#### 指示調整データ

指示調整（IT版）では以下の3データセットを使用：

1. **Gemma-2-LMSYS-Chat-1M-Synth**: LMSYS-Chat-1Mの対話データをGemma 2 27Bで合成
2. **Swallow-Magpie-Ultra-v0.1**: Magpie手法による高品質指示データ
3. **Swallow-Gemma-Magpie-v0.1**: Gemma特化のMagpie指示データ

指示調整では「模倣学習（imitation learning）」を採用し、Gemma 2 27Bの応答品質を小サイズモデルに蒸留することで、計算コストを削減しつつ出力品質を維持した。

### TPU学習の最適化

学習はGoogle Cloud TPU v6eクラスタ上でMaxTextフレームワークを使用して実施された。以下の最適化が適用されている：

```python
# 学習設定の概要
training_config = {
    "framework": "MaxText",
    "hardware": "TPU v6e cluster",
    "sharding": "FSDP Stage 3",  # Fully Sharded Data Parallel
    "throughput_improvement": "~30%",  # 最適化による改善
    "tokenization": "on-the-fly",  # ストリーミング方式
}
```

**主要な最適化手法**:

1. **On-the-flyトークナイゼーション**: 事前トークナイズを行わず、Arレコードから直接ストリーミングしてGPU/TPUに供給。データ前処理のボトルネックを解消
2. **非同期チェックポイント転送**: Google Cloud Storageへのチェックポイント保存をバックグラウンドスレッドで実行し、TPUのアイドル時間を最小化
3. **プリエンプション対応**: TPUプリエンプション通知を受けた際に即座にチェックポイントを保存する機構を実装

これらの最適化により、約30%のスループット向上を達成した。

### トークナイザ修正が不要だった理由

従来のLlama Swallowシリーズでは、Llamaのトークナイザに日本語トークンを約5,000-10,000個追加する必要があった。これにはEmbedding層の拡張と追加学習が必要で、計算コストが大きかった。

$$
C_{\text{tokenizer\_extend}} = C_{\text{embed\_init}} + C_{\text{embed\_train}} + C_{\text{vocab\_align}}
$$

Gemma 2のトークナイザは設計段階から多言語対応しており、日本語の文字・語彙が十分に含まれていた。これにより上記コスト$C_{\text{tokenizer\_extend}}$がゼロとなり、継続事前学習のみに計算資源を集中できた。

## パフォーマンス最適化（Performance）

### ベンチマーク結果

#### 事前学習モデル（PT）: 日本語理解・生成タスク平均スコア

| モデル | パラメータ | 日本語平均 | 英語平均 | 特記事項 |
|--------|----------|-----------|---------|---------|
| Gemma 2 2B（ベース） | 2.6B | 0.348 | 0.439 | ベースライン |
| **Gemma-2-Llama Swallow 2B PT** | 2.6B | **0.421** | 0.426 | **+7.3pt向上** |
| Gemma 3 4B | 4B | 0.417 | 0.501 | 比較対象 |
| Sarashina2.2 3B | 3B | 0.516 | - | 2Bクラス最高 |
| Gemma 2 9B（ベース） | 9B | - | 0.597 | ベースライン |
| **Gemma-2-Llama Swallow 9B PT** | 9B | **0.558** | 0.595 | **Gemma 3 12Bを上回る** |
| Gemma 3 12B | 12B | 0.518 | - | 比較対象 |
| Gemma 2 27B（ベース） | 27B | - | 0.645 | ベースライン |
| **Gemma-2-Llama Swallow 27B PT** | 27B | **0.594** | 0.655 | **70Bクラスに匹敵** |
| Llama 3.1 Swallow 70B v0.1 | 70B | ~0.59 | - | 比較対象 |

#### 指示調整モデル（IT）: 日本語MT-Benchスコア

| モデル | パラメータ | 日本語MT-Bench |
|--------|----------|---------------|
| Gemma-2-Llama Swallow 2B IT | 2.6B | 0.597 |
| Gemma-2-Llama Swallow 9B IT | 9B | 0.749 |
| Gemma-2-Llama Swallow 27B IT | 27B | 0.759 |
| GPT-4o | - | 1位 |
| Llama 3.3 Swallow 70B IT | 70B | 2位 |

### 英語性能のトレードオフ

継続事前学習では英語性能とのトレードオフが発生する：

| サイズ | 英語性能変化 | 分析 |
|--------|------------|------|
| 2B | -1.3pt（0.439→0.426） | 小モデルでは影響大 |
| 9B | -0.2pt（0.597→0.595） | ほぼ影響なし |
| 27B | **+1.0pt**（0.645→0.655） | 日本語追加が英語にも好影響 |

27Bモデルでは日本語データの追加が英語性能をも向上させた。これは大規模モデルにおける多言語転移学習の正の効果を示唆している。

### 「1サイズ上」に匹敵する性能

Swallowチームの分析によると、各サイズのモデルが1つ上のサイズクラスのモデルに匹敵する性能を示す：

$$
\text{Perf}(\text{Swallow}_{n\text{B}}) \approx \text{Perf}(\text{Base}_{2n \sim 3n\text{B}})
$$

具体的には：
- 2B → 4Bクラス（Gemma 3 4Bと同等）
- 9B → 12Bクラス（Gemma 3 12Bを上回る）
- 27B → 70Bクラス（Llama 3.1 Swallow 70Bに匹敵）

これは推論コストの観点から極めて重要であり、同等性能をより少ないパラメータ数で達成できることを意味する。

## 運用での学び（Production Lessons）

### モデル選定における実践的知見

1. **ベースモデルの日本語適性が最重要**: トークナイザの日本語カバレッジが高いベースモデルを選ぶことで、継続事前学習のコストと品質が大幅に改善される

2. **データ品質フィルタリングの効果**: Swallow Corpus V2の上位10%のみを使用する戦略は、全データ使用と比較して学習効率を向上させた。品質分類器の投資は十分に回収される

3. **合成データの活用**: Llama 3.3 70Bによるコードデータの合成や、Magpie手法による指示データの生成など、大規模モデルを教師とする蒸留的アプローチが実用的

4. **TPU学習の信頼性**: On-the-flyトークナイゼーションとプリエンプション対応チェックポイントにより、長時間のTPU学習の安定性を確保

### デプロイメント形態

HuggingFace上で公開されたモデルは、以下の形態で利用可能：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "tokyotech-llm/Gemma-2-Llama-Swallow-27b-it-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "日本語LLMの選定基準を教えてください"}
]
input_ids = tokenizer.apply_chat_template(
    messages, return_tensors="pt"
).to(model.device)

output = model.generate(input_ids, max_new_tokens=512)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### ライセンスの注意事項

Gemma Terms of UseとLlama 3.3 Licenseの両方に準拠する必要がある。商用利用は可能だが、デュアルライセンスの制約に注意が必要。

## 学術研究との関連（Academic Connection）

Gemma-2-Llama Swallowは以下の学術的文脈に位置づけられる：

- **継続事前学習の理論**: Gupta et al. (2023) "Continual Pre-Training of Large Language Models" の手法を日本語に特化して適用
- **Swallowシリーズの発展**: Llama 2 Swallow → Llama 3 Swallow → Llama 3.1 Swallow → Gemma-2-Llama Swallowと、ベースモデルの選択を戦略的に変更
- **日本語ベンチマーク**: 10種類の日本語理解・生成タスクでの体系的評価により、日本語LLM研究の比較基盤を提供
- **主権AI**: 自国語処理能力を外部依存せずに確保する「主権AI」の実践例として、日本の国立研究機関が主導

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 構成 | トラフィック | サービス | 月額コスト |
|------|------------|---------|-----------|
| Small | ~100 req/日 | Lambda + Bedrock (Claude 3.5 Haiku) | $50-150 |
| Medium | ~1,000 req/日 | ECS Fargate + SageMaker Endpoint (Swallow 9B) | $400-900 |
| Large | 10,000+ req/日 | EKS + g5.12xlarge + Spot (Swallow 27B) | $1,500-4,000 |

**Small構成（~100 req/日）**:
- 日本語タスクにはBedrock経由でClaude 3.5 Haikuを使用（Swallowと同等の日本語品質をマネージドで）
- Lambda（256MB, 30秒）→ Bedrock → DynamoDB（結果キャッシュ）
- **月額内訳**: Bedrock $30-80、Lambda $5、DynamoDB $5-10、CloudWatch $5

**Medium構成（~1,000 req/日）**:
- SageMaker上にGemma-2-Llama Swallow 9Bをデプロイ（ml.g5.2xlarge）
- ECS Fargateでリクエストルーティング + ElastiCacheでKVキャッシュ
- **月額内訳**: SageMaker $250-500、Fargate $50-100、ElastiCache $30-60、ALB $20

**Large構成（10,000+ req/日）**:
- EKS上にSwallow 27BをvLLMでセルフホスト（g5.12xlarge × 2, 4xA10G GPU）
- Karpenter + Spot Instancesで自動スケーリング
- **月額内訳**: EC2 GPU $1,000-3,000（Spot後）、EKS $75、S3 $20、CloudWatch $50

**コスト削減テクニック**:
- Spot Instances: g5インスタンスで最大70%削減
- SageMaker Savings Plans: 1年コミットで最大64%削減
- 量子化（AWQ/GPTQ）: 27Bモデルを単一GPU（A10G 24GB）にフィット可能
- バッチ推論: 非リアルタイム処理でスループット3-5倍向上

**コスト試算の注意事項**: 上記は2026年2月時点のAWS ap-northeast-1概算値。SageMakerエンドポイントは常時起動のため、アイドル時のコストに注意。Auto-scalingのmin_instances=0設定で待機コストを削減可能。

### Terraformインフラコード

**Small構成（Serverless: Lambda + Bedrock）**:

```hcl
# --- IAM（最小権限） ---
resource "aws_iam_role" "swallow_lambda" {
  name = "swallow-inference-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "bedrock_access" {
  name = "bedrock-invoke-policy"
  role = aws_iam_role.swallow_lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["bedrock:InvokeModel"]
      Resource = "arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.*"
    }]
  })
}

# --- Lambda ---
resource "aws_lambda_function" "swallow_inference" {
  function_name = "swallow-japanese-inference"
  runtime       = "python3.12"
  handler       = "handler.lambda_handler"
  role          = aws_iam_role.swallow_lambda.arn
  memory_size   = 256
  timeout       = 30
  filename      = "lambda.zip"

  environment {
    variables = {
      MODEL_ID    = "anthropic.claude-3-5-haiku-20241022-v1:0"
      CACHE_TABLE = aws_dynamodb_table.response_cache.name
    }
  }
}

# --- DynamoDB（キャッシュ、KMS暗号化） ---
resource "aws_dynamodb_table" "response_cache" {
  name         = "swallow-response-cache"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "prompt_hash"

  attribute {
    name = "prompt_hash"
    type = "S"
  }

  server_side_encryption { enabled = true }
  ttl { attribute_name = "expires_at"; enabled = true }
}

# --- CloudWatch アラーム ---
resource "aws_cloudwatch_metric_alarm" "cost_alert" {
  alarm_name          = "swallow-daily-cost-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Invocations"
  namespace           = "AWS/Lambda"
  period              = 86400
  statistic           = "Sum"
  threshold           = 200
  dimensions          = { FunctionName = aws_lambda_function.swallow_inference.function_name }
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
```

**Large構成（Container: EKS + Spot + vLLM）**:

```hcl
# --- EKS Cluster ---
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "~> 20.0"
  cluster_name    = "swallow-inference-cluster"
  cluster_version = "1.31"
  vpc_id          = aws_vpc.main.id
  subnet_ids      = aws_subnet.private[*].id

  cluster_endpoint_public_access = false
}

# --- Karpenter NodePool（GPU Spot優先） ---
resource "kubectl_manifest" "gpu_nodepool" {
  yaml_body = yamlencode({
    apiVersion = "karpenter.sh/v1"
    kind       = "NodePool"
    metadata   = { name = "swallow-gpu" }
    spec = {
      template = {
        spec = {
          requirements = [
            { key = "karpenter.sh/capacity-type", operator = "In", values = ["spot", "on-demand"] },
            { key = "node.kubernetes.io/instance-type", operator = "In",
              values = ["g5.12xlarge", "g5.48xlarge"] },
          ]
          nodeClassRef = { name = "default" }
        }
      }
      limits = { cpu = "192", "nvidia.com/gpu" = "16" }
      disruption = {
        consolidationPolicy = "WhenEmptyOrUnderutilized"
        consolidateAfter    = "30s"
      }
    }
  })
}

# --- AWS Budgets ---
resource "aws_budgets_budget" "monthly_budget" {
  name         = "swallow-inference-monthly"
  budget_type  = "COST"
  limit_amount = "4000"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_sns_topic_arns = [aws_sns_topic.alerts.arn]
  }
}
```

### 運用・監視設定

**CloudWatch Logs Insights クエリ**:

```
# 日本語推論のレイテンシ分析
fields @timestamp, model_name, latency_ms, input_tokens, output_tokens
| filter model_name like /swallow/
| stats avg(latency_ms) as avg_lat, pct(latency_ms, 95) as p95,
        avg(output_tokens) as avg_out_tok by bin(1h)
| sort @timestamp desc

# トークン使用量の異常検知
fields @timestamp, input_tokens, output_tokens
| stats sum(input_tokens + output_tokens) as total_tokens by bin(1h)
| filter total_tokens > 50000
```

**CloudWatch アラーム設定（Python）**:

```python
import boto3

cloudwatch = boto3.client("cloudwatch", region_name="ap-northeast-1")

# SageMaker推論レイテンシ監視
cloudwatch.put_metric_alarm(
    AlarmName="swallow-inference-latency-high",
    MetricName="ModelLatency",
    Namespace="AWS/SageMaker",
    Statistic="p99",
    Period=300,
    EvaluationPeriods=3,
    Threshold=5000,  # 5秒超過でアラート
    ComparisonOperator="GreaterThanThreshold",
    Dimensions=[{"Name": "EndpointName", "Value": "swallow-27b-endpoint"}],
    AlarmActions=["arn:aws:sns:ap-northeast-1:ACCOUNT:swallow-alerts"],
)
```

**X-Ray トレーシング設定**:

```python
from aws_xray_sdk.core import xray_recorder, patch_all

patch_all()

@xray_recorder.capture("swallow_inference")
def invoke_swallow(prompt: str, model_size: str = "27b") -> dict:
    """Swallowモデル推論のトレーシング"""
    subsegment = xray_recorder.current_subsegment()
    subsegment.put_annotation("model_size", model_size)
    subsegment.put_annotation("language", "ja")
    subsegment.put_metadata("prompt_length", len(prompt))

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=f"swallow-{model_size}-endpoint",
        Body=prompt.encode("utf-8"),
        ContentType="application/json",
    )
    return response
```

**Cost Explorer 日次レポート**:

```python
import boto3
from datetime import date, timedelta

ce = boto3.client("ce", region_name="us-east-1")

def get_swallow_daily_cost() -> dict:
    """Swallow推論関連の日次コスト取得"""
    today = date.today()
    resp = ce.get_cost_and_usage(
        TimePeriod={"Start": str(today - timedelta(1)), "End": str(today)},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
        Filter={"Tags": {"Key": "Project", "Values": ["swallow-inference"]}},
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )
    return resp["ResultsByTime"][0]["Groups"]
```

### コスト最適化チェックリスト

**アーキテクチャ選択**:
- [ ] トラフィック量でServerless/SageMaker/EKSを選択
- [ ] モデルサイズ選定：簡単なタスクは9B、複雑なタスクは27B
- [ ] 量子化検討：AWQ/GPTQで27Bを単一GPU化

**リソース最適化**:
- [ ] Spot Instances優先（g5/g6で最大70%削減）
- [ ] SageMaker Savings Plans（1年コミットで64%削減）
- [ ] Reserved Instances: GPU 1年コミット
- [ ] SageMaker Auto-scaling（min_instances=0で待機コスト削減）
- [ ] EKS Karpenter自動スケーリング

**LLMコスト削減**:
- [ ] バッチ推論活用（非リアルタイムで3-5倍効率化）
- [ ] KVキャッシュ（ElastiCache）で重複推論回避
- [ ] プロンプト最適化（トークン数削減）
- [ ] モデルサイズ自動選択（タスク複雑度に応じて2B/9B/27B切替）

**監視・アラート**:
- [ ] AWS Budgets（月額上限80%で通知）
- [ ] CloudWatchアラーム（レイテンシP99、GPU使用率）
- [ ] Cost Anomaly Detection有効化
- [ ] 日次コストレポート（SNS送信）

**リソース管理**:
- [ ] 未使用SageMakerエンドポイント自動停止
- [ ] タグ戦略（Project, Environment, ModelSize）
- [ ] S3モデルアーティファクト管理（バージョニング）
- [ ] 開発環境の夜間・週末自動停止
- [ ] EBSスナップショットライフサイクル

## まとめと実践への示唆

Gemma-2-Llama Swallowは、日本語LLM選定において以下の重要な示唆を提供する：

1. **パラメータ効率**: 27Bパラメータで70Bクラスの日本語性能を達成。推論コスト（GPU使用量、メモリ、レイテンシ）を約60%削減可能
2. **ベースモデル選定の重要性**: トークナイザの日本語カバレッジが高いGemma 2を選択したことで、継続事前学習のコストと品質が大幅に改善された。Zenn記事で比較した各モデルの性能差は、この設計判断に大きく依存する
3. **オープンモデルの進化速度**: 2025年5月リリースで既にGemma 3ベースのSwallowも計画中。日本語LLM選定は半年単位での再評価が必要

## 参考文献

- **DeepMind Blog**: [https://deepmind.google/models/gemma/gemmaverse/gemma-2-llama-swallow/](https://deepmind.google/models/gemma/gemmaverse/gemma-2-llama-swallow/)
- **Swallow Project**: [https://swallow-llm.github.io/gemma2-llama-swallow.en.html](https://swallow-llm.github.io/gemma2-llama-swallow.en.html)
- **HuggingFace**: [https://huggingface.co/tokyotech-llm/Gemma-2-Llama-Swallow-27b-pt-v0.1](https://huggingface.co/tokyotech-llm/Gemma-2-Llama-Swallow-27b-pt-v0.1)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/3a4f2089113d8e](https://zenn.dev/0h_n0/articles/3a4f2089113d8e)
