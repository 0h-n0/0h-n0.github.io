---
layout: post
title: "AWS公式ブログ解説: Bedrock Intelligent Prompt Routingのコスト・レイテンシ最適化戦略"
description: "AWS Machine Learning Blogで紹介されたBedrock IPRの技術的背景、ベンチマーク結果、実装パターンを修士学生レベルで解説"
categories: [blog, tech_blog]
tags: [AWS, Bedrock, LLM, routing, cost-optimization, RAG, prompt-routing]
date: 2026-02-23 10:00:00 +0900
source_type: tech_blog
source_domain: aws.amazon.com
source_url: https://aws.amazon.com/blogs/machine-learning/use-amazon-bedrock-intelligent-prompt-routing-for-cost-and-latency-benefits/
zenn_article: f5fa165860f5e8
zenn_url: https://zenn.dev/0h_n0/articles/f5fa165860f5e8
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Use Amazon Bedrock Intelligent Prompt Routing for cost and latency benefits](https://aws.amazon.com/blogs/machine-learning/use-amazon-bedrock-intelligent-prompt-routing-for-cost-and-latency-benefits/)（AWS Machine Learning Blog）の解説記事です。

## ブログ概要（Summary）

AWS Machine Learning Blogは、Amazon Bedrock Intelligent Prompt Routing（IPR）の技術的詳細とコスト最適化効果を解説している。IPRは、同一モデルファミリー内の2つのモデル間でリクエストごとに最適なモデルを自動選択するサーバーレスルーティング機能であり、AWSの内部ベンチマークではAnthropicファミリーで平均63.6%のコスト削減が報告されている。ブログでは、ルーティング判定のアーキテクチャ、`responseQualityDifference`パラメータの調整方法、Cross-Region InferenceやPrompt Cachingとの組み合わせによる多層最適化戦略が詳述されている。

この記事は [Zenn記事: Bedrock Intelligent Prompt Routingで社内RAGコスト最大60%削減](https://zenn.dev/0h_n0/articles/f5fa165860f5e8) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（AWS Machine Learning Blog）
- **URL**: [https://aws.amazon.com/blogs/machine-learning/use-amazon-bedrock-intelligent-prompt-routing-for-cost-and-latency-benefits/](https://aws.amazon.com/blogs/machine-learning/use-amazon-bedrock-intelligent-prompt-routing-for-cost-and-latency-benefits/)
- **組織**: Amazon Web Services (AWS)
- **発表日**: 2025年4月（GA発表）

## 技術的背景（Technical Background）

### なぜLLMルーティングが必要か

LLMの推論コストはモデルパラメータ数に概ね比例する。たとえば、Claude 3.5 Sonnet v2の入力トークン単価は$3.00/MTokであるのに対し、Claude 3.5 Haikuは$0.80/MTokenである（2026年2月時点、AWS Bedrock On-Demand料金）。両者の価格差は約3.75倍にのぼる。

しかし、すべてのリクエストがSonnetクラスの推論能力を必要とするわけではない。社内FAQの「有給休暇の残日数は？」のような定型的なクエリは、Haikuでも十分な品質で回答可能である。AWS公式ブログによると、Anthropicファミリーでは**87%のプロンプトがHaikuにルーティング可能**であり、その結果として**平均63.6%のコスト削減**が達成できると報告されている。

### 学術的背景

IPRの動作原理は、LLMルーティングの学術研究と密接に関連する。特に以下の研究が理論的基盤を提供している：

- **RouteLLM** (Ong et al., 2024, arXiv:2406.18665): 選好データに基づくルーター学習のフレームワーク。MT-Benchで85%超のコスト削減を報告
- **FrugalGPT** (Chen et al., 2023): カスケード方式でのLLMルーティング。コスト制約付き品質最大化の定式化を提案
- **Toward Optimal LLM Routing** (Ding et al., 2024): ルーティングのParetoフロンティア分析

これらの研究で確立された「プロンプト難易度推定→モデル選択」のパラダイムが、Bedrock IPRの製品設計に反映されていると考えられる。

## 実装アーキテクチャ（Architecture）

### IPRのルーティング判定フロー

AWS公式ドキュメントに基づくIPRの内部アーキテクチャは以下の通りである。

```
┌──────────────────────────────────────────────────────────────┐
│                     API Gateway / SDK                         │
│                     Converse API Request                      │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│              Prompt Router (modelId = routerARN)              │
│                                                              │
│   1. プロンプト解析（入力の複雑さを評価）                      │
│   2. 品質予測（各モデルの期待応答品質を推定）                  │
│   3. 品質差判定（quality_diff ≤ threshold ?）                 │
│                                                              │
│   threshold = responseQualityDifference                       │
└──────────┬──────────────────────────────────┬────────────────┘
           │                                  │
    quality_diff ≤ τ                   quality_diff > τ
    （品質差が小さい）                 （品質差が大きい）
           │                                  │
           ▼                                  ▼
┌─────────────────────┐        ┌──────────────────────────┐
│   Fallback Model    │        │    Primary Model          │
│   (例: Haiku)       │        │    (例: Sonnet)           │
│   低コスト・高速     │        │    高品質・高コスト        │
└─────────────────────┘        └──────────────────────────┘
```

### responseQualityDifferenceの動作

`responseQualityDifference`パラメータ（以下、RQD）は $[0.0, 1.0]$ の範囲で設定する。IPRは各リクエストに対して、プライマリモデルとフォールバックモデルの応答品質差 $\Delta q$ を内部的に推定し、以下の判定を行う：

$$
\text{routed\_model} = \begin{cases} \text{fallback (Haiku)} & \text{if } \Delta q \leq \text{RQD} \\ \text{primary (Sonnet)} & \text{if } \Delta q > \text{RQD} \end{cases}
$$

ここで、
- $\Delta q$: プライマリモデルとフォールバックモデルの推定品質差
- $\text{RQD}$: ユーザーが設定する品質差閾値

AWS公式ブログでは、RQDの設定による効果を以下のように報告している：

| RQD値 | Haikuルーティング率 | コスト削減率 | 品質影響 |
|-------|-------------------|------------|---------|
| 0.0 | 0% | 0% | なし（全てSonnet） |
| 0.1 | 約50-60% | 約30-40% | 軽微 |
| 0.25 | 約70-80% | 約50-63.6% | 中程度 |
| 0.5 | 約85-90% | 約65-75% | タスク依存 |
| 1.0 | 100% | 最大 | 品質低下あり |

### 対応モデルファミリー

IPRは同一ファミリー内の2モデル間ルーティングのみ対応している。

| ファミリー | プライマリモデル | フォールバックモデル | 主な用途 |
|-----------|-----------------|-------------------|---------|
| **Anthropic** | Claude 3.5 Sonnet v2 | Claude 3.5 Haiku | 汎用テキスト生成 |
| **Meta Llama** | Llama 3.3 70B | Llama 3.1 8B | オープンモデル活用 |
| **Amazon Nova** | Nova Pro | Nova Lite | AWSネイティブ |

**制約**: クロスファミリールーティング（例: Claude → Llama）は非対応。

### 実装例: Converse API統合

```python
import boto3

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# ルーターARNをmodelIdに指定するだけでIPRが有効になる
response = bedrock_runtime.converse(
    modelId="arn:aws:bedrock:us-east-1::default-prompt-router/anthropic.claude",
    system=[{"text": "社内ドキュメントに基づいて回答してください。"}],
    messages=[
        {
            "role": "user",
            "content": [{"text": "有給休暇の申請手順を教えてください"}],
        }
    ],
    inferenceConfig={"maxTokens": 1024, "temperature": 0.1},
)

# どのモデルが選択されたか確認
trace = response.get("trace", {})
routed_model = trace.get("promptRouter", {}).get("invokedModelId", "unknown")
print(f"ルーティング先: {routed_model}")
```

この実装の重要な点は、**既存のConverse APIコードの変更が `modelId` の差し替えのみ**で済むことである。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

IPRを本番環境に導入する際の推奨構成をトラフィック量別に示す。

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト概算 |
|------|--------------|---------|-------------|
| **Small** | ~3,000 (100/日) | Lambda + Bedrock IPR | $50-150 |
| **Medium** | ~30,000 (1,000/日) | Lambda + IPR + ElastiCache | $300-800 |
| **Large** | 300,000+ (10,000/日) | ECS Fargate + IPR + Redis | $2,000-5,000 |

**Small構成の詳細**（月額$50-150）:
- **Lambda**: ルーティング実行、1GB RAM、30秒タイムアウト（$20/月）
- **Bedrock IPR**: Anthropicファミリー（$80-120/月、使用量依存）
- **DynamoDB**: メトリクス記録、On-Demand（$5/月）
- **CloudWatch**: 基本監視（$5/月）

**コスト削減テクニック**:
- **Prompt Caching**: 同一システムプロンプトの繰り返し利用で30-90%削減
- **Cross-Region Inference**: 追加コストなしでスロットリング回避
- **Batch API**: 非リアルタイム処理で50%削減

**コスト試算の注意事項**: 上記は2026年2月時点のAWS us-east-1リージョン料金に基づく概算値です。実際のコストはトラフィックパターンにより変動します。最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください。

### Terraformインフラコード（Small構成）

```hcl
# --- Lambda + Bedrock IPR構成 ---
resource "aws_iam_role" "lambda_bedrock_router" {
  name = "lambda-bedrock-router-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "bedrock_router_invoke" {
  role = aws_iam_role.lambda_bedrock_router.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:GetPromptRouter",
        "bedrock:ListPromptRouters"
      ]
      Resource = "*"
    }]
  })
}

resource "aws_lambda_function" "ipr_handler" {
  filename      = "lambda.zip"
  function_name = "bedrock-ipr-handler"
  role          = aws_iam_role.lambda_bedrock_router.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 60
  memory_size   = 1024

  environment {
    variables = {
      ROUTER_ARN         = "arn:aws:bedrock:us-east-1::default-prompt-router/anthropic.claude"
      DYNAMODB_TABLE     = aws_dynamodb_table.routing_metrics.name
      ENABLE_PROMPT_CACHE = "true"
    }
  }
}

resource "aws_dynamodb_table" "routing_metrics" {
  name         = "bedrock-ipr-routing-metrics"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "request_id"
  range_key    = "timestamp"

  attribute {
    name = "request_id"
    type = "S"
  }
  attribute {
    name = "timestamp"
    type = "N"
  }

  ttl {
    attribute_name = "expire_at"
    enabled        = true
  }
}
```

### 運用・監視設定

**ルーティング比率の可視化**（CloudWatch Logs Insights）:

```sql
fields @timestamp, routed_model, input_tokens, output_tokens
| stats count(*) as requests,
        sum(input_tokens) as total_input,
        sum(output_tokens) as total_output
  by routed_model
| sort requests desc
```

**コスト異常検知**:

```python
import boto3

cloudwatch = boto3.client('cloudwatch')
cloudwatch.put_metric_alarm(
    AlarmName='ipr-sonnet-routing-spike',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=2,
    MetricName='SonnetRoutingRate',
    Namespace='Custom/BedrockIPR',
    Period=3600,
    Statistic='Average',
    Threshold=0.5,
    AlarmDescription='Sonnetへのルーティング率が50%超過（コスト増加の兆候）'
)
```

### コスト最適化チェックリスト

- [ ] `responseQualityDifference`を0.25（バランス型）に設定
- [ ] Prompt Cachingを有効化（システムプロンプト固定化）
- [ ] Cross-Region Inference有効化（`us.`プレフィックス使用）
- [ ] ルーティングメトリクスをCloudWatchダッシュボードに統合
- [ ] AWS Budgets月額予算アラート設定（80%/100%通知）

## パフォーマンス最適化（Performance）

### ルーターのオーバーヘッド

AWS公式ドキュメントによると、IPRのルーティング判定によるレイテンシオーバーヘッドは**P90で約85ms**と報告されている。これは、LLMの推論レイテンシ（通常500ms-数秒）と比較して十分に小さい。

### Prompt Cachingとの組み合わせ

Prompt Cachingは、Bedrock IPRと組み合わせることでさらなる効果を発揮する。AWSの公式発表によると、Prompt Cachingの効果は以下の通りである：

| 指標 | Prompt Caching有効時の改善 |
|------|--------------------------|
| コスト | 最大90%削減 |
| レイテンシ | 最大85%削減 |

IPRでHaikuにルーティングされたリクエストにPrompt Cachingを適用すると、コスト削減効果が掛け算的に作用する。たとえば、IPRで63.6%削減 × Prompt Cachingで追加30%削減 = 理論上約75%の総合削減が見込める。

## 運用での学び（Production Lessons）

### 日本語ワークロードでの注意点

IPRは英語プロンプトに最適化されている。AWS公式ドキュメントでは日本語でのルーティング精度について明示的な保証がなされていない。社内RAGが日本語中心の場合は以下の対策が有効である：

1. **RQDを低めに設定**（0.1〜0.15）：品質差に敏感に反応し、Sonnetへのルーティング率を上げる
2. **モニタリング強化**: 日本語クエリとルーティング先の相関を分析し、品質低下を早期検知
3. **ハイブリッドアプローチ**: 日本語クエリが多い場合、前段で言語判定を行い、日本語はSonnet固定、英語はIPR活用

### Claude 3.5 Haikuの非推奨化への対応

Claude 3.5 Haikuは2026年7月5日にシャットダウン予定である。IPRを運用中の場合、Claude Haiku 4.5（`anthropic.claude-haiku-4-5-20251001-v1:0`）への移行計画が必要である。ただし、2026年2月時点ではIPRの対応モデルリストにHaiku 4.5は未掲載のため、最新の[公式ドキュメント](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-routing.html)で対応状況を定期的に確認することを推奨する。

## 学術研究との関連（Academic Connection）

Bedrock IPRの「品質差閾値に基づくルーティング」は、RouteLLM（arXiv:2406.18665）が提案する閾値ベースのルーティングフレームワークと概念的に対応する。RouteLLMではMT-Benchで85%超のコスト削減が報告されており、Bedrock IPRの63.6%削減はより保守的（品質重視）な設定に相当すると解釈できる。

また、FrugalGPT（Chen et al., 2023）が提案するカスケード方式は、Bedrock IPRの「プライマリ/フォールバック」モデル構成の理論的基盤を提供している。

## まとめと実践への示唆

Bedrock IPRは、学術研究で確立されたLLMルーティングの概念を、マネージドサービスとして製品化したものである。Converse APIの`modelId`を変更するだけで導入できる低い導入障壁と、`responseQualityDifference`による品質/コストの柔軟な制御が実用上の大きな利点となっている。Cross-Region InferenceやPrompt Cachingとの組み合わせにより、さらなる最適化が可能である。

## 参考文献

- **Blog URL**: [https://aws.amazon.com/blogs/machine-learning/use-amazon-bedrock-intelligent-prompt-routing-for-cost-and-latency-benefits/](https://aws.amazon.com/blogs/machine-learning/use-amazon-bedrock-intelligent-prompt-routing-for-cost-and-latency-benefits/)
- **Related Blog**: [Effective cost optimization strategies for Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/effective-cost-optimization-strategies-for-amazon-bedrock/)
- **AWS Documentation**: [Prompt Routing](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-routing.html)
- **Related Papers**: [RouteLLM (arXiv:2406.18665)](https://arxiv.org/abs/2406.18665)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/f5fa165860f5e8](https://zenn.dev/0h_n0/articles/f5fa165860f5e8)
