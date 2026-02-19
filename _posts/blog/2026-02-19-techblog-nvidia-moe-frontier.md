---
layout: post
title: "NVIDIA解説: Mixture of Expertsが最先端AIモデルを駆動する仕組み"
description: "NVIDIAが解説するMoEアーキテクチャの技術的詳細、Blackwell NVL72でのExpert Parallelism最適化、主要モデルの性能比較を徹底解説"
categories: [blog, tech_blog]
tags: [MoE, LLM, NVIDIA, Blackwell, inference, deepseek, llm]
date: 2026-02-19 09:00:00 +0900
source_type: tech_blog
source_domain: blogs.nvidia.com
source_url: https://blogs.nvidia.com/blog/mixture-of-experts-frontier-models/
zenn_article: 3a4f2089113d8e
zenn_url: https://zenn.dev/0h_n0/articles/3a4f2089113d8e
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

NVIDIAが公開した技術ブログでは、2023年初頭以降のフロンティアAIモデルにおけるMixture of Experts（MoE）アーキテクチャの急速な普及と、NVIDIA Blackwell NVL72プラットフォームにおける推論最適化について詳細に解説している。2025年時点でオープンソースAIモデルリリースの60%以上がMoEを採用し、Artificial Analysisリーダーボードのトップ10モデルはすべてMoEアーキテクチャを使用している。Blackwell NVL72では前世代H200比で10倍の性能向上と1/10のトークンコストを実現した。

この記事は [Zenn記事: 2026年2月版 日本語LLM選定ガイド：ベンチマーク・料金・用途別に徹底比較](https://zenn.dev/0h_n0/articles/3a4f2089113d8e) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://blogs.nvidia.com/blog/mixture-of-experts-frontier-models/](https://blogs.nvidia.com/blog/mixture-of-experts-frontier-models/)
- **補助情報**: [https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/)
- **組織**: NVIDIA
- **発表日**: 2025年12月（最終更新）

## 技術的背景（Technical Background）

### なぜMoEが必要になったのか

従来のdenseモデルのスケーリングは、すべてのパラメータを各トークン生成に使用するため、計算コストがパラメータ数に比例して増大する。MetaのLlama 2の学習には約330万NVIDIA A100 GPU時間を要したことからも、dense手法のスケーリング限界は明白である。

MoEは人間の脳の動作原理に着想を得ている。脳はタスクに応じて特定の領域のみを活性化するように、MoEモデルは数千億パラメータを持ちながらも、各トークン生成時には数百億パラメータのサブセットのみを使用する。これにより、計算コストの線形増加を回避しつつモデルの表現力を向上させることが可能になった。

### 学術研究との関連

MoEの概念自体は1991年のJacobsらの研究に遡るが、LLMへの本格的な適用はGoogleのSwitch Transformer（2021年）やGShard（2020年）から始まった。Mistral AIのMixtral 8x7B（2023年末）がオープンソースMoEモデルの先駆けとなり、以降DeepSeek-V2/V3、Qwen2-MoEなど多数のモデルが登場した。

## 実装アーキテクチャ（Architecture）

### MoEの基本構造

MoEレイヤーはTransformerブロック内のMLPレイヤーを複数の「エキスパート」ネットワークに置き換える構造を持つ。各エキスパートは独立したFFN（Feed-Forward Network）であり、ルーターネットワークが入力トークンに基づいてどのエキスパートを活性化するかを決定する。

$$
y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)
$$

ここで、
- $x$: 入力トークンの隠れ表現
- $N$: エキスパートの総数
- $g_i(x)$: ルーターが出力するエキスパート$i$のゲート値
- $E_i(x)$: エキスパート$i$の出力

Sparse MoEでは、Top-Kルーティングにより$K$個のエキスパートのみ活性化する：

$$
g_i(x) = \begin{cases} \text{softmax}(W_r x)_i & \text{if } i \in \text{Top-K}(W_r x) \\ 0 & \text{otherwise} \end{cases}
$$

ここで$W_r$はルーターの重み行列、$K$は活性化エキスパート数（Mixtral 8x7Bでは$K=2$）。

### Mixtral 8x7Bの具体的構造

NVIDIAの技術ブログで詳細に解析されたMixtral 8x7Bの構造：

- **総パラメータ数**: 約470億
- **活性パラメータ数**: 約129億（各トークンあたり）
- **Transformerブロック数**: 32
- **各ブロックのMoEレイヤー**: 8エキスパート、Top-2ルーティング
- **共有レイヤー**: Attention層、正規化層はすべてのトークンで共有
- **エキスパートの組み合わせ**: 理論上 $28^{32}$ 通りのフルスタック組み合わせ

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMoELayer(nn.Module):
    """Sparse Mixture of Experts Layer

    Args:
        d_model: モデルの隠れ次元数
        d_ff: FFNの中間次元数
        num_experts: エキスパートの総数
        top_k: 活性化するエキスパート数
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # ルーターネットワーク
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # エキスパートFFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MoE Forward Pass

        Args:
            x: 入力テンソル (batch_size, seq_len, d_model)

        Returns:
            出力テンソル (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # ルーティングスコア計算
        router_logits = self.router(x)  # (B, S, num_experts)

        # Top-K選択
        top_k_logits, top_k_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # エキスパート出力の重み付き和
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]  # (B, S)
            weight = top_k_weights[:, :, k].unsqueeze(-1)  # (B, S, 1)

            for i in range(self.num_experts):
                mask = (expert_idx == i)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[i](expert_input)
                    output[mask] += weight[mask] * expert_output

        return output
```

### エキスパートの専門化パターン

NVIDIAの分析により、各エキスパートがドメイン固有の専門化を示すことが確認された：

| ドメイン | 主に活性化されるエキスパート | Layer 32での観察 |
|----------|---------------------------|-----------------|
| Abstract Algebra | Expert 3, Expert 8 | 高い活性化率 |
| Professional Law | Expert 4 | 支配的な活性化 |
| World Religions | Expert 8が低活性化 | 最活性の1/5以下 |

この専門化は明示的に設計されたものではなく、学習過程で自然に発生する。ただし、負荷分散は完全ではなく、最も忙しいエキスパートは最も暇なエキスパートの40-60%多くのトークンを処理する不均衡が残る。

### 負荷分散の数式

学習時にエキスパート間の負荷を分散するため、auxiliary loss（補助損失）を使用する：

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i
$$

ここで、
- $\alpha$: 補助損失の重み（一般に0.01程度）
- $N$: エキスパート数
- $f_i$: エキスパート$i$にルーティングされたトークンの割合
- $P_i$: エキスパート$i$へのルーティング確率の平均値

この損失は$f_i$と$P_i$が均一分布に近づくよう誘導する。DeepSeek-V3ではこの補助損失を排除し、代わりにバイアス項による負荷分散を実現している。

## Expert Parallelism：Blackwell NVL72の革新

### H200世代の課題

8GPU構成のH200では、Expert Parallelism（EP）に2つの根本的なボトルネックがあった：

1. **HBM帯域幅の逼迫**: 各GPUが多数のエキスパートを保持するため、選択されたエキスパートのパラメータをHBMから動的にロードする際、メモリ帯域幅が逼迫
2. **スケールアウトネットワークのレイテンシ**: 8GPUを超えるEPではNVLink外のネットワーク通信が必要となり、All-to-All通信パターンのレイテンシが急増

### NVL72の解決策

Blackwell NVL72は72基のBlackwell GPUを単一システムとして接続する：

- **AI性能**: 1.4エクサフロップス
- **共有高速メモリ**: 30TB
- **NVLink接続帯域**: 130TB/sのNVLink Switch接続
- **NVFP4対応**: 精度を維持しつつ推論性能を向上

```
┌─────────────────────────────────────────────┐
│           NVLink Switch Fabric               │
│         130 TB/s 総帯域幅                     │
├─────┬─────┬─────┬─────┬─────┬─────┬─────────┤
│GPU 1│GPU 2│GPU 3│GPU 4│ ... │GPU72│          │
│E1,E2│E3,E4│E5,E6│E7,E8│     │     │          │
│     │     │     │     │     │     │          │
│ HBM │ HBM │ HBM │ HBM │     │ HBM │          │
└─────┴─────┴─────┴─────┴─────┴─────┴─────────┘
  ↑各GPUのエキスパート数が減少 → HBM帯域に余裕
  ↑NVLink直結 → All-to-All通信が低レイテンシ
```

72GPUに分散することで各GPUあたりのエキスパート数が大幅に減少し、HBMの空き容量が増え、より多くの同時ユーザーや長いコンテキストをサポート可能になった。NVLink Switchはエキスパート出力の合成計算の一部をインターコネクト上で実行する。

### NVIDIA Dynamoフレームワーク

NVIDIA Dynamoは**Disaggregated Serving**を実現するフレームワークである。Prefill（入力処理）とDecode（出力生成）を異なるGPUに割り当て、それぞれに最適な並列化戦略を適用する：

- **Prefill**: Tensor Parallelismやバッチ並列化に適した戦略
- **Decode**: 大規模Expert Parallelism（72GPU活用）に最適化

## パフォーマンス最適化（Performance）

### 実測値：H200 vs NVL72

| モデル | H200 基準 | NVL72 性能 | 改善倍率 |
|--------|----------|-----------|---------|
| DeepSeek-R1 | 1x | 10x+ | 10倍 |
| Kimi K2 Thinking | 1x | 10x | 10倍 |
| Mistral Large 3 | 1x | 10x | 10倍 |

**コスト効率**:
- トークンあたりコスト: H200比で**1/10に削減**
- ワットあたり性能: **10倍向上**
- 同一レイテンシでのスループット: **10倍**

SemiAnalysis InferenceMaxベンチマークによると、DeepSeek-R1はNVIDIA H200システム比で100万トークンあたりのコストを10倍以上削減した。

### チューニング手法

MoE推論の最適化では以下のアプローチが重要：

1. **Expert Parallelism度の最適化**: エキスパート数とGPU数のマッピング
2. **NVFP4量子化**: FP16/BF16からNVFP4への変換で、精度維持しつつメモリ使用量とレイテンシを改善
3. **KVキャッシュ最適化**: Shared Attention層のKVキャッシュはGPU間で共有可能
4. **Disaggregated Serving**: PrefillとDecodeの非同期パイプライン化

## 運用での学び（Production Lessons）

### 主要デプロイメントパートナーの知見

**Together AI**のDeepSeek-V3カスタム最適化事例：
- GB200 NVL72上でのExpert Parallelism最適化
- トークン生成レイテンシの大幅削減
- TensorRT-LLMおよびSGLang/vLLMフレームワークとの統合

**Fireworks AI**のKimi K2 on B200デプロイメント：
- 推論最適化によりB200単体でも高効率運用を実現

**CoreWeave**のエージェントワークフロー構築：
- MoEモデルの低レイテンシ特性を活かしたマルチステップ推論

### 推論フレームワークの選択

NVIDIAが公式に推奨する3つの推論フレームワーク：

| フレームワーク | 特徴 | 最適ユースケース |
|--------------|------|----------------|
| TensorRT-LLM | NVIDIA最適化、最高性能 | 本番環境、最大スループット |
| SGLang | 柔軟なスケジューリング | 複雑なプロンプト処理 |
| vLLM | オープンソース、汎用性 | 研究・プロトタイプ |

## 学術研究との関連（Academic Connection）

MoEアーキテクチャの発展は学術研究と産業応用の密接な連携によって推進されてきた：

- **Switch Transformer**（Google, 2021）: FFN層をTop-1ルーティングに置き換え、MoEのスケーラビリティを実証
- **Mixtral 8x7B**（Mistral AI, 2023）: Top-2ルーティングのオープンソースMoEモデルの先駆け
- **DeepSeek-V2/V3**（DeepSeek, 2024-2025）: Fine-grained Expert Segmentationと補助損失なし負荷分散
- **Qwen2-MoE**（Alibaba, 2024）: 14.3B全パラメータ中2.7B活性でMixtral 7Bを上回る性能

NVIDIAのブログはこれら学術成果の産業実装を包括的に俯瞰し、推論インフラの視点から最適化を論じている点が独自の貢献である。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 構成 | トラフィック | サービス | 月額コスト |
|------|------------|---------|-----------|
| Small | ~100 req/日 | Lambda + Bedrock (Claude/Mistral MoE) | $50-150 |
| Medium | ~1,000 req/日 | ECS Fargate + vLLM on Bedrock | $300-800 |
| Large | 10,000+ req/日 | EKS + p5.48xlarge (H100) + Spot | $2,000-5,000 |

**Small構成（~100 req/日）**:
- AWS Lambda（512MB, 30秒タイムアウト） → Amazon Bedrock（Mistral Large 2 / Claude 3.5 Sonnet）
- DynamoDB（On-Demand）でルーティング結果キャッシュ
- **月額内訳**: Bedrock推論 $30-80、Lambda $5-10、DynamoDB $5-10、CloudWatch $5

**Medium構成（~1,000 req/日）**:
- ECS Fargate（2vCPU, 8GB RAM）でルーティングロジック実行
- Bedrock経由でMoEモデル呼び出し + ElastiCache（Redis）でKVキャッシュ
- **月額内訳**: Bedrock $200-500、Fargate $50-100、ElastiCache $30-60、ALB $20

**Large構成（10,000+ req/日）**:
- EKS + Karpenter + p5.48xlarge Spot Instances（可用時）
- vLLM/TensorRT-LLM でセルフホストMoEモデル（Mixtral 8x22B等）
- **月額内訳**: EC2 GPU $1,500-3,500（Spot適用後）、EKS $75、S3 $20-50、CloudWatch $50

**コスト削減テクニック**:
- Spot Instances: p5インスタンスで最大60-70%削減（MoE推論はステートレスなため中断耐性が高い）
- Bedrock Batch API: 非同期処理で50%削減
- Prompt Caching: 繰り返しプレフィックスで30-90%削減
- Reserved Instances: 1年コミットで最大40%削減

**コスト試算の注意事項**: 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン概算値。実際のコストはトラフィックパターン、Spot可用性、モデルサイズにより変動する。最新料金はAWS料金計算ツールで確認を推奨。

### Terraformインフラコード

**Small構成（Serverless: Lambda + Bedrock）**:

```hcl
# --- VPC & ネットワーク ---
resource "aws_vpc" "moe_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags = { Name = "moe-inference-vpc" }
}

resource "aws_subnet" "private" {
  vpc_id            = aws_vpc.moe_vpc.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "ap-northeast-1a"
  tags = { Name = "moe-private-subnet" }
}

# --- IAM（最小権限） ---
resource "aws_iam_role" "lambda_moe" {
  name = "moe-inference-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "bedrock_invoke" {
  name = "bedrock-invoke-policy"
  role = aws_iam_role.lambda_moe.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["bedrock:InvokeModel"]
      Resource = "arn:aws:bedrock:ap-northeast-1::foundation-model/*"
    }]
  })
}

# --- Lambda ---
resource "aws_lambda_function" "moe_inference" {
  function_name = "moe-inference"
  runtime       = "python3.12"
  handler       = "handler.lambda_handler"
  role          = aws_iam_role.lambda_moe.arn
  memory_size   = 512  # MoEルーティングロジック用
  timeout       = 30
  filename      = "lambda.zip"

  environment {
    variables = {
      BEDROCK_MODEL_ID = "mistral.mistral-large-2407-v1:0"
      CACHE_TABLE      = aws_dynamodb_table.routing_cache.name
    }
  }
}

# --- DynamoDB（On-Demand、KMS暗号化） ---
resource "aws_dynamodb_table" "routing_cache" {
  name         = "moe-routing-cache"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "request_hash"

  attribute {
    name = "request_hash"
    type = "S"
  }

  server_side_encryption { enabled = true }
  ttl { attribute_name = "expires_at"; enabled = true }
}

# --- CloudWatch アラーム ---
resource "aws_cloudwatch_metric_alarm" "lambda_cost" {
  alarm_name          = "moe-lambda-invocations-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Invocations"
  namespace           = "AWS/Lambda"
  period              = 3600
  statistic           = "Sum"
  threshold           = 500  # 1時間500回超過で警告
  dimensions          = { FunctionName = aws_lambda_function.moe_inference.function_name }
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
```

**Large構成（Container: EKS + Karpenter + Spot）**:

```hcl
# --- EKS Cluster ---
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "~> 20.0"
  cluster_name    = "moe-inference-cluster"
  cluster_version = "1.31"
  vpc_id          = aws_vpc.moe_vpc.id
  subnet_ids      = aws_subnet.private[*].id

  cluster_endpoint_public_access = false  # プライベートアクセスのみ
}

# --- Karpenter Provisioner（Spot優先） ---
resource "kubectl_manifest" "karpenter_nodepool" {
  yaml_body = yamlencode({
    apiVersion = "karpenter.sh/v1"
    kind       = "NodePool"
    metadata   = { name = "moe-gpu-pool" }
    spec = {
      template = {
        spec = {
          requirements = [
            { key = "karpenter.sh/capacity-type", operator = "In", values = ["spot", "on-demand"] },
            { key = "node.kubernetes.io/instance-type", operator = "In", values = ["p5.48xlarge", "p4d.24xlarge"] },
          ]
          nodeClassRef = { name = "default" }
        }
      }
      limits   = { cpu = "384", "nvidia.com/gpu" = "32" }
      disruption = { consolidationPolicy = "WhenEmptyOrUnderutilized" }
    }
  })
}

# --- Secrets Manager ---
resource "aws_secretsmanager_secret" "model_config" {
  name       = "moe-model-config"
  kms_key_id = aws_kms_key.moe_key.arn
}

# --- AWS Budgets ---
resource "aws_budgets_budget" "moe_monthly" {
  name         = "moe-inference-monthly"
  budget_type  = "COST"
  limit_amount = "5000"
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
# MoE推論のトークン使用量異常検知（1時間あたり）
fields @timestamp, @message
| filter @message like /token_count/
| stats sum(token_count) as total_tokens by bin(1h)
| filter total_tokens > 100000

# Expert Parallelism レイテンシ分析
fields @timestamp, expert_id, latency_ms
| stats avg(latency_ms) as avg_lat, pct(latency_ms, 95) as p95, pct(latency_ms, 99) as p99
  by expert_id
| sort p99 desc
```

**CloudWatch アラーム設定（Python）**:

```python
import boto3

cloudwatch = boto3.client("cloudwatch", region_name="ap-northeast-1")

# Bedrockトークン使用量スパイク検知
cloudwatch.put_metric_alarm(
    AlarmName="moe-bedrock-token-spike",
    MetricName="InputTokenCount",
    Namespace="AWS/Bedrock",
    Statistic="Sum",
    Period=3600,
    EvaluationPeriods=1,
    Threshold=50000,
    ComparisonOperator="GreaterThanThreshold",
    AlarmActions=["arn:aws:sns:ap-northeast-1:ACCOUNT:moe-alerts"],
)
```

**X-Ray トレーシング設定**:

```python
from aws_xray_sdk.core import xray_recorder, patch_all

patch_all()  # boto3自動計装

@xray_recorder.capture("moe_inference")
def invoke_moe_model(prompt: str, model_id: str) -> dict:
    """MoEモデル推論をX-Rayでトレース"""
    subsegment = xray_recorder.current_subsegment()
    subsegment.put_annotation("model_id", model_id)
    subsegment.put_metadata("prompt_length", len(prompt))

    response = bedrock.invoke_model(modelId=model_id, body=prompt)
    subsegment.put_metadata("output_tokens", response["usage"]["output_tokens"])
    return response
```

**Cost Explorer 日次レポート**:

```python
import boto3
from datetime import date, timedelta

ce = boto3.client("ce", region_name="us-east-1")

def get_daily_moe_cost() -> dict:
    """MoE関連の日次コストレポート取得"""
    today = date.today()
    yesterday = today - timedelta(days=1)

    response = ce.get_cost_and_usage(
        TimePeriod={"Start": str(yesterday), "End": str(today)},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
        Filter={"Tags": {"Key": "Project", "Values": ["moe-inference"]}},
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )
    return response["ResultsByTime"][0]["Groups"]
```

### コスト最適化チェックリスト

**アーキテクチャ選択**:
- [ ] トラフィック量に応じた構成選択（~100 req/日: Serverless、~1000: Hybrid、10000+: Container）
- [ ] MoEモデルサイズに応じたインスタンスタイプ選定（Mixtral 8x7B: p4d、8x22B: p5）
- [ ] Disaggregated Serving（Prefill/Decode分離）の検討

**リソース最適化**:
- [ ] Spot Instances優先（MoE推論はステートレスで中断耐性が高い）
- [ ] Reserved Instances: GPU 1年コミットで40%削減
- [ ] Savings Plans: Compute Savings Plans検討
- [ ] Lambda: メモリサイズ最適化（Power Tuning）
- [ ] EKS: Karpenter自動スケーリング、夜間スケールダウン

**LLMコスト削減**:
- [ ] Bedrock Batch API使用（非リアルタイム処理で50%削減）
- [ ] Prompt Caching有効化（繰り返しプレフィックスで30-90%削減）
- [ ] モデル選択ロジック（簡単なタスクにはMixtral 8x7B、複雑なタスクにはLarge）
- [ ] トークン数制限（max_tokens設定の最適化）
- [ ] NVFP4量子化モデルの利用（メモリ50%削減）

**監視・アラート**:
- [ ] AWS Budgets設定（月額上限の80%で通知）
- [ ] CloudWatch アラーム（トークン使用量、レイテンシP99）
- [ ] Cost Anomaly Detection有効化
- [ ] 日次コストレポート自動送信（SNS）

**リソース管理**:
- [ ] 未使用GPUインスタンス自動停止
- [ ] タグ戦略（Project, Environment, Team）
- [ ] S3ライフサイクルポリシー（モデルアーティファクト）
- [ ] 開発環境の夜間・週末自動停止
- [ ] EBSボリュームスナップショット管理

## まとめと実践への示唆

NVIDIAのブログは、MoEアーキテクチャがフロンティアAIモデルの標準となった現状を包括的に記述している。日本語LLM選定の観点からは以下が重要である：

1. **コスト効率**: MoEモデルは活性パラメータ数が少ないため、同等性能のdenseモデルと比較して推論コストが大幅に低い。Zenn記事で比較したDeepSeek-V3のAPI料金の安さはMoEアーキテクチャに起因する
2. **推論インフラの進化**: NVL72によりExpert Parallelismが72GPUまでスケールし、MoEモデルの推論性能が飛躍的に向上。今後さらにコスト低下が見込まれる
3. **モデル選定基準**: 活性パラメータ数（≠総パラメータ数）とルーティング品質がMoEモデルの実効性能を決定する

## 参考文献

- **Blog URL**: [https://blogs.nvidia.com/blog/mixture-of-experts-frontier-models/](https://blogs.nvidia.com/blog/mixture-of-experts-frontier-models/)
- **Technical Blog**: [https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/)
- **Related Papers**: Mixtral 8x7B (arXiv: 2401.04088), DeepSeek-V3 (arXiv: 2412.19437), Switch Transformer (arXiv: 2101.03961)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/3a4f2089113d8e](https://zenn.dev/0h_n0/articles/3a4f2089113d8e)
