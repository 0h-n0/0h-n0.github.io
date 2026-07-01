---
layout: post
title: "ICLR 2025論文解説: NV-Embed — Latent Attentionと2段階訓練によるLLMベース汎用埋め込みモデル"
description: "NVIDIAが提案したLatent Attention Layerと2段階contrastive instruction-tuningによりMTEB 72.31を達成した手法を解説"
categories: [blog, paper, conference]
tags: [embedding, nlp, NVIDIA, NV-Embed, MTEB, LLM, retrieval, machinelearning]
date: 2026-07-02 09:40:00 +0900
source_type: conference
conference: ICLR 2025
arxiv_id: "2405.17428"
source_url: https://arxiv.org/abs/2405.17428
zenn_article: b70b9c19e0a825
zenn_url: https://zenn.dev/0h_n0/articles/b70b9c19e0a825
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## 論文概要（Abstract）

本記事は[NV-Embed論文](https://arxiv.org/abs/2405.17428)の解説記事です。

NV-Embedは、NVIDIAが提案したdecoder-only LLMをベースとした汎用埋め込みモデルである。著者らは、decoder最終層の隠れ状態から高品質なシーケンス表現を抽出するLatent Attention Layerと、検索タスクと非検索タスクを段階的に学習する2段階contrastive instruction-tuning手法を提案している。これらの技術によりNV-Embed-v2はMassive Text Embedding Benchmark（MTEB）の56タスクで総合スコア72.31を達成し、オープンウェイトモデルとして当時の最高性能を記録したと報告されている。

この記事は [Zenn記事: Embeddingモデル精度評価の実践：MTEB指標の読み方と最新モデル比較](https://zenn.dev/0h_n0/articles/b70b9c19e0a825) の深掘りです。

## 情報源

- **会議名**: ICLR 2025（International Conference on Learning Representations）
- **年**: 2025
- **URL**: [https://arxiv.org/abs/2405.17428](https://arxiv.org/abs/2405.17428)
- **著者**: Chankyu Lee, Rajarshi Roy, Mengyao Xu, et al.（7名、NVIDIA所属）
- **発表形式**: Spotlight

## カンファレンス情報

ICLR（International Conference on Learning Representations）は、深層学習・表現学習分野の最高峰国際会議の1つである。2025年のICLRではSpotlight採択が全投稿の上位約5%に相当し、高い評価を受けた論文に限定される。NV-EmbedがSpotlightとして採択された点は、LLMベース埋め込みモデルの研究が深層学習コミュニティにおいて高い関心を集めていることを示している。

## 背景と動機（Background & Motivation）

テキスト埋め込みモデルは、検索・分類・クラスタリング・文類似度判定など多岐にわたるNLPタスクの基盤技術である。従来はBERT系のencoder-onlyモデルが主流であったが、LLMの大規模化に伴い、decoder-only LLMを埋め込みモデルとして活用する研究が増加している。

E5-Mistral-7bやGTE-Qwen2など先行研究では、decoder-only LLMの最終トークン（EOS）の隠れ状態をシーケンス表現として抽出する手法が採用されていた。しかし著者らは、この方法には2つの課題があると指摘している。

**第1の課題**: EOSトークンや単純なmean poolingによるシーケンス表現は、長い入力系列の情報を十分に圧縮できない。特にmean poolingでは、各トークンの寄与が均等になるため重要な情報が希釈される。

**第2の課題**: 検索タスクで有効なin-batch negativeが、分類やクラスタリングでは同クラスのペアを誤って負例として扱う問題（false negative）を引き起こす。単一段階の訓練では、検索タスクと非検索タスクの間でこの矛盾を解消できない。

NV-Embedは、これら2つの課題に対してLatent Attention Layerと2段階訓練という明確な解決策を提示している。

## 主要な貢献（Key Contributions）

- **Latent Attention Layer**: dictionary learningに着想を得たcross-attention機構により、decoder最終層の隠れ状態から情報密度の高いシーケンス表現を抽出する新しいpooling手法を提案
- **2段階contrastive instruction-tuning**: 検索タスク（in-batch negative有効）と非検索タスク（in-batch negative無効）を段階的に学習することで、タスク間の訓練目標の矛盾を解消
- **双方向attention**: decoder-only LLMのcausal attention maskを除去し、双方向のself-attentionを適用することでMTEBスコアを大幅に改善
- **MTEB最高性能**: 56タスク総合で72.31を達成し、当時のオープンウェイトモデルで1位を記録
- **モデル圧縮**: SparseGPT pruning + quantizationにより7Bパラメータを3.5B（56%削減）まで圧縮しつつ性能を維持

## 技術的詳細（Technical Details）

### Latent Attention Layer

NV-Embedの中核技術であるLatent Attention Layerは、decoder最終層の隠れ状態$\mathbf{Q} \in \mathbb{R}^{l \times d}$から、固定長の潜在表現を抽出するcross-attention機構である。ここで$l$はシーケンス長、$d$はモデルの隠れ次元（4096）を表す。

この機構では、訓練可能な潜在配列$\mathbf{K} = \mathbf{V} \in \mathbb{R}^{r \times d}$を用意する。$r$は潜在表現の個数で、著者らは$r = 512$を採用している。出力は以下のように計算される:

$$
\mathbf{O} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_h}}\right)\mathbf{V} + \text{MLP}(\text{GELU}(\cdot))
$$

ここで、
- $\mathbf{Q}$: decoderの最終層hidden states（形状: $(l, d)$）、入力テキスト全体のトークン表現
- $\mathbf{K}, \mathbf{V}$: 訓練可能な潜在配列（形状: $(r, d)$）、$r = 512$
- $d_h$: 各attention headの次元数（$d_h = d / h$、$h = 8$ heads）
- MLP: 2層のフィードフォワードネットワーク（GELU活性化関数付き）

8つのattention headで計算された出力に対してmean poolingを適用し、さらにMLPで変換して最終的な埋め込みベクトルを得る。

この設計はdictionary learningに着想を得ている。潜在配列$\mathbf{K}$は「辞書」として機能し、入力シーケンスの各トークンがどの辞書要素に対応するかをattention weightで決定する。これにより単純なmean poolingと異なり、重要なトークンに対して適応的に高い重みを割り当てることが可能になる。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentAttentionLayer(nn.Module):
    """NV-EmbedのLatent Attention Layer

    decoder最終層のhidden statesから固定長の潜在表現を抽出する
    cross-attention機構。dictionary learningに着想を得た設計。

    Args:
        d_model: モデルの隠れ次元数
        n_heads: attention headの数
        n_latents: 潜在配列の要素数（辞書サイズ）
    """

    def __init__(
        self,
        d_model: int = 4096,
        n_heads: int = 8,
        n_latents: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_latents = n_latents

        # 訓練可能な潜在配列（K=V）
        self.latent_array = nn.Parameter(
            torch.randn(n_latents, d_model) * 0.02
        )

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, d_model, bias=False)

        # 出力MLP（GELU活性化）
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Cross-attentionによる潜在表現の抽出

        Args:
            hidden_states: decoder最終層の出力 (batch_size, seq_len, d_model)

        Returns:
            pooled: 固定長の埋め込みベクトル (batch_size, d_model)
        """
        batch_size = hidden_states.size(0)

        # Q: hidden statesから、K/V: 潜在配列から生成
        q = self.q_proj(hidden_states)  # (B, L, D)
        latent = self.latent_array.unsqueeze(0).expand(batch_size, -1, -1)
        kv = self.kv_proj(latent)  # (B, R, D)

        # Multi-head reshaping
        q = q.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        kv = kv.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, kv.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, L, R)
        attn_output = torch.matmul(attn_weights, kv)  # (B, H, L, D_h)

        # Head連結
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # Mean pooling + MLP + residual
        pooled = attn_output.mean(dim=1)  # (B, D)
        pooled = self.layer_norm(pooled + self.mlp(pooled))

        return pooled
```

### 双方向Attention

decoder-only LLMは通常causal attention mask（下三角マスク）を使用するが、埋め込み生成では未来方向のトークンも参照できることが望ましい。著者らはcausal maskを除去して双方向attentionを適用した結果、MTEBスコアがcausal（68.16）から双方向（69.32）へ+1.16ポイント改善したと報告している（論文Table 5より）。

### 2段階Contrastive Instruction-Tuning

**Stage 1: Retrieval Focus**

16種類の検索データセット（MSMARCO、HotpotQA、Natural Questions等）を使用し、contrastive learningを実施する。この段階ではin-batch negativesとcurated hard negativesの両方を活用する。

contrastive lossは以下の形式をとる:

$$
\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(\mathbf{q}, \mathbf{d}^+) / \tau)}{\exp(\text{sim}(\mathbf{q}, \mathbf{d}^+) / \tau) + \sum_{j=1}^{N_{\text{neg}}} \exp(\text{sim}(\mathbf{q}, \mathbf{d}_j^-) / \tau)}
$$

ここで、
- $\mathbf{q}$: クエリの埋め込みベクトル
- $\mathbf{d}^+$: 正例ドキュメントの埋め込みベクトル
- $\mathbf{d}_j^-$: $j$番目の負例ドキュメントの埋め込みベクトル
- $\tau$: 温度パラメータ
- $\text{sim}(\cdot, \cdot)$: コサイン類似度

命令プレフィックスは `"Instruct: {task_description} Query: {text}"` の形式で付与され、タスク固有の情報をモデルに伝達する。

**Stage 2: Generalist**

Stage 1で訓練されたモデルに対し、非検索タスク（分類10種、クラスタリング6種、STS 3種）をブレンドして追加学習する。この段階では**in-batch negativesを無効化**する点が重要である。

理由は明確で、分類タスクやSTSタスクでは同一バッチ内に同じクラスに属するペアが含まれる確率が高い。in-batch negativesを有効にすると、これらの正例ペアを誤って負例として扱ってしまう（false negative問題）。著者らのablation study（論文Table 3より）では、2段階に分離することで単一段階（in-batch on: 70.83、off: 71.94）より高いスコア（72.31）が得られている。

また、非検索タスクではexample-based labelingを採用している。これは、タスクの説明文ではなく具体的な入出力例を命令プレフィックスに含める手法で、モデルがタスク意図をより正確に理解できるよう設計されている。

### Hard Negative Mining

hard negative（正例に近いが負例であるサンプル）の品質が検索タスクの性能を大きく左右する。著者らは以下のしきい値を使用している:

$$
s_{\text{neg}} < s_{\text{pos}} \times 0.95
$$

ここで$s_{\text{pos}}$は正例のスコア、$s_{\text{neg}}$は負例候補のスコアである。このしきい値により、正例に十分近いが正例ではないサンプルをhard negativeとして選択する。さらに、Mixtral-8x22Bを用いて120K件の合成データを生成し、訓練データを拡張している。

hard negative miningの段階的な改善効果は以下のとおりである（論文Table 6より）:
- ベースライン: 70.73
- +hard negatives: 71.83（+1.10）
- +追加データ: 72.07（+0.24）
- +合成データ: 72.31（+0.24）

## 実装のポイント（Implementation）

### HuggingFaceでの利用

NV-Embed-v2は[HuggingFace Model Hub](https://huggingface.co/nvidia/NV-Embed-v2)で公開されている。推論時のポイントを以下に示す。

```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F


def get_nv_embed_embeddings(
    texts: list[str],
    instruction: str = "",
    model_name: str = "nvidia/NV-Embed-v2",
    max_length: int = 32768,
) -> torch.Tensor:
    """NV-Embed-v2で埋め込みベクトルを取得する

    Args:
        texts: 埋め込み対象のテキスト群
        instruction: タスク固有の命令プレフィックス
        model_name: HuggingFaceモデル名
        max_length: 最大トークン長

    Returns:
        L2正規化された埋め込みベクトル (num_texts, 4096)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    # instructionがある場合はプレフィックスとして付与
    if instruction:
        texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]

    with torch.no_grad():
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        outputs = model(**inputs)
        # Latent Attention Layerによるpooled output
        embeddings = outputs.last_hidden_state.mean(dim=1)

    # L2正規化（コサイン類似度計算の前処理）
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings
```

**推論時の注意点**:
- **GPU要件**: 7Bモデルのため、推論にはA100 40GB以上のGPUが推奨される。FP16で約14GB、INT8量子化で約7GBのVRAMを使用する
- **最大入力長**: 32768トークンまで対応。長文埋め込みにも適用可能
- **出力次元**: 4096次元。下流タスクによってはPCAやMatryoshka表現学習で次元削減が有効な場合がある
- **命令プレフィックス**: 検索タスクでは必ず`Instruct:`プレフィックスを付与する。ドキュメント側にはプレフィックスを付けない非対称設計

## Production Deployment Guide

NV-Embed-v2は7Bパラメータの大規模モデルであり、プロダクション環境ではGPU推論が必須となる。以下にAWS上でのデプロイパターンをトラフィック量別に示す。

### AWS実装パターン（GPU推論・コスト最適化重視）

NV-Embed-v2（7B）はGPUインスタンスでの推論が前提となる。圧縮版（3.5B、SparseGPT pruning + INT8量子化）を使用すればVRAM要件を緩和できる。

**トラフィック量別の推奨構成**:

| 構成 | トラフィック | インフラ | GPU | 月額概算 |
|------|-------------|---------|-----|---------|
| Small | ~500 req/日 | SageMaker Serverless | g5.xlarge (A10G 24GB) | $200-400 |
| Medium | ~5,000 req/日 | ECS + g5.xlarge | A10G 24GB × 1 | $800-1,500 |
| Large | 50,000+ req/日 | EKS + Karpenter + Spot | g5.xlarge × 複数 | $3,000-8,000 |

**Small構成**: SageMaker Serverless Inference（圧縮3.5B版）。コールドスタートに10-30秒かかるが、低頻度のバッチ処理には十分である。月額コスト内訳: SageMaker推論 $150-300、S3モデルストレージ $5、CloudWatch $10-20。

**Medium構成**: ECS Fargate + g5.xlarge GPU。常時稼働の1インスタンスで、平均レイテンシ50-100ms（バッチサイズ32時）を実現する。月額コスト内訳: g5.xlarge On-Demand $800/月、ECS Fargate管理 $50、ALB $20-30、CloudWatch $20。

**Large構成**: EKS + Karpenter + g5.xlarge Spot Instances。Spot活用で最大60-70%のコスト削減が可能（g5.xlargeのSpot割引率は変動するが、2026年7月時点でap-northeast-1リージョンでは60-70%程度）。Karpenterによる自動スケーリングで、トラフィックの増減に柔軟に対応する。

**コスト試算の注意事項**: 上記はAWS ap-northeast-1（東京）リージョンの2026年7月時点の概算値である。実際のコストはトラフィックパターン、リージョン、Spotインスタンスの可用性により変動する。最新料金はAWS料金計算ツールで確認を推奨する。

**コスト削減テクニック**:
- **Spot Instances**: g5.xlarge Spotで最大60-70%削減（中断時のフォールバック設計が必要）
- **Reserved Instances**: 1年コミットで最大40%削減
- **モデル量子化**: INT8量子化でg5.xlarge 1台に収まり、g5.2xlargeを回避
- **動的バッチ処理**: リクエストを一定時間バッファリングし、バッチ推論でスループットを向上

### Terraformインフラコード

#### Small構成（SageMaker Serverless）

```hcl
# NV-Embed-v2 SageMaker Serverless推論
# 圧縮3.5B版を使用、低頻度バッチ処理向け

terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}

provider "aws" {
  region = "ap-northeast-1"
}

# IAMロール（最小権限）
resource "aws_iam_role" "sagemaker_execution" {
  name = "nv-embed-sagemaker-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "sagemaker_s3_access" {
  name = "nv-embed-s3-access"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.model_artifacts.arn,
          "${aws_s3_bucket.model_artifacts.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        # ECRからコンテナイメージをpullするための権限
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer"
        ]
        Resource = "*"
      }
    ]
  })
}

# モデルアーティファクト用S3バケット
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "nv-embed-model-artifacts-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# SageMakerモデル
resource "aws_sagemaker_model" "nv_embed" {
  name               = "nv-embed-v2-compressed"
  execution_role_arn = aws_iam_role.sagemaker_execution.arn

  primary_container {
    image          = "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu22.04-sagemaker"
    model_data_url = "s3://${aws_s3_bucket.model_artifacts.bucket}/nv-embed-v2-compressed/model.tar.gz"

    environment = {
      SAGEMAKER_MODEL_SERVER_TIMEOUT = "300"
      TS_MAX_RESPONSE_SIZE           = "104857600"
    }
  }
}

# SageMaker Serverless Endpointの設定
resource "aws_sagemaker_endpoint_configuration" "nv_embed" {
  name = "nv-embed-v2-serverless-config"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.nv_embed.name
    serverless_config {
      max_concurrency       = 5
      memory_size_in_mb     = 6144
      provisioned_concurrency = 0  # コスト削減: コールドスタート許容
    }
  }
}

resource "aws_sagemaker_endpoint" "nv_embed" {
  name                 = "nv-embed-v2-serverless"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.nv_embed.name
}

# CloudWatchアラーム（コスト監視）
resource "aws_cloudwatch_metric_alarm" "invocation_spike" {
  alarm_name          = "nv-embed-invocation-spike"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Invocations"
  namespace           = "AWS/SageMaker"
  period              = 3600
  statistic           = "Sum"
  threshold           = 1000
  alarm_description   = "NV-Embed invocation spike detection"

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.nv_embed.name
    VariantName  = "primary"
  }

  alarm_actions = [aws_sns_topic.alerts.arn]
}

resource "aws_sns_topic" "alerts" {
  name = "nv-embed-alerts"
}

data "aws_caller_identity" "current" {}
```

#### Large構成（EKS + Karpenter + Spot）

```hcl
# NV-Embed-v2 EKS + Karpenter GPU推論クラスタ
# 大規模トラフィック向け、Spot優先でコスト最適化

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.24"

  cluster_name    = "nv-embed-inference"
  cluster_version = "1.31"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Karpenter用のIAM設定
  enable_cluster_creator_admin_permissions = true

  # パブリックアクセス最小化
  cluster_endpoint_public_access = false

  # EKSアドオン
  cluster_addons = {
    coredns    = { most_recent = true }
    kube-proxy = { most_recent = true }
    vpc-cni    = { most_recent = true }

    # GPU対応: NVIDIA device plugin
    eks-pod-identity-agent = { most_recent = true }
  }
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.13"

  name = "nv-embed-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["ap-northeast-1a", "ap-northeast-1c", "ap-northeast-1d"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true  # コスト削減: 本番ではAZ毎に推奨
}

# Karpenter Provisioner（Spot優先、g5.xlarge GPU）
module "karpenter" {
  source  = "terraform-aws-modules/eks/aws//modules/karpenter"
  version = "~> 20.24"

  cluster_name = module.eks.cluster_name

  node_iam_role_additional_policies = {
    AmazonSSMManagedInstanceCore = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }
}

# Karpenter NodePool（GPU Spot優先）
resource "kubectl_manifest" "karpenter_node_pool" {
  yaml_body = yamlencode({
    apiVersion = "karpenter.sh/v1"
    kind       = "NodePool"
    metadata = {
      name = "gpu-inference"
    }
    spec = {
      template = {
        spec = {
          requirements = [
            {
              key      = "karpenter.sh/capacity-type"
              operator = "In"
              values   = ["spot", "on-demand"]  # Spot優先
            },
            {
              key      = "node.kubernetes.io/instance-type"
              operator = "In"
              values   = ["g5.xlarge", "g5.2xlarge"]
            },
            {
              key      = "kubernetes.io/arch"
              operator = "In"
              values   = ["amd64"]
            }
          ]
          nodeClassRef = {
            group = "karpenter.k8s.aws"
            kind  = "EC2NodeClass"
            name  = "gpu-default"
          }
        }
      }
      disruption = {
        consolidationPolicy = "WhenEmptyOrUnderutilized"
        consolidateAfter    = "60s"
      }
      limits = {
        cpu    = "64"
        memory = "256Gi"
        "nvidia.com/gpu" = "8"  # 最大GPUノード数
      }
    }
  })
}

# Secrets Manager（モデル設定）
resource "aws_secretsmanager_secret" "model_config" {
  name        = "nv-embed/model-config"
  description = "NV-Embed-v2 model configuration"
  kms_key_id  = aws_kms_key.secrets.arn
}

resource "aws_kms_key" "secrets" {
  description             = "KMS key for NV-Embed secrets"
  deletion_window_in_days = 7
  enable_key_rotation     = true
}

# AWS Budgets（予算アラート）
resource "aws_budgets_budget" "nv_embed" {
  name         = "nv-embed-monthly-budget"
  budget_type  = "COST"
  limit_amount = "5000"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["ml-ops@example.com"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = ["ml-ops@example.com"]
  }
}
```

### 運用・監視設定

#### CloudWatch Logs Insightsクエリ

```
# GPU推論レイテンシ分析（P95, P99）
fields @timestamp, @message
| filter @message like /inference_latency/
| stats avg(inference_latency_ms) as avg_ms,
        percentile(inference_latency_ms, 95) as p95_ms,
        percentile(inference_latency_ms, 99) as p99_ms,
        count(*) as request_count
  by bin(1h) as period
| sort period desc

# GPU utilization監視
fields @timestamp, gpu_utilization, gpu_memory_used_mb
| filter gpu_utilization > 0
| stats avg(gpu_utilization) as avg_util,
        max(gpu_memory_used_mb) as max_vram_mb
  by bin(5m)
| sort @timestamp desc
```

#### CloudWatchアラーム設定

```python
import boto3


def create_gpu_inference_alarms(
    endpoint_name: str,
    sns_topic_arn: str,
    region: str = "ap-northeast-1",
) -> list[str]:
    """GPU推論エンドポイントの監視アラームを作成する

    Args:
        endpoint_name: SageMaker/ECSエンドポイント名
        sns_topic_arn: 通知先SNSトピックARN
        region: AWSリージョン

    Returns:
        作成されたアラームARNのリスト
    """
    cw = boto3.client("cloudwatch", region_name=region)
    alarm_arns: list[str] = []

    # GPU推論レイテンシ異常検知（P99 > 500ms）
    cw.put_metric_alarm(
        AlarmName=f"{endpoint_name}-latency-p99",
        MetricName="ModelLatency",
        Namespace="AWS/SageMaker",
        Statistic="p99",
        Period=300,
        EvaluationPeriods=3,
        Threshold=500000,  # マイクロ秒単位
        ComparisonOperator="GreaterThanThreshold",
        AlarmActions=[sns_topic_arn],
        Dimensions=[
            {"Name": "EndpointName", "Value": endpoint_name},
            {"Name": "VariantName", "Value": "primary"},
        ],
    )
    alarm_arns.append(f"{endpoint_name}-latency-p99")

    # 推論スループット低下検知
    cw.put_metric_alarm(
        AlarmName=f"{endpoint_name}-throughput-drop",
        MetricName="InvocationsPerInstance",
        Namespace="AWS/SageMaker",
        Statistic="Average",
        Period=600,
        EvaluationPeriods=2,
        Threshold=10,
        ComparisonOperator="LessThanThreshold",
        AlarmActions=[sns_topic_arn],
        Dimensions=[
            {"Name": "EndpointName", "Value": endpoint_name},
        ],
    )
    alarm_arns.append(f"{endpoint_name}-throughput-drop")

    return alarm_arns
```

#### Cost Explorer自動レポート

```python
import boto3
from datetime import datetime, timedelta


def get_daily_gpu_inference_cost(
    service_filter: str = "Amazon SageMaker",
    region: str = "ap-northeast-1",
    alert_threshold_usd: float = 100.0,
) -> dict:
    """GPU推論の日次コストを取得し、しきい値超過時にアラートを返す

    Args:
        service_filter: AWSサービス名でのフィルタ
        region: AWSリージョン
        alert_threshold_usd: アラートしきい値（USD/日）

    Returns:
        日次コスト情報と超過フラグ
    """
    ce = boto3.client("ce", region_name=region)

    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    response = ce.get_cost_and_usage(
        TimePeriod={"Start": start_date, "End": end_date},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
        Filter={
            "Dimensions": {
                "Key": "SERVICE",
                "Values": [service_filter],
            }
        },
        GroupBy=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}],
    )

    daily_costs: list[dict] = []
    alert_triggered = False

    for result in response["ResultsByTime"]:
        date = result["TimePeriod"]["Start"]
        total = sum(
            float(g["Metrics"]["UnblendedCost"]["Amount"])
            for g in result["Groups"]
        )
        daily_costs.append({"date": date, "cost_usd": round(total, 2)})

        if total > alert_threshold_usd:
            alert_triggered = True

    return {
        "daily_costs": daily_costs,
        "alert_triggered": alert_triggered,
        "threshold_usd": alert_threshold_usd,
    }
```

### コスト最適化チェックリスト

**アーキテクチャ選択**:
- [ ] トラフィック量に応じた構成を選定（Small: SageMaker Serverless / Medium: ECS GPU / Large: EKS + Karpenter）
- [ ] バッチ処理が可能な場合はリアルタイム推論を回避し、SageMaker Batch Transformを検討

**リソース最適化**:
- [ ] g5.xlarge Spot Instancesを優先使用（On-Demand比60-70%削減）
- [ ] Reserved Instances: 安定トラフィックのベースライン分を1年コミット（最大40%削減）
- [ ] Savings Plans: EC2/SageMakerのCompute Savings Plansを検討
- [ ] モデル量子化: INT8量子化でVRAM使用量を半減（7B → 3.5B相当）
- [ ] Karpenter consolidation: アイドルGPUノードの自動縮退を60秒に設定

**推論コスト削減**:
- [ ] 動的バッチ処理: リクエストを50-100ms間バッファリングしバッチ推論
- [ ] 埋め込みキャッシュ: 同一テキストの再計算を避けるためRedis/DynamoDBにキャッシュ
- [ ] 次元削減: 4096次元から1024次元にPCA圧縮し、ストレージ・検索コストを削減
- [ ] 段階的推論: 軽量モデル（E5-small等）でフィルタリング後、NV-Embedで精密推論

**監視・アラート**:
- [ ] AWS Budgets: 月額予算を設定し、80%/100%でアラート
- [ ] CloudWatch: GPU utilization、推論レイテンシ、スループットを監視
- [ ] Cost Anomaly Detection: MLベースのコスト異常検知を有効化
- [ ] 日次コストレポート: Cost Explorer APIで自動取得、$100/日超過でSNS通知

**リソース管理**:
- [ ] 未使用SageMakerエンドポイントの定期削除
- [ ] タグ戦略: `project:nv-embed`、`env:prod/dev`でコスト按分
- [ ] S3ライフサイクル: 古いモデルアーティファクトを90日後にGlacierへ移行
- [ ] 開発環境: 営業時間外のGPUインスタンス自動停止（EventBridge + Lambda）
- [ ] ECRイメージ: 古いイメージのライフサイクルポリシー設定（最新5世代保持）

## 実験結果（Results）

### MTEBベンチマーク

著者らは56タスクで構成されるMTEB（Massive Text Embedding Benchmark）でNV-Embed-v2を評価している。カテゴリ別のスコアは以下のとおりである（論文Table 1より）:

| カテゴリ | タスク数 | スコア | 指標 |
|---------|---------|--------|------|
| Retrieval | 15 | 62.65 | nDCG@10 |
| Reranking | 4 | 60.65 | MAP |
| Clustering | 11 | 58.46 | V-Measure |
| Pair Classification | 3 | 88.67 | AP |
| Classification | 12 | 90.37 | Accuracy |
| STS | 10 | 84.31 | Spearman |
| Summarization | 1 | 30.70 | — |
| **Overall** | **56** | **72.31** | — |

### 競合モデルとの比較

論文Table 1およびMTEBリーダーボードに基づく主要モデルとの比較を示す:

| モデル | パラメータ数 | MTEB Overall | 備考 |
|--------|-------------|-------------|------|
| **NV-Embed-v2** | **7B** | **72.31** | **Spotlight@ICLR 2025** |
| BGE-en-icl | 7B | 71.24 | BAAI |
| Stella-1.5B-v5 | 1.5B | 71.19 | 小型モデル |
| SFR-Embedding-2R | 7B | 70.31 | Salesforce |
| GTE-Qwen2-7B | 7B | 70.24 | Alibaba |
| E5-Mistral-7b | 7B | 66.63 | Microsoft |
| text-embed-3-large | — | 64.59 | OpenAI API |

NV-Embed-v2は7Bクラスのモデルの中で最高スコアを達成しており、特にRetrievalとClassificationで競合を上回っている。

### Ablation Study

pooling手法の比較（論文Table 4より）:

| Pooling手法 | MTEB Overall |
|------------|-------------|
| EOS-last token | 71.63 |
| Mean pooling | 71.71 |
| Self-attention pooling | 71.61 |
| **Latent Attention** | **72.31** |

Latent Attentionがmean poolingに対して+0.60ポイントの改善を示している。EOS-lastトークン方式やself-attention poolingと比較しても一貫して優れた性能を達成している。

## 実運用への応用（Practical Applications）

NV-Embed-v2は多様なNLPタスクに適用可能であるが、プロダクション環境での利用では以下の点が重要となる。

**RAGパイプラインへの適用**: 関連Zenn記事で解説されているMTEB指標を用いたモデル選定において、NV-Embed-v2はnDCG@10で62.65を記録しており、検索精度の高さが確認されている。RAGの検索段階でのドキュメント埋め込みに適しているが、4096次元のベクトルはストレージと検索のコストが高いため、Matryoshka表現学習やPCAによる次元削減の検討が推奨される。

**セマンティック検索**: 32768トークンの最大入力長により、長文ドキュメントの埋め込みが可能である。技術文書、法律文書、学術論文など、コンテキスト長が長いドメインでの検索に適している。

**テキスト分類・クラスタリング**: Classification 90.37（Accuracy）、Clustering 58.46（V-Measure）のスコアが示すように、教師あり分類と教師なしクラスタリングの両方で高い性能を発揮する。命令プレフィックスによるタスク切り替えにより、1つのモデルで複数タスクに対応可能である。

**レイテンシ・スループットの考慮**: 7Bモデルの推論には数十ミリ秒（バッチ処理時）から数百ミリ秒（単一リクエスト時）を要する。リアルタイム性が求められるアプリケーションでは、軽量モデル（Stella-1.5B等）とのカスケード構成や、圧縮版（3.5B）の活用を検討すべきである。

## 関連研究（Related Work）

- **E5-Mistral-7b**（Wang et al., 2024）: Mistral-7Bをベースとした埋め込みモデルの先駆的研究。NV-Embedと同じベースモデルを使用するが、pooling手法としてEOS-lastトークンを採用しており、NV-EmbedのLatent Attentionはこの手法を上回っている
- **GTE-Qwen2-7B**（Alibaba, 2024）: Qwen2-7Bベースの埋め込みモデル。bidirectional attentionの適用など共通するアプローチがあるが、2段階訓練は採用していない
- **BGE-en-icl**（BAAI, 2024）: in-context learning方式で検索タスクの性能を向上させたモデル。NV-Embed-v2はBGE-en-iclを+1.07ポイント上回っている
- **Sentence-BERT**（Reimers & Gurevych, 2019）: BERT系encoder-onlyモデルによる文埋め込みの基礎研究。NV-Embedはdecoder-only LLMへのパラダイムシフトを示す位置づけにある

## まとめと今後の展望

NV-Embedは、decoder-only LLMを汎用埋め込みモデルとして活用するための実践的な手法を提示した論文である。Latent Attention Layerによる適応的なpoolingと、2段階contrastive instruction-tuningによるタスク間の訓練目標の分離が、MTEB 72.31という当時の最高スコアの達成に寄与している。

ICLR 2025でSpotlightとして採択されたことは、LLMベース埋め込みモデルの研究が実用性と学術的新規性の両面で高く評価されていることを示している。今後は、より効率的なモデル圧縮手法の探索、多言語対応の強化、そしてマルチモーダル埋め込みへの拡張が研究の方向性として考えられる。

## 参考文献

- **arXiv**: [https://arxiv.org/abs/2405.17428](https://arxiv.org/abs/2405.17428)
- **HuggingFace**: [https://huggingface.co/nvidia/NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2)
- **MTEB Leaderboard**: [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/b70b9c19e0a825](https://zenn.dev/0h_n0/articles/b70b9c19e0a825)
