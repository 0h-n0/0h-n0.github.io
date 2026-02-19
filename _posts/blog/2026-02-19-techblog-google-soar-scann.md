---
layout: post
title: "Google Research解説: SOAR — ScaNNを加速する直交残差スピリングアルゴリズム"
description: "Google ResearchのSOAR（Spilling with Orthogonality-Amplified Residuals）を徹底解説。ScaNNの検索精度を冗長性の数学的最適化で向上させ、NeurIPS 2023で発表"
categories: [blog, tech_blog]
tags: [ScaNN, SOAR, approximate-nearest-neighbor, google, NeurIPS, quantization, vectordb, rag]
date: 2026-02-19 13:00:00 +0900
source_type: tech_blog
source_domain: research.google
source_url: https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/
zenn_article: 8c8bb192985b64
zenn_url: https://zenn.dev/0h_n0/articles/8c8bb192985b64
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

Google Researchが公開したSOAR（Spilling with Orthogonality-Amplified Residuals）は、ScaNN（Scalable Nearest Neighbors）ライブラリの検索精度を大幅に向上させる新アルゴリズムです。NeurIPS 2023で発表され、従来のIVF（Inverted File）ベースの検索における「クエリがクラスタ境界に近い場合の精度低下」という根本的な課題を、数学的に最適化された冗長性（spilling）によって解決します。Big-ANN 2023ベンチマークでout-of-distributionトラックとstreamingトラックの双方でSOTAを達成しています。

この記事は [Zenn記事: 2026年版ベクトルDB選定ガイド：pgvector・Qdrant・Pineconeを本番ベンチマークで比較](https://zenn.dev/0h_n0/articles/8c8bb192985b64) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（Google Research Blog）
- **URL**: [Google Research: SOAR](https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/)
- **組織**: Google Research
- **発表日**: 2024年4月10日
- **著者**: Philip Sun, Ruiqi Guo (Software Engineers, Google Research)
- **関連論文**: NeurIPS 2023採択

## 技術的背景（Technical Background）

### ScaNNの基本アーキテクチャ

ScaNN（Scalable Nearest Neighbors）は、Googleが開発したオープンソースのベクトル検索ライブラリです。Google検索、YouTube推薦、Gmail等の大規模サービスで使用されています。

ScaNNの検索は3段階のパイプラインで構成されます：

1. **パーティショニング**: k-meansでベクトル空間を $C$ 個のクラスタに分割
2. **スコアリング**: 量子化ベースの近似距離計算（Asymmetric Hashing）
3. **リスコアリング**: 上位候補に対する正確な距離再計算

**従来のIVFの課題**:

各ベクトル $\mathbf{x}$ は最も近いクラスタセントロイド $\mu_c$ に割り当てられます。検索時、クエリ $\mathbf{q}$ に最も近いクラスタのみを探索しますが、クラスタ境界付近のベクトルは見逃されるリスクがあります。

$$
\text{error}(\mathbf{q}, \mathbf{x}) = |d(\mathbf{q}, \mathbf{x}) - \hat{d}(\mathbf{q}, \mathbf{x})|
$$

ここで $\hat{d}$ は残差ベクトル $\mathbf{r} = \mathbf{x} - \mu_c$ を用いた近似距離です。

**問題の核心**: クエリ $\mathbf{q}$ が残差 $\mathbf{r}$ と平行に近い場合、内積の近似誤差が最大化されます。

$$
\langle \mathbf{q}, \mathbf{x} \rangle = \langle \mathbf{q}, \mu_c \rangle + \langle \mathbf{q}, \mathbf{r} \rangle
$$

第2項 $\langle \mathbf{q}, \mathbf{r} \rangle$ は、$\mathbf{q}$ と $\mathbf{r}$ が平行のとき最大になり、量子化誤差が最も大きくなります。

### Spilling（スピリング）の概念

SpillingはIVFの精度問題に対する古典的なアプローチで、各ベクトルを複数のクラスタに割り当てます。これにより、クラスタ境界での見逃しリスクが低下しますが、メモリ使用量とインデックスサイズが増加するトレードオフがあります。

従来のSpillingは「2番目に近いクラスタ」に機械的に割り当てていたため、冗長性の効果が最適ではありませんでした。

## 実装アーキテクチャ（Architecture）

### SOARの核心アイデア

SOARは、**二次クラスタ割り当てを数学的に最適化**します。キーとなる洞察は：

> クエリ $\mathbf{q}$ が一次残差 $\mathbf{r}$ と平行のとき（近似誤差が最大）、$\mathbf{q}$ は二次残差 $\mathbf{r}'$ と直交する。この直交性を利用すれば、二次クラスタでの近似誤差を最小化できる。

**一次残差と二次残差の直交性**:

$$
\mathbf{r} = \mathbf{x} - \mu_{c_1} \quad (\text{一次残差})
$$

$$
\mathbf{r}' = \mathbf{x} - \mu_{c_2} \quad (\text{二次残差})
$$

SOARは、$\mathbf{r}$ と $\mathbf{r}'$ が可能な限り直交するように $c_2$ を選択します：

$$
c_2^* = \arg\min_{c \neq c_1} \frac{|\langle \mathbf{r}, \mathbf{x} - \mu_c \rangle|}{\|\mathbf{r}\| \cdot \|\mathbf{x} - \mu_c\|}
$$

これにより、一次クラスタでの近似が悪いケース（$\mathbf{q} \parallel \mathbf{r}$）で、二次クラスタが正確な近似を提供する「保険」として機能します。

### 数学的な直感

内積の近似誤差は、クエリと残差の平行度に比例します：

$$
\text{error} \propto |\langle \mathbf{q}, \mathbf{r} \rangle| = \|\mathbf{q}\| \cdot \|\mathbf{r}\| \cdot |\cos\theta|
$$

$\theta$ はクエリと残差のなす角度です。

- $\theta \approx 0$（平行）: 一次クラスタの誤差が最大 → **二次クラスタが救済**
- $\theta \approx \pi/2$（直交）: 一次クラスタの誤差が最小 → 二次クラスタは不要

SOARは $\mathbf{r} \perp \mathbf{r}'$ を保証するため、一次と二次のどちらかが必ず低誤差となります。

### アルゴリズムの擬似コード

```python
import numpy as np
from typing import NamedTuple

class SOARIndex(NamedTuple):
    centroids: np.ndarray       # shape: (C, D)
    primary_assign: np.ndarray  # shape: (N,) - 一次クラスタ割り当て
    secondary_assign: np.ndarray # shape: (N,) - 二次クラスタ割り当て（SOAR最適化）
    residuals: dict[int, np.ndarray]  # クラスタ→残差ベクトル

def soar_build_index(
    vectors: np.ndarray,  # (N, D)
    centroids: np.ndarray,  # (C, D)
) -> SOARIndex:
    """SOARインデックスの構築

    Args:
        vectors: データベクトル
        centroids: k-meansクラスタセントロイド

    Returns:
        SOARIndex: 最適化されたインデックス
    """
    N, D = vectors.shape
    C = len(centroids)

    # 一次クラスタ割り当て（最近傍セントロイド）
    distances = np.linalg.norm(
        vectors[:, None, :] - centroids[None, :, :], axis=2
    )  # (N, C)
    primary_assign = distances.argmin(axis=1)

    # SOAR: 直交性最大化による二次クラスタ割り当て
    secondary_assign = np.zeros(N, dtype=int)
    for i in range(N):
        c1 = primary_assign[i]
        r1 = vectors[i] - centroids[c1]  # 一次残差

        best_score = float('inf')
        best_c2 = -1

        for c in range(C):
            if c == c1:
                continue
            r2 = vectors[i] - centroids[c]  # 候補二次残差

            # 直交性スコア（小さいほど直交に近い）
            cos_sim = abs(np.dot(r1, r2)) / (
                np.linalg.norm(r1) * np.linalg.norm(r2) + 1e-8
            )

            if cos_sim < best_score:
                best_score = cos_sim
                best_c2 = c

        secondary_assign[i] = best_c2

    return SOARIndex(
        centroids=centroids,
        primary_assign=primary_assign,
        secondary_assign=secondary_assign,
        residuals={},  # 省略
    )


def soar_search(
    index: SOARIndex,
    query: np.ndarray,
    k: int = 10,
    n_probe: int = 20,
) -> list[tuple[int, float]]:
    """SOAR検索

    Args:
        query: クエリベクトル (D,)
        k: 返却件数
        n_probe: 探索するクラスタ数

    Returns:
        (id, distance) のリスト
    """
    # 1. クエリに近いn_probeクラスタを選択
    distances_to_centroids = np.linalg.norm(
        index.centroids - query, axis=1
    )
    probe_clusters = distances_to_centroids.argsort()[:n_probe]

    # 2. 一次割り当て + 二次割り当て（SOAR）のベクトルを収集
    candidates = set()
    for c in probe_clusters:
        # 一次割り当てがこのクラスタのベクトル
        primary_mask = index.primary_assign == c
        candidates.update(np.where(primary_mask)[0])

        # SOAR: 二次割り当てがこのクラスタのベクトルも追加
        secondary_mask = index.secondary_assign == c
        candidates.update(np.where(secondary_mask)[0])

    # 3. 候補に対してリスコアリング（正確な距離計算）
    # ... (省略)

    return sorted_results[:k]
```

### ScaNN全体のパイプライン

```
クエリ q
  │
  ▼
┌──────────────────────┐
│ 1. パーティショニング   │  ← k-meansセントロイドとの距離計算
│    n_probe クラスタ選択 │     O(C × D)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 2. SOAR スピリング    │  ← 一次 + 二次クラスタのベクトルを収集
│    候補セット構築      │     メモリオーバーヘッド: ~1.3x
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 3. 量子化スコアリング  │  ← Asymmetric Hashing
│    近似距離計算        │     O(候補数 × m)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 4. リスコアリング      │  ← 上位候補のfloat32正確距離
│    Top-k 返却         │     O(rerank_count × D)
└──────────────────────┘
```

## パフォーマンス最適化（Performance）

### Big-ANN 2023 ベンチマーク結果

SOARは Big-ANN 2023 で以下のトラックでSOTAを達成：

**Out-of-Distribution トラック**:
- 学習データとクエリデータの分布が異なる設定
- 従来手法比で20-30%のQPS向上（同Recall条件）

**Streaming トラック**:
- リアルタイムデータ更新がある設定
- 更新と検索の同時実行で、従来手法比15-25%のQPS向上

### SOARの性能特性

| 指標 | 標準IVF | IVF + 2x Spilling | IVF + SOAR |
|------|---------|-------------------|------------|
| Recall@10 (n_probe=20) | 0.88 | 0.93 | 0.95 |
| メモリオーバーヘッド | 1.0x | 2.0x | 1.3x |
| インデックス構築時間 | 1.0x | 1.5x | 1.2x |
| QPS (Recall=0.95) | 800 | 1,200 | 1,800 |

SOARは標準的な2倍Spillingと比較して、メモリオーバーヘッドを65%削減しながら、QPSを50%向上させています。

### ハードウェア親和性

SOARの設計は、現代のCPU/GPUアーキテクチャに最適化されています：

- **SIMD命令**: 量子化距離計算をSIMD（AVX-512等）で並列化
- **キャッシュ効率**: クラスタ内のベクトルが連続メモリに配置されるため、L1/L2キャッシュヒット率が高い
- **メモリアクセスパターン**: 二次割り当てのメモリオーバーヘッドが小さい（1.3x）ため、メモリ帯域がボトルネックになりにくい

## 運用での学び（Production Lessons）

### Googleでの本番運用

ScaNN（SOAR含む）は以下のGoogleサービスで運用されています：

- **Google検索**: ウェブページの埋め込みベクトルからの関連ページ検索
- **YouTube推薦**: 動画の埋め込みベクトルからのレコメンデーション
- **Google Cloud Vertex AI Vector Search**: マネージドベクトル検索サービス
- **AlloyDB**: PostgreSQL互換DBのScaNNインデックス

**スケーリング実績**:
- 数十億件のベクトルインデックス
- 数百万QPS
- p99レイテンシ 10ms未満

### 学術研究からプロダクションへの橋渡し

SOARの特筆すべき点は、NeurIPS 2023で発表された学術研究が、すぐにGoogle Cloud（Vertex AI）で利用可能になったことです。これは、研究チーム（Philip Sun, Ruiqi Guo）がプロダクションチームと密接に連携している Google Research の組織的な強みを反映しています。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $60-150 | Lambda + EC2 (ScaNN) |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $300-700 | ECS Fargate + EC2 (ScaNN cluster) |
| **Large** | 300,000+ (10,000/日) | Container | $2,000-5,000 | EKS + EC2 c6i/r6i cluster |

**Small構成の詳細**（月額$60-150）:
- **EC2 c6i.large**: 2 vCPU, 4GB RAM ($60/月) — ScaNNはCPU最適化（AVX-512活用）
- **Lambda**: 検索API ($20/月)
- **S3**: インデックスストレージ ($5/月)
- **CloudWatch**: 監視 ($5/月)

**Medium構成の詳細**（月額$300-700）:
- **EC2 c6i.xlarge × 2**: 4 vCPU, 8GB RAM各 ($240/月)
- **ECS Fargate**: API 0.5 vCPU × 2 ($120/月)
- **ElastiCache Redis**: クエリキャッシュ ($15/月)
- **ALB**: ($20/月)

**コスト削減テクニック**:
- ScaNNはメモリ効率が良いため、c6iファミリー（CPU最適化）で十分
- SOAR spillingのメモリオーバーヘッドはわずか1.3x
- Spot Instances（c6i）で最大70%削減（ScaNNはステートレス化可能）
- S3からインデックスロード → Spot中断時の復旧が容易

**コスト試算の注意事項**:
- 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です
- ScaNNはCPU集約型のため、GPU instancesは不要（c6i/r6iで十分）
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "scann-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["ap-northeast-1a", "ap-northeast-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true
}

resource "aws_instance" "scann_server" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "c6i.xlarge"  # CPU最適化（AVX-512対応）
  subnet_id     = module.vpc.private_subnets[0]

  root_block_device {
    volume_type = "gp3"
    volume_size = 50
    encrypted   = true
  }

  user_data = <<-EOF
    #!/bin/bash
    pip install scann tensorflow
    # ScaNN + SOARインデックスをS3からダウンロード
    aws s3 cp s3://scann-indexes/production/ /data/scann/ --recursive
  EOF

  tags = {
    Name = "scann-search-server"
    Env  = "production"
  }
}

# --- Auto Scaling Group (Spot対応) ---
resource "aws_launch_template" "scann" {
  name_prefix   = "scann-"
  instance_type = "c6i.xlarge"
  image_id      = data.aws_ami.ubuntu.id

  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price = "0.10"  # オンデマンドの約70%
    }
  }
}

resource "aws_autoscaling_group" "scann" {
  desired_capacity = 2
  max_size         = 6
  min_size         = 1

  launch_template {
    id      = aws_launch_template.scann.id
    version = "$Latest"
  }

  vpc_zone_identifier = module.vpc.private_subnets
}

resource "aws_budgets_budget" "scann" {
  name         = "scann-monthly"
  budget_type  = "COST"
  limit_amount = "1000"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = ["ops@example.com"]
  }
}
```

### 運用・監視設定

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# ScaNN検索レイテンシ監視
cloudwatch.put_metric_alarm(
    AlarmName='scann-search-latency-p99',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=2,
    MetricName='SearchLatencyP99',
    Namespace='ScaNN',
    Period=300,
    Statistic='Maximum',
    Threshold=20,  # 20ms
    AlarmDescription='ScaNN検索p99レイテンシ20ms超過'
)

# CPU使用率監視（ScaNNはCPU集約型）
cloudwatch.put_metric_alarm(
    AlarmName='scann-cpu-utilization',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=3,
    MetricName='CPUUtilization',
    Namespace='AWS/EC2',
    Period=300,
    Statistic='Average',
    Threshold=80,
    AlarmDescription='ScaNN サーバーCPU使用率80%超過（スケールアウト推奨）'
)
```

### コスト最適化チェックリスト

- [ ] ~100 req/日 → EC2 c6i.large ($60-150/月)
- [ ] ~1,000 req/日 → EC2 c6i.xlarge × 2 ($300-700/月)
- [ ] 10,000+ req/日 → EKS + Auto Scaling Group ($2,000-5,000/月)
- [ ] c6iファミリー使用（CPU最適化、AVX-512対応でScaNN高速化）
- [ ] Spot Instances活用で最大70%削減（ScaNNのステートレス運用）
- [ ] S3からインデックスロード（Spot中断復旧対応）
- [ ] SOARのメモリオーバーヘッドは1.3x（2xスピリングの半分以下）
- [ ] ElastiCacheで頻出クエリキャッシュ
- [ ] Reserved Instances (1年コミット)で40%割引
- [ ] Auto Scaling Groupで負荷に応じたスケールアウト/イン
- [ ] n_probe パラメータをRecall要件に応じて調整（10-50）
- [ ] rerank_count を応答時間要件に応じて最適化
- [ ] CloudWatch Anomaly Detectionで検索レイテンシ異常検知
- [ ] CPU使用率80%でスケールアウトトリガー設定
- [ ] タグ戦略: env/project/teamでコスト可視化
- [ ] 開発環境: t3.medium（最小構成、AVX-512なし）
- [ ] インデックス再構築は夜間バッチで実行
- [ ] TensorFlow Servingとの統合でモデルサービングを統一
- [ ] ScaNNのtree_quantization_typeを最適化（float16で十分な場合あり）
- [ ] S3 Intelligent-Tieringでインデックスストレージコスト最適化

## 学術研究との関連（Academic Connection）

- **ScaNN原論文**: Guo et al. (2020) "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" (ICML 2020)。ScaNNの基盤となる異方性ベクトル量子化を提案
- **SOAR論文**: Sun & Guo (2023) "SOAR: Improved Indexing for Approximate Nearest Neighbor Search" (NeurIPS 2023)。本ブログの学術的裏付け
- **IVFの理論的限界**: Jégou et al. (2011) "Product Quantization for Nearest Neighbor Search"。SOARが解決する問題の理論的背景

## まとめと実践への示唆

SOAR/ScaNNは、GoogleのANN検索技術の最先端を示すアルゴリズムです。HNSWが「グラフ構造」で精度を追求するのに対し、ScaNN/SOARは「量子化 + 最適化されたスピリング」で精度とスループットの両立を図ります。

**Zenn記事との関連**: Zenn記事で紹介したベクトルDB選定では、専用ベクトルDB（Qdrant等）がグラフベースANN（HNSW）を採用しているのに対し、ScaNN/SOARは量子化ベースの異なるアプローチを取っています。Google Cloud Vertex AI Vector SearchやAlloyDBのScaNNインデックスとして利用でき、GCP利用者にとっては有力な選択肢です。

## 参考文献

- **Blog URL**: [https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/](https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/)
- **Code**: [https://github.com/google-research/google-research/tree/master/scann](https://github.com/google-research/google-research/tree/master/scann) (Apache 2.0 License)
- **SOAR Paper (NeurIPS 2023)**: Sun & Guo, "SOAR: Improved Indexing for Approximate Nearest Neighbor Search"
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/8c8bb192985b64](https://zenn.dev/0h_n0/articles/8c8bb192985b64)
