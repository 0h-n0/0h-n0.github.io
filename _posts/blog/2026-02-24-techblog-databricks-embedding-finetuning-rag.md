---
layout: post
title: "Databricks解説: Embeddingモデルのファインチューニングで検索・RAG精度を向上させる"
description: "Databricksが公開したEmbeddingモデルFine-tuning手法とRAG精度改善の実践知見を詳細に解説する"
categories: [blog, tech_blog]
tags: [embedding, fine-tuning, RAG, retrieval, evaluation, NDCG, Databricks]
date: 2026-02-24 10:00:00 +0900
source_type: tech_blog
source_domain: www.databricks.com
source_url: https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning
zenn_article: db325cb1cb2e24
zenn_url: https://zenn.dev/0h_n0/articles/db325cb1cb2e24
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Databricks Blog: Improving Retrieval and RAG with Embedding Model Finetuning](https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning) の解説記事です。

## ブログ概要（Summary）

Databricksは、汎用Embeddingモデルをドメイン特化データでファインチューニングすることにより、ベクトル検索およびRAGパイプラインの精度を大幅に向上させる手法を紹介している。同ブログではDatabricksプラットフォーム上でのファインチューニング、デプロイ、評価の一連のワークフローを解説しており、ドメイン特化データでのファインチューニングによりNDCG@10が10〜30%向上する事例が報告されている。また、ファインチューニング済みモデルがリランキングと同等以上の精度を達成するケースがあることも示されている。

この記事は [Zenn記事: 自社データで実践するEmbeddingモデル精度評価パイプライン構築](https://zenn.dev/0h_n0/articles/db325cb1cb2e24) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning](https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning)
- **組織**: Databricks
- **発表日**: 2025年

## 技術的背景（Technical Background）

RAGシステムの検索精度は、使用するEmbeddingモデルの品質に直結する。しかし、汎用モデルがMTEBなどの公開ベンチマークで高スコアを記録していても、特定ドメインのデータに対しては期待どおりの精度が得られないケースがある。Zenn記事でも指摘されているように、ドメイン固有の語彙分布・文書構造・クエリ特性の違いが原因となる。

Databricksのブログでは、この課題に対してファインチューニングが有効であることを実証している。コントラスト学習（Contrastive Learning）を基盤とし、Hard Negative Mining（困難な負例の発掘）を組み合わせることで、ドメイン特化の検索精度改善を実現している。

この技術は学術的には以下の研究に基づいている。
- **DPR（Karpukhin et al., 2020）**: Dense Passage Retrievalにおけるコントラスト学習の基礎
- **E5（Wang et al., 2024）**: 大規模言語モデルを用いたテキスト埋め込みの改善

## 実装アーキテクチャ（Architecture）

### ファインチューニングパイプライン

Databricksのブログに記載されているファインチューニングのアーキテクチャは、以下の3段階で構成されている。

```mermaid
graph TD
    A[ドメインデータ<br/>クエリ+文書ペア] --> B[訓練データ準備<br/>Hard Negative Mining]
    B --> C[コントラスト学習<br/>Embedding Fine-tuning]
    C --> D[評価<br/>NDCG@10, Recall@10]
    D -->|精度不足| B
    D -->|合格| E[デプロイ<br/>Mosaic AI Model Serving]
    E --> F[RAGパイプライン<br/>ベクトル検索統合]
```

### コントラスト学習の定式化

ファインチューニングでは、InfoNCE損失関数を用いたコントラスト学習が基盤となっている。

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\exp(\text{sim}(q, d^+) / \tau) + \sum_{j=1}^{N} \exp(\text{sim}(q, d_j^-) / \tau)}
$$

ここで、
- $q$: クエリの埋め込みベクトル
- $d^+$: 正例（関連文書）の埋め込みベクトル
- $d_j^-$: 負例（非関連文書）の埋め込みベクトル（$j = 1, ..., N$）
- $\text{sim}(\cdot, \cdot)$: コサイン類似度
- $\tau$: 温度パラメータ（通常 $\tau = 0.05$）
- $N$: 負例の数

この損失関数は、正例ペアの類似度を高め、負例ペアの類似度を低くするようにモデルパラメータを最適化する。

### Hard Negative Mining

ファインチューニングの精度改善において重要な技術がHard Negative Mining（困難な負例の発掘）である。単純なランダム負例ではなく、クエリに対して「類似しているが不正解」な文書を負例として選択する。

```python
def mine_hard_negatives(
    query: str,
    positive_doc_id: str,
    corpus: dict[str, str],
    model: EmbeddingModel,
    top_k: int = 10,
    num_negatives: int = 5,
) -> list[str]:
    """Hard Negative Miningの実装

    Args:
        query: クエリテキスト
        positive_doc_id: 正解文書のID
        corpus: 検索対象コーパス
        model: 現在のEmbeddingモデル
        top_k: 検索上位件数
        num_negatives: 選択する負例数

    Returns:
        Hard Negative文書のIDリスト
    """
    # 現在のモデルで検索
    query_emb = model.encode(query)
    scores = {}
    for doc_id, doc_text in corpus.items():
        doc_emb = model.encode(doc_text)
        scores[doc_id] = cosine_similarity(query_emb, doc_emb)

    # 上位top_kから正解を除外 → Hard Negatives
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    hard_negatives = [
        doc_id for doc_id, _ in ranked
        if doc_id != positive_doc_id
    ][:num_negatives]

    return hard_negatives
```

Hard Negative Miningのポイントは、**現在のモデルが間違えやすい文書**を優先的に負例に含めることで、モデルの弁別能力を効果的に向上させる点にある。DatabricksのブログではNVIDIA NeMo Curatorの手法（top-K selection, threshold-based selection, positive-aware mining）も言及されている。

### スケーリング戦略

Databricksプラットフォームでのファインチューニングは以下の構成で実行される。

- **計算基盤**: Mosaic AI（旧MosaicML）上のGPUクラスタ
- **分散学習**: PyTorch DistributedDataParallel（DDP）
- **データローダー**: Apache Spark経由のPetastormで大規模データセットに対応
- **モデル保存**: Unity Catalog上のMLflowモデルレジストリ

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

EmbeddingファインチューニングパイプラインをAWS上で構築する場合の構成を示す。

**トラフィック量別の推奨構成**:

| 規模 | 訓練データ規模 | 推奨構成 | 月額コスト | 主要サービス |
|------|-------------|---------|-----------|------------|
| **Small** | ~10K ペア | Serverless | $100-300 | SageMaker Training Job (Spot) |
| **Medium** | ~100K ペア | Hybrid | $500-1,500 | SageMaker + S3 + ECR |
| **Large** | 1M+ ペア | Container | $3,000-8,000 | EKS + Multi-GPU + S3 |

**Small構成の詳細**（月額$100-300）:
- **SageMaker Training**: ml.g5.xlarge Spot Instance（$40/回 × 月2-3回）
- **S3**: 訓練データ・モデルアーティファクト保存（$10/月）
- **ECR**: コンテナイメージ（$5/月）
- **SageMaker Endpoint**: リアルタイム推論 ml.g5.xlarge（$150/月、Serverless Inferenceで削減可）

**Medium構成の詳細**（月額$500-1,500）:
- **SageMaker Training**: ml.g5.2xlarge × 2 Spot（$200/回 × 月2-4回）
- **SageMaker Endpoint**: ml.g5.xlarge × 2（Auto Scaling, $500/月）
- **S3 + FSx for Lustre**: 高速データアクセス（$100/月）
- **SageMaker Experiments**: 実験追跡（$20/月）

**Large構成の詳細**（月額$3,000-8,000）:
- **EKS + Karpenter**: GPU Spot管理（$72/月 + Spot $1,500/月）
- **Multi-GPU Training**: g5.12xlarge × 2-4台（$2,000/月）
- **S3 + FSx**: 大規模データパイプライン（$300/月）
- **SageMaker Endpoint**: Multi-Model Endpoint（$1,500/月）

**コスト削減テクニック**:
- SageMaker Managed Spot Trainingで最大90%削減
- SageMaker Serverless Inferenceでアイドル時コスト$0
- S3 Intelligent-Tieringで保存コスト自動最適化
- Graviton Instancesで推論コスト20%削減（CPUベース埋め込みの場合）

**コスト試算の注意事項**:
- 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です
- 実際のコストは訓練頻度、データサイズ、推論トラフィックにより変動します
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

**Small構成（Serverless）: SageMaker Training + Endpoint**

```hcl
resource "aws_s3_bucket" "training_data" {
  bucket = "embedding-finetune-data"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "training_data" {
  bucket = aws_s3_bucket.training_data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_iam_role" "sagemaker_exec" {
  name = "embedding-finetune-sagemaker-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker_exec.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_sagemaker_endpoint_configuration" "embedding" {
  name = "embedding-finetune-endpoint"

  production_variants {
    variant_name           = "primary"
    model_name             = "embedding-finetuned-model"
    initial_instance_count = 1
    instance_type          = "ml.g5.xlarge"
  }
}

resource "aws_cloudwatch_metric_alarm" "sagemaker_cost" {
  alarm_name          = "sagemaker-training-cost-spike"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "TrainingJobDuration"
  namespace           = "AWS/SageMaker"
  period              = 86400
  statistic           = "Sum"
  threshold           = 36000
  alarm_description   = "SageMaker訓練時間が10時間/日超過（コスト急増）"
}
```

**Large構成（Container）: EKS + Multi-GPU Training**

```hcl
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "~> 20.0"
  cluster_name    = "embedding-finetune-cluster"
  cluster_version = "1.31"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets
  cluster_endpoint_public_access = true
  enable_cluster_creator_admin_permissions = true
}

resource "kubectl_manifest" "karpenter_gpu" {
  yaml_body = <<-YAML
    apiVersion: karpenter.sh/v1alpha5
    kind: Provisioner
    metadata:
      name: gpu-training-provisioner
    spec:
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot"]
        - key: node.kubernetes.io/instance-type
          operator: In
          values: ["g5.xlarge", "g5.2xlarge", "g5.12xlarge"]
      limits:
        resources:
          cpu: "96"
          memory: "384Gi"
          nvidia.com/gpu: "8"
      ttlSecondsAfterEmpty: 120
  YAML
}

resource "aws_budgets_budget" "finetune_monthly" {
  name         = "embedding-finetune-monthly"
  budget_type  = "COST"
  limit_amount = "8000"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["ops@example.com"]
  }
}
```

### セキュリティベストプラクティス

- **IAMロール**: SageMaker実行ロールはS3バケットとECRリポジトリのみにアクセスを制限
- **ネットワーク**: SageMaker VPCモード有効化、プライベートサブネット内で訓練実行
- **シークレット管理**: APIキー（Anthropic, Voyage等）はSecrets Managerに保存
- **暗号化**: S3 KMS暗号化、SageMaker Volume暗号化を有効化
- **監査**: CloudTrail有効化、SageMaker Experiments でバージョン管理

### 運用・監視設定

```sql
-- CloudWatch Logs Insights: 訓練ジョブの異常検知
fields @timestamp, training_job_name, loss, learning_rate
| stats avg(loss) as avg_loss by bin(5m)
| filter avg_loss > 1.0
```

```python
import boto3

cloudwatch = boto3.client('cloudwatch')
cloudwatch.put_metric_alarm(
    AlarmName='embedding-finetune-loss-diverge',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=3,
    MetricName='TrainingLoss',
    Namespace='Embedding/FineTuning',
    Period=300,
    Statistic='Average',
    Threshold=2.0,
    AlarmDescription='訓練損失が発散（学習率調整が必要）'
)
```

### コスト最適化チェックリスト

**アーキテクチャ選択**:
- [ ] ~10Kペア → SageMaker Spot Training（$100-300/月）
- [ ] ~100Kペア → SageMaker + Auto Scaling（$500-1,500/月）
- [ ] 1M+ペア → EKS Multi-GPU Spot（$3,000-8,000/月）

**リソース最適化**:
- [ ] SageMaker Managed Spot Trainingで最大90%削減
- [ ] Serverless Inferenceでアイドル時$0
- [ ] S3 Intelligent-Tieringで保存コスト自動最適化
- [ ] Graviton Instancesで推論20%削減
- [ ] 訓練完了後のGPUインスタンス自動終了

**監視・アラート**:
- [ ] AWS Budgets: 月額予算設定
- [ ] CloudWatch: 訓練損失監視、推論レイテンシ監視
- [ ] Cost Anomaly Detection: 自動異常検知
- [ ] 日次コストレポート: SNS/Slackへ自動送信

**リソース管理**:
- [ ] 未使用エンドポイント削除
- [ ] S3ライフサイクルポリシー: 古いモデルアーティファクト30日でGlacier移行
- [ ] タグ戦略: 実験名・日付でコスト可視化
- [ ] 開発環境エンドポイント夜間停止

## パフォーマンス最適化（Performance）

Databricksのブログで報告されている性能改善の知見を以下にまとめる。

**ファインチューニング前後の精度比較**:

Databricksのブログによると、ドメイン特化データでのファインチューニングにより以下の改善が報告されている。

| 指標 | ベースライン（汎用モデル） | ファインチューニング後 | 改善幅 |
|------|------------------------|--------------------|----|
| NDCG@10 | 0.55-0.70 | 0.65-0.90 | +10〜30% |
| Recall@10 | 0.60-0.75 | 0.75-0.95 | +10〜25% |

**リランキングとの比較**:

ブログの報告によると、ファインチューニング済みEmbeddingモデルは多くのケースでリランキングと同等以上の精度を達成する。リランキングはレイテンシの増加（50-200ms追加）を伴うため、ファインチューニングによる検索段階での精度改善はレイテンシ面でも有利となる。

## 運用での学び（Production Lessons）

### ファインチューニング時の注意点

**カタストロフィック・フォーゲッティング（壊滅的忘却）**: ドメイン特化ファインチューニングにより、一般ドメインでの性能が低下するリスクがある。Zenn記事でも言及されているように、学習率を低く設定（$1 \times 10^{-6}$ 〜 $5 \times 10^{-6}$）し、元データの一部を混ぜて学習することで軽減可能である。

**訓練データの品質がボトルネック**: モデルアーキテクチャやハイパーパラメータの調整よりも、訓練データ（特にHard Negatives）の品質が最終的な精度に与える影響が大きい。ブログではNVIDIA NeMo Curatorのパイプラインを訓練データ品質向上に活用する方法も紹介されている。

**評価データの厳密な分離**: Zenn記事で詳述されているように、学習データ・検証データ・テストデータの厳密な分離が不可欠である。テストデータを繰り返し参照すると過学習の検出ができなくなるため、テストデータでの評価は最終段階で1回のみ実施する。

## 学術研究との関連（Academic Connection）

Databricksのファインチューニング手法は以下の学術研究に基づいている。

- **DPR（Karpukhin et al., 2020）**: InfoNCE損失によるコントラスト学習の基礎。Dense Passage Retrievalにおける正例・負例ペアの学習フレームワーク
- **E5（Wang et al., 2024, arXiv:2401.00368）**: 大規模言語モデルを活用したテキスト埋め込み改善。弱教師ありの事前学習とcontrastive fine-tuningの2段階訓練
- **ANCE（Xiong et al., 2021）**: Approximate Nearest Neighbor Negative Contrastive Estimation。Hard Negative Miningの効率的なアプローチ

## まとめと実践への示唆

Databricksのブログは、Embeddingモデルのファインチューニングが検索・RAG精度向上に有効であることを実証的に示した。ドメイン特化データでのNDCG@10が10〜30%向上するとの報告は、Zenn記事で紹介した評価パイプラインで計測可能な改善幅として現実的である。実務では、まず自社データでの評価パイプラインを構築してベースラインを計測し、その結果に基づいてファインチューニングの要否を判断するワークフローが推奨される。

## 参考文献

- **Blog URL**: [https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning](https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning)
- **Related Papers**: [DPR (arXiv:2004.04906)](https://arxiv.org/abs/2004.04906), [E5 (arXiv:2401.00368)](https://arxiv.org/abs/2401.00368)
- **NVIDIA NeMo Curator**: [https://developer.nvidia.com/blog/boost-embedding-model-accuracy-for-custom-information-retrieval/](https://developer.nvidia.com/blog/boost-embedding-model-accuracy-for-custom-information-retrieval/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/db325cb1cb2e24](https://zenn.dev/0h_n0/articles/db325cb1cb2e24)
