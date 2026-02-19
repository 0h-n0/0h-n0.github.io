---
layout: post
title: "AWS公式解説: pgvectorインデックス最適化ガイド — IVFFlatとHNSWの深掘り"
description: "AWSデータベースブログによるpgvectorのIVFFlatとHNSWインデックスの詳細解説。パラメータチューニング、ベンチマーク、Aurora PostgreSQL上での最適化手法を網羅"
categories: [blog, tech_blog]
tags: [pgvector, postgresql, HNSW, IVFFlat, aws, aurora, benchmark, rag, vectordb]
date: 2026-02-19 11:00:00 +0900
source_type: tech_blog
source_domain: aws.amazon.com
source_url: https://aws.amazon.com/blogs/database/optimize-generative-ai-applications-with-pgvector-indexing-a-deep-dive-into-ivfflat-and-hnsw-techniques/
zenn_article: 8c8bb192985b64
zenn_url: https://zenn.dev/0h_n0/articles/8c8bb192985b64
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

AWSデータベースブログが公開したpgvectorインデックスの包括的な解説記事です。Amazon Aurora PostgreSQLおよびRDS for PostgreSQL上でのベクトル検索を最適化するため、IVFFlat（Inverted File Flat）とHNSW（Hierarchical Navigable Small World）の2つのインデックスタイプについて、アルゴリズムの仕組み・パラメータ設定・性能特性・使い分けを詳細に解説しています。AWSマネージド環境特有の最適化手法やモニタリング戦略も含まれており、実務での pgvector 運用に直結する内容です。

この記事は [Zenn記事: 2026年版ベクトルDB選定ガイド：pgvector・Qdrant・Pineconeを本番ベンチマークで比較](https://zenn.dev/0h_n0/articles/8c8bb192985b64) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（AWS Database Blog）
- **URL**: [AWS Database Blog: pgvector indexing deep dive](https://aws.amazon.com/blogs/database/optimize-generative-ai-applications-with-pgvector-indexing-a-deep-dive-into-ivfflat-and-hnsw-techniques/)
- **組織**: Amazon Web Services (AWS)
- **発表日**: 2024年

## 技術的背景（Technical Background）

生成AIアプリケーション、特にRAG（Retrieval-Augmented Generation）パイプラインでは、数百万〜数千万件のベクトル埋め込みからリアルタイムで類似検索を実行する必要があります。PostgreSQLのpgvector拡張はこの需要に応えるソリューションですが、データ量の増加に伴い、適切なインデックス設計なしではクエリレイテンシが劇的に悪化します。

pgvectorは2つのANN（Approximate Nearest Neighbor）インデックスタイプを提供します：

1. **IVFFlat**: ベクトル空間をk-meansクラスタリングで分割し、クエリに近いクラスタのみを探索する手法
2. **HNSW**: 多層グラフ構造を構築し、粗い探索から精密な探索へ段階的に絞り込む手法

AWS Aurora PostgreSQLとRDS for PostgreSQLでは、これらのインデックスがネイティブにサポートされており、マネージド環境での運用が可能です。

## 実装アーキテクチャ（Architecture）

### IVFFlat インデックスの仕組み

IVFFlatは、ベクトル空間を $C$ 個のVoronoiセル（クラスタ）に分割し、クエリ時には最も近い $n\_probe$ 個のクラスタのみを探索します。

**k-meansクラスタリングによる空間分割**:

$$
\text{argmin}_{\{\mu_1, \ldots, \mu_C\}} \sum_{i=1}^{N} \min_{j \in \{1, \ldots, C\}} \|\mathbf{x}_i - \mu_j\|^2
$$

- $\mathbf{x}_i$: $i$ 番目のベクトル
- $\mu_j$: $j$ 番目のクラスタセントロイド
- $C$: クラスタ数（推奨値: $\sqrt{N}$、$N$ はベクトル数）

**検索プロセス**:

1. クエリベクトル $\mathbf{q}$ と全セントロイド $\{\mu_1, \ldots, \mu_C\}$ の距離を計算
2. 最も近い $n\_probe$ 個のクラスタを選択
3. 選択されたクラスタ内の全ベクトルに対して正確な距離計算
4. 上位 $k$ 件を返却

**Recall近似式**:

$$
\text{Recall} \approx 1 - \left(1 - \frac{n\_probe}{C}\right)^k
$$

例えば、$C = 1000$, $n\_probe = 50$, $k = 10$ の場合、Recall ≈ 0.40 です。高いRecallを得るには $n\_probe$ を増やす必要がありますが、レイテンシとのトレードオフが発生します。

```sql
-- IVFFlatインデックスの作成
CREATE INDEX idx_embedding_ivf
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 1000);  -- lists = sqrt(N)

-- 検索時のprobeパラメータ設定
SET ivfflat.probes = 50;  -- 探索するクラスタ数

-- コサイン類似度検索
SELECT id, content, 1 - (embedding <=> query_vec::vector) AS similarity
FROM documents
ORDER BY embedding <=> query_vec::vector
LIMIT 10;
```

**IVFFlatのパラメータガイド**:

| データ量 | lists (C) | probes (n_probe) | Recall@10 | 備考 |
|---------|-----------|------------------|-----------|------|
| 10万件 | 316 | 16 | ~0.85 | $\sqrt{N}$ ルール |
| 100万件 | 1,000 | 50 | ~0.90 | バランス型 |
| 1,000万件 | 3,162 | 100 | ~0.92 | メモリ効率重視 |

### HNSW インデックスの仕組み

HNSWは、多層のNavigable Small Worldグラフを構築します。各層はスキップリストのように機能し、上位層で大まかな探索、下位層で精密な探索を行います。

**グラフ構造**:

$$
L_{\max} = \lfloor \log_{m_L}(N) \rfloor
$$

- $L_{\max}$: 最大層数
- $m_L$: 層間のスケーリング係数（デフォルト: $1 / \ln(M)$）
- $M$: 各層でのノードの最大接続数

**挿入アルゴリズム（概要）**:

```python
def hnsw_insert(graph: HNSWGraph, q: Vector, M: int, ef_construction: int) -> None:
    """HNSWグラフへのベクトル挿入

    Args:
        graph: HNSWグラフ構造
        q: 挿入するベクトル
        M: 最大接続数
        ef_construction: 構築時の探索幅
    """
    # ランダムに挿入層を決定（指数分布）
    l = floor(-log(random()) * m_L)

    # 上位層でのGreedy Search（エントリポイントから）
    ep = graph.entry_point
    for layer in range(graph.max_level, l, -1):
        ep = greedy_search(graph, q, ep, layer, ef=1)

    # 下位層でのef_construction個の近傍探索とエッジ追加
    for layer in range(min(l, graph.max_level), -1, -1):
        neighbors = search_layer(graph, q, ep, layer, ef=ef_construction)
        selected = select_neighbors(q, neighbors, M)  # M個に絞り込み
        add_bidirectional_edges(graph, q, selected, layer)
        ep = neighbors[0]  # 次の層のエントリポイント
```

**検索アルゴリズム**:

```python
def hnsw_search(
    graph: HNSWGraph, q: Vector, k: int, ef_search: int
) -> list[tuple[int, float]]:
    """HNSWグラフでの近傍検索

    Args:
        q: クエリベクトル
        k: 返却件数
        ef_search: 検索時の探索幅（>= k）

    Returns:
        (id, distance) のリスト
    """
    ep = graph.entry_point

    # 上位層: 各層でGreedy Search（ef=1）
    for layer in range(graph.max_level, 0, -1):
        ep = greedy_search(graph, q, ep, layer, ef=1)

    # 最下層: ef_search個の候補を探索
    candidates = search_layer(graph, q, ep, layer=0, ef=ef_search)

    # 上位k件を返却
    return sorted(candidates, key=lambda x: x[1])[:k]
```

```sql
-- HNSWインデックスの作成
CREATE INDEX idx_embedding_hnsw
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- 検索時のefパラメータ設定
SET hnsw.ef_search = 100;

-- コサイン類似度検索（クエリは同一）
SELECT id, content, 1 - (embedding <=> query_vec::vector) AS similarity
FROM documents
ORDER BY embedding <=> query_vec::vector
LIMIT 10;
```

**HNSWのパラメータガイド**:

| パラメータ | 推奨値 | 影響 |
|-----------|--------|------|
| M | 16 (標準) / 32 (高精度) | メモリ: M比例、Recall: M↑で向上 |
| ef_construction | 128-256 | 構築時間: ef↑で増加、品質: ef↑で向上 |
| ef_search | 64-200 | レイテンシ: ef↑で増加、Recall: ef↑で向上 |

### IVFFlat vs HNSW: 使い分けガイド

| 観点 | IVFFlat | HNSW |
|------|---------|------|
| **Recall@10** | 0.85-0.92 | 0.95-0.99 |
| **QPS (100万件)** | 2,000-5,000 | 800-2,000 |
| **メモリ使用量** | 低い（ベクトル+セントロイドのみ） | 高い（ベクトル+グラフ構造） |
| **構築時間** | 短い（k-means収束後） | 長い（逐次挿入） |
| **動的更新** | 弱い（新データでクラスタが劣化） | 強い（逐次挿入可能） |
| **フィルタ付き検索** | WHERE句で可能だがRecall低下 | WHERE句で可能、Recall安定 |
| **推奨ユースケース** | バッチ処理、メモリ制約環境 | リアルタイム検索、高精度要件 |

**AWSの推奨**: 「多くの生成AIアプリケーションではHNSWを推奨します。Recall@10が95%以上を安定的に達成でき、動的なデータ更新にも対応できるためです。IVFFlatは、メモリ制約が厳しい場合やバッチ処理ワークロードに適しています。」

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $70-180 | Lambda + Aurora Serverless v2 (pgvector) |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $350-800 | ECS Fargate + Aurora PostgreSQL + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,000-5,000 | EKS + Aurora PostgreSQL Multi-AZ + ElastiCache Cluster |

**Small構成の詳細**（月額$70-180）:
- **Aurora Serverless v2 (PostgreSQL 16 + pgvector 0.8)**: 0.5〜2 ACU ($50-100/月)
- **Lambda**: 検索API 1GB RAM ($20/月)
- **API Gateway**: REST API ($5/月)
- **CloudWatch**: 基本監視 ($5/月)

**Medium構成の詳細**（月額$350-800）:
- **Aurora PostgreSQL db.r6g.large**: 2 vCPU, 16GB RAM ($200/月)
- **Aurora Reader Replica**: 読み取り分散 ($200/月)
- **ECS Fargate**: API 0.5 vCPU × 2 ($120/月)
- **ElastiCache Redis (cache.t3.micro)**: ($15/月)
- **ALB**: ($20/月)

**コスト削減テクニック**:
- Aurora Serverless v2でアイドル時0.5 ACUまでスケールダウン
- IVFFlatインデックスでメモリ使用量を抑制（HNSW比50%削減）
- ElastiCacheで頻出クエリをキャッシュ（Aurora負荷70%削減）
- Aurora Reserved Instances (1年)で40%割引
- Reader Replicaで読み取りを分散

**コスト試算の注意事項**:
- 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です
- Aurora Serverless v2のACU課金はワークロードにより変動します
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

**Small構成: Aurora Serverless v2 + pgvector**

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "pgvector-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["ap-northeast-1a", "ap-northeast-1c"]
  private_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  database_subnets = ["10.0.3.0/24", "10.0.4.0/24"]

  enable_nat_gateway     = false
  enable_dns_hostnames   = true
  create_database_subnet_group = true
}

resource "aws_rds_cluster" "pgvector" {
  cluster_identifier = "pgvector-cluster"
  engine             = "aurora-postgresql"
  engine_version     = "16.4"
  engine_mode        = "provisioned"
  database_name      = "vectordb"
  master_username    = "dbadmin"
  master_password    = aws_secretsmanager_secret_version.db_pass.secret_string

  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.aurora.id]

  serverlessv2_scaling_configuration {
    min_capacity = 0.5
    max_capacity = 4.0
  }

  storage_encrypted = true
  kms_key_id       = aws_kms_key.aurora.arn
}

resource "aws_rds_cluster_instance" "pgvector" {
  cluster_identifier = aws_rds_cluster.pgvector.id
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.pgvector.engine
  engine_version     = aws_rds_cluster.pgvector.engine_version
}

resource "aws_lambda_function" "search_api" {
  filename      = "lambda.zip"
  function_name = "pgvector-search"
  role          = aws_iam_role.lambda.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 30
  memory_size   = 1024

  vpc_config {
    subnet_ids         = module.vpc.private_subnets
    security_group_ids = [aws_security_group.lambda.id]
  }

  environment {
    variables = {
      DB_CLUSTER_ENDPOINT = aws_rds_cluster.pgvector.endpoint
      DB_SECRET_ARN       = aws_secretsmanager_secret.db_pass.arn
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "aurora_acr" {
  alarm_name          = "aurora-acu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "ServerlessDatabaseCapacity"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 3.0
  alarm_description   = "Aurora ACU 3.0超過（コスト注意）"

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.pgvector.cluster_identifier
  }
}
```

### 運用・監視設定

```sql
-- Aurora Performance Insights: pgvectorクエリ分析
-- 最も遅いベクトル検索クエリを特定
SELECT query, calls, mean_exec_time, total_exec_time
FROM pg_stat_statements
WHERE query LIKE '%<=>%' OR query LIKE '%<->%'
ORDER BY mean_exec_time DESC
LIMIT 10;

-- インデックス使用状況の確認
SELECT schemaname, relname, indexrelname, idx_scan, idx_tup_read,
       pg_size_pretty(pg_relation_size(indexrelid)) as idx_size
FROM pg_stat_user_indexes
WHERE indexrelname LIKE '%hnsw%' OR indexrelname LIKE '%ivf%';
```

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Aurora pgvector 検索レイテンシ監視
cloudwatch.put_metric_alarm(
    AlarmName='pgvector-query-latency',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=2,
    MetricName='ReadLatency',
    Namespace='AWS/RDS',
    Period=300,
    Statistic='p99',
    Threshold=0.05,  # 50ms
    AlarmDescription='pgvectorベクトル検索p99レイテンシ50ms超過',
    AlarmActions=['arn:aws:sns:ap-northeast-1:123456789:db-alerts']
)

# Aurora ACU使用量監視（コスト直結）
cloudwatch.put_metric_alarm(
    AlarmName='aurora-acu-cost',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=3,
    MetricName='ServerlessDatabaseCapacity',
    Namespace='AWS/RDS',
    Period=300,
    Statistic='Average',
    Threshold=3.0,
    AlarmDescription='Aurora Serverless ACU 3.0超過（コスト増加注意）'
)
```

### コスト最適化チェックリスト

- [ ] ~100 req/日 → Aurora Serverless v2 + Lambda ($70-180/月)
- [ ] ~1,000 req/日 → Aurora Provisioned + ECS Fargate ($350-800/月)
- [ ] 10,000+ req/日 → Aurora Multi-AZ + EKS ($2,000-5,000/月)
- [ ] Aurora Serverless v2: min_capacity=0.5でアイドルコスト最小化
- [ ] IVFFlatインデックス: HNSW比50%のメモリ削減（メモリ制約環境向け）
- [ ] HNSWインデックス: Recall優先の場合はM=16, ef_construction=200
- [ ] ElastiCacheキャッシュ: 頻出クエリのDB負荷を70%削減
- [ ] Aurora Reserved Instances (1年コミット)で40%割引
- [ ] Reader Replicaで読み取り分散
- [ ] pg_stat_statementsで低速クエリを定期分析
- [ ] VACUUM ANALYZEの定期実行（pg_cronで自動化）
- [ ] 不要なインデックスの削除（IVFFlat→HNSW移行後）
- [ ] CloudWatch Anomaly Detectionでクエリレイテンシ異常検知
- [ ] Aurora Performance Insightsで負荷パターンを分析
- [ ] タグ戦略: env/project/teamでコスト可視化
- [ ] 開発環境: Aurora Serverless v2のmax_capacity=1に制限
- [ ] HNSW ef_searchをリアルタイム(64)/バッチ(200)で使い分け
- [ ] IVFFlat probesを精度要件に応じて調整（10-100）
- [ ] Connection Pooling（RDS Proxy）で接続数を最適化
- [ ] Enhanced Monitoring有効化で詳細なOS・DB メトリクス取得

## パフォーマンス最適化（Performance）

### pgvector 0.8.0 の改善点

pgvector 0.8.0（Aurora PostgreSQL対応）では以下の性能改善が実施されています：

- **iterative_scan機能**: フィルタ付きクエリのRecallを大幅改善。従来版では WHERE句との組み合わせで不完全な結果が返される問題があった
- **HNSWインデックスの並列構築**: maintenance_work_memの範囲でマルチスレッド構築
- **量子化サポートの強化**: halfvec（16bit浮動小数点）による メモリ50%削減

### チューニングのベストプラクティス

```sql
-- 作業メモリの設定（HNSW構築時に重要）
SET maintenance_work_mem = '2GB';

-- 並列ワーカー数の設定
SET max_parallel_maintenance_workers = 4;

-- HNSWインデックス構築（並列）
CREATE INDEX CONCURRENTLY idx_embedding_hnsw
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- halfvecによるメモリ最適化
ALTER TABLE documents
  ALTER COLUMN embedding TYPE halfvec(768);
```

## 運用での学び（Production Lessons）

**AWSマネージド環境特有の考慮事項**:

- **Aurora Serverless v2のACU変動**: ベクトル検索の負荷により ACU が急増する場合がある。max_capacity の設定とCloudWatchアラームで監視が必須
- **メンテナンスウィンドウ**: HNSW インデックスの再構築は maintenance_work_mem を大量に消費するため、メンテナンスウィンドウ内で実行を推奨
- **Read Replicaの活用**: ベクトル検索クエリをReader Replicaにルーティングし、Writer Instanceの負荷を軽減

## 学術研究との関連（Academic Connection）

- **HNSWアルゴリズム**: Malkov & Yashunin (2020) "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" の実装。pgvectorはこの論文のアルゴリズムをC言語で実装
- **IVFFlatアルゴリズム**: Jégou et al. (2011) "Product quantization for nearest neighbor search" のIVF部分を実装（PQ部分はなし）
- **ANN-Benchmarks**: pgvectorの性能はANN-Benchmarks上でも追跡されており、バージョンアップごとに性能向上が確認されている

## まとめと実践への示唆

AWSのpgvectorインデックスガイドは、PostgreSQLベースのベクトル検索を本番運用する際の実践的な指針を提供しています。HNSWが多くのユースケースで推奨され、IVFFlatはメモリ制約環境やバッチ処理向けという使い分けが明確です。Aurora Serverless v2との組み合わせにより、コスト効率の良いベクトル検索基盤を構築できます。

## 参考文献

- **Blog URL**: [https://aws.amazon.com/blogs/database/optimize-generative-ai-applications-with-pgvector-indexing-a-deep-dive-into-ivfflat-and-hnsw-techniques/](https://aws.amazon.com/blogs/database/optimize-generative-ai-applications-with-pgvector-indexing-a-deep-dive-into-ivfflat-and-hnsw-techniques/)
- **pgvector 0.8.0**: [https://aws.amazon.com/blogs/database/supercharging-vector-search-performance-and-relevance-with-pgvector-0-8-0-on-amazon-aurora-postgresql/](https://aws.amazon.com/blogs/database/supercharging-vector-search-performance-and-relevance-with-pgvector-0-8-0-on-amazon-aurora-postgresql/)
- **Related Papers**: Malkov & Yashunin (2020) arXiv:1603.09320
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/8c8bb192985b64](https://zenn.dev/0h_n0/articles/8c8bb192985b64)
