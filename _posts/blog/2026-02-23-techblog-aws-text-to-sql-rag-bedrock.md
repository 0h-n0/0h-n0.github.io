---
layout: post
title: "AWS解説: Amazon Bedrock×RAGによるText-to-SQLアプリケーション構築 — エンティティ抽出・SQL・ベクトル検索の統合"
description: "Amazon Bedrock AgentsとAurora PostgreSQL pgvectorを使い、エンティティ抽出・SQL検索・ベクトル検索を統合したRAGドキュメントアシスタントの構築手法を解説"
categories: [blog, tech_blog]
tags: [aws, bedrock, text-to-sql, RAG, vector-search, pgvector, agents, langgraph, sql]
date: 2026-02-23 11:00:00 +0900
source_type: tech_blog
source_domain: aws.amazon.com
source_url: https://aws.amazon.com/blogs/machine-learning/boosting-rag-based-intelligent-document-assistants-using-entity-extraction-sql-querying-and-agents-with-amazon-bedrock/
zenn_article: 58dc3076d2ffba
zenn_url: https://zenn.dev/0h_n0/articles/58dc3076d2ffba
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [AWS Machine Learning Blog: Boosting RAG-based intelligent document assistants using entity extraction, SQL querying, and agents with Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/boosting-rag-based-intelligent-document-assistants-using-entity-extraction-sql-querying-and-agents-with-amazon-bedrock/) の解説記事です。

## ブログ概要（Summary）

AWSのMachine Learning Blogで公開されたこのブログ記事では、RAGベースのドキュメントアシスタントを「エンティティ抽出」「SQL検索」「ベクトル検索」の3つの検索手法で強化する手法を紹介している。Amazon Bedrock Agentsがオーケストレーションレイヤーとして、クエリの性質に応じて最適な検索パスを自動選択する。Amazon Aurora PostgreSQL with pgvector拡張を使用し、SQL検索とベクトル検索の両方を単一のデータベースで実現するアーキテクチャが特徴的である。

この記事は [Zenn記事: LangGraph×Claude Sonnet 4.6でSQL統合Agentic RAGを実装する](https://zenn.dev/0h_n0/articles/58dc3076d2ffba) の深掘りです。Zenn記事ではLangGraphでSQL検索とベクトル検索を統合していますが、AWSブログではBedrock Agentsによるマネージドなオーケストレーションでこれを実現するアプローチが紹介されています。

## 情報源

- **種別**: 企業テックブログ（AWS公式）
- **URL**: [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/boosting-rag-based-intelligent-document-assistants-using-entity-extraction-sql-querying-and-agents-with-amazon-bedrock/)
- **組織**: Amazon Web Services
- **発表日**: 2024年

## 技術的背景（Technical Background）

従来のRAGシステムはベクトル検索のみに依存しており、「昨月の売上トップ5」や「営業部の社員一覧」といった構造化データへのクエリには回答できない。この課題はZenn記事で詳細に議論されているものと同一であり、AWSブログでは以下の3つの技術を組み合わせてこの問題に対処している。

1. **エンティティ抽出**: 非構造化ドキュメントから構造化情報（人名、組織名、日付、金額等）を抽出し、RDBのテーブルに格納する
2. **SQL検索**: 抽出されたエンティティおよび既存の構造化データに対してText-to-SQLでクエリを実行する
3. **ベクトル検索**: 非構造化テキストに対してセマンティック検索を行う

**なぜこのアプローチが重要か**: 企業データの多くは構造化データ（RDB）と非構造化データ（ドキュメント、メール、会議録等）の両方に分散している。どちらか一方の検索手法だけでは、ユーザーの質問に完全に回答することができない。

## 実装アーキテクチャ（Architecture）

### システム構成

AWSブログで紹介されているアーキテクチャは以下の主要コンポーネントで構成される。

```
ユーザークエリ
    ↓
[Amazon Bedrock Agent]（Claude 3.5 Sonnet / Haiku）
    │
    ├─→ [Action Group 1] RetrieveFromKnowledgeBase
    │       → Amazon Bedrock Knowledge Base
    │       → Aurora PostgreSQL pgvector（ベクトル検索）
    │
    ├─→ [Action Group 2] FetchTableSchema
    │       → Aurora PostgreSQL（スキーマ取得）
    │
    ├─→ [Action Group 3] GenerateSQLQuery
    │       → Claude 3.5 Sonnet（Text-to-SQL生成）
    │       → Aurora PostgreSQL（SQL実行）
    │
    └─→ [Action Group 4] ExtractEntities
            → Amazon Textract（ドキュメント処理）
            → Claude（エンティティ抽出）
            → Aurora PostgreSQL（構造化データ格納）
```

**注目点**: Aurora PostgreSQL with pgvectorを使用することで、ベクトル検索とSQL検索を**同一のデータベースエンジン**で実行できる。Zenn記事ではChromaDB（ベクトル検索）とSQLite/PostgreSQL（SQL検索）を別々に管理しているが、AWSのアプローチでは運用の一元化が可能になる。

### Bedrock Agentsのオーケストレーション

Amazon Bedrock Agentsは、Zenn記事のLangGraph StateGraphと類似した役割を果たすが、以下の点で異なる。

| 機能 | LangGraph StateGraph | Bedrock Agents |
|------|---------------------|----------------|
| ルーティング | `add_conditional_edges`で宣言的 | LLMがAction Groupを自動選択 |
| 状態管理 | TypedDictで明示的 | セッション管理はマネージド |
| ツール定義 | Python関数 | OpenAPI仕様のAction Group |
| デプロイ | 自前インフラ | フルマネージド |
| カスタマイズ性 | 高い | Action Groupの範囲内 |

Bedrock Agentsの利点は運用コストの低さにある。一方、Zenn記事のLangGraphアプローチは、ルーティングロジックの細かな制御や、ノード間のデータフローのカスタマイズが可能であり、複雑な検索パイプラインでは有利である。

### Aurora PostgreSQL with pgvector

pgvector拡張を使用することで、PostgreSQLテーブルに`vector`型カラムを追加し、ベクトル類似度検索をSQLで実行できる。

```sql
-- pgvectorによるベクトル検索（コサイン類似度）
SELECT content, metadata,
       1 - (embedding <=> query_embedding) AS similarity
FROM documents
WHERE 1 - (embedding <=> query_embedding) > 0.7
ORDER BY embedding <=> query_embedding
LIMIT 5;

-- 同一DBでのSQL検索（構造化データ）
SELECT e.name, e.email, d.department_name
FROM employees e
JOIN departments d ON e.department_id = d.id
WHERE d.department_name = '営業部';
```

この統合により、「営業部のメンバー（SQL）が最近書いたドキュメント（ベクトル検索）」のようなハイブリッドクエリを単一のDB接続で実行できる。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $80-200 | Lambda + Bedrock + Aurora Serverless v2 |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $400-1,000 | Lambda + Bedrock Agents + Aurora Provisioned |
| **Large** | 300,000+ (10,000/日) | Container | $2,500-6,000 | ECS Fargate + Bedrock + Aurora Multi-AZ |

**Small構成の詳細** (月額$80-200):
- **Lambda**: Text-to-SQL生成ロジック ($20/月)
- **Bedrock**: Claude 3.5 Haiku、Prompt Caching有効 ($30-80/月)
- **Aurora Serverless v2**: 0.5-2 ACU、pgvector有効 ($25-50/月)
- **Bedrock Knowledge Base**: ベクトルインデックス管理 ($5/月)
- **CloudWatch**: 基本監視 ($5/月)

**Medium構成の詳細** (月額$400-1,000):
- **Bedrock Agents**: オーケストレーション ($50/月)
- **Lambda**: Action Group実行 ($30/月)
- **Bedrock**: Claude 3.5 Sonnet、Batch API活用 ($200-500/月)
- **Aurora Provisioned**: db.r6g.large、pgvector ($120/月)
- **ElastiCache Redis**: スキーマキャッシュ ($15/月)

**コスト削減テクニック**:
- Aurora Serverless v2でアイドル時のコストを最小化（0.5 ACUまでスケールダウン）
- Bedrock Prompt Cachingでシステムプロンプト部分のコストを30-90%削減
- Bedrock Batch APIで非リアルタイム処理を50%割引
- pgvectorのIVFFlat/HNSWインデックスで検索パフォーマンスを最適化

**コスト試算の注意事項**:
- 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です
- 実際のコストはトラフィックパターン、クエリ複雑度、Bedrockトークン消費量により変動します
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

**Small構成 (Serverless): Lambda + Bedrock + Aurora Serverless v2**

```hcl
# --- VPC基盤 ---
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "text-to-sql-rag-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["ap-northeast-1a", "ap-northeast-1c"]
  private_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  database_subnets = ["10.0.10.0/24", "10.0.11.0/24"]

  enable_nat_gateway = false
  enable_dns_hostnames = true

  create_database_subnet_group = true
}

# --- Aurora Serverless v2 with pgvector ---
resource "aws_rds_cluster" "knowledge_db" {
  cluster_identifier = "text-to-sql-rag"
  engine             = "aurora-postgresql"
  engine_version     = "15.4"
  engine_mode        = "provisioned"
  database_name      = "knowledge"
  master_username    = "app_readonly"
  manage_master_user_password = true

  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.aurora.id]

  serverlessv2_scaling_configuration {
    min_capacity = 0.5  # コスト最適化: 最小0.5 ACU
    max_capacity = 4.0
  }

  storage_encrypted = true
}

resource "aws_rds_cluster_instance" "writer" {
  cluster_identifier = aws_rds_cluster.knowledge_db.id
  instance_class     = "db.serverless"
  engine             = "aurora-postgresql"
}

# --- Lambda関数（Text-to-SQL） ---
resource "aws_lambda_function" "text_to_sql" {
  filename      = "lambda.zip"
  function_name = "text-to-sql-handler"
  role          = aws_iam_role.lambda_role.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 60
  memory_size   = 512

  vpc_config {
    subnet_ids         = module.vpc.private_subnets
    security_group_ids = [aws_security_group.lambda.id]
  }

  environment {
    variables = {
      DB_CLUSTER_ARN    = aws_rds_cluster.knowledge_db.arn
      DB_SECRET_ARN     = aws_rds_cluster.knowledge_db.master_user_secret[0].secret_arn
      BEDROCK_MODEL_ID  = "anthropic.claude-3-5-haiku-20241022-v1:0"
    }
  }
}

# --- IAMロール（最小権限） ---
resource "aws_iam_role" "lambda_role" {
  name = "text-to-sql-lambda-role"

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
  role = aws_iam_role.lambda_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["bedrock:InvokeModel"]
        Resource = "arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.claude-3-5-haiku*"
      },
      {
        Effect = "Allow"
        Action = ["rds-data:ExecuteStatement", "rds-data:BatchExecuteStatement"]
        Resource = aws_rds_cluster.knowledge_db.arn
      },
      {
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue"]
        Resource = aws_rds_cluster.knowledge_db.master_user_secret[0].secret_arn
      }
    ]
  })
}

# --- CloudWatchアラーム ---
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "text-to-sql-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "Text-to-SQLエラー急増検知"

  dimensions = {
    FunctionName = aws_lambda_function.text_to_sql.function_name
  }
}
```

### セキュリティベストプラクティス

1. **ネットワーク**: Aurora/LambdaはVPCプライベートサブネット内に配置。パブリックアクセス無効化
2. **認証**: IAMロール最小権限。Bedrockモデルは使用するモデルIDのみ許可
3. **シークレット**: DBパスワードはAWS Secrets Manager管理。環境変数ハードコード禁止
4. **暗号化**: Aurora storage_encrypted有効。転送中はTLS 1.2以上
5. **監査**: CloudTrail有効化。RDS監査ログ出力

### 運用・監視設定

**CloudWatch Logs Insights クエリ**:

```sql
-- Text-to-SQL生成レイテンシ分析
fields @timestamp, @duration
| stats avg(@duration) as avg_ms, pct(@duration, 95) as p95_ms,
        pct(@duration, 99) as p99_ms by bin(5m)
| filter @message like /text-to-sql/

-- Bedrockトークン使用量異常検知
fields @timestamp, input_tokens, output_tokens
| stats sum(input_tokens + output_tokens) as total_tokens by bin(1h)
| filter total_tokens > 100000
```

**CloudWatchアラーム設定**:

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Aurora ACU使用量アラート
cloudwatch.put_metric_alarm(
    AlarmName='aurora-acu-spike',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=2,
    MetricName='ServerlessDatabaseCapacity',
    Namespace='AWS/RDS',
    Period=300,
    Statistic='Average',
    Threshold=3.0,  # 3 ACU超過でアラート
    AlarmDescription='Aurora Serverless v2 ACU使用量異常'
)
```

### コスト最適化チェックリスト

**アーキテクチャ選択**:
- [ ] ~100 req/日 → Lambda + Aurora Serverless v2 - $80-200/月
- [ ] ~1000 req/日 → Bedrock Agents + Aurora Provisioned - $400-1,000/月
- [ ] 10000+ req/日 → ECS + Aurora Multi-AZ - $2,500-6,000/月

**リソース最適化**:
- [ ] Aurora Serverless v2: min_capacity=0.5でアイドルコスト最小化
- [ ] Lambda: メモリ512MB（CloudWatch Insights分析で最適化）
- [ ] pgvector: HNSWインデックスで検索速度向上

**LLMコスト削減**:
- [ ] Bedrock Prompt Caching: スキーマ情報のキャッシュで30-90%削減
- [ ] モデル選択: 簡易クエリはHaiku ($0.25/MTok)、複雑クエリはSonnet ($3/MTok)
- [ ] Batch API: 非リアルタイム処理で50%割引

**監視・アラート**:
- [ ] AWS Budgets: 月額予算設定（80%警告、100%アラート）
- [ ] CloudWatch: ACU使用量、Lambda実行時間、Bedrockトークン数
- [ ] Cost Anomaly Detection: 自動異常検知有効化

**リソース管理**:
- [ ] pgvectorインデックス: 定期的なVACUUM ANALYZE実行
- [ ] Lambda Layers: 共通ライブラリの共有でデプロイサイズ削減
- [ ] ライフサイクルポリシー: 古いベクトルデータの自動アーカイブ

## パフォーマンス最適化（Performance）

ブログ記事で紹介されているアーキテクチャのパフォーマンス特性を整理する。

**pgvectorの検索方式比較**:

| 検索方式 | 精度 | レイテンシ（1M行） | インデックスサイズ |
|---------|------|-------------------|-----------------|
| Exact (Sequential) | 100% | ~500ms | なし |
| IVFFlat | ~95% | ~10ms | 中 |
| HNSW | ~99% | ~5ms | 大 |

HNSWインデックスはビルド時間が長い（1M行で約30分）が、検索レイテンシとrecallのバランスが優れている。

**Text-to-SQL生成のレイテンシ内訳**:
- Bedrock Agent判断: ~200ms
- スキーマ取得: ~50ms
- SQL生成（Claude 3.5 Haiku）: ~500ms
- SQL実行（Aurora）: ~100ms
- 合計: ~850ms（SQL検索パス）

Zenn記事で報告されているSQLite構成での約800msと同等の水準であり、Aurora PostgreSQLを使用しても大きなレイテンシ増加は発生しない。

## 運用での学び（Production Lessons）

AWSブログで紹介されている本番運用の知見を整理する。

1. **スキーマキャッシュの重要性**: Text-to-SQL生成のたびにスキーマをDBから取得するとレイテンシが増加する。ElastiCache Redisでスキーマ情報をキャッシュし、TTL付きで管理することが推奨されている
2. **pgvectorのメモリ管理**: HNSWインデックスはメモリを大量に消費する。Aurora Serverless v2のACU上限を適切に設定しないと、OOMによる接続断が発生する可能性がある
3. **Bedrock Agentsのハルシネーション対策**: エージェントがAction Groupを誤選択するケースがある。Action GroupのOpenAPI仕様に詳細な説明とパラメータ制約を記述することで、選択精度が向上する

## 学術研究との関連（Academic Connection）

- **CHESS** (Talaei et al., 2024): CHESSのEntity Retrieval（BM25+FAISS）はAWSブログのエンティティ抽出と類似する。CHESSはオフラインインデックス、AWSはリアルタイム抽出という違いがある
- **DIN-SQL** (Pourreza & Rafiei, 2023): DIN-SQLのSchema Linking → SQL GenerationパイプラインはBedrock Agentsの FetchTableSchema → GenerateSQLQuery Action Groupと対応する
- **Q4 Inc.事例** (AWS Blog): Q4社がBedrock + SQLDatabaseChainで構築した本番QAチャットボットの事例も公開されており、SQLDatabaseToolkitの実運用知見が得られる

## まとめと実践への示唆

AWSブログで紹介されている「エンティティ抽出 + SQL検索 + ベクトル検索」のハイブリッドアーキテクチャは、Zenn記事のLangGraphベースのSQL統合Agentic RAGと設計思想が共通している。Aurora PostgreSQL with pgvectorを使用することで、SQL検索とベクトル検索を同一DBで実行できるアプローチは、運用の簡素化において注目に値する。

Zenn記事のアーキテクチャを本番環境にデプロイする際は、AWSブログのBedrock Agents構成を参考にすることで、マネージドサービスの活用によるインフラ管理コストの削減が期待できる。一方、ルーティングの細粒度な制御が必要な場合は、Zenn記事のLangGraph StateGraphアプローチが優位である。

## 参考文献

- **Blog URL**: [AWS Machine Learning Blog - Boosting RAG with Entity Extraction, SQL, and Agents](https://aws.amazon.com/blogs/machine-learning/boosting-rag-based-intelligent-document-assistants-using-entity-extraction-sql-querying-and-agents-with-amazon-bedrock/)
- **関連AWS Blog**: [Build a conversational data assistant with Text-to-SQL](https://aws.amazon.com/blogs/machine-learning/build-a-conversational-data-assistant-part-1-text-to-sql-with-amazon-bedrock-agents/)
- **関連AWS Blog**: [Custom text-to-sql agent using Converse API](https://aws.amazon.com/blogs/machine-learning/building-a-custom-text-to-sql-agent-using-amazon-bedrock-and-converse-api/)
- **pgvector**: [https://github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/58dc3076d2ffba](https://zenn.dev/0h_n0/articles/58dc3076d2ffba)
