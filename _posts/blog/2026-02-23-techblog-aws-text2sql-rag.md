---
layout: post
title: "AWS解説: Amazon Bedrock×RAGによるエンタープライズText-to-SQLの構築パターン"
description: "Amazon BedrockのClaude 3.5 SonnetとRAGを組み合わせたText-to-SQLアプリケーション構築パターンの解説"
categories: [blog, tech_blog]
tags: [AWS, Amazon-Bedrock, Text-to-SQL, RAG, Claude, LangGraph, sql, NL2SQL]
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

本記事は [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/boosting-rag-based-intelligent-document-assistants-using-entity-extraction-sql-querying-and-agents-with-amazon-bedrock/) の解説記事です。

## ブログ概要（Summary）

AWSは、RAGベースのドキュメントアシスタントにおいて、ベクトル検索だけでは構造化データの正確な問合せが困難であるという課題に対し、**エンティティ抽出+SQLクエリ+エージェント**を統合したアーキテクチャを提案している。Amazon Bedrock上のClaude 3.5 Sonnetと、Amazon Aurora PostgreSQLのpgvector拡張を組み合わせ、非構造化ドキュメントと構造化データの横断検索を実現する構成が紹介されている。

この記事は [Zenn記事: LangGraph×Claude Sonnet 4.6でSQL統合Agentic RAGを実装する](https://zenn.dev/0h_n0/articles/58dc3076d2ffba) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（AWS Machine Learning Blog）
- **URL**: [https://aws.amazon.com/blogs/machine-learning/boosting-rag-based-intelligent-document-assistants-using-entity-extraction-sql-querying-and-agents-with-amazon-bedrock/](https://aws.amazon.com/blogs/machine-learning/boosting-rag-based-intelligent-document-assistants-using-entity-extraction-sql-querying-and-agents-with-amazon-bedrock/)
- **組織**: Amazon Web Services (AWS)
- **関連ブログ**: [Build gen AI text-to-SQL with RAG + Bedrock](https://aws.amazon.com/blogs/machine-learning/build-your-gen-ai-based-text-to-sql-application-using-rag-powered-by-amazon-bedrock-claude-3-sonnet-and-amazon-titan-for-embedding/)、[Enterprise NL2SQL Best Practices](https://aws.amazon.com/blogs/machine-learning/generating-value-from-enterprise-data-best-practices-for-text2sql-and-generative-ai/)

## 技術的背景（Technical Background）

### ベクトル検索単体の限界

RAGの標準的な構成であるベクトル検索（Embedding → Vector Store → LLM生成）は、非構造化テキストの意味検索に優れるが、以下の課題がある：

1. **数値クエリへの対応不足**: 「先月の売上トップ5は？」のようなSQLの集計・ソートを必要とするクエリに対応できない
2. **厳密なフィルタリングの困難**: 「営業部かつ入社3年以上」のようなAND条件のフィルタリングはベクトル類似度では不正確
3. **リレーション結合の不在**: テーブルJOINに相当する構造化操作がベクトル検索にはない

これらの課題は、Zenn記事で指摘した「ベクトル検索だけでは社員情報やチケットデータなどの構造化データに対応できない」という問題と完全に一致する。

### AWSによるハイブリッドアプローチ

AWSブログでは、この課題を以下の3層アーキテクチャで解決する：

1. **エンティティ抽出層**: Amazon Textractでドキュメントから構造化エンティティ（人名、日付、金額等）を抽出
2. **ハイブリッド検索層**: Aurora PostgreSQL（pgvector）でベクトル検索とSQLクエリを同一DB上で実行
3. **エージェント層**: Amazon Bedrock AgentsがClaude 3.5 Sonnetを用いて、クエリの意図に応じて検索パスを決定

## 実装アーキテクチャ（Architecture）

### システム構成

```
ユーザークエリ
    ↓
[Amazon Bedrock Agent]
    ├── Claude 3.5 Sonnet（意図分類・SQL生成・回答合成）
    ↓
[Lambda Orchestrator]
    ├── エンティティ抽出（Amazon Textract）
    ├── SQLクエリ（Aurora PostgreSQL）
    └── ベクトル検索（Aurora PostgreSQL + pgvector）
    ↓
[回答生成]
    ↓
最終回答
```

### 主要AWSサービスと役割

| サービス | 役割 | 備考 |
|---------|------|------|
| **Amazon Bedrock** | LLMホスティング + Agent機能 | Claude 3.5 Sonnet使用 |
| **Amazon Bedrock Agents** | エージェントループ、ツール選択 | ReActパターン |
| **AWS Lambda** | オーケストレーション | 各サービス連携 |
| **Aurora PostgreSQL** | RDB + ベクトルDB | pgvector拡張で両機能を統合 |
| **Amazon Textract** | ドキュメントからの構造化データ抽出 | OCR + エンティティ抽出 |
| **Amazon Titan Embeddings** | テキスト→ベクトル変換 | 1536次元 |
| **Amazon S3** | ドキュメントストレージ | ソースドキュメント格納 |

### Zenn記事との対比

| 側面 | Zenn記事の実装 | AWSブログの実装 |
|------|-------------|---------------|
| LLM | Claude Sonnet 4.6 (直接API) | Claude 3.5 Sonnet (Bedrock経由) |
| ルーティング | LangGraph StateGraph + with_structured_output | Bedrock Agents (ReActパターン) |
| SQL | SQLDatabaseToolkit + SQLite | Aurora PostgreSQL |
| ベクトル検索 | ChromaDB + HuggingFace Embeddings | Aurora pgvector + Titan Embeddings |
| 統合パターン | ノードベースの宣言的ルーティング | エージェントベースのReActループ |

**設計思想の違い**:

Zenn記事のLangGraph StateGraphアプローチは**宣言的（declarative）**で、ルーティングロジックを`add_conditional_edges`で明示的に定義する。一方、AWSのBedrock Agentsアプローチは**命令的（imperative）**で、LLMがReActパターンで動的にツールを選択する。

宣言的アプローチの利点：
- ルーティングのテスト・デバッグが容易
- 実行パスが予測可能
- 各ノードを独立に差し替え可能

ReActアプローチの利点：
- より柔軟なツール選択
- 未知のクエリタイプへの対応力
- 追加ツールの統合が容易

### pgvectorによるSQL+ベクトル検索統合

AWSアーキテクチャの特徴的な点は、Aurora PostgreSQLのpgvector拡張により**同一データベース上でSQLクエリとベクトル検索の両方を実行**する設計である：

```sql
-- pgvectorによるハイブリッド検索の例
-- Step 1: ベクトル類似度検索
SELECT id, content, embedding <=> $1 AS distance
FROM documents
WHERE embedding <=> $1 < 0.5
ORDER BY distance
LIMIT 10;

-- Step 2: 構造化データのSQL結合
SELECT d.content, e.entity_name, e.entity_type
FROM documents d
JOIN entities e ON d.id = e.document_id
WHERE e.entity_type = 'PERSON'
  AND e.entity_name ILIKE '%田中%';

-- Step 3: ハイブリッド（ベクトル+SQL）
SELECT d.content, e.entity_name,
       d.embedding <=> $1 AS semantic_distance
FROM documents d
JOIN entities e ON d.id = e.document_id
WHERE e.entity_type = 'AMOUNT'
  AND CAST(e.entity_value AS numeric) > 1000000
ORDER BY semantic_distance
LIMIT 5;
```

Zenn記事ではChromaDB（ベクトル検索専用）とSQLite（SQL専用）を分離しているが、AWSアーキテクチャではpgvectorで統合している。これにより：

- **利点**: 単一DB接続で両方の検索が可能、トランザクション整合性を確保
- **制約**: pgvectorのベクトル検索性能は専用ベクトルDB（Pinecone, Qdrant等）には劣る可能性がある

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $80-200 | Lambda + Bedrock + Aurora Serverless v2 |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $400-1,000 | Lambda + Bedrock + Aurora + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,500-6,000 | EKS + Bedrock + Aurora Multi-AZ |

**Small構成の詳細** (月額$80-200):
- **Lambda**: 1GB RAM, 120秒タイムアウト ($25/月)
- **Bedrock**: Claude 3.5 Haiku + Titan Embeddings ($80/月)
- **Aurora Serverless v2**: PostgreSQL 15 + pgvector, 0.5 ACU最小 ($40/月)
- **S3**: ドキュメントストレージ ($5/月)

**コスト削減テクニック**:
- Aurora Serverless v2のACU自動スケーリング（アイドル時0.5 ACU）
- Bedrock Prompt Caching（スキーマ+ドキュメントメタデータのキャッシュ）
- Lambda Provisioned Concurrency不使用（コールドスタート許容）
- S3 Intelligent-Tiering（アクセス頻度に応じた自動階層化）

**コスト試算の注意事項**:
上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です。実際のコストはトラフィックパターン、クエリ複雑度、ドキュメント数により変動します。最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください。

### Terraformインフラコード

```hcl
# --- Aurora PostgreSQL + pgvector ---
resource "aws_rds_cluster" "hybrid_search" {
  cluster_identifier = "hybrid-search-db"
  engine             = "aurora-postgresql"
  engine_mode        = "provisioned"
  engine_version     = "15.4"
  database_name      = "knowledge"

  serverlessv2_scaling_configuration {
    min_capacity = 0.5
    max_capacity = 8.0
  }

  storage_encrypted = true
}

resource "aws_rds_cluster_instance" "hybrid_search" {
  cluster_identifier = aws_rds_cluster.hybrid_search.id
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.hybrid_search.engine
  engine_version     = aws_rds_cluster.hybrid_search.engine_version
}

# --- Lambda (Orchestrator) ---
resource "aws_lambda_function" "orchestrator" {
  filename      = "orchestrator.zip"
  function_name = "hybrid-search-orchestrator"
  role          = aws_iam_role.lambda_role.arn
  handler       = "handler.main"
  runtime       = "python3.12"
  timeout       = 120
  memory_size   = 1024

  environment {
    variables = {
      DB_SECRET_ARN    = aws_secretsmanager_secret.db.arn
      BEDROCK_MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"
    }
  }

  vpc_config {
    subnet_ids         = module.vpc.private_subnets
    security_group_ids = [aws_security_group.lambda.id]
  }
}

# --- Bedrock Agent ---
resource "aws_bedrockagent_agent" "sql_rag_agent" {
  agent_name                  = "hybrid-search-agent"
  agent_resource_role_arn     = aws_iam_role.bedrock_agent.arn
  foundation_model            = "anthropic.claude-3-5-sonnet-20241022-v2:0"
  idle_session_ttl_in_seconds = 600

  instruction = <<-EOT
    あなたは社内ナレッジ検索アシスタントです。
    ユーザーの質問に対して、SQLクエリとベクトル検索を
    適切に使い分けて回答してください。
  EOT
}
```

### 運用・監視設定

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Aurora pgvector検索レイテンシ監視
cloudwatch.put_metric_alarm(
    AlarmName='aurora-query-latency',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=3,
    MetricName='SelectLatency',
    Namespace='AWS/RDS',
    Period=300,
    Statistic='p99',
    Threshold=5000,  # P99 5秒超過でアラート
    AlarmDescription='Aurora pgvector Query Latency'
)

# Bedrock Agent実行時間監視
cloudwatch.put_metric_alarm(
    AlarmName='bedrock-agent-duration',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=2,
    MetricName='InvocationLatency',
    Namespace='AWS/Bedrock',
    Period=300,
    Statistic='Average',
    Threshold=30000,  # 平均30秒超過
    AlarmDescription='Bedrock Agent Invocation Latency'
)
```

### コスト最適化チェックリスト

- [ ] Aurora Serverless v2: 最小ACU 0.5設定
- [ ] Bedrock Prompt Caching: スキーマ情報のキャッシュ有効化
- [ ] Lambda: メモリサイズ最適化（CloudWatch Insights分析）
- [ ] S3 Intelligent-Tiering: ドキュメントストレージ最適化
- [ ] Bedrock Batch API: 非リアルタイム処理で50%削減
- [ ] AWS Budgets: 月額予算設定（80%で警告）
- [ ] Cost Anomaly Detection: 自動異常検知有効化
- [ ] タグ戦略: 環境別（dev/staging/prod）でコスト可視化

## パフォーマンス最適化（Performance）

### ハイブリッド検索のレイテンシ

AWSブログで報告されている構成および関連ブログの情報に基づく参考値：

- **SQL検索パス**: Aurora PostgreSQLでの単純SELECTは数十ms、JOIN+集計は100-500ms程度
- **ベクトル検索パス**: pgvectorでの類似度検索はインデックス（IVFFlat/HNSW）により100-300ms程度
- **エージェントループ全体**: Bedrock Agentの1ターンは5-15秒程度（LLM推論含む）

**チューニング手法**:
- pgvectorのHNSWインデックス使用（IVFFlatより高速だがメモリ消費大）
- Aurora Reader Instanceの追加（読み取りスケールアウト）
- Lambda Provisioned Concurrencyでコールドスタート回避

## 運用での学び（Production Lessons）

### AWSブログから読み取れる実運用の知見

1. **スキーマ情報のRAGが重要**: 単純にSQLを生成するのではなく、テーブル定義、カラム説明、データサンプルをRAGで提供することで精度が向上する
2. **pgvectorの一元管理**: RDBとベクトルDBを分離するより、pgvectorで統合した方が運用負荷が低い
3. **Bedrock Agentsの制御**: ReActパターンは柔軟だが、ルーティングの予測可能性ではLangGraphのStateGraphが優れる

## 学術研究との関連（Academic Connection）

- **TAG** (Biswal et al., 2024): AWSのハイブリッドアーキテクチャはTAGの実装パターンの一つと見なせる。Stage 1をBedrock Agentが、Stage 2をAurora PostgreSQLが、Stage 3をClaude Sonnetが担当
- **NL2SQL Survey** (Cui et al., 2024): AWSのアプローチはサーベイのTool-Augmented + RAG-based手法の実例

## まとめと実践への示唆

- AWSはAurora PostgreSQL + pgvectorにより、**単一DB上でSQLとベクトル検索を統合**するアーキテクチャを推奨している
- Bedrock AgentsのReActパターンとLangGraph StateGraphの宣言的ルーティングは、それぞれ異なるトレードオフを持つ
- Zenn記事の実装をAWSにデプロイする場合、ChromaDB→Aurora pgvector、SQLite→Aurora PostgreSQLへの移行が推奨される

## 参考文献

- **Blog URL**: [https://aws.amazon.com/blogs/machine-learning/boosting-rag-based-intelligent-document-assistants-using-entity-extraction-sql-querying-and-agents-with-amazon-bedrock/](https://aws.amazon.com/blogs/machine-learning/boosting-rag-based-intelligent-document-assistants-using-entity-extraction-sql-querying-and-agents-with-amazon-bedrock/)
- **関連ブログ**: [Build Text-to-SQL with RAG + Bedrock](https://aws.amazon.com/blogs/machine-learning/build-your-gen-ai-based-text-to-sql-application-using-rag-powered-by-amazon-bedrock-claude-3-sonnet-and-amazon-titan-for-embedding/)
- **関連ブログ**: [Enterprise NL2SQL Best Practices](https://aws.amazon.com/blogs/machine-learning/generating-value-from-enterprise-data-best-practices-for-text2sql-and-generative-ai/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/58dc3076d2ffba](https://zenn.dev/0h_n0/articles/58dc3076d2ffba)
