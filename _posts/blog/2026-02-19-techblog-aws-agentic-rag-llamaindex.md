---
layout: post
title: "AWS公式解説: LlamaIndex×Amazon Bedrockで構築するAgentic RAGアプリケーション"
description: "AWS公式ブログのAgentic RAG実装ガイドを詳細解説。LlamaIndex+Mistral+Bedrock Knowledge Basesによる本番RAG構築パターン"
categories: [blog, tech_blog]
tags: [RAG, AWS, LlamaIndex, Amazon-Bedrock, agentic-ai, LLM, python]
date: 2026-02-19 23:50:00 +0900
source_type: tech_blog
source_domain: aws.amazon.com
source_url: https://aws.amazon.com/blogs/machine-learning/create-an-agentic-rag-application-for-advanced-knowledge-discovery-with-llamaindex-and-mistral-in-amazon-bedrock/
zenn_article: 62e946539206db
zenn_url: https://zenn.dev/0h_n0/articles/62e946539206db
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

AWS公式Machine Learningブログの本記事は、LlamaIndexフレームワークとAmazon Bedrockを組み合わせたAgentic RAGアプリケーションの構築方法を解説する。Amazon Bedrock Knowledge Basesによるマネージドなベクトル検索基盤の上に、LlamaIndexのエージェント機能を統合することで、従来のNaive RAGを超える高度な知識発見アプリケーションを実現する。AWSのマネージドサービスを活用した本番環境構築のベストプラクティスが含まれる。

この記事は [Zenn記事: LlamaIndex v0.14実践ガイド：AgentWorkflowで本番RAGを構築する](https://zenn.dev/0h_n0/articles/62e946539206db) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: https://aws.amazon.com/blogs/machine-learning/create-an-agentic-rag-application-for-advanced-knowledge-discovery-with-llamaindex-and-mistral-in-amazon-bedrock/
- **組織**: Amazon Web Services (AWS) Machine Learning Blog
- **発表日**: 2024

## 技術的背景（Technical Background）

### なぜAWSマネージドサービス + LlamaIndexなのか

Zenn記事ではLlamaIndex v0.14のAgentWorkflowとAgentic Retrievalを中心に解説しているが、本番環境で運用する際には以下のインフラ課題が発生する。

**本番化の壁:**
1. **ベクトルDBの運用**: 自前でPinecone/Qdrant/Chromaを運用するコストと管理負荷
2. **LLMのスケーリング**: OpenAI APIへの依存度とレート制限
3. **セキュリティ**: VPC内でのデータ処理、IAMによるアクセス制御
4. **コスト管理**: トークン使用量の可視化と予算管理

AWSのマネージドサービスを活用することで、これらの課題をインフラレベルで解決できる。

| 課題 | LlamaIndex単体 | AWS + LlamaIndex |
|------|---------------|------------------|
| ベクトルDB | ChromaDB（ローカル） | **Bedrock Knowledge Bases**（マネージド） |
| LLM | OpenAI API | **Amazon Bedrock**（マネージド、VPC内） |
| スケーリング | 手動 | **Auto Scaling + Bedrock** |
| セキュリティ | アプリケーション層 | **IAM + VPC + KMS** |
| コスト可視化 | カスタム実装 | **CloudWatch + Cost Explorer** |

### Amazon Bedrock Knowledge Basesとは

Bedrock Knowledge Basesは、S3にアップロードしたドキュメントを自動的にチャンク分割→ベクトル化→インデックス格納するマネージドRAGサービスである。

**サポートするベクトルDB:**
- Amazon OpenSearch Serverless（デフォルト）
- Amazon Aurora PostgreSQL（pgvector）
- Pinecone
- Redis Enterprise Cloud
- MongoDB Atlas

**データソース:**
- Amazon S3（PDF, DOCX, HTML, TXT, CSV, MD等）
- Web Crawler
- Confluence
- SharePoint

### Agentic RAGアーキテクチャ

AWSブログが提案するアーキテクチャは、LlamaIndexのエージェント機能とBedrock Knowledge Basesを組み合わせた構成である。

```
ユーザークエリ
    │
    ▼
┌─────────────────────┐
│  LlamaIndex Agent   │  AgentWorkflow / ReActAgent
│  (オーケストレーター)  │
└──────┬──────────────┘
       │
  ┌────┴────┐
  │         │
  ▼         ▼
┌──────┐  ┌──────────────────┐
│ Tool │  │ Bedrock Knowledge │
│ Use  │  │ Bases Retriever   │
│      │  │ (ベクトル検索)      │
└──────┘  └──────────────────┘
              │
              ▼
       ┌─────────────┐
       │ Amazon      │
       │ Bedrock LLM │  Mistral / Claude / Llama
       │ (推論)       │
       └─────────────┘
```

**処理フロー:**
1. ユーザークエリをLlamaIndex Agentが受信
2. Agentがクエリを分析し、Bedrock Knowledge Basesへの検索が必要か判断
3. 必要に応じて複数のKnowledge Basesを検索（Multi-Index Routing相当）
4. 検索結果をAgentが評価し、不十分なら追加検索（Agentic Retrieval相当）
5. 十分な情報が集まったらBedrock LLMで最終回答を生成

## 実装アーキテクチャ（Architecture）

### LlamaIndex + Bedrock統合コード

```python
import boto3
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core import VectorStoreIndex, Settings

# Bedrock LLMの設定
llm = Bedrock(
    model="mistral.mistral-large-2402-v1:0",
    region_name="us-east-1",
    context_size=32000,
    temperature=0.1,
)

# Bedrock Embeddingの設定
embed_model = BedrockEmbedding(
    model="amazon.titan-embed-text-v2:0",
    region_name="us-east-1",
)

# グローバル設定
Settings.llm = llm
Settings.embed_model = embed_model
```

### Bedrock Knowledge Bases Retrieverの統合

```python
from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever

# Knowledge Bases Retrieverの作成
kb_retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="XXXXXXXXXX",  # Knowledge Base ID
    retrieval_config={
        "vectorSearchConfiguration": {
            "numberOfResults": 10,
            "overrideSearchType": "HYBRID",  # ハイブリッド検索（キーワード+セマンティック）
        }
    },
    region_name="us-east-1",
)
```

### AgentWorkflowによるAgentic RAG構成

Zenn記事のAgentWorkflowパターンをBedrock上で実装する。

```python
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool

# Knowledge Bases を QueryEngineTool として登録
tech_kb_tool = QueryEngineTool.from_defaults(
    query_engine=tech_kb_index.as_query_engine(),
    name="tech_knowledge_base",
    description="技術文書・設計書・API仕様書を検索する",
)

finance_kb_tool = QueryEngineTool.from_defaults(
    query_engine=finance_kb_index.as_query_engine(),
    name="finance_knowledge_base",
    description="財務報告書・予算計画を検索する",
)

# Web検索ツール（Knowledge Basesで不十分な場合のフォールバック）
web_search_tool = FunctionTool.from_defaults(
    fn=web_search,
    name="web_search",
    description="Knowledge Basesに情報がない場合にWeb検索する",
)

# Multi-Index Routing Agent
router_agent = FunctionAgent(
    name="RouterAgent",
    description="クエリの性質を分析し、最適なKnowledge Baseを選択する",
    tools=[tech_kb_tool, finance_kb_tool, web_search_tool],
    system_prompt="""質問に最も関連するKnowledge Baseを選んで検索してください。
    技術的な質問は tech_knowledge_base を、
    財務に関する質問は finance_knowledge_base を使用してください。
    Knowledge Basesに情報がない場合のみ web_search を使用してください。""",
    can_handoff_to=["QualityChecker"],
)

# 検索品質チェックAgent
quality_agent = FunctionAgent(
    name="QualityChecker",
    description="検索結果の品質を評価し、不十分なら再検索を指示する",
    tools=[evaluate_relevance],
    system_prompt="検索結果がクエリに十分な情報を含むか評価してください。",
    can_handoff_to=["RouterAgent", "Synthesizer"],
)

# 回答生成Agent
synthesizer_agent = FunctionAgent(
    name="Synthesizer",
    description="検索結果を統合して最終回答を生成する",
    tools=[],
    system_prompt="検索結果を元に、正確で簡潔な回答を生成してください。",
    can_handoff_to=[],
)

# Agentic RAG Workflow
workflow = AgentWorkflow(
    agents=[router_agent, quality_agent, synthesizer_agent],
    root_agent="RouterAgent",
)

# 実行
response = await workflow.run(
    user_msg="Q4の売上データと技術ロードマップの関連を分析して"
)
```

### Bedrock Knowledge Bases のハイブリッド検索

AWSブログでは、Knowledge BasesのHYBRID検索モードの活用を推奨している。これはキーワード検索（BM25）とセマンティック検索（ベクトル類似度）を組み合わせたもので、Zenn記事で言及されている検索精度向上のアプローチに直接対応する。

```python
# ハイブリッド検索の設定
retrieval_config = {
    "vectorSearchConfiguration": {
        "numberOfResults": 10,
        "overrideSearchType": "HYBRID",  # SEMANTIC / HYBRID
        "filter": {
            "equals": {
                "key": "document_type",
                "value": "technical"
            }
        }
    }
}
```

**検索モードの比較:**

| モード | 手法 | 適用ケース |
|--------|------|----------|
| SEMANTIC | ベクトル類似度のみ | 意味的に近い文書の検索 |
| **HYBRID** | BM25 + ベクトル類似度 | **推奨: キーワードマッチ + 意味的類似度の両方** |

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $100-250 | Lambda + Bedrock + KB + OpenSearch Serverless |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $500-1,200 | ECS Fargate + Bedrock + KB + OpenSearch |
| **Large** | 300,000+ (10,000/日) | Container | $3,000-8,000 | EKS + Bedrock + KB + OpenSearch Managed |

**Small構成の詳細** (月額$100-250):
- **Lambda**: 1GB RAM, 120秒タイムアウト ($25/月)
- **Amazon Bedrock**: Mistral Large ($80/月) — エージェント推論
- **Bedrock Knowledge Bases**: マネージドRAGパイプライン ($20/月) — チャンク分割・埋め込み自動
- **OpenSearch Serverless**: ベクトル検索 ($50/月)
- **S3**: ドキュメントストレージ ($5/月)
- **CloudWatch**: 監視 ($5/月)

**Bedrock Knowledge Basesのコスト利点**:
- ベクトルDBの運用管理不要 → 運用工数ゼロ
- ドキュメント追加時の自動再インデックス → 差分更新相当の機能をマネージドで提供
- IAM統合 → アプリケーション層のセキュリティ実装不要

**コスト試算の注意事項**:
- 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値
- Bedrock Knowledge Basesは米国東部リージョンが最安（東京リージョンは10-20%高い場合あり）
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

**Small構成 (Serverless): Lambda + Bedrock + Knowledge Bases**

```hcl
# --- S3（ドキュメントソース） ---
resource "aws_s3_bucket" "documents" {
  bucket = "agentic-rag-documents"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

# --- IAMロール（Bedrock Knowledge Bases用） ---
resource "aws_iam_role" "bedrock_kb" {
  name = "bedrock-knowledge-base-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "bedrock.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "bedrock_kb_s3" {
  role = aws_iam_role.bedrock_kb.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.documents.arn,
          "${aws_s3_bucket.documents.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "aoss:APIAccessAll"
        ]
        Resource = "*"
      }
    ]
  })
}

# --- Bedrock Knowledge Base ---
resource "aws_bedrockagent_knowledge_base" "tech_docs" {
  name     = "tech-documents-kb"
  role_arn = aws_iam_role.bedrock_kb.arn

  knowledge_base_configuration {
    type = "VECTOR"
    vector_knowledge_base_configuration {
      embedding_model_arn = "arn:aws:bedrock:ap-northeast-1::foundation-model/amazon.titan-embed-text-v2:0"
    }
  }

  storage_configuration {
    type = "OPENSEARCH_SERVERLESS"
    opensearch_serverless_configuration {
      collection_arn    = aws_opensearchserverless_collection.rag.arn
      vector_index_name = "tech-docs-index"
      field_mapping {
        vector_field   = "embedding"
        text_field     = "text"
        metadata_field = "metadata"
      }
    }
  }
}

# --- Knowledge Base Data Source (S3) ---
resource "aws_bedrockagent_data_source" "tech_docs_s3" {
  knowledge_base_id = aws_bedrockagent_knowledge_base.tech_docs.id
  name              = "tech-docs-s3"

  data_source_configuration {
    type = "S3"
    s3_configuration {
      bucket_arn = aws_s3_bucket.documents.arn
    }
  }
}

# --- Lambda関数（Agentic RAGハンドラ） ---
resource "aws_lambda_function" "agentic_rag" {
  filename      = "agentic_rag.zip"
  function_name = "agentic-rag-llamaindex"
  role          = aws_iam_role.agentic_rag_lambda.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 120
  memory_size   = 1024

  environment {
    variables = {
      KNOWLEDGE_BASE_ID = aws_bedrockagent_knowledge_base.tech_docs.id
      BEDROCK_MODEL_ID  = "mistral.mistral-large-2402-v1:0"
      SEARCH_TYPE       = "HYBRID"
      MAX_RESULTS       = "10"
    }
  }
}

# --- OpenSearch Serverless ---
resource "aws_opensearchserverless_collection" "rag" {
  name = "agentic-rag-collection"
  type = "VECTORSEARCH"
}

# --- CloudWatch アラーム ---
resource "aws_cloudwatch_metric_alarm" "bedrock_latency" {
  alarm_name          = "agentic-rag-bedrock-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ModelLatency"
  namespace           = "AWS/Bedrock"
  period              = 300
  statistic           = "Average"
  threshold           = 10000
  alarm_description   = "Bedrockモデルレイテンシ異常（10秒超過）"
}
```

### セキュリティベストプラクティス

**本番環境での推奨設定**:

1. **ネットワークセキュリティ**:
   - Lambda: VPC内配置、プライベートサブネット使用
   - Bedrock: VPCエンドポイント経由でプライベートアクセス
   - OpenSearch Serverless: VPCエンドポイント、ネットワークポリシー設定

2. **認証・認可**:
   - IAMロール: 最小権限の原則（PoLP）
   - Bedrock: モデル単位のアクセス制御
   - Knowledge Bases: データソース単位のアクセス制御

3. **データ保護**:
   - S3: KMS暗号化（保管時）
   - Bedrock: TLS 1.2以上（転送中）
   - OpenSearch: KMS暗号化

### 運用・監視設定

**CloudWatch Logs Insights — Agentic RAG監視**:
```sql
fields @timestamp, knowledge_base_id, search_type, result_count, latency_ms
| stats avg(latency_ms) as avg_latency,
        avg(result_count) as avg_results,
        count(*) as query_count
  by knowledge_base_id, search_type
| sort avg_latency desc
```

**Bedrock使用量モニタリング（Python）**:
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Bedrockトークン使用量アラート
cloudwatch.put_metric_alarm(
    AlarmName='bedrock-agentic-rag-tokens',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='InputTokenCount',
    Namespace='AWS/Bedrock',
    Period=3600,
    Statistic='Sum',
    Threshold=1000000,
    AlarmDescription='Bedrockトークン使用量異常（100万トークン/時間）'
)
```

### コスト最適化チェックリスト

- [ ] Bedrock Knowledge BasesのHYBRID検索を使用（SEMANTIC単体より精度向上）
- [ ] Knowledge Basesの自動再インデックスを活用（手動グラフ再構築不要）
- [ ] Bedrock Prompt Caching有効化（システムプロンプト部分で30-90%削減）
- [ ] OpenSearch Serverlessの最小容量設定（アイドル時コスト削減）
- [ ] Lambda関数のメモリ最適化（CloudWatch Insights分析）
- [ ] Bedrock Batch API活用（非リアルタイム処理で50%割引）
- [ ] VPCエンドポイント使用（NAT Gatewayコスト回避）
- [ ] S3 Intelligent-Tiering（アクセス頻度に応じた自動ストレージクラス最適化）
- [ ] AWS Budgets月額予算設定（80%警告、100%アラート）

## パフォーマンス最適化（Performance）

### Bedrock Knowledge Bases vs 自前ベクトルDB

| 項目 | 自前構築 (ChromaDB等) | Bedrock Knowledge Bases |
|------|---------------------|------------------------|
| **セットアップ時間** | 数時間〜数日 | **数分** |
| **ベクトルDB運用** | 自己責任 | **AWS管理** |
| **ドキュメント更新** | 手動再インデックス | **S3アップロードで自動** |
| **検索レイテンシ** | ~50ms | ~100ms（ネットワーク経由） |
| **スケーリング** | 手動 | **自動** |
| **コスト** | EC2/Fargate + DB料金 | **Knowledge Bases料金のみ** |

**チューニング手法:**
- `numberOfResults`（top-k）を5-15の範囲で調整
- HYBRID検索を使用（キーワード+セマンティック）
- メタデータフィルタでドキュメントタイプを絞り込み

## 運用での学び（Production Lessons）

### Knowledge Bases活用のベストプラクティス

1. **ドキュメント分類**: 技術文書と財務文書は別々のKnowledge Baseに分離（Multi-Index Routingの実現）
2. **チャンク設定**: デフォルト（300トークン）で開始し、QA精度を見ながら調整
3. **メタデータ活用**: ドキュメントの作成日・カテゴリ・著者をメタデータに設定し、フィルタリングに活用
4. **モデル選択**: 推論にはMistral Large（コスト効率）、埋め込みにはTitan Embed v2（AWS最適化）

### 障害パターンと対策

| 障害パターン | 原因 | 対策 |
|-------------|------|------|
| Knowledge Base同期遅延 | S3→KB同期に数分かかる | CloudWatch Eventsで同期完了を監視 |
| Bedrockレート制限 | バーストトラフィック | エクスポネンシャルバックオフ + SQSバッファ |
| OpenSearch容量超過 | ドキュメント急増 | Auto Scaling + 容量アラート設定 |

## 学術研究との関連（Academic Connection）

AWSブログのアーキテクチャは、以下の学術研究の成果を本番環境に適用したものと位置づけられる。

| 学術概念 | AWS実装 |
|---------|---------|
| **Agentic RAG** (2501.15228) | LlamaIndex Agent + Bedrock |
| **Multi-Index Routing** | 複数Knowledge Bases + AgentWorkflow |
| **Hybrid Search** | Knowledge Bases HYBRID mode (BM25 + Vector) |
| **Corrective RAG** | Agent品質チェック + Web検索フォールバック |

LlamaIndex v0.14のAgentic Retrieval 3段階（Auto-Routed → Multi-Index → PropertyGraph）のうち、Bedrock Knowledge Basesは**Stage 1-2をマネージドサービスで実現**する。Stage 3（PropertyGraphIndex）は別途Neptune等の構築が必要。

## まとめと実践への示唆

AWS公式ブログのAgentic RAGアーキテクチャは、Zenn記事で紹介されているLlamaIndex v0.14の機能を本番環境で運用するための具体的なリファレンス実装である。

**主要なテイクアウェイ:**
1. **Bedrock Knowledge Basesでベクトル検索基盤を簡素化** — 自前でChromaDB/Qdrantを運用する必要がない
2. **LlamaIndex AgentWorkflowでAgentic RAGを宣言的に構築** — Multi-Index Routing、品質チェック、フォールバックを3エージェント構成で実現
3. **HYBRID検索モードの活用** — キーワードマッチ + セマンティック検索の併用で精度向上
4. **マネージドサービスの活用でTotal Cost of Ownership削減** — 運用工数ゼロの検索基盤

LlamaIndex v0.14で学んだ設計パターン（AgentWorkflow、Agentic Retrieval）を、AWSのマネージドサービスで本番化するための実践的ガイドとして活用してほしい。

## 参考文献

- **Blog URL**: https://aws.amazon.com/blogs/machine-learning/create-an-agentic-rag-application-for-advanced-knowledge-discovery-with-llamaindex-and-mistral-in-amazon-bedrock/
- **Amazon Bedrock Knowledge Bases**: https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html
- **LlamaIndex Bedrock Integration**: https://docs.llamaindex.ai/en/stable/examples/llm/bedrock/
- **Related Zenn article**: https://zenn.dev/0h_n0/articles/62e946539206db
