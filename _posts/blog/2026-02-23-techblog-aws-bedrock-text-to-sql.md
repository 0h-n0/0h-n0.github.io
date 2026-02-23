---
layout: post
title: "AWS解説: Amazon Bedrock Knowledge Basesによる構造化データの自然言語クエリ — マネージドNL2SQLの実装パターン"
description: "Amazon Bedrock Knowledge Basesの構造化データ対応機能を解説。RAGベースのスキーマ理解、NL2SQL変換、自己修正パイプラインの実装パターンを詳述。"
categories: [blog, tech_blog]
tags: [aws, bedrock, text-to-sql, rag, knowledge-bases, claude, nlp]
date: 2026-02-23 13:00:00 +0900
source_type: tech_blog
source_domain: aws.amazon.com
source_url: https://aws.amazon.com/blogs/machine-learning/build-conversational-interfaces-for-structured-data-using-amazon-bedrock-knowledge-bases/
zenn_article: 58dc3076d2ffba
zenn_url: https://zenn.dev/0h_n0/articles/58dc3076d2ffba
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [AWS Machine Learning Blog: Build conversational interfaces for structured data using Amazon Bedrock Knowledge Bases](https://aws.amazon.com/blogs/machine-learning/build-conversational-interfaces-for-structured-data-using-amazon-bedrock-knowledge-bases/) の解説記事です。

## ブログ概要（Summary）

AWSは2025年にAmazon Bedrock Knowledge Basesに構造化データ対応機能を追加し、マネージドなNL2SQL（Natural Language to SQL）モジュールを提供している。これにより、データベース構造やSQL構文を理解していないユーザーでも、自然言語で構造化データにクエリを実行できるようになった。AWSのブログ記事群では、RAGを活用したスキーマ理解の強化、Claude 3 Sonnetを使ったSQL生成、自己修正パイプラインの実装パターンが複数回にわたって紹介されている。

この記事は [Zenn記事: LangGraph×Claude Sonnet 4.6でSQL統合Agentic RAGを実装する](https://zenn.dev/0h_n0/articles/58dc3076d2ffba) の深掘りです。Zenn記事がLangGraph+Claude Sonnet 4.6でのカスタム実装を扱うのに対し、AWSのアプローチはBedrock Knowledge BasesによるマネージドサービスとしてのText-to-SQL実装を提示しており、「自作 vs マネージド」の設計判断に必要な情報を提供する。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://aws.amazon.com/blogs/machine-learning/build-conversational-interfaces-for-structured-data-using-amazon-bedrock-knowledge-bases/](https://aws.amazon.com/blogs/machine-learning/build-conversational-interfaces-for-structured-data-using-amazon-bedrock-knowledge-bases/)
- **組織**: Amazon Web Services (AWS)
- **関連記事群**:
  - [Text-to-SQL with RAG using Bedrock Claude 3 Sonnet](https://aws.amazon.com/blogs/machine-learning/build-your-gen-ai-based-text-to-sql-application-using-rag-powered-by-amazon-bedrock-claude-3-sonnet-and-amazon-titan-for-embedding/)
  - [Robust Text-to-SQL with self-correction](https://aws.amazon.com/blogs/machine-learning/build-a-robust-text-to-sql-solution-generating-complex-queries-self-correcting-and-querying-diverse-data-sources/)
  - [Conversational data assistant with Bedrock Agents](https://aws.amazon.com/blogs/machine-learning/build-a-conversational-data-assistant-part-1-text-to-sql-with-amazon-bedrock-agents/)

## 技術的背景（Technical Background）

### エンタープライズデータアクセスの課題

企業のデータ資産は多くの場合、RDB（Amazon Redshift、Aurora、RDS等）に格納されている。しかし、以下の理由で非技術者のデータアクセスが制限されている。

1. **SQL知識の壁**: ビジネスユーザーはSQLを書けない
2. **スキーマ理解の壁**: 数百テーブル・数千カラムのDBスキーマを把握するのは専門家でも困難
3. **セキュリティの壁**: 直接的なDB接続はセキュリティリスクを伴う

AWSはこれらの課題に対し、Bedrock Knowledge Basesの構造化データ対応として、マネージドNL2SQLモジュールを提供している。

### マネージドNL2SQLの位置づけ

AWSのブログによると、Amazon Bedrock Knowledge Basesは「最初の完全マネージドなout-of-the-boxのRAGソリューションの一つであり、構造化データがある場所でネイティブにクエリできる」と位置づけられている。これにより、生成AIアプリケーションの構築期間を「1ヶ月以上から数日」に短縮できるとされている。

## 実装アーキテクチャ（Architecture）

### Bedrock Knowledge Bases構造化データ対応

AWSが提供するNL2SQLの処理フローは以下のとおりである。

```
ユーザーの自然言語クエリ
       │
       ▼
┌─────────────────────────────────────┐
│  Amazon Bedrock Knowledge Bases     │
│  ┌─────────────────────────────┐   │
│  │ 1. スキーマ理解              │   │
│  │    - データカタログ参照       │   │
│  │    - RAGによるスキーマ補完    │   │
│  └──────────┬──────────────────┘   │
│             ▼                       │
│  ┌─────────────────────────────┐   │
│  │ 2. SQL生成                   │   │
│  │    - Foundation Model        │   │
│  │    - (Claude 3 Sonnet等)     │   │
│  └──────────┬──────────────────┘   │
│             ▼                       │
│  ┌─────────────────────────────┐   │
│  │ 3. SQL実行                   │   │
│  │    - Query Engine経由        │   │
│  └──────────┬──────────────────┘   │
│             ▼                       │
│  ┌─────────────────────────────┐   │
│  │ 4. 回答生成                  │   │
│  │    - LLMによる自然言語変換    │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
       │
       ▼
ユーザーへの自然言語回答
```

### RAGベースのスキーマ理解

AWSのアプローチの特徴は、RAGを使ってLLMにスキーマ情報を提供する点にある。

**データカタログ連携:**
- AWS Glueデータカタログのメタ情報（テーブル名、カラム名、データ型、説明文）をベクトルエンベディング化
- カラムのシノニム（別名）情報をRAGコンテキストに含めることで、ビジネス用語とDB用語のギャップを埋める
- サンプルクエリをベクトルストアに格納し、類似クエリのFew-shot例として活用

**処理の流れ:**

```python
from typing import TypedDict

class NL2SQLState(TypedDict):
    """NL2SQL処理の状態"""
    user_query: str
    relevant_tables: list[str]
    relevant_columns: list[str]
    sample_queries: list[str]
    generated_sql: str
    execution_result: str
    natural_language_answer: str


async def schema_retrieval(state: NL2SQLState) -> dict:
    """RAGベースのスキーマ検索

    ユーザークエリに基づいて関連するテーブル・カラム情報を
    ベクトル検索で取得する。
    """
    query_embedding = embed_model.encode(state["user_query"])

    # データカタログのベクトルインデックスから類似スキーマを検索
    relevant_schemas = vector_store.similarity_search(
        query_embedding,
        k=10,
        filter={"type": "schema_metadata"},
    )

    # 類似のサンプルクエリを検索
    sample_queries = vector_store.similarity_search(
        query_embedding,
        k=3,
        filter={"type": "sample_query"},
    )

    return {
        "relevant_tables": extract_tables(relevant_schemas),
        "relevant_columns": extract_columns(relevant_schemas),
        "sample_queries": [q.page_content for q in sample_queries],
    }
```

### Claude 3 SonnetによるSQL生成

AWSのブログでは、Claude 3 Sonnet（Amazon Titan for Embedding）を使ったText-to-SQL実装が紹介されている。

```python
async def generate_sql(state: NL2SQLState) -> dict:
    """Claude 3 SonnetによるSQL生成

    RAGで取得したスキーマ情報とサンプルクエリを
    コンテキストとしてLLMに渡し、SQLを生成する。
    """
    prompt = f"""You are an expert SQL query generator.
Based on the following database schema and sample queries,
generate a SQL query to answer the user's question.

## Database Schema
Tables: {state['relevant_tables']}
Columns: {state['relevant_columns']}

## Sample Queries (Few-shot examples)
{chr(10).join(state['sample_queries'])}

## Rules
- Generate only SELECT statements
- Use LIMIT 100 for safety
- Use explicit column names (avoid SELECT *)

## User Question
{state['user_query']}

Generate the SQL query:"""

    response = await bedrock_client.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body={"prompt": prompt},
    )

    return {"generated_sql": response["completion"]}
```

### 自己修正パイプライン

AWSの「Robust Text-to-SQL」ブログ記事では、生成されたSQLの自己修正メカニズムが詳述されている。

```python
async def execute_and_correct(
    state: NL2SQLState,
    max_retries: int = 3
) -> dict:
    """SQL実行と自己修正ループ

    生成SQLを実行し、エラー時はLLMに修正を依頼する。
    Amazon Athenaを使った多様なデータソースへの対応も可能。
    """
    current_sql = state["generated_sql"]

    for attempt in range(max_retries):
        try:
            result = await query_engine.execute(current_sql)
            if result and len(result) > 0:
                return {
                    "execution_result": format_result(result),
                    "generated_sql": current_sql,
                }
        except Exception as e:
            error_msg = str(e)

        # LLMに修正を依頼
        correction_prompt = f"""The SQL query failed with error:
{error_msg}

Original question: {state['user_query']}
Failed SQL: {current_sql}

Generate a corrected SQL query."""

        response = await bedrock_client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body={"prompt": correction_prompt},
        )
        current_sql = response["completion"]

    return {
        "execution_result": "Query could not be executed after retries",
        "generated_sql": current_sql,
    }
```

## パフォーマンス最適化（Performance）

### セマンティックキャッシング

AWSの別のブログ記事（[AI-powered text-to-SQL chatbot](https://aws.amazon.com/blogs/database/build-an-ai-powered-text-to-sql-chatbot-using-amazon-bedrock-amazon-memorydb-and-amazon-rds/)）では、Amazon MemoryDBを使ったセマンティックキャッシングが紹介されている。

- 過去のクエリとその結果をベクトルインデックスに格納
- 類似クエリが来た場合、LLM呼び出しをスキップしてキャッシュから結果を返す
- LLM APIコストの削減とレスポンス時間の短縮を実現

### AWS Glue連携のメリット

- **自動メタデータ収集**: AWS Glueクローラーがスキーマ情報を自動収集
- **データカタログの一元管理**: 複数DBのスキーマを統合的に管理
- **カラムシノニム**: ビジネス用語とDB用語のマッピングを管理

## 運用での学び（Production Lessons）

### Zenn記事との比較

| 項目 | Zenn記事（カスタム実装） | AWS Bedrock（マネージド） |
|------|------------------------|------------------------|
| 実装工数 | 高（LangGraph設計・実装） | 低（設定ベース） |
| 柔軟性 | 高（任意のロジック実装可） | 中（AWSサービスの範囲内） |
| スケーリング | 手動（インフラ管理必要） | 自動（マネージド） |
| コスト | LLM API + インフラ | Bedrock従量課金 |
| ルーティング | カスタムロジック | Bedrock Agents |
| データソース | SQLite/PostgreSQL | Redshift/Aurora/Athena |

### マネージド vs カスタムの選択基準

**Bedrock Knowledge Basesが適するケース:**
- AWSインフラを既に使用している
- Redshift/Aurora上に構造化データがある
- 運用の手間を最小化したい
- SQLスキーマが安定している

**LangGraphカスタム実装が適するケース:**
- 複雑なルーティングロジックが必要（SQL/ベクトル/両方の動的判定）
- マルチクラウドまたはオンプレミス環境
- Claude Sonnet 4.6の最新機能（structured output等）を活用したい
- カスタムの自己修正ロジックや後処理が必要

## 学術研究との関連（Academic Connection）

AWSのText-to-SQLアプローチは、以下の学術研究に基づいている。

- **RAGによるスキーマ理解**: CHESS (Talaei et al., 2024) のスキーマ選択と同様のRAG活用パターン
- **Few-shot例の選択**: DAIL-SQL (Gao et al., 2023) のFew-shot選択戦略を、ベクトル検索によるサンプルクエリ取得として実装
- **自己修正ループ**: Self-Refine (Madaan et al., NeurIPS 2023) のフィードバックループをSQL実行結果に適用

## まとめと実践への示唆

AWSのBedrock Knowledge Bases構造化データ対応は、マネージドサービスとしてのText-to-SQL実装を提供する。RAGベースのスキーマ理解、Claude 3 SonnetによるSQL生成、自己修正ループの組み合わせは、Zenn記事のLangGraph実装と技術的に共通する要素が多い。

主な違いは「マネージド vs カスタム」の設計判断にあり、AWSインフラを活用する場合はBedrock Knowledge Basesが迅速な立ち上げに有利である一方、LangGraphによるカスタム実装はルーティングロジックの柔軟性で優位性がある。両アプローチの技術的な基盤は、CHESSやDAIL-SQL等の学術研究に共通しており、スキーマ選択と自己修正がText-to-SQL精度向上の鍵であるという点で一致している。

## 参考文献

- **Blog URL (構造化データ対応)**: [https://aws.amazon.com/blogs/machine-learning/build-conversational-interfaces-for-structured-data-using-amazon-bedrock-knowledge-bases/](https://aws.amazon.com/blogs/machine-learning/build-conversational-interfaces-for-structured-data-using-amazon-bedrock-knowledge-bases/)
- **Blog URL (Text-to-SQL with RAG)**: [https://aws.amazon.com/blogs/machine-learning/build-your-gen-ai-based-text-to-sql-application-using-rag-powered-by-amazon-bedrock-claude-3-sonnet-and-amazon-titan-for-embedding/](https://aws.amazon.com/blogs/machine-learning/build-your-gen-ai-based-text-to-sql-application-using-rag-powered-by-amazon-bedrock-claude-3-sonnet-and-amazon-titan-for-embedding/)
- **Amazon Bedrock**: [https://aws.amazon.com/bedrock/](https://aws.amazon.com/bedrock/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/58dc3076d2ffba](https://zenn.dev/0h_n0/articles/58dc3076d2ffba)
