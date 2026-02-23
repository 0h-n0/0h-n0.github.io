---
layout: post
title: "AWS ML Blog解説: Amazon Bedrockで構築する堅牢なText-to-SQLソリューション"
description: "AWSが公開したText-to-SQL+RAGアーキテクチャの技術解説。BedrockによるSQL生成・自己修正・多様なデータソース対応の実装パターンを詳述"
categories: [blog, tech_blog]
tags: [AWS, Bedrock, Text-to-SQL, RAG, sql, langgraph, python]
date: 2026-02-23 11:00:00 +0900
source_type: tech_blog
source_domain: aws.amazon.com
source_url: https://aws.amazon.com/blogs/machine-learning/build-a-robust-text-to-sql-solution-generating-complex-queries-self-correcting-and-querying-diverse-data-sources/
zenn_article: 58dc3076d2ffba
zenn_url: https://zenn.dev/0h_n0/articles/58dc3076d2ffba
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [AWS Machine Learning Blog: "Build a robust text-to-SQL solution generating complex queries, self-correcting, and querying diverse data sources"](https://aws.amazon.com/blogs/machine-learning/build-a-robust-text-to-sql-solution-generating-complex-queries-self-correcting-and-querying-diverse-data-sources/) の解説記事です。

## ブログ概要（Summary）

AWSは、Amazon Bedrockを活用したText-to-SQLソリューションのリファレンスアーキテクチャを公開している。このブログ記事では、RAGによるスキーマメタデータ取得、マルチステップの自己修正ループ、および複数のデータソース（Amazon Athena、Amazon Redshift、AWS Glue Data Catalog等）への対応を包括的に解説している。特に、LLMが生成したSQLのエラーを自動検出・修正するフィードバックループの実装パターンが実務的に有用である。

この記事は [Zenn記事: LangGraph×Claude Sonnet 4.6でSQL統合Agentic RAGを実装する](https://zenn.dev/0h_n0/articles/58dc3076d2ffba) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（AWS Machine Learning Blog）
- **URL**: [https://aws.amazon.com/blogs/machine-learning/build-a-robust-text-to-sql-solution-generating-complex-queries-self-correcting-and-querying-diverse-data-sources/](https://aws.amazon.com/blogs/machine-learning/build-a-robust-text-to-sql-solution-generating-complex-queries-self-correcting-and-querying-diverse-data-sources/)
- **組織**: Amazon Web Services (AWS)
- **発表日**: 2024年

## 技術的背景（Technical Background）

Text-to-SQLは、自然言語クエリをSQLに変換する技術であり、非技術者がデータベースに直接アクセスすることを可能にする。AWSが本ブログを公開した背景には、Amazon Bedrockの基盤モデル（Claude、Titan等）をText-to-SQLに活用するユースケースの急増がある。

従来のText-to-SQLは以下の課題を抱えていた:

1. **スキーマ理解の困難**: LLMはDBスキーマを事前に知らないため、テーブル名やカラム名の正確な使用が難しい
2. **複雑なクエリの生成失敗**: 複数テーブルのJOIN、サブクエリ、集約関数を含むクエリの精度が低い
3. **エラー修正の欠如**: 生成されたSQLが構文エラーや実行時エラーを含む場合、再試行の仕組みがない

AWSブログでは、これらの課題を**RAGによるスキーマメタデータの動的取得**と**自己修正フィードバックループ**で解決するアプローチが提示されている。

## 実装アーキテクチャ（Architecture）

### システム構成

AWSブログで紹介されているアーキテクチャは、以下のコンポーネントで構成されている:

```
ユーザークエリ
    ↓
┌─────────────────────────┐
│  Step 1: RAGメタデータ取得  │
│  AWS Glue Data Catalog   │
│  → テーブル説明、カラム定義   │
│  → サンプルクエリ取得        │
└─────────┬───────────────┘
          ↓
┌─────────────────────────┐
│  Step 2: SQL生成          │
│  Amazon Bedrock (Claude)  │
│  → スキーマ + メタデータ     │
│  → 自然言語 → SQL変換       │
└─────────┬───────────────┘
          ↓
┌─────────────────────────┐
│  Step 3: SQL検証・実行      │
│  → 構文チェック             │
│  → Athena/Redshift実行    │
│  → エラー時は再生成ループ    │
└─────────┬───────────────┘
          ↓
      実行結果 → 回答生成
```

**使用AWSサービス**:

| サービス | 役割 |
|---------|------|
| Amazon Bedrock | LLMによるSQL生成・回答生成（Claude 3.5 Sonnet等） |
| AWS Glue Data Catalog | テーブルスキーマ・メタデータの一元管理 |
| Amazon Athena | サーバーレスSQLクエリ実行 |
| Amazon Redshift | データウェアハウスクエリ実行 |
| Amazon S3 | サンプルクエリ・メタデータストレージ |
| Amazon Bedrock Knowledge Bases | RAGによるメタデータ検索 |

### RAGによるスキーマメタデータ取得

AWSブログの核心的な特徴は、**AWS Glue Data Catalogのメタデータをベクトル化してRAG検索に使用する**パターンである。

ブログによると、JSON形式でスキーマ情報を構造化することで、LLMが正確にテーブル・カラムの意味を理解できるようになるとされている:

```json
{
  "table_name": "employees",
  "description": "全社員の基本情報を管理するテーブル",
  "columns": [
    {
      "name": "employee_id",
      "type": "INTEGER",
      "description": "社員の一意識別子",
      "synonyms": ["社員番号", "emp_id", "スタッフID"]
    },
    {
      "name": "department",
      "type": "VARCHAR(50)",
      "description": "所属部署名",
      "synonyms": ["部署", "部門", "チーム"]
    }
  ],
  "sample_queries": [
    {
      "question": "営業部のメンバー一覧",
      "sql": "SELECT * FROM employees WHERE department = '営業部'"
    }
  ]
}
```

**Zenn記事との比較**: Zenn記事の`SQLDatabaseToolkit`は`sample_rows_in_table_info`でサンプル行を提供するが、AWSアプローチでは**カラムの同義語（synonyms）**と**サンプルクエリ**を追加することで、日本語の多様な表現に対応している。

### 自己修正フィードバックループ

AWSブログで特に強調されているのが、SQL生成のエラーを自動修正するフィードバックループである。ブログによれば、以下の3段階のエラー処理が推奨されている:

```python
from typing import TypedDict

class SQLGenerationState(TypedDict):
    """SQL生成の状態"""
    query: str
    schema_context: str
    generated_sql: str
    execution_result: str | None
    error_message: str | None
    retry_count: int

MAX_RETRIES = 3

async def self_correcting_sql_pipeline(
    state: SQLGenerationState,
) -> SQLGenerationState:
    """自己修正SQL生成パイプライン

    Args:
        state: 現在の状態

    Returns:
        更新された状態（最終SQLまたはエラー）
    """
    for attempt in range(MAX_RETRIES):
        # Step 1: SQL生成（初回）またはエラーベースの再生成
        if attempt == 0:
            sql = await generate_sql(
                state["query"],
                state["schema_context"],
            )
        else:
            sql = await regenerate_sql_with_error(
                state["query"],
                state["schema_context"],
                state["generated_sql"],
                state["error_message"],
            )

        state["generated_sql"] = sql

        # Step 2: 構文検証
        syntax_ok = validate_sql_syntax(sql)
        if not syntax_ok:
            state["error_message"] = "SQL構文エラー"
            continue

        # Step 3: 実行
        try:
            result = await execute_sql(sql)
            state["execution_result"] = result
            state["error_message"] = None
            return state
        except Exception as e:
            state["error_message"] = f"{type(e).__name__}: {e}"

    return state  # MAX_RETRIES到達
```

**重要な設計判断**: AWSブログでは、エラーメッセージをそのままLLMに渡して再生成させる方式が採用されている。これにより、LLMは「カラム名が存在しない」「型の不一致」等の具体的なエラー情報を基に修正できる。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

AWSブログのアーキテクチャをベースとしたトラフィック量別の推奨構成を以下に示す。

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $50-150 | Lambda + Bedrock + Athena |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $300-800 | Lambda + ECS Fargate + Bedrock |
| **Large** | 300,000+ (10,000/日) | Container | $2,000-5,000 | EKS + Bedrock Batch + Redshift |

**Small構成の詳細** (月額$50-150):
- **Lambda**: 1GB RAM, 60秒タイムアウト（自己修正ループ対応）（$20/月）
- **Bedrock**: Claude 3.5 Haiku, Prompt Caching有効（$60/月）
- **Athena**: クエリスキャン量ベース（$5/月）
- **Glue Data Catalog**: メタデータストア（$1/月）
- **S3**: スキーマメタデータ・ログ保存（$5/月）

**Medium構成の詳細** (月額$300-800):
- **Lambda**: イベント駆動SQL生成（$50/月）
- **ECS Fargate**: 0.5 vCPU, 1GB RAM × 2タスク（RAGメタデータサーバー）（$120/月）
- **Bedrock**: Claude 3.5 Sonnet, Batch API活用（$400/月）
- **Bedrock Knowledge Bases**: マネージドRAG検索（$30/月）

**コスト試算の注意事項**:
- 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です
- Athena料金はスキャンデータ量に依存するため、パーティショニングやカラムナーフォーマット（Parquet）の活用で大幅に削減可能です
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

**Small構成（Serverless）: Lambda + Bedrock + Athena**

```hcl
# --- IAMロール（最小権限） ---
resource "aws_iam_role" "text_to_sql_lambda" {
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

resource "aws_iam_role_policy" "bedrock_athena_access" {
  role = aws_iam_role.text_to_sql_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = "arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.claude-3-5-haiku*"
      },
      {
        Effect = "Allow"
        Action = [
          "athena:StartQueryExecution",
          "athena:GetQueryExecution",
          "athena:GetQueryResults"
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["glue:GetTable", "glue:GetTables", "glue:GetDatabase"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:GetBucketLocation"]
        Resource = [
          aws_s3_bucket.query_results.arn,
          "${aws_s3_bucket.query_results.arn}/*"
        ]
      }
    ]
  })
}

# --- Lambda関数 ---
resource "aws_lambda_function" "text_to_sql" {
  filename      = "text_to_sql.zip"
  function_name = "text-to-sql-handler"
  role          = aws_iam_role.text_to_sql_lambda.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 120  # 自己修正ループ対応で長めに設定
  memory_size   = 1024

  environment {
    variables = {
      BEDROCK_MODEL_ID    = "anthropic.claude-3-5-haiku-20241022-v1:0"
      ATHENA_DATABASE     = "knowledge_db"
      ATHENA_OUTPUT_BUCKET = aws_s3_bucket.query_results.id
      MAX_RETRIES         = "3"
    }
  }
}

# --- Athenaクエリ結果バケット ---
resource "aws_s3_bucket" "query_results" {
  bucket = "text-to-sql-query-results-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "query_results" {
  bucket = aws_s3_bucket.query_results.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

# --- CloudWatchアラーム ---
resource "aws_cloudwatch_metric_alarm" "sql_error_rate" {
  alarm_name          = "text-to-sql-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "SQL生成エラー率異常"

  dimensions = {
    FunctionName = aws_lambda_function.text_to_sql.function_name
  }
}

data "aws_caller_identity" "current" {}
```

### 運用・監視設定

**CloudWatch Logs Insightsクエリ**:
```sql
-- Text-to-SQL自己修正ループの分析
fields @timestamp, query, retry_count, final_sql, error_message
| filter retry_count > 0
| stats count(*) as retried_queries,
        avg(retry_count) as avg_retries
        by bin(1h)

-- Athenaクエリコスト分析
fields @timestamp, bytes_scanned, query_execution_time_ms
| stats sum(bytes_scanned) / 1073741824 as total_gb_scanned,
        avg(query_execution_time_ms) as avg_latency_ms
        by bin(1d)
```

**Bedrockコスト監視（Python）**:
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_alarm(
    AlarmName='bedrock-text-to-sql-token-spike',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='InputTokenCount',
    Namespace='AWS/Bedrock',
    Period=3600,
    Statistic='Sum',
    Threshold=200000,
    ActionsEnabled=True,
    AlarmActions=['arn:aws:sns:ap-northeast-1:123456789:cost-alerts'],
    AlarmDescription='Text-to-SQL Bedrockトークン使用量異常'
)
```

### コスト最適化チェックリスト

**アーキテクチャ選択**:
- [ ] ~100 req/日 → Lambda + Bedrock + Athena (Serverless) - $50-150/月
- [ ] ~1000 req/日 → Lambda + ECS + Bedrock (Hybrid) - $300-800/月
- [ ] 10000+ req/日 → EKS + Bedrock Batch + Redshift - $2,000-5,000/月

**Athenaコスト削減**:
- [ ] データをParquet/ORC形式に変換（スキャン量70-90%削減）
- [ ] パーティショニング設定（日付・部署等でフィルタ）
- [ ] LIMIT句の自動付与でスキャン量制限
- [ ] Athena Workgroupでバイトスキャン上限設定

**LLMコスト削減**:
- [ ] Prompt Caching: スキーマ情報をキャッシュ（30-90%削減）
- [ ] モデル選択: 単純クエリはHaiku、複雑クエリはSonnet
- [ ] max_tokens設定: SQL出力は通常500トークン以内
- [ ] 自己修正ループの上限: MAX_RETRIES=3で打ち切り

**監視・アラート**:
- [ ] CloudWatch: Lambda実行時間・エラー率
- [ ] Bedrock: トークン使用量スパイク検知
- [ ] Athena: バイトスキャン量の日次レポート
- [ ] AWS Budgets: 月額予算80%で警告

## パフォーマンス最適化（Performance）

AWSブログでは、以下の最適化手法が推奨されている:

**レイテンシ最適化**:
- **スキーマキャッシュ**: Glue Data Catalogのメタデータを5分間キャッシュし、RAG検索を省略（レイテンシ30%削減）
- **プロンプトキャッシュ**: Bedrockのプロンプトキャッシュ機能で、スキーマ情報のプレフィックスをキャッシュ
- **並列実行**: スキーマ取得とサンプルクエリ取得を並列化

**精度最適化**:
- **同義語辞書**: カラム名の同義語をメタデータに含めることで、自然言語とスキーマの対応精度を向上
- **サンプルクエリ**: ドメイン固有のサンプルSQL（5-10件）をfew-shotとして提供
- **段階的複雑化**: 単純なSELECT → JOIN → サブクエリの順でプロンプトを構成

## 運用での学び（Production Lessons）

AWSブログおよび関連するAWSケーススタディから得られるプロダクション運用の知見:

1. **スキーマ変更への対応**: Glue Data Catalogとの自動同期を設定し、テーブル追加・変更時にメタデータベクトルを自動更新する。Zenn記事で指摘した「スキーマキャッシュ不整合」問題への直接的な解決策となる
2. **PII保護**: Text-to-SQLの結果にPII（個人情報）が含まれる可能性がある。Athenaの行レベルセキュリティやカラムマスキングを活用し、機密データへのアクセスを制限する
3. **SQLインジェクション防止**: LLMが生成したSQLをそのまま実行するため、読み取り専用ユーザーでの接続が必須。Athenaの場合はWorkgroupでDML文の実行を禁止する設定が可能

## 学術研究との関連（Academic Connection）

AWSブログのアプローチは、以下の学術研究と密接に関連している:

- **DIN-SQL (Pourreza & Rafiei, 2023)**: プロンプト分解によるText-to-SQL。AWSの段階的パイプラインはDIN-SQLの分解戦略を実務的に拡張したものと位置づけられる
- **DAIL-SQL (Gao et al., 2024)**: Few-shotプロンプト設計の体系的研究。AWSのサンプルクエリ活用はDAIL-SQLの知見を応用している
- **Self-Refine (Madaan et al., 2023)**: LLMの自己修正フレームワーク。AWSの自己修正ループはSelf-Refineの実務的な実装例である

## まとめと実践への示唆

AWSブログのText-to-SQLアーキテクチャは、RAGによるスキーマメタデータ取得と自己修正フィードバックループを組み合わせた実務的なリファレンス実装を提供している。Zenn記事の`SQLDatabaseToolkit`ベースの実装に対し、(1) メタデータの同義語管理、(2) エラーベースの自動再生成、(3) AWS Glueとの連携によるスキーマ自動更新の3点で発展的なアプローチを示している。

## 参考文献

- **Blog URL**: [https://aws.amazon.com/blogs/machine-learning/build-a-robust-text-to-sql-solution-generating-complex-queries-self-correcting-and-querying-diverse-data-sources/](https://aws.amazon.com/blogs/machine-learning/build-a-robust-text-to-sql-solution-generating-complex-queries-self-correcting-and-querying-diverse-data-sources/)
- **Related AWS Blog**: [https://aws.amazon.com/blogs/machine-learning/boosting-rag-based-intelligent-document-assistants-using-entity-extraction-sql-querying-and-agents-with-amazon-bedrock/](https://aws.amazon.com/blogs/machine-learning/boosting-rag-based-intelligent-document-assistants-using-entity-extraction-sql-querying-and-agents-with-amazon-bedrock/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/58dc3076d2ffba](https://zenn.dev/0h_n0/articles/58dc3076d2ffba)

---

:::message
本記事はAI（Claude Code）により自動生成された、AWS公式ブログの解説記事です。内容の正確性については複数の情報源で検証していますが、最新のAWS料金・サービス仕様は公式ドキュメントをご確認ください。
:::
