---
layout: post
title: "CIDR 2025論文解説: Text2SQL is Not Enough — TAGフレームワークによるDB×LLM推論の統合"
description: "Text2SQLとRAGの限界を克服し、データベースクエリ実行とLLM推論を統合するTAGフレームワークの解説"
categories: [blog, paper, conference]
tags: [Text-to-SQL, RAG, LLM, database, LangGraph, agentic-rag, sql]
date: 2026-02-23 09:00:00 +0900
source_type: conference
conference: CIDR 2025
source_url: https://arxiv.org/abs/2408.14717
zenn_article: 58dc3076d2ffba
zenn_url: https://zenn.dev/0h_n0/articles/58dc3076d2ffba
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [arXiv:2408.14717](https://arxiv.org/abs/2408.14717)（CIDR 2025採択）の解説記事です。

## 論文概要（Abstract）

Biswalらは、自然言語によるデータベース質問応答において、Text2SQLとRAGはそれぞれ相補的な強みを持つが、単独では実世界の複雑な質問の20〜80%に回答できないことを示した。著者らは**Table-Augmented Generation（TAG）**を提案し、データベースのクエリ実行能力とLLMの世界知識・意味推論能力を統合する一般化フレームワークを構築している。BIRD-TAGベンチマークにおいて、TAGは正答率を19%（RAG単体）から50%へ向上させたと報告されている。

この記事は [Zenn記事: LangGraph×Claude Sonnet 4.6でSQL統合Agentic RAGを実装する](https://zenn.dev/0h_n0/articles/58dc3076d2ffba) の深掘りです。

## 情報源

- **会議名**: CIDR 2025（Conference on Innovative Data Systems Research）
- **URL**: [https://arxiv.org/abs/2408.14717](https://arxiv.org/abs/2408.14717)
- **著者**: Asim Biswal, Liana Patel, Siddarth Jha, Amog Kamsetty, Shu Liu, Joseph E. Gonzalez, Carlos Guestrin, Matei Zaharia（全員UC Berkeley）
- **発表形式**: CIDR 2025 採択論文

## カンファレンス情報

**CIDRについて**:
CIDRはデータベース・データシステム分野の招待制カンファレンスであり、革新的なデータシステム研究を対象とする。VLDB/SIGMODと並ぶ主要会議の一つで、特に新しいパラダイムやビジョン論文が重視される。本論文はText2SQL/RAGの限界を指摘し、新しいパラダイムとしてTAGを提案するビジョン論文として採択されている。

## 背景と動機

### Text2SQLの限界

Text2SQL（自然言語→SQL変換）は、構造化データへのアクセスを可能にする手法として広く研究されている。しかし著者らは、以下の限界を指摘している：

- **世界知識の欠如**: 「この映画監督の年収より興行収入は高いか？」のような質問では、DB内に「監督の年収」が存在しない。Text2SQLはDBスキーマ外の知識を持たないため回答不能となる
- **意味推論の不足**: 「このレストランの雰囲気はデートに適しているか？」のような主観的・意味的判断を含む質問にSQLは対応できない
- **著者らの分析によると、BIRD-SQLベンチマークで20〜30%の質問がText2SQL単体では回答不能**

### RAGの限界

RAG（Retrieval-Augmented Generation）はベクトル検索で外部知識にアクセスできるが、構造化データの正確なクエリに弱い：

- **精確な集計・フィルタリングの困難**: 「売上トップ5の商品」のようなクエリでは、SQLのGROUP BY・ORDER BY・LIMITによる正確な集計がRAGでは困難
- **JOIN操作の不在**: 複数テーブルにまたがるリレーション結合はベクトル検索の範囲外
- **著者らの分析によると、BIRD-TAGベンチマークでRAGは60〜80%の質問に回答不能**

この「Text2SQLは構造化クエリに強いが世界知識が無い」「RAGは世界知識にアクセスできるが構造化クエリが弱い」というギャップが、TAG提案の動機である。

## 主要な貢献（Key Contributions）

論文の主要な貢献は以下の3点である：

- **貢献1**: Text2SQLとRAGを特殊ケースとして内包する一般化フレームワーク**TAG（Table-Augmented Generation）**の提案
- **貢献2**: DB計算とLLM推論を明示的にオーケストレーションする**TAGクエリオペレータ**の形式的定義
- **貢献3**: 既存ベンチマーク（BIRD-SQL）がDB単独/LLM単独で解ける問題とTAGが必要な問題を混在させていた欠陥を修正した**BIRD-TAGベンチマーク**の構築

## 技術的詳細（Technical Details）

### TAGオペレータモデルの構造

TAGは3段階のパイプラインとして定式化される。

```
自然言語クエリ q
    ↓
[Stage 1] Query Synthesis（クエリ合成）
    → LLMがqを分析し、SQLクエリ s と推論プロンプト p を生成
    ↓
[Stage 2] Query Execution（クエリ実行）
    → SQLエンジンが s を実行し、構造化結果 R を取得
    ↓
[Stage 3] LLM Reasoning（LLM推論）
    → LLMが R と自身の世界知識を統合し、最終回答 a を生成
```

形式的には以下のように定義される：

$$
\text{TAG}(q) = \text{LLM}_{\text{reason}}(R, p, \mathcal{K})
$$

ここで、
- $q$: 自然言語クエリ
- $R = \text{DB}.\text{execute}(s)$: SQLクエリ $s$ の実行結果
- $s, p = \text{LLM}_{\text{synth}}(q, \mathcal{S})$: LLMがスキーマ $\mathcal{S}$ を参照して合成したSQLと推論プロンプト
- $\mathcal{K}$: LLMが持つ世界知識（パラメトリック知識）
- $a$: 最終回答

### Text2SQL・RAGとの関係

TAGはText2SQLとRAGの**上位互換**として設計されている：

| 側面 | Text2SQL | RAG | TAG |
|------|----------|-----|-----|
| 処理範囲 | SQLで表現可能な質問のみ | ベクトル類似検索可能な質問 | DB+LLM推論が必要な複合質問 |
| 世界知識 | 使用しない | 外部ドキュメントから取得 | LLMパラメトリック知識を活用 |
| 構造化クエリ | SQL全機能 | 不可（近似検索のみ） | SQL全機能 |
| 集計・フィルタ | 高精度 | 困難 | 高精度（DB側で処理） |
| セマンティック推論 | 不可 | LLMに依存 | LLMが明示的に担当 |

重要な点として、Text2SQLはTAGの特殊ケース（Stage 3の推論が不要な場合）であり、RAGもTAGの特殊ケース（Stage 2のSQL実行がベクトル検索に置き換わる場合）として位置づけられる。

### TAGが解決する質問タイプの具体例

著者らは以下のカテゴリを定義している：

**Type 1: DB単独で解ける（Text2SQL十分）**
- 例: 「営業部の社員数は？」→ `SELECT COUNT(*) FROM employees WHERE dept='営業'`

**Type 2: LLM単独で解ける（RAG十分）**
- 例: 「SQLインジェクションとは何か？」→ LLMの世界知識で回答

**Type 3: DB+LLM両方が必要（TAGが必要）**
- 例: 「このデータベースに含まれる映画の中で、監督の推定年収を超える興行収入を持つ作品は？」
  - Step 1（SQL）: 映画タイトルと興行収入をDBから取得
  - Step 2（LLM）: 各監督の推定年収を世界知識から推定
  - Step 3（LLM推論）: 興行収入と年収を比較して回答

### アルゴリズム

```python
from typing import TypedDict
from dataclasses import dataclass


@dataclass
class TAGResult:
    """TAGパイプラインの実行結果"""
    sql_query: str
    db_result: list[dict]
    reasoning_prompt: str
    final_answer: str


class TAGPipeline:
    """Table-Augmented Generation パイプライン

    Text2SQLとRAGを統合し、DB計算+LLM推論の
    ハイブリッド質問応答を実現する。
    """

    def __init__(self, llm, db_engine, schema: str):
        self.llm = llm
        self.db_engine = db_engine
        self.schema = schema

    def query_synthesis(self, question: str) -> tuple[str, str]:
        """Stage 1: 自然言語をSQLと推論プロンプトに分解

        Args:
            question: ユーザーの自然言語クエリ

        Returns:
            (sql_query, reasoning_prompt) のタプル
        """
        synthesis_prompt = f"""以下の質問を分析し、2つの要素を生成してください：
1. データベースから取得すべき情報のSQL（SELECT文のみ）
2. DB結果とLLM知識を統合するための推論プロンプト

スキーマ: {self.schema}
質問: {question}"""

        response = self.llm.invoke(synthesis_prompt)
        sql_query = extract_sql(response)
        reasoning_prompt = extract_reasoning_prompt(response)
        return sql_query, reasoning_prompt

    def query_execution(self, sql_query: str) -> list[dict]:
        """Stage 2: SQLをデータベースで実行

        Args:
            sql_query: 実行するSQL文

        Returns:
            クエリ結果の辞書リスト
        """
        return self.db_engine.execute(sql_query)

    def llm_reasoning(
        self,
        db_result: list[dict],
        reasoning_prompt: str,
        question: str,
    ) -> str:
        """Stage 3: DB結果とLLM世界知識を統合して回答生成

        Args:
            db_result: SQL実行結果
            reasoning_prompt: Stage 1で生成した推論プロンプト
            question: 元の質問

        Returns:
            最終回答テキスト
        """
        context = f"""データベース検索結果:
{format_results(db_result)}

推論タスク: {reasoning_prompt}
元の質問: {question}

上記のDB結果とあなたの知識を組み合わせて回答してください。
DB結果に含まれない情報は、あなたの知識で補完してください。
知識が不確かな場合はその旨を明記してください。"""

        return self.llm.invoke(context)

    def run(self, question: str) -> TAGResult:
        """TAGパイプライン全体を実行

        Args:
            question: 自然言語クエリ

        Returns:
            TAGResult（SQL、DB結果、推論プロンプト、最終回答）
        """
        sql_query, reasoning_prompt = self.query_synthesis(question)
        db_result = self.query_execution(sql_query)
        final_answer = self.llm_reasoning(
            db_result, reasoning_prompt, question
        )
        return TAGResult(
            sql_query=sql_query,
            db_result=db_result,
            reasoning_prompt=reasoning_prompt,
            final_answer=final_answer,
        )
```

## 実装のポイント（Implementation）

TAGを実装する際の要点は以下の通りである：

1. **Query Synthesisの品質がボトルネック**: Stage 1でLLMがSQLと推論プロンプトを正しく分離できるかが全体の精度を決定する。著者らはGPT-4oを使用しているが、プロンプト設計が重要
2. **スキーマ情報の提供方法**: LLMにDBスキーマを渡す際、テーブル定義+サンプル行（3〜5行）を含めると合成精度が向上する。これはZenn記事の`SQLDatabaseToolkit`が`sample_rows_in_table_info`パラメータで実現している手法と同一
3. **SQLの安全性**: Stage 2でLLM生成SQLを実行するため、読み取り専用接続（READ ONLY）が必須。`include_tables`でアクセス可能テーブルを制限する設計も推奨
4. **LLM推論の制御**: Stage 3で世界知識を使う際、ハルシネーションリスクがある。回答に「確信度」や「情報源の種別（DB/LLM知識）」を明示させるプロンプト設計が有効

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

TAGパイプラインをAWS上にデプロイする場合のトラフィック量別推奨構成を示す。

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $50-150 | Lambda + Bedrock + DynamoDB |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $300-800 | Lambda + ECS Fargate + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,000-5,000 | EKS + Karpenter + EC2 Spot |

**Small構成の詳細** (月額$50-150):
- **Lambda**: 1GB RAM, 60秒タイムアウト, TAGの3ステージを逐次実行 ($20/月)
- **Bedrock**: Claude 3.5 Haiku（Query Synthesis + LLM Reasoning）, Prompt Caching有効 ($80/月)
- **Aurora Serverless v2**: PostgreSQL互換, 0.5 ACU最小 ($30/月)
- **DynamoDB**: スキーマキャッシュ用, On-Demand ($10/月)

**コスト削減テクニック**:
- Spot Instances使用で最大90%削減（EKS + Karpenter構成）
- Bedrock Batch API使用で50%削減（非リアルタイム処理）
- Prompt Caching有効化でスキーマ情報の再送を30-90%削減
- Query Synthesisの結果キャッシュ（同一クエリパターンの再利用）

**コスト試算の注意事項**:
上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です。実際のコストはトラフィックパターン、リージョン、バースト使用量により変動します。最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください。

### Terraformインフラコード

**Small構成 (Serverless): Lambda + Bedrock + Aurora Serverless**

```hcl
# --- VPC基盤 ---
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "tag-pipeline-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["ap-northeast-1a", "ap-northeast-1c"]
  private_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  database_subnets = ["10.0.3.0/24", "10.0.4.0/24"]

  enable_nat_gateway   = false
  enable_dns_hostnames = true
}

# --- IAMロール（最小権限） ---
resource "aws_iam_role" "tag_lambda" {
  name = "tag-pipeline-lambda-role"

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
  role = aws_iam_role.tag_lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
      Resource = "arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.claude-3-5-haiku*"
    }]
  })
}

# --- Lambda関数（TAGパイプライン） ---
resource "aws_lambda_function" "tag_handler" {
  filename      = "tag_pipeline.zip"
  function_name = "tag-pipeline-handler"
  role          = aws_iam_role.tag_lambda.arn
  handler       = "handler.main"
  runtime       = "python3.12"
  timeout       = 120
  memory_size   = 1024

  environment {
    variables = {
      BEDROCK_MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"
      DB_SECRET_ARN    = aws_secretsmanager_secret.db_credentials.arn
      ENABLE_CACHE     = "true"
    }
  }

  vpc_config {
    subnet_ids         = module.vpc.private_subnets
    security_group_ids = [aws_security_group.lambda_sg.id]
  }
}

# --- Aurora Serverless v2（READ ONLY接続推奨） ---
resource "aws_rds_cluster" "tag_db" {
  cluster_identifier = "tag-pipeline-db"
  engine             = "aurora-postgresql"
  engine_mode        = "provisioned"
  engine_version     = "15.4"
  database_name      = "knowledge"

  serverlessv2_scaling_configuration {
    min_capacity = 0.5
    max_capacity = 4.0
  }

  storage_encrypted = true
  kms_key_id        = aws_kms_key.db_encryption.arn
}

# --- CloudWatch アラーム ---
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "tag-pipeline-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "TAG Pipeline Lambda Error Rate"
  dimensions = {
    FunctionName = aws_lambda_function.tag_handler.function_name
  }
}
```

### セキュリティベストプラクティス

**本番環境での推奨設定**:

1. **DB接続**: READ ONLYユーザーでの接続を必須とする。`include_tables`相当の設定でアクセステーブルを制限
2. **IAMロール**: Bedrockの`InvokeModel`権限のみ付与。特定モデルIDにリソース制限
3. **シークレット管理**: DB接続情報はSecrets Managerに格納。環境変数ハードコード禁止
4. **暗号化**: Aurora StorageのKMS暗号化、転送中のTLS 1.2以上を必須化
5. **監査**: CloudTrailで全API呼び出しを記録

### 運用・監視設定

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# TAGパイプラインのレイテンシ監視
cloudwatch.put_metric_alarm(
    AlarmName='tag-pipeline-latency',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=2,
    MetricName='Duration',
    Namespace='AWS/Lambda',
    Period=300,
    Statistic='p99',
    Threshold=60000,  # P99 60秒超過でアラート
    AlarmDescription='TAG Pipeline P99 Latency Alert'
)

# Bedrockトークン使用量監視
cloudwatch.put_metric_alarm(
    AlarmName='tag-bedrock-token-spike',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='InputTokenCount',
    Namespace='AWS/Bedrock',
    Period=3600,
    Statistic='Sum',
    Threshold=500000,
    AlarmDescription='Bedrock Token Usage Spike'
)
```

### コスト最適化チェックリスト

- [ ] ~100 req/日 → Lambda + Bedrock (Serverless) - $50-150/月
- [ ] ~1000 req/日 → ECS Fargate + Aurora (Hybrid) - $300-800/月
- [ ] 10000+ req/日 → EKS + Spot Instances (Container) - $2,000-5,000/月
- [ ] Aurora Serverless v2: 最小ACU 0.5設定（アイドル時コスト削減）
- [ ] Bedrock Prompt Caching: スキーマ情報のキャッシュで30-90%削減
- [ ] Query Synthesis結果キャッシュ: DynamoDB TTL 24h
- [ ] Lambda: メモリ1024MB最適化（CloudWatch Insights分析）
- [ ] Bedrock Batch API: 非リアルタイム処理で50%削減
- [ ] AWS Budgets: 月額予算設定（80%で警告）
- [ ] Cost Anomaly Detection: 自動異常検知
- [ ] タグ戦略: 環境別（dev/staging/prod）でコスト可視化
- [ ] 未使用リソース定期削除: Trusted Advisor活用

## 実験結果（Results）

### BIRD-TAGベンチマーク

著者らは既存のBIRD-SQLベンチマークを拡張し、TAGが必要な質問群を分離したBIRD-TAGベンチマークを構築した。以下は論文で報告されている主要結果である：

| 手法 | BIRD-TAG正答率 | 備考 |
|------|---------------|------|
| RAG | 19% | ベクトル検索のみ |
| Text2SQL (GPT-4o) | ~40%台 | SQL実行のみ、世界知識不使用 |
| **TAG（提案手法）** | **50%** | SQL実行+LLM推論の統合 |

（数値は論文のAbstract・実験セクションより引用）

**分析ポイント**:
- TAGはRAGに対して+31ポイントの大幅改善を達成している
- Text2SQLと比較しても約10ポイントの改善が報告されている
- 50%という正答率は、実世界の複雑な質問（DB外知識+セマンティック推論が必要）に対する数値であり、DB単独で解ける問題を含む通常のBIRD-SQLスコアとは比較対象が異なる点に注意

### 既存手法の失敗分析

著者らはText2SQLとRAGの失敗パターンを定量分析している：

- **Text2SQL**: 20〜30%の質問で回答不能。主因はDBスキーマに存在しない世界知識への依存
- **RAG**: 60〜80%の質問で回答不能。主因は構造化クエリ（集計・JOIN・フィルタ）の処理不足

この分析は、Zenn記事で実装したSQL+ベクトル検索統合のアプローチが正しい方向性であることを裏付けている。ただし、TAGはさらにLLMの世界知識を明示的に組み込む点で一歩先を行っている。

## 実運用への応用（Practical Applications）

### Zenn記事のアーキテクチャとの対応

Zenn記事で実装したLangGraph StateGraphによるSQL+ベクトル検索統合は、TAGフレームワークの部分的な実装と見なせる：

| Zenn記事の構成要素 | TAGの対応ステージ |
|-------------------|-----------------|
| ルーターノード（Claude + with_structured_output） | Stage 1: Query Synthesis |
| SQL検索ノード（SQLDatabaseToolkit） | Stage 2: Query Execution |
| ベクトル検索ノード（ChromaDB） | Stage 2の別パス（非構造化検索） |
| 回答生成ノード | Stage 3: LLM Reasoning |

**TAGの視点から見たZenn記事の拡張方向**:

1. **世界知識の明示的統合**: 現在の回答生成ノードにLLMの世界知識を積極的に活用する指示を追加
2. **Query Synthesisの強化**: ルーターの出力にSQL設計意図と推論プロンプトを含める
3. **BIRD-TAGベンチマークでの評価**: 自社データでTAGスタイルの評価ベンチマークを構築

### 制約と限界

- TAGの正答率50%は改善の余地が大きい（著者ら自身が認めている）
- Query Synthesisの品質がLLMに依存し、安定性に課題がある
- LLMの世界知識はハルシネーションリスクを伴う
- 計算コストがText2SQL単体の約2倍（LLM呼び出しが2回）

## 関連研究（Related Work）

- **DAIL-SQL** (Gao et al., 2023): In-context learningベースのText2SQL手法。デモ選択アルゴリズムが特徴。TAGはDAIL-SQLのSQL生成能力をStage 2に組み込みつつ、Stage 3でLLM推論を追加する上位互換として位置づけられる
- **DIN-SQL** (Pourreza & Rafiei, 2023): 分割統治法によるSQL生成。複雑なクエリを分解する点がTAGのQuery Synthesisと共通
- **DeepRAG** (Guan et al., 2025): MDP定式化によるステップ単位検索制御。TAGのStage 1（検索の要否判断）と設計思想が類似

## まとめと今後の展望

- TAGはText2SQLとRAGを包含する一般化フレームワークとして、DB計算能力とLLM推論能力の統合を実現した
- BIRD-TAGベンチマークで正答率を19%から50%に向上させたが、実用化には更なる改善が必要
- Zenn記事のLangGraph×Claude Sonnet実装は、TAGの部分的実現と位置づけられ、Query SynthesisとLLM Reasoningの強化で拡張可能
- 今後の研究方向として、著者らはQuery Synthesis精度の向上、マルチモーダルデータへの拡張、より大規模なベンチマーク構築を挙げている

## 参考文献

- **arXiv**: [https://arxiv.org/abs/2408.14717](https://arxiv.org/abs/2408.14717)
- **CIDR 2025 Proceedings**: [https://vldb.org/cidrdb/papers/2025/p11-biswal.pdf](https://vldb.org/cidrdb/papers/2025/p11-biswal.pdf)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/58dc3076d2ffba](https://zenn.dev/0h_n0/articles/58dc3076d2ffba)
- **BIRD-SQL Benchmark**: Li et al., 2024 — TAGの評価基盤となったNL2SQLベンチマーク
