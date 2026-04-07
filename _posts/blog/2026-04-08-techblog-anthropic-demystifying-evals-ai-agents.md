---
layout: post
title: "Anthropic Engineering解説: AIエージェント評価（Evals）の体系的設計手法"
description: "Anthropicが公開したAIエージェント評価ガイドを詳細解説。3種のGrader設計、pass@k/pass^kメトリクス、エージェント種別ごとの評価戦略を整理する"
categories: [blog, tech_blog]
tags: [LLM, agent, evaluation, observability, langsmith, langchain]
date: 2026-04-08 09:00:00 +0900
source_type: tech_blog
source_domain: anthropic.com
source_url: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
zenn_article: 734ae787f0cc54
zenn_url: https://zenn.dev/0h_n0/articles/734ae787f0cc54
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Demystifying evals for AI agents（Anthropic Engineering Blog, 2026年1月9日公開）](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) の解説記事です。

## ブログ概要（Summary）

Anthropicのエンジニアリングチーム（Mikaela Grace, Jeremy Hadfield, Rodrigo Olivares, Jiri De Jonghe）が、AIエージェントの評価（Evals）を体系的に設計・運用するための実践ガイドを公開した。記事では、評価の基本概念（Task, Trial, Grader, Transcript, Outcome）を定義した上で、3種類のGrader（Code-Based, Model-Based, Human）の使い分け、エージェント種別（コーディング、会話、リサーチ、コンピュータ操作）ごとの評価戦略、非決定性に対応するpass@k / pass^kメトリクス、そして段階的な導入ロードマップを提示している。

この記事は [Zenn記事: LangSmithでLLMエージェントをデバッグする実践ガイド 2026年版](https://zenn.dev/0h_n0/articles/734ae787f0cc54) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- **組織**: Anthropic Engineering
- **発表日**: 2026年1月9日

## 技術的背景（Technical Background）

LLMエージェントの品質保証において、従来のソフトウェアテストとは根本的に異なる課題が存在する。エージェントの出力は非決定的であり、同一入力に対して複数の正解パスが存在しうる。また、マルチターンのインタラクション、ツール呼び出し、外部環境とのやり取りが絡むため、単純な入出力テストでは品質を測定できない。

Anthropicのブログでは、この課題に対して「Swiss Cheese Model」（複数の評価手法を重ね合わせることで、個々の手法の穴を補い合う）というアプローチを提唱している。これはLangSmithが提供するトレーシング・評価機能と密接に関連しており、LangSmithのMulti-turn Evalsが解決しようとする課題の理論的背景を理解するのに有用である。

## 評価の基本構造（Evaluation Fundamentals）

Anthropicのブログでは、評価の構成要素を以下のように定義している。

| 用語 | 定義 | LangSmithでの対応概念 |
|------|------|----------------------|
| **Task** | 入力と成功基準が定義された単一テスト | Dataset内のExample |
| **Trial** | Taskに対する個別の実行試行 | 個別のRun |
| **Grader** | エージェント出力を採点するロジック | Evaluator |
| **Transcript** | 出力・ツール呼び出し・推論の完全記録 | Trace |
| **Outcome** | Trialが完了した後の環境最終状態 | Runの最終出力 |
| **Evaluation harness** | 評価をE2Eで実行するインフラ | LangSmith Experiments |

著者らは、「シングルターン評価はシンプル（プロンプト→応答→採点）だが、マルチターン評価では複数インタラクションにわたるエージェントの振る舞いを追跡する必要がある」と述べている。

## 3種類のGrader設計

ブログの中核は、3種類のGraderの使い分けに関する詳細なガイダンスである。

### 1. Code-Based Grader

決定的な基準で自動採点する方式。コーディングエージェントではユニットテスト通過率、状態検証エージェントでは環境変化の確認に用いる。

**手法**: 文字列マッチング、バイナリテスト、静的解析、ツール呼び出し検証

**利点**: 高速、低コスト、再現性100%、デバッグ容易

**欠点**: 著者らは「正解の変形を受け入れられない脆さがある」と指摘している。例えば、同一の正解であっても表記揺れや同義表現で不合格になるケースがある。

```python
# Code-Based Graderの例: ツール呼び出し検証
def grade_tool_calls(transcript: dict) -> dict:
    """エージェントが適切なツールを呼び出したか検証する"""
    expected_tools = {"search_documents", "get_weather"}
    actual_tools = {
        call["tool_name"]
        for step in transcript["steps"]
        for call in step.get("tool_calls", [])
    }

    missing = expected_tools - actual_tools
    extra = actual_tools - expected_tools

    return {
        "score": 1.0 if not missing else len(actual_tools & expected_tools) / len(expected_tools),
        "missing_tools": list(missing),
        "unexpected_tools": list(extra),
    }
```

### 2. Model-Based Grader（LLM-as-a-Judge）

LLMを評価者として使用する方式。LangSmithのMulti-turn Evalsでも採用されている手法である。

**手法**: ルーブリック評価、自然言語アサーション、ペアワイズ比較、リファレンスベース評価

**利点**: 柔軟性が高く、オープンエンドなタスクにも適用可能

**欠点**: 著者らは「非決定的であり、人間の判断との較正（calibration）が必要」と注意喚起している。

```python
# Model-Based Graderの例: ルーブリック評価
RUBRIC_PROMPT = """
以下のエージェントの応答を評価してください。

評価基準:
1. ユーザー意図の理解度（1-5点）
2. 情報の正確性（1-5点）
3. ツール使用の効率性（1-5点）

応答:
{agent_response}

JSON形式で回答:
{"intent": <1-5>, "accuracy": <1-5>, "efficiency": <1-5>, "reasoning": "<理由>"}
"""
```

### 3. Human Grader

SME（Subject Matter Expert）レビュー、クラウドソーシング、A/Bテスト等で人間が評価する方式。

著者らは「ゴールドスタンダードの品質を提供するが、スケーリングにはコストと時間がかかる」と述べている。主にModel-Based Graderの較正に使用することを推奨している。

## エージェント種別ごとの評価戦略

### コーディングエージェント

著者らは「ソフトウェアは評価が比較的容易であるため、決定的Graderが自然に適合する」と述べている。SWE-bench VerifiedやTerminal-Benchといったベンチマークがユニットテストを主要シグナルとして使用している。

推奨される評価の組み合わせ:
1. パス/フェイルテスト（主要成果の検証）
2. LLMルーブリック（コード品質の評価）
3. 静的解析ツール（ruff, mypy, bandit）
4. 状態検証（環境変更の確認）

### 会話エージェント

多次元の成功基準が必要となる。

- 検証可能なエンドステート（チケット解決、返金処理等）
- ターン数制約（X回以内での完了）
- LLMルーブリックによるインタラクション品質評価
- ユーザーペルソナのシミュレーション

これはLangSmithのMulti-turn Evalsが対象としている領域と一致する。

### リサーチエージェント

著者らは、主観的な品質評価と変化するグラウンドトゥルースが課題であると指摘している。以下を組み合わせる:

- Groundednessチェック（ソースに裏付けがあるか）
- カバレッジチェック（重要事実の漏れがないか）
- ソース品質の検証
- 専門家による較正

## 非決定性メトリクス: pass@kとpass^k

エージェントの出力は非決定的であるため、単一の試行結果だけでは信頼性の高い評価ができない。著者らは2つの補完的メトリクスを提示している。

**pass@k**: $k$回の試行のうち、少なくとも1回正解する確率。

$$
\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

ここで、$n$は総試行数、$c$は正解数、$k$は選択する試行数。$k$が増えるほどスコアは上昇する。

**pass^k**: $k$回の試行がすべて正解する確率。

$$
\text{pass}^k = \left(\frac{c}{n}\right)^k
$$

$k$が増えるほどスコアは低下し、一貫性を要求する。

著者らは「pass@kが100%に近づく一方でpass^kが0%に向かって低下するケースは、エージェントが正解を出せる能力は持っているが一貫性に欠けることを示す」と分析している。この乖離がLangSmithのInsights Agentが検出しようとする障害パターンの一つである。

## 実装ロードマップ（Step 0-8）

著者らは段階的な導入を推奨している。

**Step 0-3: データセット構築**
- 20-50件の実際の失敗事例からスタート（数百件を待たない）
- 手動チェックをテストケースに変換
- 曖昧さのないタスク仕様と参照解を準備
- 陽性・陰性の両方をテストするバランスの取れた問題セット

**Step 4-5: ハーネスとGrader設計**
- テスト環境の隔離（Trial間の干渉防止）
- 特定パスではなく成果（Outcome）を採点（脆さの回避）
- 複合タスクへの部分点の実装
- LLM-as-judge Graderの人間専門家との較正

**Step 6-8: 長期保守**
- Transcriptの定期レビュー（採点の公正性確認）
- 評価飽和の監視（pass rateが100%に近づいたら）
- ドメイン専門家の貢献を含む専任オーナーシップ
- eval-driven development（実装前に成功メトリクスを定義）

## Swiss Cheese Model: 評価手法の組み合わせ

著者らは、単一の評価手法に依存せず複数を重ね合わせる「Swiss Cheese Model」を提唱している。

| 手法 | 主な価値 |
|------|---------|
| 自動Evals | 高速イテレーション、再現性、CI/CD統合 |
| 本番監視 | 実世界のグラウンドトゥルース |
| A/Bテスト | 統計的制御下でのユーザー成果検証 |
| ユーザーフィードバック | 予期しない障害の発見 |
| 手動Transcriptレビュー | 微妙な品質評価、直感の構築 |
| 体系的ヒューマンスタディ | 較正のためのゴールドスタンダード |

LangSmithはこのうち「自動Evals」「本番監視」「手動Transcriptレビュー」をカバーしており、Polly（AIアシスタント）が手動レビューの効率化を担っている。

## 評価フレームワークの比較

著者らはブログの付録で、主要な評価フレームワークを紹介している。

| フレームワーク | 特徴 |
|---------------|------|
| **Harbor** | コンテナベース、標準化されたTask/Grader形式 |
| **Braintrust** | オフライン評価＋本番オブザーバビリティの統合 |
| **LangSmith** | LangChainエコシステム統合、トレーシング＋評価 |
| **Langfuse** | セルフホスト対応、データレジデンシー制御 |
| **Arize Phoenix/AX** | LLMトレーシング＋最適化 |

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

エージェント評価パイプラインをAWS上で構築する場合の推奨構成を示す。

**トラフィック量別の推奨構成**:

| 規模 | 評価頻度 | 推奨構成 | 月額コスト | 主要サービス |
|------|---------|---------|-----------|------------|
| **Small** | ~100 eval/日 | Serverless | $80-200 | Lambda + Bedrock + DynamoDB |
| **Medium** | ~1,000 eval/日 | Hybrid | $500-1,200 | Lambda + ECS Fargate + ElastiCache |
| **Large** | 10,000+ eval/日 | Container | $3,000-8,000 | EKS + Karpenter + Spot Instances |

**Small構成の詳細**（月額$80-200）:
- **Lambda**: 1GB RAM, 120秒タイムアウト（eval実行用）$30/月
- **Bedrock**: Claude 3.5 Haiku（Model-Based Grader用）$100/月
- **DynamoDB**: On-Demand（eval結果保存）$10/月
- **S3**: Transcript保存 $5/月
- **CloudWatch**: 基本監視 $5/月

**Medium構成の詳細**（月額$500-1,200）:
- **Lambda**: イベントトリガー $50/月
- **ECS Fargate**: 1 vCPU, 2GB RAM × 2タスク（eval実行）$200/月
- **Bedrock**: Claude 3.5 Sonnet（高品質Grader用）$600/月
- **ElastiCache Redis**: cache.t3.micro（結果キャッシュ）$15/月
- **SQS**: eval ジョブキュー $5/月

**コスト削減テクニック**:
- Bedrock Batch API使用で50%削減（非リアルタイム評価に最適）
- Prompt Caching有効化で30-90%削減（同一ルーブリックの再利用）
- Spot Instances使用で最大90%削減（EKS + Karpenter）

**コスト試算の注意事項**:
- 上記は2026年4月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です
- 実際のコストはeval頻度、モデル選択、Transcript長により変動します
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

**Small構成（Serverless）: Lambda + Bedrock + DynamoDB**

```hcl
# --- IAMロール（最小権限） ---
resource "aws_iam_role" "eval_lambda" {
  name = "eval-pipeline-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "bedrock_grader" {
  role = aws_iam_role.eval_lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ]
      Resource = "arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.claude-3-5-haiku*"
    }]
  })
}

# --- Lambda関数（eval実行） ---
resource "aws_lambda_function" "eval_runner" {
  filename      = "eval_runner.zip"
  function_name = "agent-eval-runner"
  role          = aws_iam_role.eval_lambda.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 120
  memory_size   = 1024

  environment {
    variables = {
      BEDROCK_MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"
      DYNAMODB_TABLE   = aws_dynamodb_table.eval_results.name
      S3_BUCKET        = aws_s3_bucket.transcripts.id
    }
  }
}

# --- DynamoDB（eval結果保存） ---
resource "aws_dynamodb_table" "eval_results" {
  name         = "agent-eval-results"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "eval_id"
  range_key    = "trial_id"

  attribute {
    name = "eval_id"
    type = "S"
  }
  attribute {
    name = "trial_id"
    type = "S"
  }

  ttl {
    attribute_name = "expire_at"
    enabled        = true
  }
}

# --- S3（Transcript保存） ---
resource "aws_s3_bucket" "transcripts" {
  bucket = "agent-eval-transcripts"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "transcripts" {
  bucket = aws_s3_bucket.transcripts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

# --- CloudWatchアラーム ---
resource "aws_cloudwatch_metric_alarm" "eval_errors" {
  alarm_name          = "eval-runner-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "Eval実行エラー率異常"
  dimensions = {
    FunctionName = aws_lambda_function.eval_runner.function_name
  }
}
```

### セキュリティベストプラクティス

- **IAMロール**: 最小権限の原則（Bedrockモデル固有のARN指定）
- **シークレット管理**: AWS Secrets Manager使用、環境変数ハードコード禁止
- **暗号化**: S3/DynamoDB全てKMS暗号化
- **ネットワーク**: Lambda VPC内配置、パブリックアクセス不要

### 運用・監視設定

**CloudWatch Logs Insights クエリ**:
```sql
-- eval失敗パターン分析
fields @timestamp, eval_id, grader_type, score
| filter score < 0.5
| stats count(*) as fail_count by grader_type, bin(1h)
| sort fail_count desc

-- Grader別レイテンシ分析
fields @timestamp, grader_type, duration_ms
| stats pct(duration_ms, 95) as p95, pct(duration_ms, 99) as p99 by grader_type
```

**CloudWatch アラーム（コスト重視）**:
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_alarm(
    AlarmName='bedrock-eval-token-spike',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='TokenUsage',
    Namespace='AWS/Bedrock',
    Period=3600,
    Statistic='Sum',
    Threshold=200000,
    AlarmDescription='Bedrock eval用トークン使用量異常'
)
```

### コスト最適化チェックリスト

**アーキテクチャ選択**:
- [ ] ~100 eval/日 → Lambda + Bedrock (Serverless) - $80-200/月
- [ ] ~1,000 eval/日 → ECS Fargate + Bedrock (Hybrid) - $500-1,200/月
- [ ] 10,000+ eval/日 → EKS + Spot (Container) - $3,000-8,000/月

**LLMコスト削減**:
- [ ] Code-Based Grader優先（LLM呼び出しゼロ）
- [ ] Bedrock Batch API使用で50%削減
- [ ] Prompt Caching有効化（ルーブリック固定部分のキャッシュ）
- [ ] モデル選択: 簡易eval→Haiku、複雑eval→Sonnet

**監視・アラート**:
- [ ] AWS Budgets: 月額予算設定（80%で警告）
- [ ] CloudWatch: eval失敗率・レイテンシ監視
- [ ] Cost Anomaly Detection: 自動異常検知
- [ ] 日次コストレポート: SNS通知

## 学術研究との関連（Academic Connection）

Anthropicの評価フレームワークは、LLM-as-a-Judge（Zheng et al., 2023, MT-Bench）の概念を実践レベルに落とし込んだものと位置づけられる。特にpass@k / pass^kメトリクスはコード生成分野（Chen et al., 2021, HumanEval）で確立された手法をエージェント評価に拡張したものである。LangSmithのMulti-turn EvalsはこのModel-Based Graderを会話全体に適用した実装例である。

## まとめと実践への示唆

Anthropicの評価ガイドは、LangSmithのようなオブザーバビリティプラットフォームを「なぜ使うべきか」の理論的根拠を提供している。著者らが強調するeval-driven development（実装前に評価基準を定義する）は、LangSmithのDataset → Experiments → Trace分析というワークフローと自然に対応する。LLMエージェントの品質保証を体系化するための実践的な出発点として、このブログは有用な指針となる。

## 参考文献

- **Blog URL**: [Demystifying evals for AI agents（Anthropic Engineering）](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- **Related Papers**: Zheng et al., "Judging LLM-as-a-Judge" (2023), Chen et al., "Evaluating Large Language Models Trained on Code" (2021)
- **Related Zenn article**: [LangSmithでLLMエージェントをデバッグする実践ガイド 2026年版](https://zenn.dev/0h_n0/articles/734ae787f0cc54)
