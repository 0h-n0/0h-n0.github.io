---
layout: post
title: "ICLR 2025 Spotlight論文解説: TheAgentCompany — 実世界タスクでのLLMエージェントベンチマーク"
description: "ソフトウェア企業環境をシミュレーションしLLMエージェントの実タスク完遂能力を評価するベンチマーク"
categories: [blog, paper, conference]
tags: [LLM-agent, benchmark, evaluation, ICLR, real-world, langgraph, rag]
date: 2026-02-22 13:00:00 +0900
source_type: conference
conference: ICLR 2025
arxiv_id: "2412.14161"
source_url: https://arxiv.org/abs/2412.14161
zenn_article: 88cd951a1ec060
zenn_url: https://zenn.dev/0h_n0/articles/88cd951a1ec060
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## 論文概要（Abstract）

TheAgentCompanyは、LLMエージェントが実世界の業務タスクをどの程度自律的に遂行できるかを評価するベンチマークである。ソフトウェア企業の環境をセルフホスト型のインフラで再現し、GitLab・RocketChat・ownCloud・Planeなど実務で使われるツール群を統合した175の業務タスクを提供する。最も高性能なエージェントでも自律的に完了できるタスクは全体の約30%にとどまり、現状のLLMエージェントの能力と限界を定量的に示した研究である。

この記事は [Zenn記事: LangGraphマルチエージェントRAGの評価フレームワーク設計と協調品質の定量化](https://zenn.dev/0h_n0/articles/88cd951a1ec060) の深掘りです。

## 情報源

- **会議名**: ICLR 2025（Spotlight）
- **年**: 2025
- **URL**: [https://arxiv.org/abs/2412.14161](https://arxiv.org/abs/2412.14161)
- **著者**: Frank F. Xu, Yufan Song, Boxuan Li et al.（Carnegie Mellon University）
- **発表形式**: Spotlight

## カンファレンス情報

**ICLR（International Conference on Learning Representations）について**: ICLRは機械学習・深層学習分野における最高峰の国際会議の一つであり、表現学習を中心に幅広いAI研究を扱う。Spotlight採択は上位数パーセントの論文に与えられる高い評価であり、TheAgentCompanyはLLMエージェント評価の重要性が認められた形となっている。

## 技術的詳細（Technical Details）

### シミュレーション環境アーキテクチャ

TheAgentCompanyは、小規模ソフトウェア企業を模した自己完結型のシミュレーション環境を構築している。環境は以下の4つのセルフホスト型OSSで構成される。

| サービス | 役割 | 対応する実務ツール |
|----------|------|-------------------|
| GitLab | バージョン管理、CI/CD、コードレビュー | GitHub, GitLab SaaS |
| RocketChat | チーム内メッセージング、同僚との対話 | Slack, Teams |
| ownCloud | ファイルストレージ、ドキュメント共同編集 | Google Drive, SharePoint |
| Plane | イシュートラッキング、スプリント管理 | Jira, Linear |

エージェントには3つのインターフェースが提供される。(1) bashシェル、(2) Jupyter IPythonサーバ、(3) Playwright経由のChromiumブラウザである。これにより、コード実行・データ分析・Webブラウジング・チャットコミュニケーションのすべてが可能となる。

### タスクカテゴリと分布

175タスクは以下の5つの職種カテゴリに分類される。

| カテゴリ | タスク数 | 例 |
|----------|---------|-----|
| SDE（ソフトウェア開発） | 69 | GitLabリポジトリへのバグ修正PR作成 |
| HR（人事） | 29 | 従業員情報の集約・レポート作成 |
| PM（プロジェクト管理） | 28 | Planeでのスプリント計画・イシュー整理 |
| Admin（管理） | 15 | ownCloudのアクセス権限設定 |
| DS（データサイエンス） | 14 | データ分析・可視化レポート作成 |
| Finance（財務） | 12 | 経費精算・財務レポート作成 |

### 評価メトリクス

TheAgentCompanyは2種類の完了スコアを定義している。

**完全完了スコア（Full Completion Score）**:

$$
S_{\text{full}} = \begin{cases} 1 & \text{if all checkpoints pass} \\ 0 & \text{otherwise} \end{cases}
$$

**部分完了スコア（Partial Completion Score）**:

$$
S_{\text{partial}} = 0.5 \times \frac{\sum_{i} c_i}{\sum_{i} t_i} + 0.5 \times S_{\text{full}}
$$

ここで、
- $c_i$: チェックポイント$i$で獲得したポイント
- $t_i$: チェックポイント$i$の最大ポイント
- $S_{\text{full}}$: 完全完了スコア（0または1）

部分完了スコアは、途中まで正しく進められたケースにも一定の評価を与えつつ、完全完了に対して0.5のボーナスを加算することで、最後までタスクを遂行するインセンティブを設計している。

### 効率メトリクス

タスク遂行の効率性は以下の2指標で測定される。

- **ステップ数**: LLM呼び出しの総回数
- **コスト**: トークン消費量に基づくAPI利用料金（プロンプトキャッシュなし前提）

$$
\text{Cost} = n_{\text{prompt}} \times r_{\text{prompt}} + n_{\text{completion}} \times r_{\text{completion}}
$$

ここで、$n_{\text{prompt}}$はプロンプトトークン数、$r_{\text{prompt}}$はプロンプト単価、$n_{\text{completion}}$は生成トークン数、$r_{\text{completion}}$は生成単価である。

### 同僚NPC（Simulated Colleagues）

RocketChatを通じたコミュニケーションタスクでは、Claude 3.5 Sonnetを用いたNPC（Non-Player Character）がSotopiaプラットフォーム上で同僚役を務める。タスクの初期指示には含まれない情報をNPCから引き出す必要があり、エージェントの情報収集・対話能力が試される。

## 査読者の評価（Peer Review Insights）

OpenReviewでの査読情報によると、TheAgentCompanyはベンチマーク設計の網羅性と実務タスクの再現性が高く評価された。セルフホスト環境による再現可能性、チェックポイントベースの部分評価メトリクス、および同僚NPCを介したコミュニケーション評価が新規性として認められている。一方で、タスクの難易度分布の偏り（SDEタスクが約40%を占める）や、NPC応答品質のLLM依存性について改善提案がなされた。

## 実装のポイント（Implementation）

TheAgentCompanyの環境構築にはDockerとDocker Composeが必要であり、30GB以上のディスク容量が求められる。評価はAmazon EC2 t3.2xlargeインスタンスが推奨されている。

```python
# タスク実行と評価の基本フロー（擬似コード）
from typing import TypedDict

class TaskResult(TypedDict):
    """タスク評価結果"""
    task_id: str
    checkpoints_passed: list[bool]
    total_steps: int
    total_cost: float

def evaluate_task(task_id: str, agent_output: dict) -> TaskResult:
    """チェックポイントベースのタスク評価

    Args:
        task_id: タスク識別子
        agent_output: エージェントの出力（ファイル、コミット、メッセージ等）

    Returns:
        TaskResult: チェックポイント通過状況とコスト
    """
    checkpoints = load_checkpoints(task_id)
    results = []
    for cp in checkpoints:
        if cp.eval_type == "deterministic":
            passed = cp.check(agent_output)
        else:
            # LLM-as-Judge: Claude 3.5 Sonnetで評価
            passed = llm_judge_evaluate(cp.rubric, agent_output)
        results.append(passed)

    return TaskResult(
        task_id=task_id,
        checkpoints_passed=results,
        total_steps=agent_output["step_count"],
        total_cost=agent_output["token_cost"],
    )

def compute_partial_score(result: TaskResult) -> float:
    """部分完了スコアの計算

    Args:
        result: タスク評価結果

    Returns:
        float: 0.0-1.0の部分完了スコア
    """
    n_passed = sum(result["checkpoints_passed"])
    n_total = len(result["checkpoints_passed"])
    s_full = 1.0 if all(result["checkpoints_passed"]) else 0.0
    return 0.5 * (n_passed / n_total) + 0.5 * s_full
```

各タスクはDockerイメージとして配布され、`/utils/init.sh`で環境初期化、`/instruction/task.md`からタスク指示を取得し、`/utils/eval.py`で自動採点される。ホストネットワーキングモードでの実行が必須である点に注意が必要である。

## Production Deployment Guide

TheAgentCompanyの評価基盤は、LLMエージェントの自動テスト・CI/CDパイプラインとして実運用に転用できる。以下にAWS上でエージェント評価パイプラインをデプロイするための構成を示す。

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 構成 | ユースケース | 主要サービス | 月額コスト概算 |
|------|-------------|-------------|---------------|
| Small (~100 eval/日) | 個人開発・PoC | Lambda + Bedrock + ECR | $50-150 |
| Medium (~1,000 eval/日) | チーム開発・CI連携 | ECS Fargate + Bedrock + RDS | $300-800 |
| Large (10,000+ eval/日) | 組織横断ベンチマーク | EKS + Spot + Bedrock Batch | $2,000-5,000 |

**Small構成（~100 eval/日）**:
- AWS Lambda: タスク実行のオーケストレーション（メモリ1024MB、タイムアウト900秒）
- Amazon Bedrock: Claude 3.5 Sonnet（NPC応答・LLM-as-Judge）
- Amazon ECR: タスクDockerイメージの管理
- Amazon DynamoDB: 評価結果・チェックポイント状態の永続化
- Amazon S3: タスク成果物（コード差分、レポート等）の保存
- 月額内訳: Lambda $5 + Bedrock $30-100 + DynamoDB $5 + S3 $5 + ECR $5

**Medium構成（~1,000 eval/日）**:
- ECS Fargate: GitLab・RocketChat・ownCloud・Planeのコンテナ群
- Application Load Balancer: サービスディスカバリとルーティング
- Amazon RDS (PostgreSQL): GitLab・PlaneのバックエンドDB
- ElastiCache (Redis): セッション管理・キャッシュ
- 月額内訳: Fargate $150-300 + Bedrock $100-400 + RDS $50 + ALB $30

**Large構成（10,000+ eval/日）**:
- EKS + Karpenter: タスクコンテナの自動スケーリング（Spot優先）
- Bedrock Batch API: 非同期評価で50%コスト削減
- Amazon MSK: タスクキュー・イベント駆動パイプライン
- 月額内訳: EKS $70 + EC2 Spot $500-1,500 + Bedrock Batch $500-2,000 + MSK $200

**コスト削減テクニック**:
- Spot Instances活用でEC2コストを最大90%削減（t3.2xlarge Spot: ~$0.10/hr vs On-Demand: $0.3328/hr）
- Reserved Instances 1年コミットで最大72%削減
- Bedrock Batch APIで非リアルタイム評価を50%削減
- Prompt Caching有効化でNPC応答コストを30-90%削減

> **注意**: 上記コストはAWS ap-northeast-1（東京）リージョンの2026年2月時点の概算値です。実際のコストはトラフィックパターン、リージョン、バースト使用量により変動します。最新料金は[AWS料金計算ツール](https://calculator.aws/)で確認してください。

### Terraformインフラコード

**Small構成（Serverless）**: Lambda + Bedrock + DynamoDB

```hcl
# TheAgentCompany評価パイプライン - Small構成
# Lambda + Bedrock + DynamoDB (Serverless)

terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }
}

provider "aws" {
  region = "ap-northeast-1"
}

# --- IAM Role (最小権限) ---
resource "aws_iam_role" "eval_lambda" {
  name = "theagentcompany-eval-lambda"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "eval_lambda_policy" {
  name = "theagentcompany-eval-policy"
  role = aws_iam_role.eval_lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = "arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.claude-3-5-sonnet*"
      },
      {
        Effect   = "Allow"
        Action   = ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query"]
        Resource = aws_dynamodb_table.eval_results.arn
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:GetObject"]
        Resource = "${aws_s3_bucket.artifacts.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# --- DynamoDB (On-Demand, KMS暗号化) ---
resource "aws_dynamodb_table" "eval_results" {
  name         = "theagentcompany-eval-results"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "task_id"
  range_key    = "eval_timestamp"

  attribute {
    name = "task_id"
    type = "S"
  }
  attribute {
    name = "eval_timestamp"
    type = "S"
  }

  server_side_encryption {
    enabled = true  # AWS managed KMS
  }

  tags = {
    Project = "theagentcompany"
    Env     = "production"
  }
}

# --- S3 (成果物保存、暗号化) ---
resource "aws_s3_bucket" "artifacts" {
  bucket = "theagentcompany-eval-artifacts"
  tags   = { Project = "theagentcompany" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

# --- Lambda関数 ---
resource "aws_lambda_function" "eval_orchestrator" {
  function_name = "theagentcompany-eval"
  role          = aws_iam_role.eval_lambda.arn
  runtime       = "python3.12"
  handler       = "handler.lambda_handler"
  timeout       = 900       # 15分（最大）
  memory_size   = 1024      # 1GB

  filename         = "lambda_package.zip"
  source_code_hash = filebase64sha256("lambda_package.zip")

  environment {
    variables = {
      DYNAMODB_TABLE = aws_dynamodb_table.eval_results.name
      S3_BUCKET      = aws_s3_bucket.artifacts.id
      BEDROCK_MODEL  = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    }
  }

  tags = { Project = "theagentcompany" }
}

# --- CloudWatch アラーム (コスト監視) ---
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "theagentcompany-lambda-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "Lambda evaluation errors exceed threshold"

  dimensions = {
    FunctionName = aws_lambda_function.eval_orchestrator.function_name
  }
}
```

**Large構成（Container）**: EKS + Karpenter + Spot Instances

```hcl
# TheAgentCompany評価パイプライン - Large構成
# EKS + Karpenter + Spot Instances

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.31"

  cluster_name    = "theagentcompany-eval"
  cluster_version = "1.31"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Karpenter用のIRSA
  enable_irsa = true

  eks_managed_node_groups = {
    # 最小限の管理ノード（システムPod用）
    system = {
      instance_types = ["t3.medium"]
      min_size       = 1
      max_size       = 2
      desired_size   = 1
    }
  }

  tags = { Project = "theagentcompany" }
}

# --- Karpenter Provisioner (Spot優先) ---
resource "kubectl_manifest" "karpenter_provisioner" {
  yaml_body = yamlencode({
    apiVersion = "karpenter.sh/v1"
    kind       = "NodePool"
    metadata   = { name = "eval-workers" }
    spec = {
      template = {
        spec = {
          requirements = [
            { key = "karpenter.sh/capacity-type", operator = "In", values = ["spot", "on-demand"] },
            { key = "node.kubernetes.io/instance-type", operator = "In",
              values = ["t3.2xlarge", "t3a.2xlarge", "m5.2xlarge", "m5a.2xlarge"] },
          ]
          nodeClassRef = { name = "default" }
        }
      }
      limits   = { cpu = "128", memory = "512Gi" }
      disruption = {
        consolidationPolicy = "WhenEmptyOrUnderutilized"
        consolidateAfter    = "30s"
      }
    }
  })
}

# --- Secrets Manager (Bedrock設定) ---
resource "aws_secretsmanager_secret" "bedrock_config" {
  name = "theagentcompany/bedrock-config"
}

resource "aws_secretsmanager_secret_version" "bedrock_config" {
  secret_id = aws_secretsmanager_secret.bedrock_config.id
  secret_string = jsonencode({
    model_id    = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    region      = "ap-northeast-1"
    max_tokens  = 4096
    temperature = 0.0
  })
}

# --- AWS Budgets (予算アラート) ---
resource "aws_budgets_budget" "monthly" {
  name         = "theagentcompany-monthly"
  budget_type  = "COST"
  limit_amount = "5000"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "FORECASTED"
    subscriber_email_addresses = ["ops@example.com"]
  }
}
```

### 運用・監視設定

**CloudWatch Logs Insights クエリ**（コスト異常検知・レイテンシ分析）:

```
# 1時間あたりのBedrock トークン使用量（コスト異常検知）
fields @timestamp, @message
| filter @message like /bedrock/
| stats sum(input_tokens) as prompt_tokens,
        sum(output_tokens) as completion_tokens,
        sum(input_tokens * 0.003 / 1000 + output_tokens * 0.015 / 1000) as estimated_cost
  by bin(1h)
| sort @timestamp desc

# タスク評価レイテンシ P95/P99
fields @timestamp, task_id, duration_ms
| filter event = "task_eval_complete"
| stats percentile(duration_ms, 95) as p95,
        percentile(duration_ms, 99) as p99,
        avg(duration_ms) as avg_ms
  by bin(1h)
```

**CloudWatch アラーム設定**（Python）:

```python
import boto3

cloudwatch = boto3.client("cloudwatch", region_name="ap-northeast-1")

def create_bedrock_token_alarm() -> None:
    """Bedrockトークン使用量スパイク検知アラーム"""
    cloudwatch.put_metric_alarm(
        AlarmName="theagentcompany-bedrock-token-spike",
        MetricName="InputTokenCount",
        Namespace="AWS/Bedrock",
        Statistic="Sum",
        Period=3600,
        EvaluationPeriods=1,
        Threshold=500000,  # 1時間あたり50万トークン
        ComparisonOperator="GreaterThanThreshold",
        AlarmActions=["arn:aws:sns:ap-northeast-1:123456789012:ops-alerts"],
    )

def create_lambda_duration_alarm() -> None:
    """Lambda実行時間異常検知アラーム"""
    cloudwatch.put_metric_alarm(
        AlarmName="theagentcompany-lambda-duration",
        MetricName="Duration",
        Namespace="AWS/Lambda",
        Statistic="p99",
        Period=300,
        EvaluationPeriods=3,
        Threshold=600000,  # 10分（600秒）
        ComparisonOperator="GreaterThanThreshold",
        Dimensions=[
            {"Name": "FunctionName", "Value": "theagentcompany-eval"}
        ],
        AlarmActions=["arn:aws:sns:ap-northeast-1:123456789012:ops-alerts"],
    )
```

**X-Ray トレーシング設定**（Python）:

```python
from aws_xray_sdk.core import xray_recorder, patch_all

# boto3自動計装
patch_all()

xray_recorder.configure(
    sampling=True,
    context_missing="LOG_ERROR",
    daemon_address="127.0.0.1:2000",
)

@xray_recorder.capture("evaluate_task")
def evaluate_task_traced(task_id: str, agent_output: dict) -> dict:
    """X-Rayトレース付きタスク評価"""
    subsegment = xray_recorder.current_subsegment()
    subsegment.put_annotation("task_id", task_id)
    subsegment.put_metadata("agent_model", agent_output.get("model", "unknown"))
    subsegment.put_metadata("step_count", agent_output.get("step_count", 0))

    result = evaluate_task(task_id, agent_output)

    subsegment.put_annotation("score", result["partial_score"])
    return result
```

**Cost Explorer自動レポート**（Python）:

```python
import boto3
from datetime import datetime, timedelta

ce = boto3.client("ce", region_name="us-east-1")
sns = boto3.client("sns", region_name="ap-northeast-1")

def daily_cost_report() -> None:
    """日次コストレポート取得・SNS通知"""
    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    response = ce.get_cost_and_usage(
        TimePeriod={"Start": start, "End": end},
        Granularity="DAILY",
        Metrics=["BlendedCost"],
        Filter={
            "Tags": {
                "Key": "Project",
                "Values": ["theagentcompany"],
            }
        },
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )

    total = sum(
        float(g["Metrics"]["BlendedCost"]["Amount"])
        for r in response["ResultsByTime"]
        for g in r["Groups"]
    )

    if total > 100:
        sns.publish(
            TopicArn="arn:aws:sns:ap-northeast-1:123456789012:cost-alerts",
            Subject=f"TheAgentCompany日次コスト警告: ${total:.2f}",
            Message=f"日次コストが$100を超過しました: ${total:.2f}",
        )
```

### コスト最適化チェックリスト

**アーキテクチャ選択**:
- [ ] トラフィック量に応じた構成選定（~100 eval/日: Serverless、~1,000: Hybrid、10,000+: Container）
- [ ] 非同期評価が許容される場合はBedrock Batch APIを優先

**リソース最適化**:
- [ ] EC2: Spot Instances優先（t3.2xlarge Spot ~$0.10/hr）
- [ ] Reserved Instances: 1年コミットで最大72%削減
- [ ] Savings Plans: コンピューティング使用量に応じた柔軟な割引
- [ ] Lambda: メモリサイズ1024MBで実行時間とコストのバランス最適化
- [ ] ECS/EKS: Karpenter consolidation policyでアイドルノード自動回収

**LLMコスト削減**:
- [ ] Bedrock Batch API使用（非リアルタイム評価で50%削減）
- [ ] Prompt Caching有効化（NPC応答の共通プレフィックスで30-90%削減）
- [ ] モデル選択ロジック（簡易チェックポイントにはHaikuクラス、複雑評価にはSonnetを使い分け）
- [ ] トークン数制限（max_tokens設定で不要な長文生成を抑制）
- [ ] システムプロンプト最適化（共通部分の圧縮でプロンプトトークン削減）

**監視・アラート**:
- [ ] AWS Budgets: 月額予算上限設定（予測ベースで80%到達時に通知）
- [ ] CloudWatch アラーム: Bedrock トークン使用量・Lambda実行時間の閾値設定
- [ ] Cost Anomaly Detection: 自動異常検知の有効化
- [ ] 日次コストレポート: Cost Explorer APIによる自動集計・SNS通知

**リソース管理**:
- [ ] 未使用タスクイメージの定期削除（ECRライフサイクルポリシー）
- [ ] タグ戦略: `Project=theagentcompany`, `Env=production/staging` の必須化
- [ ] S3ライフサイクルポリシー: 評価成果物の90日後Glacier移行
- [ ] 開発環境の夜間自動停止（ECS desired_count=0）
- [ ] NAT Gateway不使用構成でネットワークコスト削減（VPCエンドポイント活用）

## 実験結果（Results）

著者らが報告した主要な実験結果を以下に示す（論文Table 1より）。

| モデル | 完全完了率 | 部分完了スコア | 平均ステップ数 | タスク平均コスト |
|--------|-----------|---------------|---------------|----------------|
| Gemini 2.5 Pro | 30.3% | 39.3% | 27.2 | $4.2 |
| Claude 3.5 Sonnet (new) | 26.3% | 36.4% | 27.8 | $4.1 |
| Claude 3.5 Sonnet (old) | 24.0% | 34.4% | 29.2 | $6.3 |
| Gemini 2.0 Flash | 11.4% | 19.0% | 39.9 | $0.6 |
| GPT-4o | 8.6% | 16.7% | 14.6 | $1.3 |
| Llama 3.1 405B | 7.4% | 14.1% | 23.0 | $3.2 |
| Llama 3.3 70B | 6.9% | 12.8% | 20.9 | $0.9 |

主な分析ポイントは以下の通りである。

- **カテゴリ別の傾向**: SDE（コーディング）タスクでは比較的高い成功率を示す一方、Finance・Admin・DSタスクではすべてのモデルで成功率が低い
- **コミュニケーションの困難さ**: RocketChatを介した同僚とのやり取りが必要なタスクは、全モデルで顕著にスコアが低下する
- **UIナビゲーションの限界**: ownCloudのWebベースオフィスUIの操作は、現状のエージェントにとって大きな障壁である
- **コストと性能のトレードオフ**: 高性能モデル（Gemini 2.5 Pro, Claude 3.5 Sonnet）はタスクあたり$4前後のコストがかかるが、軽量モデル（Gemini 2.0 Flash: $0.6）との性能差は大きい

## 実運用への応用（Practical Applications）

TheAgentCompanyの知見は、LLMエージェントの実運用における複数の側面で示唆を与える。

**エージェント評価パイプラインの構築**: TheAgentCompanyの環境構成とチェックポイントベース評価は、組織独自のLLMエージェントCI/CDパイプラインの設計指針となる。Zenn記事で解説したLangGraphマルチエージェントRAGの評価においても、部分完了スコアの考え方は協調品質の定量化に応用できる。

**タスク設計の指針**: 175タスクの難易度分析から、エージェントに委任すべきタスク（構造化されたコーディング作業）と人間が担うべきタスク（コミュニケーション集約型・UI操作集約型）の切り分けが可能となる。

**コスト効率の最適化**: タスクあたり$0.6-$6.3のコスト分析は、プロダクション環境でのエージェント運用コスト見積もりのベースラインとなる。モデル選択ロジック（簡易タスクには軽量モデル、高難度タスクにはフロンティアモデル）の設計にも活用できる。

**ベンチマーク駆動開発**: TheAgentCompanyはOSSとして公開されており、自組織のタスクを追加して独自ベンチマークを構築できる。Docker化されたタスク単位の設計により、新規タスクの追加が容易である。

## まとめと今後の展望

TheAgentCompanyは、LLMエージェントの実世界業務タスク遂行能力を評価する包括的ベンチマークである。最高性能エージェントでも30%程度の完了率にとどまる結果は、現状のLLMエージェントにはコーディング以外の業務（コミュニケーション、UI操作、管理業務）で大きな改善余地があることを示している。チェックポイントベースの部分評価と効率メトリクスの組み合わせは、エージェント開発の進捗を定量的に追跡する有用なフレームワークであり、LangGraphベースのマルチエージェント評価にも応用可能な設計思想を提供している。

## 参考文献

- **arXiv**: [https://arxiv.org/abs/2412.14161](https://arxiv.org/abs/2412.14161)
- **Code**: [https://github.com/TheAgentCompany/TheAgentCompany](https://github.com/TheAgentCompany/TheAgentCompany)
- **Project Page**: [https://the-agent-company.com/](https://the-agent-company.com/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/88cd951a1ec060](https://zenn.dev/0h_n0/articles/88cd951a1ec060)
