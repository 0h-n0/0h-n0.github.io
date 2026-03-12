---
layout: post
title: "Amazon Bedrock AgentCore Gateway 解説: AWS によるマネージド MCP Gateway の設計と機能"
description: "AWSが提供するフルマネージドMCP Gateway（AgentCore Gateway）のアーキテクチャと機能を解説"
categories: [blog, tech_blog]
tags: [mcp, aws, bedrock, agentcore, gateway, aiagent, security]
date: 2026-03-13 11:00:00 +0900
source_type: tech_blog
source_domain: aws.amazon.com
source_url: https://aws.amazon.com/blogs/machine-learning/introducing-amazon-bedrock-agentcore-gateway-transforming-enterprise-ai-agent-tool-development/
zenn_article: c55263b7af78bf
zenn_url: https://zenn.dev/0h_n0/articles/c55263b7af78bf
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

本記事は [AWS Machine Learning Blog: "Introducing Amazon Bedrock AgentCore Gateway: Transforming enterprise AI agent tool development"](https://aws.amazon.com/blogs/machine-learning/introducing-amazon-bedrock-agentcore-gateway-transforming-enterprise-ai-agent-tool-development/) および関連記事 [Streamline AI agent tool interactions: Connect API Gateway to AgentCore Gateway with MCP](https://aws.amazon.com/blogs/machine-learning/streamline-ai-agent-tool-interactions-connect-api-gateway-to-agentcore-gateway-with-mcp/) の解説記事です。

AWSは、MCP（Model Context Protocol）をネイティブサポートするフルマネージドサービス「Amazon Bedrock AgentCore Gateway」を提供しています。AWSの発表によると、このサービスはエージェントとツールの通信を抽象化し、ゼロコードでのMCPツール作成、インテリジェントなツールディスカバリ、組み込みの認証・認可機能、サーバーレスインフラを特徴としています。

この記事は [Zenn記事: MCP Gatewayで社内ツール統合エージェントを設計する実践パターン](https://zenn.dev/0h_n0/articles/c55263b7af78bf) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（AWS Machine Learning Blog）
- **URL**: [https://aws.amazon.com/blogs/machine-learning/introducing-amazon-bedrock-agentcore-gateway-transforming-enterprise-ai-agent-tool-development/](https://aws.amazon.com/blogs/machine-learning/introducing-amazon-bedrock-agentcore-gateway-transforming-enterprise-ai-agent-tool-development/)
- **組織**: Amazon Web Services (AWS)
- **発表日**: 2025年

## 技術的背景（Technical Background）

Zenn記事では、MCP Gatewayを自前で実装するアーキテクチャ（Pythonクラス `MCPGateway`）を解説しています。一方で、AWSはこのGatewayパターンをフルマネージドサービスとして提供しています。自前実装とマネージドサービスの比較は、エンタープライズ環境での設計判断において重要なポイントです。

MCPが2024年11月にAnthropicから公開されて以降、クラウドベンダー各社がMCP対応を進めています。AWSのMCP Proxy（GA: 2025年10月）に続き、AgentCore GatewayはMCPを中核に据えたエンタープライズ向けツール統合プラットフォームとして位置づけられています。

## 実装アーキテクチャ（Architecture）

### AgentCore Gateway の全体構成

AWSの発表に基づくと、AgentCore Gatewayは以下の機能を提供します。

```mermaid
graph TB
    subgraph "AIエージェント"
        AGENT1[Bedrock Agent]
        AGENT2[LangChain Agent]
        AGENT3[Custom Agent]
    end

    subgraph "AgentCore Gateway"
        MCP_EP[MCP エンドポイント]
        DISCOVERY[ツールディスカバリ]
        AUTH[認証・認可]
        TRANSFORM[API → MCP 変換]
    end

    subgraph "ツール提供元"
        APIGW[API Gateway]
        LAMBDA[Lambda Functions]
        EXISTING[既存 REST API]
    end

    AGENT1 --> MCP_EP
    AGENT2 --> MCP_EP
    AGENT3 --> MCP_EP
    MCP_EP --> DISCOVERY
    DISCOVERY --> AUTH
    AUTH --> TRANSFORM
    TRANSFORM --> APIGW
    TRANSFORM --> LAMBDA
    TRANSFORM --> EXISTING
```

### 主要機能

AWSの公式発表に基づく主要機能は以下の通りです。

#### 1. ゼロコード MCP ツール作成

既存のAPI GatewayエンドポイントやLambda関数を、コードを書かずにMCPツールとして公開できるとされています。OpenAPI仕様（Swagger）からのツール自動生成にも対応していると報告されています。

```python
# 従来の手動MCP実装（Zenn記事のパターン）
class MCPServerConfig:
    name: str
    url: str
    tools: list[str]
    allowed_roles: dict[str, list[str]]

# AgentCore Gateway の場合（概念的な比較）
# API Gateway の OpenAPI 仕様から自動的にMCPツールが生成される
# 開発者はAPI定義とIAMポリシーのみを設定
```

#### 2. インテリジェントツールディスカバリ

エージェントが利用可能なツールを自動検出する機能です。Zenn記事の`_tool_to_server`辞書による静的マッピングと比較すると、動的なツール検出がマネージドで提供される点が異なります。

#### 3. 組み込み認証・認可

AWSの発表によると、AgentCore Gatewayはインバウンドおよびアウトバウンドの認証を組み込みで提供します。

- **インバウンド認証**: MCP クライアント → Gateway（IAM / Cognito）
- **アウトバウンド認証**: Gateway → バックエンドAPI（IAM Role / Secrets Manager）

Zenn記事の`AuthMiddleware`クラスが提供するOAuth + RBAC機能に相当しますが、IAMポリシーベースの権限管理はAWS環境での標準的なパターンです。

#### 4. サーバーレスインフラ

インフラの管理が不要で、トラフィックに応じた自動スケーリングが提供されるとされています。

### API Gateway との連携パターン

関連ブログ記事では、既存のAPI GatewayをAgentCore Gatewayに接続するパターンが報告されています。

```mermaid
graph LR
    AGENT[AI Agent] --> |MCP| ACG[AgentCore Gateway]
    ACG --> |HTTP| APIGW[API Gateway]
    APIGW --> |invoke| LAMBDA[Lambda]
    APIGW --> |proxy| ECS[ECS Service]
    APIGW --> |proxy| EC2[EC2 API]
```

このパターンにより、既存のAPI Gateway経由のAPIをMCPツールとして再利用でき、Microsoftのブログで報告されているマイグレーションコスト（REST → MCP追加: 1-3週間）をさらに短縮できる可能性があります。

## 自前実装 vs マネージドサービスの比較

Zenn記事のMCP Gateway実装とAgentCore Gatewayの比較を、ブログの情報に基づいて整理します。

| 比較項目 | 自前実装（Zenn記事） | AgentCore Gateway |
|---------|-------------------|-------------------|
| **初期開発コスト** | 数日〜数週間 | 設定のみ（コード不要） |
| **スケーリング** | 自前管理（ECS/EKS） | 自動（サーバーレス） |
| **認証** | OAuth 2.0 + JWT（手動実装） | IAM/Cognito（組み込み） |
| **ツール追加** | コード変更必要 | OpenAPI仕様から自動生成 |
| **レイテンシ制御** | 完全に制御可能 | AWS側に依存 |
| **カスタマイズ性** | 無制限 | AWSサービスの範囲内 |
| **ベンダーロックイン** | なし | AWS依存 |
| **運用コスト** | インフラ管理が必要 | マネージド（AWSが管理） |

### 選択基準

**AgentCore Gatewayが適する場合**:
- AWSエコシステムに既に深く組み込まれている
- 運用チームが小規模でインフラ管理を最小化したい
- 既存のAPI GatewayやLambda関数をMCPツールとして公開したい
- IAMベースの権限管理で要件を満たせる

**自前実装が適する場合**:
- マルチクラウドまたはオンプレミス環境
- レイテンシの完全な制御が必要（Bifrost等の高速Gatewayを使用）
- カスタム認証フロー（OAuth 2.1 + PKCE等）が必要
- ベンダーロックインを避けたい

## セキュリティアーキテクチャ

### IAM ベースの認証モデル

AWSの発表に基づくと、AgentCore GatewayはIAMポリシーでツールレベルのアクセス制御を実現します。

```python
# IAM ポリシー例（概念的な構造）
# Zenn記事のRBACと同等の機能をIAMで実現

iam_policy_viewer = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": "agentcore:InvokeTool",
        "Resource": [
            "arn:aws:agentcore:ap-northeast-1:123456789:tool/slack-search",
            "arn:aws:agentcore:ap-northeast-1:123456789:tool/jira-get-issue",
        ],
    }],
}

iam_policy_admin = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": "agentcore:InvokeTool",
        "Resource": "arn:aws:agentcore:ap-northeast-1:123456789:tool/*",
    }],
}
```

Zenn記事の`TOOL_PERMISSIONS`辞書によるRBACと比較すると、IAMポリシーは以下の利点があります。

- **AWS全体との統合**: CloudTrail監査ログ、Organizations SCP、SSO連携
- **条件ベースのアクセス制御**: IPアドレス、時間帯、MFA要件等の条件付きポリシー
- **クロスアカウントアクセス**: AWS Organizations内の複数アカウントからの利用

### Secrets Manager による認証情報管理

バックエンドAPIへの認証情報はSecrets Managerで管理され、AgentCore GatewayのIAMロールのみがアクセスできます。Zenn記事の`AuthMiddleware`ではJWTトークンをコード内で処理していますが、Secrets Managerを使用することで認証情報のローテーションと暗号化が自動化されます。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

AgentCore Gatewayを中心としたAWS構成のコスト試算です。

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | AgentCore + Lambda | $30-100 | AgentCore Gateway + Lambda + Secrets Manager |
| **Medium** | ~30,000 (1,000/日) | AgentCore + API GW | $200-600 | AgentCore Gateway + API Gateway + Lambda + CloudWatch |
| **Large** | 300,000+ (10,000/日) | AgentCore + ECS | $1,500-4,000 | AgentCore Gateway + API Gateway + ECS Fargate + ElastiCache |

**Small構成の詳細** (月額$30-100):
- **AgentCore Gateway**: サーバーレス、リクエスト課金（$10-30/月）
- **Lambda**: 1GB RAM, 30秒タイムアウト ($15/月)
- **Secrets Manager**: API認証情報管理 ($5/月)
- **CloudWatch**: 基本監視 ($5/月)

**コスト試算の注意事項**:
- 上記は2026年3月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です
- AgentCore Gatewayの料金はプレビュー/GA時期により変動する可能性があります
- 実際のコストはトラフィックパターン、リージョン、バースト使用量により変動します
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

**AgentCore Gateway + Lambda構成**:

```hcl
# --- IAMロール（AgentCore Gateway用） ---
resource "aws_iam_role" "agentcore_gateway" {
  name = "agentcore-gateway-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "agentcore.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "agentcore_invoke_lambda" {
  role = aws_iam_role.agentcore_gateway.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "lambda:InvokeFunction"
      ]
      Resource = aws_lambda_function.mcp_tools.arn
    }]
  })
}

# --- Lambda関数（MCPツール実装） ---
resource "aws_lambda_function" "mcp_tools" {
  filename      = "mcp_tools.zip"
  function_name = "mcp-gateway-tools"
  role          = aws_iam_role.lambda_mcp.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 30
  memory_size   = 512

  environment {
    variables = {
      SECRETS_ARN = aws_secretsmanager_secret.api_credentials.arn
    }
  }
}

resource "aws_iam_role" "lambda_mcp" {
  name = "lambda-mcp-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

# --- Secrets Manager（バックエンドAPI認証情報） ---
resource "aws_secretsmanager_secret" "api_credentials" {
  name = "mcp-gateway-api-credentials"
}

resource "aws_secretsmanager_secret_version" "api_credentials" {
  secret_id = aws_secretsmanager_secret.api_credentials.id
  secret_string = jsonencode({
    slack_token = "xoxb-placeholder"  # 実際の値はAWS Consoleで設定
    jira_token  = "placeholder"
  })
}

# --- CloudWatch アラーム（コスト監視） ---
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "mcp-tools-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "MCPツールLambdaのエラー率異常"

  dimensions = {
    FunctionName = aws_lambda_function.mcp_tools.function_name
  }
}

# --- AWS Budgets（月額予算アラート） ---
resource "aws_budgets_budget" "mcp_gateway" {
  name         = "mcp-gateway-monthly"
  budget_type  = "COST"
  limit_amount = "200"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator = "GREATER_THAN"
    threshold           = 80
    threshold_type      = "PERCENTAGE"
    notification_type   = "ACTUAL"
    subscriber_email_addresses = ["ops@example.com"]
  }
}
```

### 運用・監視設定

**CloudWatch Logs Insights クエリ**:

```sql
-- AgentCore Gateway のツール呼び出しレイテンシ分析
fields @timestamp, tool_name, duration_ms, status_code
| stats avg(duration_ms) as avg_latency,
        pct(duration_ms, 95) as p95_latency,
        pct(duration_ms, 99) as p99_latency
  by bin(5m)
| filter status_code >= 400

-- エージェントごとのツール利用パターン
fields @timestamp, agent_id, tool_name
| stats count() as call_count by agent_id, tool_name
| sort call_count desc
| limit 20
```

### コスト最適化チェックリスト

**アーキテクチャ選択**:
- [ ] ~100 req/日 → AgentCore + Lambda (Serverless) - $30-100/月
- [ ] ~1000 req/日 → AgentCore + API Gateway (Hybrid) - $200-600/月
- [ ] 10000+ req/日 → AgentCore + ECS Fargate (Container) - $1,500-4,000/月

**リソース最適化**:
- [ ] Lambda: メモリサイズを CloudWatch Insights で分析・最適化
- [ ] API Gateway: キャッシュ有効化（繰り返しのツール呼び出し削減）
- [ ] ECS Fargate: Spot タスク活用（最大70%削減）
- [ ] Secrets Manager: 不要なシークレットの定期的な棚卸し
- [ ] CloudWatch: ログ保持期間の適切な設定（不要な長期保存を削減）

**LLMコスト削減**:
- [ ] Bedrock Batch API: 50%割引（非リアルタイム処理）
- [ ] Prompt Caching: 30-90%削減（Bedrock Prompt Caching有効化）
- [ ] モデル選択: タスク複雑度に応じたモデル自動選択
- [ ] トークン数制限: max_tokens 設定で過剰生成防止

**監視・アラート**:
- [ ] AWS Budgets: 月額予算設定（80%で警告）
- [ ] CloudWatch: レイテンシ・エラー率アラーム
- [ ] Cost Anomaly Detection: 自動異常検知
- [ ] CloudTrail: 全ツール呼び出しの監査ログ

**リソース管理**:
- [ ] 未使用ツール定義の定期的な棚卸し
- [ ] タグ戦略: 環境別（dev/staging/prod）でコスト可視化
- [ ] Lambda バージョン: 古いバージョンの自動削除

## パフォーマンス最適化（Performance）

### マネージドサービスのレイテンシ特性

AgentCore Gatewayはサーバーレスで提供されるため、コールドスタートの影響を受ける可能性があります。Microsoftのブログで報告されているMCPのコールドスタート（200-500ms）と同様の傾向が予想されますが、AWSの具体的なベンチマーク値は公開時点では確認できていません。

Zenn記事で紹介されている高速Gateway（Bifrost: オーバーヘッド3ms未満）と比較すると、マネージドサービスはレイテンシ面で不利になる可能性がありますが、運用コストとスケーラビリティのトレードオフとして判断する必要があります。

### API Gateway 統合時の最適化

AgentCore Gatewayと既存API Gatewayを連携する場合の最適化ポイント：

1. **API Gateway キャッシュ**: TTLベースのレスポンスキャッシュで重複呼び出しを削減
2. **Lambda Provisioned Concurrency**: コールドスタート回避（追加コスト発生）
3. **VPC Endpoint**: AgentCore Gateway → Lambda 間のネットワークレイテンシ削減
4. **リージョン選択**: エージェントと同一リージョンに配置

## 運用での学び（Production Lessons）

### MCP Proxy との違い

AWSは2025年10月にMCP Proxyを一般提供開始しています。AgentCore Gatewayとの違いは以下の通りです。

| 比較項目 | MCP Proxy | AgentCore Gateway |
|---------|-----------|-------------------|
| **用途** | クライアント側プロキシ | サーバー側Gateway |
| **認証** | AWS SigV4 | IAM/Cognito + Secrets Manager |
| **ツール管理** | なし（透過的プロキシ） | ツールディスカバリ・管理 |
| **API変換** | なし | REST/Lambda → MCP 自動変換 |
| **デプロイ** | クライアント側にインストール | AWSマネージド |

### Interceptor パターン

AWS AgentCore Gatewayでは、ツール呼び出しの前後にカスタムロジック（Interceptor）を挿入できるとされています。これはZenn記事のセキュリティチェック（プロンプトインジェクション対策、RBAC）をGatewayレベルで実装するパターンに相当します。

```python
# Interceptor パターンの概念実装
from typing import Any, Callable, Awaitable

ToolInterceptor = Callable[
    [str, dict[str, Any], dict[str, str]],
    Awaitable[dict[str, Any] | None]
]

async def security_interceptor(
    tool_name: str,
    args: dict[str, Any],
    context: dict[str, str],
) -> dict[str, Any] | None:
    """セキュリティチェック Interceptor

    Args:
        tool_name: 呼び出し対象ツール名
        args: ツール引数
        context: リクエストコンテキスト（user_id, role等）

    Returns:
        None（パススルー）またはエラーレスポンス
    """
    # プロンプトインジェクション検出
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 1000:
            # 異常に長い引数は検査対象
            if _contains_injection_pattern(value):
                return {
                    "error": "SecurityViolation",
                    "message": f"Suspicious input in '{key}'",
                }
    return None  # パススルー


def _contains_injection_pattern(text: str) -> bool:
    """プロンプトインジェクションパターンを検出する"""
    import re
    patterns = [
        r"(?i)ignore\s+previous",
        r"(?i)system\s*:\s*you\s+are",
        r"(?i)admin\s+override",
    ]
    return any(re.search(p, text) for p in patterns)
```

## 学術研究との関連（Academic Connection）

AgentCore Gatewayのツールディスカバリ機能は、以下の学術研究と関連しています。

- **ToolNet**（Li et al., 2025, arXiv:2502.11157）: グラフベースのツール管理。AgentCore Gatewayのツールディスカバリが静的カタログか動的探索かは公式情報では明確でないが、大規模環境ではToolNetのような構造化管理が有効
- **EASYTOOL**（Xu et al., 2024, arXiv:2405.00253）: ツール説明の簡潔化。AgentCore Gatewayの自動ツール生成において、適切な粒度のツール説明が生成されるかが品質を左右する
- **AgentTool**（2025, arXiv:2503.16557）: 自律的ツール統合。AgentCore Gatewayの「ゼロコードMCPツール作成」は、学術的には自動ツール統合の実用化に位置づけられる

## まとめと実践への示唆

AWS AgentCore Gatewayは、Zenn記事で解説されたMCP Gateway パターンをマネージドサービスとして提供するものであり、AWS環境に深く統合されたエンタープライズ向けソリューションです。

選択指針をまとめると：

1. **AWS中心のエンタープライズ環境**: AgentCore Gateway が最も低コスト・低運用負荷
2. **マルチクラウド・レイテンシ重視**: 自前実装（Bifrost等の高速Gateway）が適切
3. **段階的導入**: 既存API Gateway → AgentCore Gateway連携で段階的にMCP対応

ただし、AgentCore Gatewayの具体的なレイテンシベンチマークやスケーリング特性は公式ドキュメントで継続的に確認することを推奨します。マネージドサービスの特性として、内部実装の変更によりパフォーマンス特性が変わる可能性があります。

## 参考文献

- **Blog URL**: [Introducing Amazon Bedrock AgentCore Gateway](https://aws.amazon.com/blogs/machine-learning/introducing-amazon-bedrock-agentcore-gateway-transforming-enterprise-ai-agent-tool-development/)
- **Related Blog**: [Connect API Gateway to AgentCore Gateway with MCP](https://aws.amazon.com/blogs/machine-learning/streamline-ai-agent-tool-interactions-connect-api-gateway-to-agentcore-gateway-with-mcp/)
- **AWS MCP Proxy GA**: [Model Context Protocol Proxy for AWS](https://aws.amazon.com/about-aws/whats-new/2025/10/model-context-protocol-proxy-available/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/c55263b7af78bf](https://zenn.dev/0h_n0/articles/c55263b7af78bf)

---

:::message
この記事はAI（Claude Code）により自動生成されました。AWS サービスの料金・機能は変更される可能性があるため、最新情報は [AWS公式ドキュメント](https://docs.aws.amazon.com/) もご確認ください。
:::
