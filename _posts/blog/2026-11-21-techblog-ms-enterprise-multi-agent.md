---
layout: post
title: "エンタープライズマルチエージェントシステムの本番設計とセキュリティ：Microsoftが示す7つの原則"
description: "Microsoft Foundryチームが公開したエンタープライズ向けマルチエージェントシステムの本番運用ガイドを解説。所有権・ID連鎖・状態契約・復旧・ネットワーク・ナレッジ認可・制御マトリクスの7原則を整理する"
categories: [blog, tech_blog]
tags: [multi-agent, production, security, microsoft, autogen, agent-framework, orchestration]
date: 2026-11-21 09:00:00 +0900
source_type: tech_blog
source_domain: "techcommunity.microsoft.com"
source_url: "https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/global-enterprise-multi-agent-systems-in-production-design-and-security-part-1/4564913"
zenn_article: c5509c723c2670
zenn_url: https://zenn.dev/0h_n0/articles/c5509c723c2670
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

本記事は [Global enterprise multi-agent systems in production: Design and security (Part 1)](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/global-enterprise-multi-agent-systems-in-production-design-and-security-part-1/4564913) の解説記事です。

Microsoft Foundryチームの dhangerkapil 氏が2026年10月に公開したこのブログ記事は、エンタープライズ環境でマルチエージェントシステムを本番運用する際の設計原則とセキュリティ要件を7つの教訓（Lesson）として体系化したものである。Microsoft Agent Framework、LangGraph、またはカスタムPython/C#コードで構築されたエージェントをMicrosoft Foundryにホスティングする際の実践的なガイドラインを示している。

この記事は [Zenn記事: AutoGenからAgent Frameworkへ：自律エージェント移行実践ガイド](https://zenn.dev/0h_n0/articles/c5509c723c2670) の深掘りです。Zenn記事ではAutoGenからAgent Frameworkへの移行パターンと本番運用の設計パターンを概観しているが、本記事ではMicrosoftが具体的にどのような本番運用原則を推奨しているかを詳細に解説する。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://techcommunity.microsoft.com/.../4564913](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/global-enterprise-multi-agent-systems-in-production-design-and-security-part-1/4564913)
- **組織**: Microsoft Foundry チーム
- **著者**: dhangerkapil
- **発表日**: 2026年10月26日
- **シリーズ**: Part 1（設計とセキュリティ）/ Part 2（リリースと運用）

## 技術的背景（Technical Background）

### なぜこのガイドが必要なのか

2026年はマルチエージェントシステムの本番導入が急速に進んだ年である。しかし、プロトタイプから本番環境への移行では、単一エージェントとは質的に異なる課題が発生する。複数のエージェント間でのハンドオフ、分散状態の管理、障害からの復旧、セキュリティ境界の設計など、従来のマイクロサービスアーキテクチャとも異なる独自の設計判断が求められる。

ブログ著者は、マネージドホスティング（Microsoft Foundry）を利用しても、アプリケーション認可・ビジネス効果・データハンドリング・エビデンスの責任はチームに残ると明記している。つまり、プラットフォームが提供するサンドボックスやID管理は必要条件であって十分条件ではない。

### 学術研究との関連

本ブログの内容は、マルチエージェントシステムのオーケストレーションパターン研究（例: NeurIPS 2025 "Multi-Agent Collaboration via Evolving Orchestration"）やエージェントのツール使用に関するサーベイ（arXiv:2603.22862）で議論されている設計上のトレードオフを、エンタープライズの本番環境に落とし込んだものと位置付けられる。学術研究が性能やタスク達成率に焦点を当てるのに対し、本ブログはセキュリティ・状態管理・復旧可能性・コスト制御という本番固有の要件を体系化している。

## 7つの設計原則の詳細解説

### Lesson 1: 所有権と最小限のアーキテクチャ

著者は「最もシンプルな設計を選べ」と強調している。具体的には、以下の段階的なアプローチを推奨している。

1. **直接的なモデル呼び出し**（最もシンプル）
2. **ツール付き単一エージェント**
3. **マルチエージェント**（最も複雑）

各ステップアップは、ハンドオフ・API呼び出し・分散状態・障害モードを追加する。ブログでは「マルチエージェントをデフォルトにすることは推奨しない」と明記されている。

オーケストレーションパターンは以下の4つに分類されている。

| パターン | 適用場面 | トレードオフ |
|---------|---------|------------|
| **Sequential** | 順序依存の前提条件がある場合 | レイテンシ蓄積、初期エラーの伝搬 |
| **Concurrent** | 独立したサブタスクの場合 | キャパシティのスパイク、結果の競合時に集約が必要 |
| **Handoff** | 対話中に専門家が必要になった場合 | 明示的な所有権移転、ループ防止、エスカレーション |
| **Discussion/Iterative** | 批評が品質を改善する場合 | 完了時間・トークン・コストの予測困難 |

重要な設計判断として、**モデルにはリクエスト解釈と計画提案を任せ、データアクセス・ジョブ認可・承認・ビジネストランジションは決定論的なコードで実行する**というパターンが推奨されている。

また、**単一のエンドツーエンド予算**（デッドライン、モデル呼び出し回数、トークンまたはコスト、ツール呼び出し回数、リトライ回数、ハンドオフ回数、ファンアウト、並行数）をシステム全体で共有し、ネストされたエージェントはこの予算から引き落とす設計を推奨している。個々のエージェントに独立した予算を与えると、全体のコスト制御が困難になる。

### Lesson 2: IDチェーンとしてのアイデンティティ

著者は、マルチエージェントシステムでは5つのアイデンティティを明示的にマッピングする必要があると述べている。

1. **呼び出し元→アプリ**: ユーザーのトークン
2. **デプロイメントプリンシパル**: イメージのpush/pull
3. **ランタイムの依存アクセス**: サービス間通信
4. **エンドユーザーの委任**: OBO（On-Behalf-Of）フロー
5. **各ツールサービスの独自アクセス制御**

具体的な認証フローとして、バックエンドがユーザートークンを検証→OBOフローでダウンストリームトークン取得→Foundryにワークロードトークン + `x-client-*`ヘッダーで送信→エージェントは委任トークンのみ使用、というパターンが推奨されている。

著者は以下のアンチパターンを警告している。
- ユーザーIDを委任トークンの代替として使用する
- スプーファブルなクライアントヘッダーを信頼する
- ベアラートークンを遅延処理のために保存する
- アプリ専用アクセスへのサイレントフォールバック
- プラットフォーム分離をアプリレベルのマルチテナンシーと混同する

### Lesson 3: 状態契約（State Contracts）

マルチエージェントシステムの状態は、プロセスメモリ・セッションファイル・会話履歴・フレームワークチェックポイント・承認/外部ジョブ・ビジネスレコード・メタデータ/キャッシュの7種類に分類される。それぞれ異なる所有者（ランタイム、アプリケーション、ビジネスロジック等）を持つ。

著者は「永続化（persistence）とハイドレーション（hydration: 次のターンでのロード）は別々にテストせよ」と強調している。ある実装では、会話履歴の保存は正しく動作していたが、ワークフローアダプタが次のターンでそれをロードしなかったという事例が紹介されている。

ライフサイクルでは、アイドルタイムアウト・セッション有効期限・履歴保持期間・削除ポリシー・バックアップ・派生チャンクの有効期限をそれぞれ個別に定義する必要がある。

### Lesson 4: 復旧・承認・キャンセルのプロトコル

著者は、バックグラウンド実行、チェックポイント、外部イベント継続、クラッシュリカバリは**すべて別々の機能であり、個別にオプトインが必要**と述べている。

**ビジネス効果の復旧シーケンス**として、以下の4ステップが推奨されている。

1. 認可済みインテントと安定したオペレーションIDを永続化
2. べき等なダウンストリームアクション（`idempotency_key`にオペレーションIDを使用）
3. 権威あるレシートと結果の状態を記録
4. 不確実性を解消し、リトライ前に権限を再確認

**承認（Approval）** については、以下の要素すべてに紐付ける必要がある。
- 正確なアクション
- 引数
- リソース
- 承認者
- バージョン
- 有効期限

変更された作業は再承認が必要であり、拒否または期限切れの作業はその効果の前に停止する必要がある。承認メタデータだけではMCPの`tools/call`をブロックできないため、ランタイムでの強制が必要である。

**キャンセル**は、UI→サーバーワークフロー→ツール→リモートジョブの各レベルで設計する。それぞれのレベルで「これ以上の効果を防止」「終端状態を報告」「部分的な履歴を保持するか決定」を定義する。

### Lesson 5: ネットワークとキャパシティ

著者は、イメージビルド・プッシュ、プラットフォームプル、クライアントイングレス、ランタイムエグレス、モデル/ツール/検索アクセス、IDエンドポイント、テレメトリ、復旧のすべてのパスをマッピングし、各パスのDNS・ルート・プライベートエンドポイント・TLS・認可を検証することを推奨している。

特に重要なのは、クォータが独立している点である。セッションクォータ、モデルのトークン/RPM、ツールスループット、ストレージ、並行数は別々に管理され、一つのリソースの余裕が他の制約を緩和することはない。

### Lesson 6: ナレッジ取込みと認可

検索（Retrieval）と情報開示（Disclosure）を分離するのが重要な設計判断である。著者は3つの権限を明確に分離すべきと述べている。

1. **検索（Retrieve）する権限**: 情報にアクセスできる
2. **推論（Reason）に使用する権限**: 情報を処理に使える
3. **開示（Disclose）する権限**: 情報をユーザーに返せる

「推論には使えるが開示はできない」というケースは、検索権限だけでは表現できない。機密ラベルメタデータの伝搬と下流でのポリシー適用が必要になる。

取得したドキュメント（社内文書を含む）は**信頼されないデータとして扱う**。プロベナンス（出処）と分類を保持し、サポートされている場合はSpotlightingで低信頼コンテンツをマーキングする。

### Lesson 7: 制御カバレッジマトリクスと敵対的テスト

すべての信頼境界について、以下を記録する制御マトリクスを構築する。

| 境界 | 義務 |
|-----|------|
| **入力** | 認証、検証、必須のコンテンツ制御を適用 |
| **検索/ツール出力** | 認可とプロベナンスを保持、間接的インジェクションに対処 |
| **ハンドオフ** | スキーマ・分類・所有権・残予算を検証 |
| **ツール呼び出しと引数** | 権限・リソース・ビジネス制限・承認を強制 |
| **生成コード** | 実行・ファイルシステム・ネットワーク・リソース・認証情報を個別に制約 |
| **出力** | 開示とコンテンツをチェック、ストリーム終端エラーを処理 |
| **ネットワーク/運用** | エグレス制御、検出とレスポンスのエビデンス |

著者は、このマトリクスの各行に対してレッドチーミングを実施することを推奨している。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

ブログで紹介されたMicrosoft FoundryのアーキテクチャパターンをAWSで実現する構成を示す。

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $50-150 | Lambda + Bedrock + DynamoDB |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $300-800 | Lambda + ECS Fargate + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,000-5,000 | EKS + Karpenter + EC2 Spot |

**Small構成の詳細**（月額$50-150）:
- **Lambda**: 1GB RAM, 60秒タイムアウト（$20/月）
- **Bedrock**: Claude 3.5 Haiku, Prompt Caching有効（$80/月）
- **DynamoDB**: On-Demand、状態管理・チェックポイント保存（$10/月）
- **CloudWatch**: 基本監視（$5/月）
- **API Gateway**: REST API + IAM認証（$5/月）

**Medium構成の詳細**（月額$300-800）:
- **Lambda**: イベント処理・オーケストレーション（$50/月）
- **ECS Fargate**: 0.5 vCPU, 1GB RAM × 2タスク（$120/月）
- **Bedrock**: Claude 3.5 Sonnet, Batch API活用（$400/月）
- **ElastiCache Redis**: cache.t3.micro、セッション状態管理（$15/月）
- **Application Load Balancer**（$20/月）

**Large構成の詳細**（月額$2,000-5,000）:
- **EKS**: コントロールプレーン（$72/月）
- **EC2 Spot Instances**: g5.xlarge × 2-4台（平均$800/月）
- **Karpenter**: 自動スケーリング（追加コストなし）
- **Bedrock Batch**: 50%割引活用（$2,000/月）
- **S3**: プロンプトキャッシュ、チェックポイントストレージ（$20/月）
- **CloudWatch + X-Ray**: 詳細監視（$100/月）

**コスト削減テクニック**:
- Spot Instances使用で最大90%削減（EKS + Karpenter）
- Reserved Instances購入で最大72%削減（1年コミット）
- Bedrock Batch API使用で50%削減
- Prompt Caching有効化で30-90%削減

**コスト試算の注意事項**: 上記は2026年11月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です。実際のコストはトラフィックパターン、リージョン、バースト使用量により変動します。最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください。

### Terraformインフラコード

**Small構成（Serverless）: Lambda + Bedrock + DynamoDB**

```hcl
# --- VPC基盤 ---
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "multi-agent-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["ap-northeast-1a", "ap-northeast-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]

  enable_nat_gateway   = false
  enable_dns_hostnames = true
}

# --- IAMロール（最小権限） ---
resource "aws_iam_role" "agent_orchestrator" {
  name = "agent-orchestrator-role"

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
  role = aws_iam_role.agent_orchestrator.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
      Resource = "arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.claude-3-5-haiku*"
    }]
  })
}

# --- Lambda関数（オーケストレーター） ---
resource "aws_lambda_function" "orchestrator" {
  filename      = "orchestrator.zip"
  function_name = "agent-orchestrator"
  role          = aws_iam_role.agent_orchestrator.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 60
  memory_size   = 1024

  environment {
    variables = {
      BEDROCK_MODEL_ID    = "anthropic.claude-3-5-haiku-20241022-v1:0"
      STATE_TABLE         = aws_dynamodb_table.agent_state.name
      CHECKPOINT_TABLE    = aws_dynamodb_table.checkpoints.name
      ENABLE_PROMPT_CACHE = "true"
    }
  }
}

# --- DynamoDB（状態管理・チェックポイント） ---
resource "aws_dynamodb_table" "agent_state" {
  name         = "agent-session-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "session_id"
  range_key    = "state_type"

  attribute {
    name = "session_id"
    type = "S"
  }
  attribute {
    name = "state_type"
    type = "S"
  }

  ttl {
    attribute_name = "expire_at"
    enabled        = true
  }
}

resource "aws_dynamodb_table" "checkpoints" {
  name         = "agent-checkpoints"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "operation_id"

  attribute {
    name = "operation_id"
    type = "S"
  }

  ttl {
    attribute_name = "expire_at"
    enabled        = true
  }
}

# --- CloudWatch アラーム（コスト監視） ---
resource "aws_cloudwatch_metric_alarm" "orchestrator_cost" {
  alarm_name          = "agent-orchestrator-cost-spike"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = 3600
  statistic           = "Sum"
  threshold           = 100000
  alarm_description   = "オーケストレーター実行時間異常（コスト急増の可能性）"

  dimensions = {
    FunctionName = aws_lambda_function.orchestrator.function_name
  }
}
```

Large構成（EKS + Karpenter + Spot）は、Karpenter ProvisionerでSpot Instances（m5.xlarge/c5.xlarge）を優先し、`ttlSecondsAfterEmpty: 30`でアイドル時自動縮退する。Secrets ManagerでBedrock設定を管理し、AWS Budgetsで月額予算アラート（80%閾値）を設定する。

### セキュリティベストプラクティス

ブログのLesson 2（IDチェーン）とLesson 7（制御マトリクス）に対応するAWSでの推奨設定:

1. **ネットワークセキュリティ**: EKSは`cluster_endpoint_public_access = false`でVPN経由、Lambda はVPC内配置
2. **認証・認可**: IAMロール最小権限、IRSA（IAM Roles for Service Accounts）、MFA必須
3. **シークレット管理**: Secrets Manager使用、90日ごとローテーション、Terraform State はS3 + KMS暗号化
4. **監査**: CloudTrail全リージョン有効化、Config、GuardDuty
5. **データ保護**: TLS 1.2以上、S3/DynamoDB/EBS全てKMS暗号化

### 運用・監視設定

**CloudWatch Logs Insights クエリ**:

```sql
-- エージェント予算消費の監視
fields @timestamp, agent_name, tokens_used, budget_remaining
| stats sum(tokens_used) as total_tokens by agent_name, bin(1h)
| filter total_tokens > 50000

-- ハンドオフ失敗の検出
fields @timestamp, source_agent, target_agent, handoff_status
| filter handoff_status = "FAILED"
| stats count() as failures by source_agent, target_agent, bin(5m)
```

**CloudWatch アラーム**: `Custom/MultiAgent`名前空間で`TokenUsage`メトリクスを監視し、1時間あたり50万トークン超過でSNSアラートを送信する。エージェント予算消費とハンドオフ失敗を主要監視対象とする。

**X-Ray トレーシング**: `aws_xray_sdk`の`patch_all()`でboto3を自動計装し、`@xray_recorder.capture`でオーケストレーションパターン・予算・タスク長をアノテーション/メタデータとして記録する。

### コスト最適化チェックリスト

**アーキテクチャ選択**:
- [ ] ~100 req/日 → Lambda + Bedrock（Serverless）- $50-150/月
- [ ] ~1000 req/日 → ECS Fargate + Bedrock（Hybrid）- $300-800/月
- [ ] 10000+ req/日 → EKS + Spot Instances（Container）- $2,000-5,000/月

**リソース最適化**: Spot Instances優先（Karpenter管理）、Reserved Instances 1年コミット72%削減、Lambda メモリ最適化、夜間スケールダウン

**LLMコスト削減**: Bedrock Batch API 50%割引、Prompt Caching 30-90%削減、モデル選択ロジック（Haiku $0.25/MTok vs Sonnet $3/MTok）、ブログLesson 1の単一予算パターン実装

**監視・アラート**: AWS Budgets 80%警告、CloudWatch エージェント予算/ハンドオフ監視、Cost Anomaly Detection、日次レポート

**リソース管理**: Trusted Advisor活用、環境別タグ戦略、チェックポイント自動削除、開発環境夜間停止

## 運用での学び（Production Lessons）

### リリース前の5つのチェック

ブログでは、本番リリース前に以下の5つのテストを実施することを推奨している。これらは非本番環境で、異なる権限を持つ2人のテストユーザー、異なる開示権限を持つ情報ソース、分離されたエンドポイントを使って実行する。

| チェック項目 | 期待される動作 |
|-------------|-------------|
| スコープ内リクエスト | 権限のあるユーザーに根拠のある回答、内部知識はそのまま再現せず |
| スコープ外・拒否・失効 | スコープ外は拒否、非権限ユーザーは拒否、失効後は制限 |
| 抽出攻撃 | 直接・間接・マルチターンでの内部知識の一括取得試行を拒否 |
| ジョブコミット後にACK消失 | 不確実性を報告、オペレーションキーで照合後にリトライ |
| 作業中のキャンセル | UI→API→SDK→実行→ジョブへ停止が伝搬 |

### 障害対応パターン

ブログが特に強調しているのは、**チェックポイントは外部効果を原子的にコミットしない**という点である。チェックポイントを進捗マーカーの前に書き込む必要があり、効果の前にマーキングすると作業が失われ、効果の後にダウンストリームの重複排除なしにマーキングすると作業が重複する。

## 学術研究との関連（Academic Connection）

本ブログの内容は、以下の学術研究と密接に関連している。

- **MAFBench（arXiv:2602.03128）**: アーキテクチャ設計がパフォーマンスを決定するという主張はLesson 1と整合
- **Evolving Orchestration（NeurIPS 2025）**: 動的オーケストレーションパターンはブログのパターン表と対応
- **AWS Agentic AI Lens**: リトライ・予算管理はAGENTREL07推奨パターンとほぼ一致

ただし、学術研究がタスク達成率や効率性を主要メトリクスとするのに対し、本ブログは**セキュリティ・コンプライアンス・監査可能性**を本番環境の主要関心事として位置付けている点が大きく異なる。

## まとめと実践への示唆

Microsoftのこのブログ記事は、マルチエージェントシステムを本番環境に投入する際に避けがたい7つの設計判断を体系化した実践ガイドである。特に重要な教訓は以下の3点に集約される。

1. **「最もシンプルな設計を選べ」**: マルチエージェントはデフォルトの選択肢ではなく、より単純な設計で要件を満たせないと確認した後の選択肢である
2. **状態は「永続ストレージ」として一括りにできない**: プロセスメモリ、セッション、会話履歴、チェックポイント、ビジネスレコードはそれぞれ異なるライフサイクルと所有権を持つ
3. **セキュリティは事後付けではなく設計時に組み込む**: IDチェーン、検索/推論/開示の権限分離、制御マトリクスの各行へのレッドチーミングが必要

本ブログはPart 1（設計とセキュリティ）であり、Part 2ではリリース・評価・テレメトリ・レイテンシ・コスト・運用復旧・改善ループがカバーされている。AutoGenからAgent Frameworkへの移行を検討しているチームは、移行先のフレームワークを選択した後、本番投入前にこの7つの原則をチェックリストとして活用することを推奨する。

## 参考文献

- **Blog URL**: [Global enterprise multi-agent systems in production: Design and security (Part 1)](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/global-enterprise-multi-agent-systems-in-production-design-and-security-part-1/4564913)
- **Part 2**: [Release and Operations](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/global-enterprise-multi-agent-systems-in-production-release-and-operations-part-/4564916)
- **Microsoft Agent Framework**: [公式ドキュメント](https://learn.microsoft.com/en-us/agent-framework/)
- **OWASP Top 10 for LLM Applications**: [OWASP](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- **AWS Well-Architected Agentic AI Lens**: [AGENTREL07](https://docs.aws.amazon.com/wellarchitected/latest/agentic-ai-lens/agentrel07.html)
- **Related Zenn article**: [AutoGenからAgent Frameworkへ：自律エージェント移行実践ガイド](https://zenn.dev/0h_n0/articles/c5509c723c2670)

---

:::message
この記事はAI（Claude Code）により自動生成されました。内容の正確性については情報源のブログ記事で検証していますが、実際の利用時は公式ドキュメントもご確認ください。
:::
