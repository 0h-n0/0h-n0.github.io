---
layout: post
title: "Google ADK解説: マルチエージェントアプリケーション構築のためのオープンソースフレームワーク"
description: "Google Agent Development Kit(ADK)はGemini最適化のオープンソースフレームワークで、Sequential/Parallel/Loopワークフローを宣言的に定義しマルチエージェント協調を実現する"
categories: [blog, tech_blog]
tags: [google-adk, multi-agent, agent-orchestration, gemini, googlecloud, llm]
date: 2026-02-20 11:00:00 +0900
source_type: tech_blog
source_domain: developers.googleblog.com
source_url: https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/
zenn_article: a7935e0412571c
zenn_url: https://zenn.dev/0h_n0/articles/a7935e0412571c
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

Google Agent Development Kit（ADK）は、Google Cloud NEXT 2025で発表されたオープンソースのマルチエージェントフレームワークである。Geminiを中心としつつLiteLLMを通じたマルチモデル対応、Sequential/Parallel/Loopのワークフローエージェント、MCP（Model Context Protocol）ツール統合、そしてVertex AI Agent Engineへのマネージドデプロイメントを提供する。Google社内のAgentspaceやCustomer Engagement Suiteで実運用されているフレームワークであり、Zenn記事で紹介されているGemini 3.1 Proのマルチエージェント協調コーディングの基盤技術である。

この記事は [Zenn記事: Gemini 3.1 Proで構築するマルチエージェント協調コーディングの実践手法](https://zenn.dev/0h_n0/articles/a7935e0412571c) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（Google Developers Blog）
- **URL**: [https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/](https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/)
- **組織**: Google / Google Cloud
- **発表日**: 2025年4月（Google Cloud NEXT 2025）
- **公式ドキュメント**: [https://google.github.io/adk-docs/](https://google.github.io/adk-docs/)
- **GitHub**: [https://github.com/google/adk-python](https://github.com/google/adk-python)（Apache 2.0 License）

## 技術的背景（Technical Background）

### なぜADKが必要か

LLMエージェントの開発には、以下の技術的課題がある：

1. **メモリ管理**: 長時間の対話でコンテキストが膨張し、品質が劣化する
2. **ツールオーケストレーション**: 複数ツールの呼び出し順序・並列実行の管理が複雑
3. **マルチエージェント協調**: エージェント間のタスク委譲・結果集約のルーティング
4. **デプロイメント**: 本番環境でのスケーリング・監視・セキュリティ

従来のフレームワーク（LangChain, CrewAI等）はこれらの一部を解決するが、Gemini固有の機能（thought signatures、並列ツール呼び出し、thinking_level制御）を最大限活用する設計ではなかった。ADKはGemini APIのネイティブ機能を直接活用し、かつマルチモデル対応も保つ位置づけで設計されている。

### 学術研究との関連

ADKのアーキテクチャはMetaGPT（Hong et al., 2023）のSOP駆動設計を一般化したものと解釈できる。MetaGPTがソフトウェア開発に特化したSOP（Product Manager → Architect → Engineer → QA）を定義したのに対し、ADKは任意のドメインで同様の構造を宣言的に定義可能にしている。

## 実装アーキテクチャ（Architecture）

### コアコンセプト: 3種類のエージェント

ADKは以下の3種類のエージェントを提供する：

#### 1. LlmAgent（推論エージェント）

LLMを使って自然言語理解・推論・ツール呼び出しを行うエージェント。Gemini 3.1 Proのthought signatures、thinking_level制御、並列ツール呼び出しをネイティブにサポートする。

```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool, google_search

# コード検索エージェント
code_searcher = LlmAgent(
    model="gemini-3.1-pro-preview",
    name="code_searcher",
    description="コードベースを検索し、関連ファイルを特定するエージェント",
    instruction="""
    与えられたタスクに関連するコードファイルを検索します。
    以下の手順で作業してください：
    1. タスクのキーワードを抽出
    2. search_codebase ツールで関連ファイルを検索
    3. 各ファイルの関連度を評価
    4. 上位5ファイルのパスと関連箇所を報告
    """,
    tools=[
        FunctionTool(search_codebase),
        FunctionTool(read_file),
    ],
)
```

#### 2. ワークフローエージェント（オーケストレーション）

複数エージェントの実行順序を制御する。ADKは3つのワークフローパターンを提供する：

**SequentialAgent（直列実行）**:
```python
from google.adk.agents import SequentialAgent

# 計画 → 実装 → レビュー の直列パイプライン
coding_pipeline = SequentialAgent(
    name="coding_pipeline",
    description="タスクを計画・実装・レビューする",
    sub_agents=[planner, coder, reviewer],
)
```

**ParallelAgent（並列実行）**:
```python
from google.adk.agents import ParallelAgent

# 複数ファイルの同時編集
parallel_coders = ParallelAgent(
    name="parallel_coders",
    description="独立したファイルを並列に編集",
    sub_agents=[coder_module_a, coder_module_b, coder_module_c],
)
```

**LoopAgent（反復実行）**:
```python
from google.adk.agents import LoopAgent

# テスト→修正の反復
test_fix_loop = LoopAgent(
    name="test_fix_loop",
    description="テストが通るまでコードを修正",
    sub_agents=[test_runner, fixer],
    max_iterations=5,
)
```

#### 3. カスタムエージェント

`BaseAgent`を継承して独自のエージェントを実装可能。完全にカスタムの制御フローを実現する。

### エージェント階層とルーティング

ADKのマルチエージェントシステムはツリー構造で組織される：

```
root_agent (LlmAgent)
├── planner (LlmAgent, thinking_level=high)
├── coding_pipeline (SequentialAgent)
│   ├── parallel_coders (ParallelAgent)
│   │   ├── coder_a (LlmAgent, thinking_level=medium)
│   │   └── coder_b (LlmAgent, thinking_level=medium)
│   └── test_fix_loop (LoopAgent)
│       ├── test_runner (LlmAgent, thinking_level=low)
│       └── fixer (LlmAgent, thinking_level=high)
└── reviewer (LlmAgent, thinking_level=medium)
```

ルートエージェント（LlmAgent）は、受け取ったタスクを各サブエージェントの`description`に基づいてLLMが自動ルーティングする。このルーティングにはGemini 3.1 Proの推論能力が活用され、タスクの複雑さに応じた適切な委譲先が選択される。

### ツールエコシステム

ADKのツール統合は以下の4レイヤーで構成される：

1. **ビルトインツール**: Google Search、Code Execution（Gemini API直接統合）
2. **FunctionTool**: Python関数をツールとしてラップ
3. **MCPツール**: Model Context Protocolに準拠した外部ツール
4. **エージェント-as-ツール**: 他のエージェント自体をツールとして呼び出し

```python
from google.adk.tools import FunctionTool
from google.adk.tools.mcp import MCPTool

# Python関数をツール化
def search_codebase(query: str, max_results: int = 10) -> list[dict]:
    """コードベースをキーワード検索する

    Args:
        query: 検索クエリ
        max_results: 最大結果数

    Returns:
        検索結果のリスト
    """
    # 実装...
    return results

search_tool = FunctionTool(search_codebase)

# MCPツールの統合
mcp_github = MCPTool(
    server_url="http://localhost:3000/mcp",
    tool_name="github_issues",
)
```

### Zenn記事との技術的対応

Zenn記事のGemini 3.1 Proコード例との対応関係：

| Zenn記事の要素 | ADKの対応機能 |
|--------------|-------------|
| `types.ThinkingConfig(thinking_level=...)` | `LlmAgent`のモデル設定で制御 |
| `chat.send_message()` + thought signatures | ADKが自動管理 |
| 並列`functionCall`の順序保持 | `ParallelAgent`が内部で管理 |
| 手動ツール実行ループ | `FunctionTool`で自動化 |
| 計画層/実行層/検証層 | `SequentialAgent`で宣言的定義 |

ADKを使うことで、Zenn記事で示されている手動のツール実行ループ（`while True` + `function_call`チェック）が不要になり、宣言的にパイプラインを定義できる。

## パフォーマンス最適化（Performance）

### スケーリング戦略

ADKはVertex AI Agent Engineと統合して本番スケーリングを実現する：

- **自動スケーリング**: トラフィックに応じてエージェントインスタンスを動的に増減
- **セッション管理**: エージェントの状態（thought signatures含む）を永続化
- **コンテキスト管理**: 長時間セッションでのメモリ最適化

### コスト最適化

thinking_level制御による具体的なコスト試算：

| thinking_level | 入力トークン単価 | 出力トークン単価 | 用途 |
|---------------|--------------|--------------|------|
| high | $2/MTok | $12/MTok | 計画・設計（Planner） |
| medium | $2/MTok | $12/MTok | コード生成・レビュー |
| low | $2/MTok | $12/MTok | テスト実行・ログ解析 |

注：thinking_levelはトークン単価自体は変わらないが、推論に使うthinkingトークン数が変わるため、実効コストが異なる。highは多くのthinkingトークンを消費し、lowは最小限に抑える。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $70-200 | Lambda + Bedrock/Gemini API + DynamoDB |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $400-1,000 | ECS Fargate + Gemini API + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,500-6,000 | EKS + Karpenter + Vertex AI |

**Small構成の詳細** (月額$70-200):
- **Lambda**: 1GB RAM, 120秒タイムアウト ($25/月)
- **Gemini API**: Gemini 3.1 Pro, thinking_level制御あり ($100/月)
- **DynamoDB**: セッション状態永続化 ($10/月)
- **CloudWatch**: 基本監視 ($5/月)

**コスト削減テクニック**:
- thinking_level制御: Planner=high, Coder=medium, TestRunner=lowでthinkingトークンを最適化
- ParallelAgentの活用: 並列実行で壁時計時間を削減（コスト自体は同じだが応答時間が短縮）
- セッションキャッシュ: DynamoDBにthought signaturesを保存し、再利用
- Gemini 3 Flash併用: 単純タスクは安価なFlashモデルに委譲

**コスト試算の注意事項**:
- 上記は2026年2月時点のGemini API料金とAWS ap-northeast-1（東京）リージョン料金に基づく概算値です
- Gemini API料金はGoogle Cloud公式を確認してください
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

```hcl
# --- Lambda: ADKオーケストレータ ---
resource "aws_lambda_function" "adk_orchestrator" {
  filename      = "adk_orchestrator.zip"
  function_name = "adk-multi-agent-orchestrator"
  role          = aws_iam_role.adk_lambda.arn
  handler       = "main.handler"
  runtime       = "python3.12"
  timeout       = 300
  memory_size   = 1024

  environment {
    variables = {
      GEMINI_API_KEY     = data.aws_secretsmanager_secret_version.gemini_key.secret_string
      DYNAMODB_TABLE     = aws_dynamodb_table.sessions.name
      THINKING_PLANNER   = "high"
      THINKING_CODER     = "medium"
      THINKING_REVIEWER  = "medium"
    }
  }
}

# --- DynamoDB: セッション状態管理 ---
resource "aws_dynamodb_table" "sessions" {
  name         = "adk-agent-sessions"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "session_id"
  range_key    = "agent_name"

  attribute {
    name = "session_id"
    type = "S"
  }
  attribute {
    name = "agent_name"
    type = "S"
  }

  ttl {
    attribute_name = "expire_at"
    enabled        = true
  }
}

# --- Secrets Manager: Gemini APIキー ---
resource "aws_secretsmanager_secret" "gemini_key" {
  name = "adk-gemini-api-key"
}

# --- CloudWatch: パフォーマンス監視 ---
resource "aws_cloudwatch_metric_alarm" "agent_latency" {
  alarm_name          = "adk-agent-latency-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "p95"
  threshold           = 120000
  alarm_description   = "ADKオーケストレータのP95レイテンシが120秒を超過"
  dimensions = {
    FunctionName = aws_lambda_function.adk_orchestrator.function_name
  }
}
```

### セキュリティベストプラクティス

- **APIキー管理**: Secrets Manager使用、Lambda環境変数での直接指定は禁止
- **IAM最小権限**: DynamoDB/CloudWatch/Secrets Managerの必要な操作のみ許可
- **ネットワーク**: VPC内配置、Gemini APIへはNAT Gateway経由
- **監査**: CloudTrailで全API呼び出しを記録

### 運用・監視設定

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# エージェント別実行時間アラート
for agent_name in ['planner', 'coder', 'reviewer']:
    cloudwatch.put_metric_alarm(
        AlarmName=f'adk-{agent_name}-timeout',
        ComparisonOperator='GreaterThanThreshold',
        EvaluationPeriods=1,
        MetricName=f'{agent_name}_duration_ms',
        Namespace='Custom/ADK',
        Period=3600,
        Statistic='p95',
        Threshold=60000,
        AlarmDescription=f'{agent_name}エージェントのP95が60秒を超過'
    )
```

### コスト最適化チェックリスト

- [ ] thinking_level最適化: Planner=high, Coder=medium, Test=low
- [ ] ParallelAgent活用: 独立タスクの並列実行で応答時間短縮
- [ ] Gemini 3 Flash併用: 単純タスクは安価なモデルに委譲
- [ ] セッションキャッシュ: thought signaturesをDynamoDBに保存
- [ ] Lambda Reserved Concurrency: 不要な並列実行を防止
- [ ] DynamoDB TTL: セッション状態の自動クリーンアップ
- [ ] CloudWatch Logs: 保持期間30日に設定
- [ ] Secrets Manager: APIキーのローテーション設定
- [ ] AWS Budgets: 月額予算アラート設定
- [ ] Gemini APIクォータ: RPM/TPMリミットの監視

## 運用での学び（Production Lessons）

### エージェントルーティングの精度

ADKのLLM駆動ルーティングでは、各サブエージェントの`description`が極めて重要である。descriptionが曖昧だと、タスクが間違ったエージェントにルーティングされる。

**良いdescription例**:
```python
coder = LlmAgent(
    description="Python/TypeScriptのコードを生成・修正する。"
                "ファイル作成、関数実装、バグ修正を担当。"
                "テスト実行やコードレビューは担当しない。",
)
```

**悪いdescription例**:
```python
coder = LlmAgent(
    description="コーディングを担当するエージェント",  # 曖昧すぎる
)
```

### thought signaturesの管理

Zenn記事で指摘されている「並列ツール呼び出しの順序保持」は、ADKを使うことで自動的に解決される。ただし、以下の注意点がある：

1. **セッション跨ぎ**: thought signaturesはセッション内でのみ有効。DynamoDBにセッション状態を保存する場合、thought signaturesの再構築が必要
2. **モデル切り替え**: 同一セッション内でモデルを切り替えると、thought signaturesが無効化される

### 障害パターンと対策

| 障害パターン | 発生頻度 | 対策 |
|------------|---------|------|
| thinking_level=high のタイムアウト | 中 | Lambda タイムアウトを300秒に設定、LoopAgentで再試行 |
| エージェント間ルーティングの誤り | 低 | description の精緻化、ルーティングログの分析 |
| Gemini API レート制限 | 中 | 指数バックオフ+ジッタ、tenacityライブラリ使用 |
| thought signatures の破損 | 稀 | セッション再開（新規thought signaturesの生成） |

## 学術研究との関連（Academic Connection）

ADKの設計は以下の学術研究と密接に関連している：

- **MetaGPT** (Hong et al., 2023): SOP駆動の役割分担。ADKのSequentialAgentが同様の構造を汎用化
- **ReAct** (Yao et al., 2022): 推論と行動の交互実行。ADKのLlmAgentの基本動作原理
- **Toolformer** (Schick et al., 2023): ツール使用のためのLLM微調整。ADKはプロンプトベースでツール使用を実現（微調整不要）

ADKはこれらの研究成果をプロダクションレベルで統合した実装であり、学術研究と実運用の橋渡しとなるフレームワークである。

## まとめと実践への示唆

Google ADKは、Zenn記事で紹介されているGemini 3.1 Proのマルチエージェント協調コーディングを実現するための**公式推奨フレームワーク**である。

**実践的な導入ステップ**:

1. `pip install google-adk` でインストール
2. 単一のLlmAgentで基本動作を確認
3. SequentialAgentで計画→実装→レビューのパイプラインを構築
4. thinking_level制御でコスト最適化
5. Vertex AI Agent Engineで本番デプロイ

ADKの最大の価値は、Zenn記事で手動実装している「ツール実行ループ」「thought signatures管理」「並列呼び出しの順序保持」をフレームワークレベルで抽象化し、開発者がドメインロジックに集中できるようにする点にある。

## 参考文献

- **Blog URL**: [https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/](https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/)
- **公式ドキュメント**: [https://google.github.io/adk-docs/](https://google.github.io/adk-docs/)
- **GitHub**: [https://github.com/google/adk-python](https://github.com/google/adk-python)（Apache 2.0）
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/a7935e0412571c](https://zenn.dev/0h_n0/articles/a7935e0412571c)
