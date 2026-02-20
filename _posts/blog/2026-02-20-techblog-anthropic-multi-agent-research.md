---
layout: post
title: "Anthropic解説: マルチエージェントリサーチシステムの設計と実装"
description: "Anthropicが構築したオーケストレータ・ワーカー型マルチエージェントリサーチシステムの設計原則と実装パターンを解説。単一エージェント比90.2%の性能向上を達成"
categories: [blog, tech_blog]
tags: [anthropic, multi-agent, orchestrator-worker, claude, parallel-processing, llm]
date: 2026-02-20 12:00:00 +0900
source_type: tech_blog
source_domain: anthropic.com
source_url: https://www.anthropic.com/engineering/multi-agent-research-system
zenn_article: a7935e0412571c
zenn_url: https://zenn.dev/0h_n0/articles/a7935e0412571c
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

Anthropicが公開した「How we built our multi-agent research system」は、Claude APIを活用したオーケストレータ・ワーカー型マルチエージェントシステムの設計と実装を詳細に解説したエンジニアリングブログである。リードエージェント（Claude Opus 4）がタスクを分解し、サブエージェント（Claude Sonnet 4）が並列に情報収集を行うアーキテクチャにより、単一エージェントのClaude Opus 4と比較して**90.2%の性能向上**を達成した。トークン消費量は通常チャットの15倍に達するが、複雑なリサーチタスクにおけるROIは極めて高い。

この記事は [Zenn記事: Gemini 3.1 Proで構築するマルチエージェント協調コーディングの実践手法](https://zenn.dev/0h_n0/articles/a7935e0412571c) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（Anthropic Engineering Blog）
- **URL**: [https://www.anthropic.com/engineering/multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system)
- **組織**: Anthropic
- **発表日**: 2025年6月

## 技術的背景（Technical Background）

### マルチエージェントシステムの必要性

複雑なリサーチタスク（例：「特定の条件を満たす企業を10社見つけ、各社の技術スタック・従業員数・直近の資金調達を調査する」）は、単一のLLMコンテキストでは以下の問題が発生する：

1. **コンテキスト長の制約**: 大量の情報を1つのコンテキストに詰め込むとLLMの品質が劣化する
2. **逐次処理の遅延**: 10社を1社ずつ調査すると、壁時計時間が線形に増加する
3. **適切な作業量の判断困難**: LLMは単一タスクに対してどれだけの深さで調査すべきかの判断が苦手

Anthropicのマルチエージェントシステムは、これらの問題をオーケストレータ・ワーカーパターンで解決している。

### 学術研究との位置づけ

このシステムは、学術研究における以下のパターンを実務に適用した事例である：

- **Hierarchical Multi-Agent** (AgentOrchestra, 2024): 階層型エージェントの実プロダクト実装
- **ReAct** (Yao et al., 2022): 各サブエージェントの基本動作
- **Map-Reduce型並列処理**: サブエージェントへのタスク分散と結果集約

## 実装アーキテクチャ（Architecture）

### オーケストレータ・ワーカーパターン

Anthropicのマルチエージェントシステムは以下の2層構造で構成される：

#### リードエージェント（Orchestrator）

リードエージェントはClaude Opus 4を使用し、以下の責務を持つ：

1. **クエリ分析**: ユーザーの複雑なリサーチクエリを分析
2. **戦略立案**: 調査の方針と深さを決定
3. **タスク分解**: クエリを3-5個の独立したサブタスクに分解
4. **サブエージェント生成**: 各サブタスクに対してサブエージェントを起動
5. **結果集約**: サブエージェントの結果を統合し、最終回答を生成

```python
class LeadAgent:
    """リードエージェント（オーケストレータ）

    Claude Opus 4を使用してタスク分解と結果集約を行う
    """

    def __init__(self):
        self.model = "claude-opus-4"
        self.sub_agents: list[SubAgent] = []

    async def research(self, query: str) -> str:
        """リサーチクエリを処理

        Args:
            query: ユーザーのリサーチクエリ

        Returns:
            調査結果の統合レポート
        """
        # 1. 戦略立案とタスク分解
        plan = await self._create_plan(query)

        # 2. サブエージェントを並列起動（3-5個）
        tasks = []
        for subtask in plan.subtasks:
            agent = SubAgent(
                objective=subtask.objective,
                output_format=subtask.output_format,
                tools=subtask.tools,
                boundaries=subtask.boundaries,
            )
            tasks.append(agent.execute())

        # 3. 並列実行と結果収集
        results = await asyncio.gather(*tasks)

        # 4. 結果を統合して最終レポート生成
        report = await self._synthesize(query, results)
        return report
```

#### サブエージェント（Worker）

サブエージェントはClaude Sonnet 4を使用し、以下の4つの情報を受け取って動作する：

1. **目的（Objective）**: 何を調査するか
2. **出力フォーマット（Output Format）**: 結果をどの形式で返すか
3. **ツールと情報源（Tools & Sources）**: 使用可能なツールと検索対象
4. **タスク境界（Boundaries）**: 調査の範囲と深さの制約

```python
class SubAgent:
    """サブエージェント（ワーカー）

    Claude Sonnet 4を使用して特定のリサーチサブタスクを実行する
    """

    def __init__(
        self,
        objective: str,
        output_format: str,
        tools: list[str],
        boundaries: str,
    ):
        self.model = "claude-sonnet-4"
        self.objective = objective
        self.output_format = output_format
        self.tools = tools
        self.boundaries = boundaries

    async def execute(self) -> dict:
        """サブタスクを実行

        Returns:
            構造化された調査結果
        """
        system_prompt = f"""
あなたはリサーチサブエージェントです。

## 目的
{self.objective}

## 出力フォーマット
{self.output_format}

## 使用可能なツール
{', '.join(self.tools)}

## タスク境界
{self.boundaries}

## 重要な制約
- 指定された範囲外の情報は収集しない
- 各情報源について最低3回の検索を実行
- 結果は必ず指定フォーマットで返す
"""
        # ReActループでツールを使いながら情報収集
        result = await self._react_loop(system_prompt)
        return result
```

### 並列処理の2レベル設計

Anthropicのシステムは並列処理を2つのレベルで実装している：

**Level 1: サブエージェントレベルの並列化**

リードエージェントが3-5個のサブエージェントを同時に起動する。これにより、10社の調査を5つのサブエージェントに2社ずつ割り当て、処理時間を1/5に削減する。

**Level 2: ツール呼び出しレベルの並列化**

各サブエージェント内でも、複数のツール呼び出し（Web検索、API呼び出し等）を並列に実行する。

$$
T_{total} = T_{plan} + \max_{i \in \{1..n\}}(T_{sub_i}) + T_{synthesize}
$$

ここで、
- $T_{total}$: 全体の処理時間
- $T_{plan}$: リードエージェントの計画策定時間
- $T_{sub_i}$: $i$番目のサブエージェントの実行時間
- $T_{synthesize}$: 結果統合時間
- $n$: サブエージェント数（通常3-5）

逐次処理の場合：

$$
T_{sequential} = T_{plan} + \sum_{i=1}^{n} T_{sub_i} + T_{synthesize}
$$

並列化による高速化率：

$$
\text{Speedup} = \frac{T_{sequential}}{T_{total}} \approx \frac{\sum T_{sub_i}}{\max T_{sub_i}}
$$

5つのサブエージェントが均等な処理時間の場合、理論上5倍の高速化が得られる。実測では、複雑なクエリに対して**処理時間を最大90%削減**している。

### スケーリングルールの埋め込み

Anthropicが発見した重要な知見は、**LLMは適切な作業量の判断が苦手**という点である。単純なYes/No回答で済むタスクに対して過剰な調査を行ったり、逆に複雑なタスクに対して浅い調査で終えたりする。

この問題に対して、プロンプトにスケーリングルールを明示的に埋め込む手法を採用している：

```python
SCALING_RULES = """
## スケーリングルール（作業量の目安）

### 検索深度
- 単純な事実確認: 検索1-2回で十分
- 企業概要の調査: 検索3-5回
- 技術的詳細の調査: 検索5-10回

### 情報源の数
- 確認可能な事実: 1ソースで十分
- 議論のある主張: 最低2ソースでクロスチェック
- 重要な数値データ: 最低3ソースで検証

### 出力の長さ
- 簡潔な回答: 100-200文字
- 詳細な分析: 500-1000文字
- 包括的レポート: 2000-5000文字
"""
```

## パフォーマンス最適化（Performance）

### 実測値

Anthropicの内部評価ベンチマークでの結果：

| 構成 | スコア | 改善率 |
|------|-------|-------|
| Claude Sonnet 4（単一） | ベースライン | — |
| Claude Opus 4（単一） | ベースライン+α | — |
| **マルチエージェント**（Opus 4 lead + Sonnet 4 workers） | **+90.2%** | vs 単一Opus 4 |

### トークン経済学

マルチエージェントシステムのトークン消費は通常チャットの**約15倍**に達する：

| 構成要素 | トークン消費（概算） |
|---------|-----------------|
| リードエージェントの計画策定 | 5,000-10,000 |
| サブエージェント × 5（各20,000-40,000） | 100,000-200,000 |
| 結果統合 | 10,000-20,000 |
| **合計** | **115,000-230,000** |

通常のチャット（5,000-15,000トークン）と比較して15-20倍のコストとなる。

$$
\text{Cost}_{multi} = C_{opus} \times T_{lead} + C_{sonnet} \times \sum_{i=1}^{n} T_{sub_i}
$$

ここで、
- $C_{opus}$: Claude Opus 4のトークン単価
- $C_{sonnet}$: Claude Sonnet 4のトークン単価
- $T_{lead}$: リードエージェントのトークン消費
- $T_{sub_i}$: $i$番目のサブエージェントのトークン消費

### チューニング手法

- **リードエージェントには高能力モデル**: Opus 4の高い推論能力でタスク分解の品質を確保
- **サブエージェントにはコスト効率モデル**: Sonnet 4の十分な能力で情報収集を効率的に実行
- **サブエージェント数の最適化**: 3-5個が最適。多すぎると結果統合の品質が低下

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $150-400 | Lambda + Bedrock + Step Functions |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $800-2,000 | ECS Fargate + Bedrock + SQS |
| **Large** | 300,000+ (10,000/日) | Container | $5,000-15,000 | EKS + Bedrock + ElastiCache |

**Small構成の詳細** (月額$150-400):
- **Step Functions**: オーケストレーション（並列分岐） ($20/月)
- **Lambda**: リードエージェント + サブエージェント ($30/月)
- **Bedrock**: Claude 3.5 Sonnet（リード）+ Haiku（サブ） ($250/月)
- **DynamoDB**: 結果キャッシュ ($10/月)
- **CloudWatch**: 監視 ($5/月)

**コスト削減テクニック**:
- リードエージェント: Claude 3.5 Sonnetで十分（Opusほどのコストをかけずにタスク分解が可能）
- サブエージェント: Claude 3.5 Haikuで情報収集（高速・低コスト）
- Bedrock Batch API: 非リアルタイムのリサーチタスクで50%割引
- 結果キャッシュ: 類似クエリに対してDynamoDB TTL付きキャッシュ
- サブエージェント数の動的制御: クエリの複雑さに応じて2-5個の範囲で調整

**コスト試算の注意事項**:
- 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です
- Bedrockの料金モデルはトークン従量課金のため、実際のコストはリクエスト内容に大きく依存します
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

```hcl
# --- Step Functions: オーケストレーション ---
resource "aws_sfn_state_machine" "multi_agent" {
  name     = "multi-agent-research"
  role_arn = aws_iam_role.sfn_role.arn

  definition = jsonencode({
    StartAt = "PlanResearch"
    States = {
      PlanResearch = {
        Type     = "Task"
        Resource = aws_lambda_function.lead_agent.arn
        Next     = "ParallelResearch"
      }
      ParallelResearch = {
        Type = "Parallel"
        Branches = [
          { StartAt = "SubAgent1", States = { SubAgent1 = { Type = "Task", Resource = aws_lambda_function.sub_agent.arn, End = true } } },
          { StartAt = "SubAgent2", States = { SubAgent2 = { Type = "Task", Resource = aws_lambda_function.sub_agent.arn, End = true } } },
          { StartAt = "SubAgent3", States = { SubAgent3 = { Type = "Task", Resource = aws_lambda_function.sub_agent.arn, End = true } } },
        ]
        Next = "SynthesizeResults"
      }
      SynthesizeResults = {
        Type     = "Task"
        Resource = aws_lambda_function.lead_agent.arn
        End      = true
      }
    }
  })
}

# --- Lambda: リードエージェント ---
resource "aws_lambda_function" "lead_agent" {
  filename      = "lead_agent.zip"
  function_name = "multi-agent-lead"
  role          = aws_iam_role.lambda_role.arn
  handler       = "main.handler"
  runtime       = "python3.12"
  timeout       = 300
  memory_size   = 1024

  environment {
    variables = {
      BEDROCK_MODEL_LEAD = "anthropic.claude-3-5-sonnet-20241022-v2:0"
      BEDROCK_MODEL_SUB  = "anthropic.claude-3-5-haiku-20241022-v1:0"
    }
  }
}

# --- Lambda: サブエージェント ---
resource "aws_lambda_function" "sub_agent" {
  filename      = "sub_agent.zip"
  function_name = "multi-agent-sub"
  role          = aws_iam_role.lambda_role.arn
  handler       = "main.handler"
  runtime       = "python3.12"
  timeout       = 120
  memory_size   = 512
}

# --- DynamoDB: 結果キャッシュ ---
resource "aws_dynamodb_table" "research_cache" {
  name         = "multi-agent-research-cache"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "query_hash"

  attribute {
    name = "query_hash"
    type = "S"
  }

  ttl {
    attribute_name = "expire_at"
    enabled        = true
  }
}
```

### 運用・監視設定

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# サブエージェントのタイムアウト監視
cloudwatch.put_metric_alarm(
    AlarmName='sub-agent-timeout',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='Duration',
    Namespace='AWS/Lambda',
    Period=3600,
    Statistic='p95',
    Threshold=100000,
    AlarmDescription='サブエージェントのP95が100秒を超過'
)

# Step Functions並列実行数の監視
cloudwatch.put_metric_alarm(
    AlarmName='sfn-parallel-throttle',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='ExecutionThrottled',
    Namespace='AWS/States',
    Period=300,
    Statistic='Sum',
    Threshold=5,
    AlarmDescription='Step Functionsのスロットリング発生'
)
```

### コスト最適化チェックリスト

- [ ] リードエージェント: Sonnet使用（Opusは不要な場合が多い）
- [ ] サブエージェント: Haiku使用（情報収集には十分）
- [ ] Bedrock Batch API: 非リアルタイム処理で50%割引
- [ ] 結果キャッシュ: DynamoDB TTL付き（同一クエリの再実行防止）
- [ ] サブエージェント数の動的制御: 2-5個の範囲で調整
- [ ] Step Functions Express Workflows: 短時間のワークフローに最適
- [ ] Lambda メモリ最適化: Power Tuningで最適値を特定
- [ ] CloudWatch Logs: 保持期間14日
- [ ] AWS Budgets: 月額予算アラート設定
- [ ] Bedrock クォータ: RPM/TPMリミットの事前確認

## 運用での学び（Production Lessons）

### タスク分解の品質がボトルネック

マルチエージェントシステム全体の性能は、リードエージェントのタスク分解品質に大きく依存する。Anthropicの知見では、以下の要素が重要：

1. **明確な目的の記述**: 各サブタスクに曖昧さを残さない
2. **独立性の確保**: サブタスク間の依存関係を最小化（並列実行のため）
3. **出力フォーマットの統一**: 結果統合を容易にするため、サブエージェントの出力形式を統一

### スケーリングルールの重要性

LLMは「どの程度の深さで調査すべきか」の判断が苦手であるため、プロンプトに明示的なスケーリングルールを埋め込む必要がある。これはZenn記事のthinking_level制御と同様の課題であり、タスクの複雑さに応じたリソース配分を人間が設計する必要がある。

### Gemini 3.1 Proへの応用

Anthropicのオーケストレータ・ワーカーパターンは、Zenn記事のGemini 3.1 Pro構成に以下のように適用可能：

| Anthropicの設計 | Gemini 3.1 Pro + ADKでの実装 |
|----------------|---------------------------|
| Opus 4リードエージェント | Planner（thinking_level=high） |
| Sonnet 4サブエージェント | Coder/Reviewer（thinking_level=medium） |
| 並列サブエージェント起動 | ADKのParallelAgent |
| 結果統合 | SequentialAgentの最終ステップ |
| スケーリングルール | thinking_level制御 |

## 学術研究との関連（Academic Connection）

Anthropicのシステムは以下の研究を実運用で検証した事例である：

- **AgentOrchestra** (2024): 階層型マルチエージェントの理論的フレームワーク。Anthropicの実装は具体的な性能数値（90.2%改善）を提供
- **Map-Reduce Parallel Agent** (2024): 並列分散処理パターンの応用。2レベルの並列化は新規性がある
- **Scaling Laws for Agent Systems** (Google Research, 2024): エージェント数のスケーリングと性能の関係。Anthropicの3-5個のサブエージェントは実務的な最適値を示唆

## まとめと実践への示唆

Anthropicのマルチエージェントリサーチシステムから得られる実践的な教訓は以下の3点である：

1. **モデル選択の階層化**: 高コストモデル（Opus/高thinking_level）はオーケストレーションに、低コストモデル（Sonnet/Haiku/低thinking_level）はワーカーに使用することで、品質とコストの最適バランスを達成

2. **スケーリングルールの明示化**: LLMは自律的に作業量を判断できないため、タスクの複雑さに応じた作業量の指針をプロンプトに埋め込む必要がある

3. **2レベル並列化**: エージェントレベルとツール呼び出しレベルの両方で並列化を適用することで、処理時間を最大90%削減可能

これらの知見は、Zenn記事のGemini 3.1 Pro + ADKによるマルチエージェント協調コーディングの設計に直接活用できる。

## 参考文献

- **Blog URL**: [https://www.anthropic.com/engineering/multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system)
- **Claude Agent SDK**: [https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/a7935e0412571c](https://zenn.dev/0h_n0/articles/a7935e0412571c)
