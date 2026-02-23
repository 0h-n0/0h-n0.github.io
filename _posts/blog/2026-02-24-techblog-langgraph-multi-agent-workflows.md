---
layout: post
title: "LangChain公式解説: LangGraphマルチエージェントワークフローの3設計パターン"
description: "LangChain公式ブログが提示するマルチエージェント協調・Supervisor・階層チーム3パターンの設計原則と実装戦略を解説"
categories: [blog, tech_blog]
tags: [LangGraph, multi-agent, supervisor, hierarchical, workflow, langgraph, claude, rag, python, multiagent]
date: 2026-02-24 13:00:00 +0900
source_type: tech_blog
source_domain: blog.langchain.com
source_url: https://blog.langchain.com/langgraph-multi-agent-workflows/
zenn_article: 92f41b4fdc7b49
zenn_url: https://zenn.dev/0h_n0/articles/92f41b4fdc7b49
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [LangGraph: Multi-Agent Workflows（LangChain公式ブログ）](https://blog.langchain.com/langgraph-multi-agent-workflows/) の解説記事です。

## ブログ概要（Summary）

LangChainチームが公式ブログで発表した本記事は、LangGraphフレームワークを用いたマルチエージェントシステムの3つの主要設計パターン（Multi-Agent Collaboration、Agent Supervisor、Hierarchical Agent Teams）を体系的に解説している。各パターンのアーキテクチャ、情報共有方式、制御フロー、および適用場面の使い分けが示されており、Zenn記事で実装しているマルチエージェントRAGの設計基盤となるドキュメントである。

この記事は [Zenn記事: LangGraph×Claude 4.6で推論精度と応答速度を両立するマルチエージェントRAG](https://zenn.dev/0h_n0/articles/92f41b4fdc7b49) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://blog.langchain.com/langgraph-multi-agent-workflows/](https://blog.langchain.com/langgraph-multi-agent-workflows/)
- **組織**: LangChain
- **発表日**: 2024年1月23日（最終更新: 2025年2月26日）

## 技術的背景（Technical Background）

マルチエージェントシステムの研究は、分散AIの文脈で長い歴史を持つが、LLMの登場により新たな局面を迎えている。LangChainチームは、LLMベースのマルチエージェントシステムを「言語モデルによって駆動される複数の独立したアクターが、特定の方法で接続されたシステム」と定義している。

従来のsingle-agentアプローチでは、すべてのツール・プロンプト・推論を1つのエージェントが担当する。しかし、ツール数が増加するとプロンプトが肥大化し、エージェントの性能が劣化する。LangChainチームは、ツールと責任をグループ化して複数のエージェントに分担させることで、以下の利点が得られると述べている。

1. **専門化されたプロンプト**: 各エージェントが特定タスクに最適化されたプロンプトを持つ
2. **個別評価**: 各エージェントを独立に評価・改善でき、全体に影響を与えずにイテレーションが可能
3. **関心の分離**: 各エージェントのツールセットが小さくなり、tool selectionの精度が向上

この設計思想はZenn記事のマルチエージェントRAGアーキテクチャの根拠となっている。Triage Agent（ルーティング）、検索エージェント（文書検索）、生成エージェント（回答生成）の役割分離は、まさにこの原則の具体化である。

## 実装アーキテクチャ（Architecture）

### パターン1: Multi-Agent Collaboration（協調型）

**アーキテクチャ**: すべてのエージェントが共有スクラッチパッドにメッセージを書き込み、他のエージェントの出力が全エージェントに可視となる。ルーターはrule-basedのロジック（例: ツール呼び出しに基づくルーティング）で制御フローを管理する。

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

class CollaborationState(TypedDict):
    messages: Annotated[list, operator.add]  # 共有スクラッチパッド
    next_agent: str

def research_agent(state: CollaborationState) -> dict:
    """研究エージェント: WebSearchとDocument検索を担当"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def writing_agent(state: CollaborationState) -> dict:
    """執筆エージェント: 収集した情報を基に文章生成"""
    response = llm_writer.invoke(state["messages"])
    return {"messages": [response]}

def router(state: CollaborationState) -> str:
    """rule-basedルーター: ツール呼び出しに基づく遷移"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "research_agent"
    return "writing_agent"
```

**利点**: 実装がシンプル、全エージェントがコンテキストを共有
**欠点**: メッセージが冗長になりやすい、コンテキストウィンドウの圧迫

### パターン2: Agent Supervisor（監督型）

**アーキテクチャ**: 中央のSupervisorエージェントが各専門エージェントを「ツール」として呼び出す。各エージェントは独立したスクラッチパッド（メッセージ履歴）を持ち、Supervisorがグローバルスクラッチパッドに集約する。

LangChainチームは、Supervisorを「ツールが他のエージェントであるエージェント」と表現している。

```python
from langgraph.graph import StateGraph
from langgraph.types import Command
from typing import Literal

def supervisor(state: dict) -> Command[Literal[
    "retriever", "analyzer", "generator", "__end__"
]]:
    """Supervisorエージェント: 専門エージェントへの振り分け

    各エージェントを「ツール」として扱い、タスクを動的にルーティング
    """
    response = supervisor_llm.invoke([
        {"role": "system", "content": "あなたはタスクマネージャです。"
         "クエリを分析し、適切なエージェントに振り分けてください。"
         "retriever: 情報検索, analyzer: データ分析, "
         "generator: 回答生成, __end__: 完了"},
        {"role": "user", "content": state["query"]}
    ])

    next_agent = parse_routing_decision(response)
    return Command(goto=next_agent)
```

**利点**: 各エージェントが独立したコンテキストを持ち冗長性を排除、Supervisorが全体の制御フローを管理
**欠点**: Supervisorがボトルネックになりうる、LLMベースのルーティングにレイテンシ

Zenn記事のrouter_agentとCommand primitiveによる動的遷移は、このSupervisorパターンをfunctional APIで実装したものと位置付けられる。

### パターン3: Hierarchical Agent Teams（階層型）

**アーキテクチャ**: LangGraphオブジェクト自体をエージェントとしてネストし、Supervisorがそれらを接続する。各サブグラフが独自のノード・エッジを持ち、Supervisorはサブグラフの入出力のみを扱う。

```python
# サブグラフ: 検索チーム
search_team = StateGraph(SearchState)
search_team.add_node("vector_search", vector_search_agent)
search_team.add_node("web_search", web_search_agent)
search_team.add_node("merger", result_merger)
search_team_graph = search_team.compile()

# サブグラフ: 分析チーム
analysis_team = StateGraph(AnalysisState)
analysis_team.add_node("grader", document_grader)
analysis_team.add_node("reranker", reranker_agent)
analysis_team_graph = analysis_team.compile()

# 上位Supervisor
main_graph = StateGraph(MainState)
main_graph.add_node("search_team", search_team_graph)
main_graph.add_node("analysis_team", analysis_team_graph)
main_graph.add_node("generator", generator_agent)
main_graph.add_node("supervisor", supervisor_agent)
```

**利点**: AgentExecutor単体よりも柔軟、各チームを独立に開発・テスト・スケーリング可能
**欠点**: 設計の複雑さが増す、デバッグが困難になりうる

### パターン比較表

| 特性 | Collaboration | Supervisor | Hierarchical |
|------|-------------|------------|-------------|
| 情報共有 | 共有スクラッチパッド | 独立 + グローバル集約 | ネスト + 階層的集約 |
| ルーティング | rule-based | LLM-based | LLM-based（多層） |
| スケーラビリティ | 低（コンテキスト肥大） | 中 | 高 |
| 実装難度 | 低 | 中 | 高 |
| デバッグ容易性 | 高 | 中 | 低 |
| 適用場面 | 2-3エージェント | 3-5エージェント | 5+エージェント |

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

LangGraphマルチエージェントワークフローのデプロイでは、パターンに応じた構成選択が重要である。

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $50-150 | Lambda + Bedrock + DynamoDB |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $300-800 | Lambda + ECS Fargate + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,000-5,000 | EKS + Karpenter + EC2 Spot |

**パターン別のAWS構成推奨**:
- **Collaboration**: Lambda×2-3（各エージェント）+ Step Functions（オーケストレーション）
- **Supervisor**: Lambda（Supervisor）+ ECS Fargate（専門エージェント群）
- **Hierarchical**: EKS（各チームをPodで分離）+ Karpenter（動的スケーリング）

**コスト試算の注意事項**: 上記は2026年2月時点のAWS ap-northeast-1料金に基づく概算値です。最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください。

### Terraformインフラコード

**Supervisor構成: Lambda (Supervisor) + ECS (Agents)**

```hcl
# --- Lambda: Supervisor Agent ---
resource "aws_lambda_function" "supervisor" {
  filename      = "supervisor.zip"
  function_name = "langgraph-supervisor"
  role          = aws_iam_role.supervisor_lambda.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 30
  memory_size   = 512

  environment {
    variables = {
      BEDROCK_MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"
      ECS_CLUSTER      = aws_ecs_cluster.agents.name
    }
  }
}

# --- ECS: 専門エージェント群 ---
resource "aws_ecs_cluster" "agents" {
  name = "langgraph-agents"
}

resource "aws_ecs_service" "retriever" {
  name            = "retriever-agent"
  cluster         = aws_ecs_cluster.agents.id
  task_definition = aws_ecs_task_definition.retriever.arn
  desired_count   = 2
  launch_type     = "FARGATE"
}

resource "aws_ecs_task_definition" "retriever" {
  family                   = "retriever-agent"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024

  container_definitions = jsonencode([{
    name  = "retriever"
    image = "${aws_ecr_repository.agents.repository_url}:retriever-latest"
    portMappings = [{ containerPort = 8080 }]
    environment = [
      { name = "BEDROCK_MODEL_ID", value = "anthropic.claude-sonnet-4-6-20260514-v1:0" }
    ]
  }])
}
```

### 運用・監視設定

**CloudWatch: エージェント間通信のレイテンシ監視**

```sql
fields @timestamp, supervisor_decision, agent_name, latency_ms
| stats avg(latency_ms) as avg_latency,
        pct(latency_ms, 95) as p95
  by supervisor_decision, agent_name, bin(5m)
```

**Supervisorの判断精度モニタリング**:

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_alarm(
    AlarmName='supervisor-routing-accuracy',
    ComparisonOperator='LessThanThreshold',
    EvaluationPeriods=3,
    MetricName='RoutingAccuracy',
    Namespace='LangGraph/Supervisor',
    Period=3600,
    Statistic='Average',
    Threshold=0.8,
    AlarmDescription='Supervisorのルーティング精度が80%を下回った'
)
```

### コスト最適化チェックリスト

- [ ] Supervisorのルーティングは軽量モデル（Haiku）で実行
- [ ] 専門エージェントの回答生成はSonnetで実行
- [ ] ECS Fargate: 専門エージェントは最小構成（0.5vCPU/1GB）
- [ ] Lambda: Supervisorのメモリ最適化（512MB推奨）
- [ ] Bedrock Batch API: 非リアルタイム処理に50%割引
- [ ] Prompt Caching: Supervisorのシステムプロンプト固定
- [ ] ElastiCache: エージェント間通信結果のキャッシュ
- [ ] AWS Budgets: 月額予算設定
- [ ] CloudWatch: Supervisorルーティング精度の監視
- [ ] Cost Anomaly Detection: 自動異常検知

## パフォーマンス最適化（Performance）

LangChainチームは、マルチエージェントアーキテクチャの性能特性について以下の点を指摘している。

**ツールの専門化がもたらす改善**:
- single-agentに多数のツールを与えるよりも、少数のツールに専門化したエージェントの方がtool selectionの精度が高い
- 各エージェントのプロンプトがタスクに最適化され、LLMの応答品質が向上

**並列実行の効果**:
- 独立したエージェントの並列実行により、逐次実行と比較してレイテンシが削減される
- LangGraphのfan-out/fan-inパターンがこれを直接サポートする

**コンテキストウィンドウの効率利用**:
- Collaborationパターンでは共有メッセージが蓄積しコンテキストが肥大化するが、Supervisorパターンでは各エージェントが独立したコンテキストを持つため効率的

## 運用での学び（Production Lessons）

LangChainチームのブログとエコシステムの実例から、以下の運用上の知見が得られる。

**Supervisorのプロンプト設計**:
- Supervisorが「どのエージェントに振り分けるか」の判断プロンプトは、明確な判断基準を記述する必要がある
- 曖昧なプロンプトはルーティング精度の低下を招く

**エラーハンドリング**:
- 個別エージェントの失敗時にSupervisorがフォールバック戦略を実行する設計が重要
- LangGraphのチェックポイント機能を活用して、失敗したステップから再開可能にする

**段階的な複雑化**:
- 最初はCollaborationパターンで始め、エージェント数増加に伴いSupervisor→Hierarchicalに移行する戦略が実用的

## 学術研究との関連（Academic Connection）

LangChainチームのブログは実装寄りだが、以下の学術研究と関連がある。

- **MARA (Singh et al., 2025, arXiv:2504.04603)**: LangGraphのstate machineで3エージェントを協調させるアーキテクチャ。本ブログのSupervisorパターンの具体的実装例
- **Agentic RAG Survey (Singh et al., 2025, arXiv:2501.09136)**: single-agent / multi-agent / hierarchicalの3パターンを学術的に分類。本ブログの3パターンと対応
- **AutoGen (Wu et al., 2023)**: Microsoftのマルチエージェントフレームワーク。ブログ内でLangGraphとの比較が言及されている

## まとめと実践への示唆

LangChainチームの公式ブログは、マルチエージェントワークフローの3設計パターンを体系的に整理した実践的ガイドである。Collaboration→Supervisor→Hierarchicalの段階的な複雑化パスは、プロダクション環境での段階的採用に適している。

Zenn記事で解説しているfunctional API（`@task`/`@entrypoint`）とCommand primitiveは、これらのパターンをより簡潔に実装するための最新APIであり、本ブログで紹介されているStateGraphベースの実装パターンの進化系と位置付けられる。特にCommand primitiveの`goto`パラメータによる動的遷移は、Supervisorパターンのルーティングロジックを宣言的に記述する手段を提供する。

## 参考文献

- **Blog URL**: [https://blog.langchain.com/langgraph-multi-agent-workflows/](https://blog.langchain.com/langgraph-multi-agent-workflows/)
- **LangGraph Documentation**: [https://docs.langchain.com/oss/python/langgraph/](https://docs.langchain.com/oss/python/langgraph/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/92f41b4fdc7b49](https://zenn.dev/0h_n0/articles/92f41b4fdc7b49)

---

:::message
本記事はAI（Claude Code）により自動生成されました。内容はブログ記事に基づいていますが、実際の利用時は公式ドキュメントもご確認ください。
:::
