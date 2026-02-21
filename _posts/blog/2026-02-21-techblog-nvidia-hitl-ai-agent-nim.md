---
layout: post
title: "NVIDIA解説: LangGraphによるHuman-in-the-Loop AIエージェントの構築パターン"
description: "NVIDIA NIMマイクロサービスとLangGraphを使ったHITL承認フロー付きAIエージェントの設計・実装を詳細解説"
categories: [blog, tech_blog]
tags: [HITL, LangGraph, NVIDIA, AI-agent, human-in-the-loop, langgraph, rag, python]
date: 2026-02-21 10:00:00 +0900
source_type: tech_blog
source_domain: developer.nvidia.com
source_url: https://developer.nvidia.com/blog/build-your-first-human-in-the-loop-ai-agent-with-nvidia-nim/
zenn_article: e4a4b18478c692
zenn_url: https://zenn.dev/0h_n0/articles/e4a4b18478c692
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

NVIDIAの技術ブログ「Build Your First Human-in-the-Loop AI Agent with NVIDIA NIM」は、LangGraphを使ったHITL（Human-in-the-Loop）承認フロー付きAIエージェントの構築パターンを実装レベルで解説している。2つの専門エージェント（Content Creator + Digital Artist）をLangGraphのStateGraphで統合し、人間の意思決定者が各出力を承認・却下・修正する反復ワークフローを構築している。

この記事は [Zenn記事: LangGraphマルチソースRAGの本番構築：権限制御×HITLで社内検索を安全運用](https://zenn.dev/0h_n0/articles/e4a4b18478c692) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://developer.nvidia.com/blog/build-your-first-human-in-the-loop-ai-agent-with-nvidia-nim/](https://developer.nvidia.com/blog/build-your-first-human-in-the-loop-ai-agent-with-nvidia-nim/)
- **組織**: NVIDIA（Developer Technical Blog）
- **著者**: Zenodia Charpy, Gordana Neskovic
- **発表日**: 2024年11月21日

## 技術的背景（Technical Background）

### なぜHITLが必要なのか

LLMエージェントの自律性が向上するにつれ、「エージェントが完全に自律的に行動する」ことのリスクが顕在化している。特にエンタープライズ環境では、以下の理由からHITLが不可欠となる。

1. **コンプライアンスリスク**: 機密情報を含む回答が自動的にエンドユーザーに届くリスク
2. **品質保証**: ハルシネーションや不正確な情報のフィルタリング
3. **責任の所在**: 「AIが生成した回答」に対する組織の説明責任

NVIDIAのブログは、これらの課題に対して「人間を意思決定の中心に置く」設計パターンを具体的なコードとともに提示している。これはZenn記事で紹介されているLangGraphの`interrupt`プリミティブによるHITLフローと同じ設計思想を共有している。

### 学術研究との関連

本ブログの設計パターンは、以下の学術的背景に基づいている。

- **Confidence-based escalation**: LLMの出力確信度が閾値以下の場合に人間にエスカレーションする手法（RAGの文脈ではgraded_docs_countが少ない場合に対応）
- **Structured approval interfaces**: 人間が「承認」「却下」「修正」の選択肢から判断できる構造化インターフェース
- **Audit logging**: すべてのエージェント行動と人間の判断を記録し、事後監査を可能にする仕組み

## 実装アーキテクチャ（Architecture）

### システム構成

NVIDIAのブログでは、以下の技術スタックでHITLエージェントを構築している。

| コンポーネント | 技術 | 役割 |
|------------|------|------|
| LLMプロバイダー | NVIDIA NIM マイクロサービス | LLM推論の高速実行 |
| ベースモデル | Llama 3.1 405B, Mistral 7B, SDXL-turbo | テキスト生成、画像生成 |
| フレームワーク | LangChain + ChatNVIDIA | LLMとのインタラクション |
| オーケストレーション | LangGraph | エージェントワークフロー管理 |
| 出力フォーマット | Pydantic BaseModel | 構造化出力の型安全性 |

### 2エージェント構成

```python
from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA

class PromotionalContent(BaseModel):
    """Content Creatorエージェントの構造化出力"""
    title: str = Field(description="プロモーションタイトル")
    message: str = Field(description="本文メッセージ")
    hashtags: list[str] = Field(description="ハッシュタグリスト")

class ContentCreatorAgent:
    """プロモーションコンテンツを生成するエージェント

    Llama 3.1 405Bをバックエンドとし、
    with_structured_outputでPydanticモデルに型安全に出力する。
    """

    def __init__(self):
        self.llm = ChatNVIDIA(
            model="meta/llama-3.1-405b-instruct",
        ).with_structured_output(PromotionalContent)

    def generate(self, product_info: str) -> PromotionalContent:
        """プロモーションコンテンツを生成

        Args:
            product_info: 製品情報テキスト

        Returns:
            構造化されたプロモーションコンテンツ
        """
        return self.llm.invoke(
            f"以下の製品情報から魅力的なプロモーションを作成: {product_info}"
        )
```

**`with_structured_output`の重要性**: Zenn記事のソースルーター（`SourceRoute`モデル）と同じパターン。LLMの出力をPydanticモデルで型安全にバリデーションすることで、ダウンストリーム処理の信頼性を担保する。

### LangGraphによるHITLワークフロー

NVIDIAブログのLangGraphワークフローは3つの主要ノードで構成される。

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal


class AgentState(TypedDict):
    """エージェントワークフローの状態定義"""
    task_description: str
    selected_agent: str
    agent_output: dict
    human_decision: str  # "approve" | "reject" | "edit"
    iteration_count: int


def human_assign_to_agent(state: AgentState) -> AgentState:
    """人間がタスクを適切なエージェントに割り当てる

    HITLの第1段階: タスク選択
    - Content Creator: テキストコンテンツ生成
    - Digital Artist: 画像生成
    """
    # 実装: UIから人間の選択を受け取る
    selected = get_human_selection(
        options=["content_creator", "digital_artist"],
        context=state["task_description"],
    )
    return {"selected_agent": selected}


def agent_execute_task(state: AgentState) -> AgentState:
    """選択されたエージェントがタスクを実行"""
    if state["selected_agent"] == "content_creator":
        output = content_creator.generate(state["task_description"])
    else:
        output = digital_artist.generate(state["task_description"])

    return {
        "agent_output": output,
        "iteration_count": state["iteration_count"] + 1,
    }


def human_review(state: AgentState) -> AgentState:
    """人間が出力をレビューし、承認・却下・修正を判断

    HITLの第2段階: 品質ゲート
    - approve: 出力を確定し、次のステップへ
    - reject: エージェントを再実行（フィードバック付き）
    - edit: 人間が直接修正
    """
    decision = get_human_review(
        output=state["agent_output"],
        options=["approve", "reject", "edit"],
    )
    return {"human_decision": decision}


# グラフ構築
graph = StateGraph(AgentState)
graph.add_node("assign", human_assign_to_agent)
graph.add_node("execute", agent_execute_task)
graph.add_node("review", human_review)

graph.set_entry_point("assign")
graph.add_edge("assign", "execute")
graph.add_edge("execute", "review")

# 条件付きエッジ: 承認されるまで反復
def route_after_review(state: AgentState) -> str:
    if state["human_decision"] == "approve":
        return END
    return "assign"  # 再度エージェント選択から

graph.add_conditional_edges("review", route_after_review)
app = graph.compile()
```

### Zenn記事のinterruptプリミティブとの比較

NVIDIAブログのHITL実装と、Zenn記事で紹介されているLangGraphの`interrupt`プリミティブの比較は以下の通りである。

| 特性 | NVIDIAブログ | Zenn記事（interrupt） |
|------|------------|---------------------|
| 中断メカニズム | UIポーリング | `interrupt()`関数 |
| 状態永続化 | メモリ内 | PostgresチェックポインターDB |
| 再開方法 | 明示的コールバック | `Command(resume=decision)` |
| 適用場面 | 対話型UI | Slack Bot/API |
| タイムアウト | なし | 設定可能 |

Zenn記事の`interrupt`プリミティブは、NVIDIAブログのパターンをさらに発展させ、**非同期の承認フロー**（Slack通知→数時間後に承認→グラフ再開）に対応している点が大きな違いである。

```python
# Zenn記事のinterruptパターン（NVIDIAブログの発展形）
from langgraph.types import interrupt, Command

def check_approval(state: RAGState) -> RAGState:
    """NVIDIAブログのhuman_reviewに対応するノード

    違い:
    - interrupt()で実行を一時停止
    - 状態はPostgreSQLに永続化
    - 数時間後でも再開可能
    """
    decision = interrupt({
        "type": "approval_required",
        "reason": "機密トピック" if is_sensitive else "低確信度",
        "answer_preview": state["messages"][-1]["content"][:500],
    })

    if decision["action"] == "approve":
        return {"approval_status": "approved"}
    elif decision["action"] == "reject":
        return {"approval_status": "rejected"}
    elif decision["action"] == "edit":
        return {"approval_status": "approved",
                "messages": [{"role": "assistant",
                              "content": decision["edited_answer"]}]}
```

## パフォーマンス最適化（Performance）

### NIM マイクロサービスの利点

NVIDIA NIMは推論パフォーマンスを最大化するマイクロサービスである。

- **レイテンシ**: Llama 3.1 405BでのTime-to-First-Token（TTFT）が標準デプロイ比で最大2.5倍高速
- **スループット**: TensorRT-LLM最適化によりバッチ推論の効率が向上
- **スケーラビリティ**: Kubernetesネイティブで水平スケーリングに対応

### HITL承認フローのレイテンシ影響

HITLを導入すると、エンドツーエンドのレスポンスタイムに人間の判断時間が加わる。NVIDIAブログの設計では以下のパターンで影響を最小化している。

1. **非同期処理**: エージェントの実行結果をキューに入れ、人間の承認を非同期で待つ
2. **バッチ承認**: 複数の出力をまとめてレビューするUIを提供
3. **自動承認ルール**: 確信度が高い出力は自動承認し、HITLを必要なケースに限定

**トレードオフ分析**: Zenn記事では「HITL承認が全体の20%を超える場合は機密キーワードリストを精査」と述べているが、NVIDIAの実装では反復回数（`iteration_count`）で上限を設け、3回以上の修正要求で自動エスカレーションする設計も考えられる。

## 運用での学び（Production Lessons）

### HITL導入時の障害パターンと対策

| 障害パターン | 原因 | 対策 |
|------------|------|------|
| 承認待ちキューの滞留 | レビュー担当者の不足 | 自動承認ルールの閾値調整、担当者ローテーション |
| 反復ループの無限化 | 曖昧な修正指示 | 最大反復回数の設定（3回推奨） |
| 状態ロスト | プロセス再起動 | PostgreSQLチェックポインター使用 |
| レイテンシ増大 | 全クエリにHITL適用 | 確信度ベースの選択的HITL |

### モニタリング戦略

HITL付きエージェントでは、以下のメトリクスを追跡することが重要である。

```python
import json
import logging

logger = logging.getLogger("hitl.metrics")

def log_hitl_decision(
    session_id: str,
    decision: str,
    iteration: int,
    latency_ms: float,
) -> None:
    """HITL判断の構造化ログ

    Args:
        session_id: セッション識別子
        decision: 人間の判断（approve/reject/edit）
        iteration: 反復回数
        latency_ms: 判断にかかった時間
    """
    logger.info(json.dumps({
        "event": "hitl_decision",
        "session_id": session_id,
        "decision": decision,
        "iteration": iteration,
        "latency_ms": latency_ms,
        "ts": "2026-02-21T10:00:00+09:00",
    }))
```

**監視すべきメトリクス**:

| メトリクス | 閾値 | アクション |
|-----------|------|-----------|
| 承認率 | < 60% | プロンプト品質の見直し |
| 平均反復回数 | > 2.0 | エージェントの出力品質改善 |
| 承認待ち時間（P95） | > 30分 | 自動承認ルールの拡大 |
| HITL適用率 | > 20% | 確信度閾値の調整 |

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $80-200 | Lambda + Bedrock + SQS |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $400-1,000 | Lambda + ECS + ElastiCache + SQS |
| **Large** | 300,000+ (10,000/日) | Container | $2,500-6,000 | EKS + Karpenter + SQS + SNS |

**Small構成のHITL固有サービス**:
- **SQS**: 承認待ちキュー（$5/月）— 人間のレビュー待ち状態をキューで管理
- **DynamoDB**: HITL状態管理（$10/月）— セッション状態の永続化
- **SNS**: 承認リクエスト通知（$5/月）— Slack/メールへの通知
- **API Gateway WebSocket**: リアルタイム承認UI（$10/月）

**コスト削減テクニック**:
- 確信度ベースの選択的HITL適用でレビュー工数を80%削減
- SQSのlong pollingで不要なAPI呼び出しを削減
- DynamoDBのTTLでセッションデータを自動クリーンアップ

**コスト試算の注意事項**:
- 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値
- HITL導入による人件費（レビュー担当者の時間）は別途考慮が必要
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認

### Terraformインフラコード

**HITL承認フロー用 SQS + SNS構成**

```hcl
# --- SQS: 承認待ちキュー ---
resource "aws_sqs_queue" "hitl_approval_queue" {
  name                       = "hitl-approval-queue"
  visibility_timeout_seconds = 3600  # 1時間（レビュー時間を考慮）
  message_retention_seconds  = 86400 # 24時間で自動削除

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.hitl_dlq.arn
    maxReceiveCount     = 3  # 3回リトライ後にDLQへ
  })
}

resource "aws_sqs_queue" "hitl_dlq" {
  name                      = "hitl-approval-dlq"
  message_retention_seconds = 604800  # 7日間保持
}

# --- SNS: 承認リクエスト通知 ---
resource "aws_sns_topic" "hitl_notification" {
  name = "hitl-approval-notification"
}

resource "aws_sns_topic_subscription" "slack_webhook" {
  topic_arn = aws_sns_topic.hitl_notification.arn
  protocol  = "https"
  endpoint  = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
}

# --- DynamoDB: HITL状態管理 ---
resource "aws_dynamodb_table" "hitl_sessions" {
  name         = "hitl-session-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "session_id"

  attribute {
    name = "session_id"
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

# HITL承認キューの滞留監視
cloudwatch.put_metric_alarm(
    AlarmName='hitl-queue-depth-high',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='ApproximateNumberOfMessagesVisible',
    Namespace='AWS/SQS',
    Period=300,
    Statistic='Average',
    Threshold=50,  # 50件以上の承認待ち
    AlarmDescription='HITL承認キューが滞留しています。レビュー担当者を追加してください。',
    Dimensions=[{'Name': 'QueueName', 'Value': 'hitl-approval-queue'}],
    AlarmActions=['arn:aws:sns:ap-northeast-1:123456789:ops-alerts'],
)
```

### コスト最適化チェックリスト

- [ ] 確信度ベースの選択的HITL適用（全体の20%以下に）
- [ ] SQS long pollingでポーリングコスト削減
- [ ] DynamoDB TTLでセッション自動削除（24時間）
- [ ] SNSフィルタリングで不要通知を削減
- [ ] 自動承認ルールの定期見直し（月次）

## 学術研究との関連（Academic Connection）

NVIDIAのHITL実装パターンは、以下の学術研究と密接に関連している。

- **Human-AI Collaboration in Decision Making** (Bansal et al., 2021): AIの確信度を人間に提示し、適切な判断を支援する「complementary performance」の概念。NVIDIAの設計は、エージェント出力を人間に明示的に提示する点でこの研究を実装に落とし込んでいる
- **Selective Prediction** (Geifman & El-Yaniv, 2017): モデルが「予測しない」選択肢を持つことで、不確実な場合に人間に委ねるメカニズム。NVIDIAの確信度ベースHITLはこの概念の応用

## まとめと実践への示唆

NVIDIAブログが示すHITLエージェントの設計パターンは、LangGraphのStateGraphで自然に実装可能である。特に重要な知見は以下の3点である。

1. **人間を中心に据える設計**: エージェントの自律性と人間の判断権限のバランスを明示的に設計する
2. **構造化出力の活用**: Pydanticモデルで出力を型安全にし、レビューインターフェースの構築を容易にする
3. **反復ワークフロー**: 承認されるまで反復する設計により、出力品質を段階的に向上させる

Zenn記事の`interrupt`プリミティブは、NVIDIAブログのパターンを非同期・永続化に発展させたものであり、Slack Bot連携やバッチ承認など、エンタープライズ環境に適した拡張が可能である。

## 参考文献

- **Blog URL**: [https://developer.nvidia.com/blog/build-your-first-human-in-the-loop-ai-agent-with-nvidia-nim/](https://developer.nvidia.com/blog/build-your-first-human-in-the-loop-ai-agent-with-nvidia-nim/)
- **LangGraph HITL Documentation**: [https://docs.langchain.com/oss/python/langchain/human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/e4a4b18478c692](https://zenn.dev/0h_n0/articles/e4a4b18478c692)
