---
layout: post
title: "Anthropic解説: Effective Context Engineering for AI Agents"
description: "プロンプトエンジニアリングの進化形「コンテキストエンジニアリング」の概念・設計パターン・実装手法をAnthropic公式ブログから詳解"
categories: [blog, tech_blog]
tags: [context-engineering, prompt-engineering, LLM, agents, anthropic]
date: 2026-02-21 12:00:00 +0900
source_type: tech_blog
source_domain: anthropic.com
source_url: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
zenn_article: 8d05ea9be7e0f3
zenn_url: https://zenn.dev/0h_n0/articles/8d05ea9be7e0f3
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

Anthropicの応用AIチームが公開した「Effective Context Engineering for AI Agents」は、プロンプトエンジニアリングの次の段階として「コンテキストエンジニアリング」を体系的に解説したブログ記事です。LLMに供給するトークンの設計・管理・最適化を包括的にカバーし、Context Rot問題、システムプロンプト設計、ツール設計、長期タスク対応（Compaction, Note-Taking, Sub-Agent）の3戦略を詳述しています。

この記事は [Zenn記事: 2026年版プロンプトテクニック大全：8手法の使い分けとコンテキスト設計](https://zenn.dev/0h_n0/articles/8d05ea9be7e0f3) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **組織**: Anthropic Applied AI Team（Prithvi Rajasekaran, Ethan Dixon, Carly Ryan, Jeremy Hadfield）
- **発表日**: 2025年

## 技術的背景（Technical Background）

### なぜコンテキストエンジニアリングが必要か

LLMの活用が「単発のプロンプト」から「長時間動作するエージェント」に発展するにつれ、従来のプロンプトエンジニアリング（最適な指示文を書く技術）だけでは不十分になってきました。

Anthropicは、プロンプトエンジニアリングとコンテキストエンジニアリングを以下のように区別しています。

**プロンプトエンジニアリング**: 効果的な指示文を書く離散的なタスク。特にシステムプロンプトの設計に焦点を当てる。

**コンテキストエンジニアリング**: LLMの推論時にどのような情報環境（トークンの集合）を構成するかを設計する継続的なキュレーション戦略。システム指示、ツール定義、MCP（Model Context Protocol）、外部データ、メッセージ履歴の全体を管理対象とする。

この転換の背景には、LangChainの2025年State of Agent Engineering調査で、本番AIエージェントの品質問題の多くがコンテキスト管理の不備に起因するという知見があります。

### Context Rot問題

Anthropicが指摘する「Context Rot」は、コンテキストウィンドウ内のトークン数が増加するにつれて、モデルが情報を正確に想起する能力が低下する現象です。

これはTransformerアーキテクチャの構造的特性に起因します。全てのトークンが他の全てのトークンにAttendする設計（$O(n^2)$の計算量）により、トークン数 $n$ が増加すると注意が分散します。

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

ここで $Q, K, V$ はクエリ、キー、バリュー行列です。コンテキスト長 $n$ が大きくなるほど、softmaxの分母が大きくなり、各トークンへの注意重みが薄まります。

Anthropicはコンテキストを「有限リソース」として捉え、人間のワーキングメモリの制約と類似した設計思想を提唱しています。追加の各トークンはモデルの「注意予算」を消費し、限界的リターンが逓減します。

## 実装アーキテクチャ（Architecture）

### 効果的なコンテキストの構成要素

Anthropicは、コンテキストを4つのレイヤーに分解して設計することを推奨しています。

#### 1. システムプロンプト

最適なシステムプロンプトは「適切な抽象度」を持つ必要があります。

**アンチパターン: 過度に複雑なハードコードロジック**
```
If user asks about pricing AND they are enterprise AND region is US:
  respond with plan A
Elif user asks about pricing AND they are enterprise AND region is EU:
  respond with plan B
...（100行以上の条件分岐）
```

**アンチパターン: 曖昧すぎる指示**
```
ユーザーの質問にうまく答えてください。
```

**ベストプラクティス: 具体的かつ柔軟な指示**
```xml
<background_information>
あなたはEnterprise Sales Assistantです。
顧客の業種・規模・地域に応じた最適なプランを提案します。
</background_information>

<instructions>
1. 顧客の要件を理解するために質問してください
2. 要件に基づいて1-2個のプランを推奨してください
3. 推奨理由を具体的に説明してください
</instructions>

<tool_guidance>
- pricing_api: 最新の価格情報取得に使用
- crm_lookup: 顧客情報の確認に使用
</tool_guidance>
```

XMLタグやMarkdownヘッダーで構造化し、「期待される振る舞いを完全に表現する最小限の情報セット」を目指します。

#### 2. ツール設計

ツールはエージェントと情報・行動空間の間の契約です。Anthropicが推奨するツール設計の原則は以下の5つです。

1. **トークン効率の良い情報を返す**: 不要なメタデータを含めない
2. **効率的なエージェント行動を促進**: ツール名と説明で意図を明確にする
3. **最小限の機能重複**: 類似ツールが多いと選択ミスの原因になる
4. **自己完結的でロバスト**: エラー時にも有用な情報を返す
5. **入力パラメータが明確**: 曖昧さのないパラメータ定義

#### 3. Few-shot例示

Few-shot promptingは依然として有効なベストプラクティスですが、Anthropicは「網羅的なエッジケースリスト」ではなく「多様で代表的な例示」を推奨しています。LLMにとって「例示は千の言葉に値する絵」であると述べています。

#### 4. 動的コンテキスト検索（Agentic Search）

事前計算（全データをプロンプトに含める）ではなく、「ジャストインタイム」で必要な情報を動的に取得する戦略です。

```python
# アンチパターン: 全データを事前ロード
context = load_entire_database()  # 100万トークン消費
response = llm(system_prompt + context + user_query)

# ベストプラクティス: 動的検索
# 軽量な参照情報のみ保持
metadata = {"file_paths": [...], "api_endpoints": [...]}
# 必要時にツールで取得
response = agent.run(user_query, tools=[search_tool, file_reader])
```

Claude Codeの例として、大規模データ分析でBashコマンド（head, tail, grep）を使って的確にデータを探索し、全データを文脈に読み込むことなく分析を完了するパターンが紹介されています。

## 長期タスク対応の3戦略

エージェントが数十分〜数時間にわたって動作する場合、単一のコンテキストウィンドウを超える情報量が発生します。Anthropicは3つの戦略を提示しています。

### 戦略1: Compaction（圧縮）

コンテキストウィンドウの限界が近づいた際に、会話内容を要約して再初期化する手法です。

```python
def compact_context(messages: list[dict]) -> list[dict]:
    """コンテキストの圧縮

    Args:
        messages: 現在のメッセージ履歴

    Returns:
        圧縮後のメッセージ履歴
    """
    summary = llm.summarize(
        messages,
        instructions=(
            "以下の情報を保持してください:\n"
            "1. アーキテクチャ上の意思決定\n"
            "2. 未解決のバグ\n"
            "3. 完了したタスクのリスト\n"
            "4. 次のステップ\n"
            "冗長な出力やログは省略してください。"
        )
    )

    return [
        {"role": "system", "content": original_system_prompt},
        {"role": "assistant", "content": f"[コンテキスト圧縮]\n{summary}"},
    ]
```

**重要なトレードオフ**: 圧縮が攻撃的すぎると重要な文脈を失い、保守的すぎるとコンテキストウィンドウを圧迫します。Anthropicは「まずリコール（再現率）を最大化し、反復的に精度を改善する」アプローチを推奨しています。

最も安全な軽量圧縮は「ツール結果のクリアリング」です。古いメッセージ履歴のツール出力（検索結果、ファイル内容等）を削除し、メッセージ自体は保持します。

### 戦略2: Structured Note-Taking（構造化ノート）

エージェントがコンテキストウィンドウの外部に定期的にノートを書き、後で取得する手法です。

Anthropicは、ClaudeがPokemonをプレイする事例を紹介しています。数千ステップにわたって「直近1,234ステップでRoute 1のポケモンを訓練中、ピカチュウは目標レベル10に対して8レベル獲得済み」のような正確な記録を外部メモリに保持し続けました。

```python
import json
from pathlib import Path

class AgentNotes:
    """エージェントの外部メモリ"""

    def __init__(self, notes_path: str = "agent_notes.json"):
        self.path = Path(notes_path)
        self.notes: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"decisions": [], "progress": [], "blockers": []}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.notes, indent=2, ensure_ascii=False))

    def add_decision(self, decision: str, rationale: str) -> None:
        """アーキテクチャ決定を記録"""
        self.notes["decisions"].append({
            "decision": decision,
            "rationale": rationale,
            "timestamp": datetime.now().isoformat()
        })
        self.save()

    def add_progress(self, task: str, status: str) -> None:
        """タスク進捗を記録"""
        self.notes["progress"].append({
            "task": task,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
        self.save()

    def get_summary(self) -> str:
        """ノートの要約を取得（コンテキストに注入用）"""
        decisions = "\n".join(
            f"- {d['decision']}" for d in self.notes["decisions"][-5:]
        )
        progress = "\n".join(
            f"- {p['task']}: {p['status']}" for p in self.notes["progress"][-10:]
        )
        return f"## 最近の決定\n{decisions}\n\n## 進捗\n{progress}"
```

Anthropicは、Claude Developer Platformでメモリツール（ファイルベースの外部知識ストレージ）をパブリックベータとして提供開始しています。

### 戦略3: Sub-Agent Architecture（サブエージェント）

専門化されたサブエージェントが個別タスクをクリーンなコンテキストで処理し、メインエージェントが戦略を統括する構成です。

各サブエージェントは数万トークンの探索を行いますが、メインエージェントには1,000-2,000トークンの凝縮サマリーのみを返します。

```python
async def multi_agent_research(query: str) -> str:
    """サブエージェントによる並列リサーチ"""

    # 各サブエージェントが独立に調査
    tasks = [
        research_agent("arxiv_search", query),
        research_agent("code_search", query),
        research_agent("documentation_search", query),
    ]
    summaries = await asyncio.gather(*tasks)

    # メインエージェントがサマリーを統合
    synthesis = main_agent.synthesize(
        query=query,
        summaries=summaries  # 各1000-2000トークン
    )
    return synthesis
```

### 戦略の選択指針

| 戦略 | 適したタスク | 長所 | 短所 |
|------|------------|------|------|
| **Compaction** | 長い対話（コードレビュー等） | 会話フローを維持 | 微妙な文脈を失うリスク |
| **Note-Taking** | マイルストーンのある反復開発 | 最小オーバーヘッドで永続メモリ | ノート設計の質に依存 |
| **Sub-Agent** | 並列探索（リサーチ・分析） | 関心の分離が明確 | 統合の設計が複雑 |

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

コンテキストエンジニアリングを実践するエージェントシステムのAWS構成です。

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $80-200 | Lambda + Bedrock + DynamoDB + S3 |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $500-1,200 | ECS Fargate + ElastiCache + S3 |
| **Large** | 300,000+ (10,000/日) | Container | $3,000-8,000 | EKS + Karpenter + ElastiCache |

**コスト削減テクニック**:
- Prompt Cachingでシステムプロンプト・ツール定義をキャッシュ（30-90%削減）
- Compaction実装でコンテキスト長を制御（トークン消費の抑制）
- S3にエージェントノートを保存（DynamoDBより安価な長期ストレージ）
- Sub-Agentには低コストモデル（Haiku）を使用、メインエージェントのみSonnet

**コスト試算の注意事項**: 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です。エージェントの対話ターン数やコンテキスト長により大きく変動します。最新料金は[AWS料金計算ツール](https://calculator.aws/)で確認してください。

### Terraformインフラコード

```hcl
# エージェントノート用S3バケット
resource "aws_s3_bucket" "agent_notes" {
  bucket = "agent-context-notes"
}

resource "aws_s3_bucket_lifecycle_configuration" "notes_lifecycle" {
  bucket = aws_s3_bucket.agent_notes.id
  rule {
    id     = "expire-old-notes"
    status = "Enabled"
    expiration { days = 30 }
  }
}

# コンテキストキャッシュ用ElastiCache
resource "aws_elasticache_replication_group" "ctx_cache" {
  replication_group_id = "agent-ctx-cache"
  description          = "Context Engineering cache"
  node_type            = "cache.t3.micro"
  num_cache_clusters   = 1
  engine               = "redis"
  engine_version       = "7.1"
}

resource "aws_lambda_function" "agent_handler" {
  filename      = "lambda.zip"
  function_name = "context-eng-agent"
  role          = aws_iam_role.lambda_agent.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 300
  memory_size   = 2048
  environment {
    variables = {
      BEDROCK_MODEL_ID   = "anthropic.claude-3-5-sonnet-20241022-v2:0"
      NOTES_BUCKET       = aws_s3_bucket.agent_notes.bucket
      CACHE_ENDPOINT     = aws_elasticache_replication_group.ctx_cache.primary_endpoint_address
      COMPACTION_THRESHOLD = "150000"
    }
  }
}
```

### コスト最適化チェックリスト

- [ ] Prompt Cachingでシステムプロンプト・ツール定義をキャッシュ
- [ ] Compaction閾値の最適化（コンテキスト長の80%で発動）
- [ ] Sub-Agentに低コストモデル使用（Haiku $0.25/MTok）
- [ ] エージェントノートをS3に保存（DynamoDBよりコスト効率的）
- [ ] ElastiCacheでコンテキスト圧縮結果をキャッシュ
- [ ] 不要なツール結果のクリアリング（最も安全な軽量圧縮）
- [ ] AWS Budgets設定（エージェントは対話ターン数でコスト変動大）
- [ ] CloudWatchアラーム（コンテキスト長・トークン使用量監視）
- [ ] 日次コストレポート（Cost Explorer自動レポート）
- [ ] 未使用S3ノートの自動削除（ライフサイクルポリシー30日）

## パフォーマンス最適化（Performance）

### コンテキスト長とレイテンシの関係

Anthropicの知見によると、コンテキスト長の増加はTime-to-First-Token（TTFT）とTotal Latencyの両方に影響します。

```
コンテキスト長    TTFT       Total Latency
10K tokens      ~0.5s      ~2s
50K tokens      ~1.5s      ~5s
100K tokens     ~3s        ~10s
200K tokens     ~6s        ~20s
```

Compactionにより不要なコンテキストを削減することで、レイテンシも改善できます。

### 実測値の目安

- **Compactionのオーバーヘッド**: 1回の圧縮に約2-5秒（圧縮対象の長さに依存）
- **ノート読み書き**: S3で約50-100ms、ElastiCacheで約1-5ms
- **Sub-Agent統合**: 並列実行で各サブエージェントのレイテンシの最大値＋統合処理500ms

## 運用での学び（Production Lessons）

Anthropicが共有したプロダクション環境での知見は以下の通りです。

1. **コンテキストは有限リソース**: 「最小限の高シグナルトークンセット」を維持することが原則。追加の各トークンは注意予算を消費する
2. **ハイブリッド戦略が有効**: 高速応答が必要な情報は事前取得、詳細調査はエージェント自律探索に任せるハイブリッドアプローチが実務的
3. **Compactionはリコール優先**: 最初は保守的に（多く残す）、反復的に精度を改善する。攻撃的すぎる圧縮は微妙だが重要な文脈を失う
4. **モデルが賢くなるほどエンジニアリングは減る**: 能力が向上するにつれ、より自律的に動作可能になるが、コンテキストを有限・貴重なものとして扱う原則は変わらない

## 学術研究との関連（Academic Connection）

コンテキストエンジニアリングは、以下の学術研究と密接に関連しています。

- **Needle-in-a-Haystack (Kamradt, 2023)**: 長コンテキストでの情報検索精度テスト。Context Rotの根拠となる実験
- **Lost in the Middle (Liu et al., 2023, Stanford)**: 長コンテキストでは中間位置の情報が検索されにくいというPosition Bias問題の発見
- **Compressive Transformers (Rae et al., 2020, DeepMind)**: コンテキスト圧縮のアーキテクチャレベルでの研究。Compaction戦略の学術的基盤

## まとめと実践への示唆

Anthropicのコンテキストエンジニアリングは、「何を聞くか」から「LLMにどんな情報環境を用意するか」へのパラダイムシフトを提唱しています。特にエージェントシステムの構築においては、コンテキストの設計・管理・最適化が品質の決定的要因となります。

実践では、まずシステムプロンプトとツール設計から始め、長期タスクに進んだ際にCompaction→Note-Taking→Sub-Agentの順で戦略を追加するのが効率的です。

## 参考文献

- **Blog URL**: [https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Claude Developer Platform**: Memory and Context Management Cookbook
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/8d05ea9be7e0f3](https://zenn.dev/0h_n0/articles/8d05ea9be7e0f3)
