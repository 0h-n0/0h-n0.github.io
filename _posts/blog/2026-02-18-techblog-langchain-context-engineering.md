---
layout: post
title: "LangChain公式ブログ解説: エージェントのためのContext Engineering — Write/Select/Compress/Isolate実装ガイド"
description: "LangChain/LangGraphの実装コンポーネントと対応づけたContext Engineeringの4戦略パターンを詳細解説"
categories: [blog, tech_blog]
tags: [context-engineering, llm, agent, langchain, langgraph, rag, multi-agent]
date: 2026-02-18 13:00:00 +0900
source_type: tech_blog
source_domain: blog.langchain.com
source_url: https://blog.langchain.com/context-engineering-for-agents/
zenn_article: e918777f3aa87c
zenn_url: https://zenn.dev/0h_n0/articles/e918777f3aa87c
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

LangChain公式ブログの「Context Engineering for Agents」は、エージェントのコンテキスト管理を**Write / Select / Compress / Isolate**の4戦略に体系化し、各戦略のLangGraph実装コンポーネントとの対応を詳細に解説したガイドである。Drew Breunigの**コンテキスト失敗モード**（Poisoning / Distraction / Confusion / Clash）の分類を紹介し、LangSmithによるオブザーバビリティを前提とした実践的な最適化フローを提示している。Anthropicのマルチエージェント研究やCursorのRAG実装など、プロダクション事例も豊富に引用されている。

この記事は [Zenn記事: LLMエージェントのContext Engineering実践：4戦略でトークンコスト50%削減](https://zenn.dev/0h_n0/articles/e918777f3aa87c) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（LangChain Blog）
- **URL**: [https://blog.langchain.com/context-engineering-for-agents/](https://blog.langchain.com/context-engineering-for-agents/)
- **組織**: LangChain Inc.（LangGraph, LangSmith開発元）
- **発表日**: 2025年

## 技術的背景（Technical Background）

### Context Engineeringの位置づけ

LangChainは、Context Engineeringを「**エージェントの軌跡（trajectory）の各ステップで、コンテキストウィンドウに適切な情報を配置する技術と科学**」と定義する。これは単なるプロンプト最適化ではなく、エージェントの全ライフサイクルにわたるデータフロー設計である。

### コンテキスト失敗モード

Drew Breunigが整理したコンテキストの4つの失敗モードは、設計時の重要なチェックリストとなる：

| 失敗モード | 説明 | 例 |
|-----------|------|-----|
| **Poisoning**（汚染） | ハルシネーションがコンテキストに侵入 | エージェントの誤った推論が次のステップの入力に |
| **Distraction**（注意散漫） | コンテキストがモデルの学習済み知識を上書き | 無関係なドキュメントが正しい推論を妨害 |
| **Confusion**（混乱） | 余分な情報がレスポンスに影響 | 古いAPI仕様と新しい仕様が混在 |
| **Clash**（衝突） | 矛盾するコンテキスト要素 | 異なるツールが矛盾する情報を返す |

これらの失敗モードは、「情報が足りない」だけでなく「**情報が多すぎる・間違っている**」ことがエージェントの性能を劣化させることを示している。

## 実装アーキテクチャ（Architecture）

### Write戦略: 外部への書き出し

エージェントが重要な情報をコンテキストウィンドウの外に永続化する戦略。

#### Scratchpad（スクラッチパッド）

タスク実行中にエージェントが参照するための一時的なノート。ファイルやランタイム状態オブジェクトへの書き込みで実装する。

**プロダクション事例**: Anthropicのマルチエージェント研究者は、トークン制限に近づいたときに計画をメモリに保存する。

```python
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph

# LangGraphでのScratchpad実装
store = InMemoryStore()

def save_scratchpad(state, config, store):
    """タスク進捗をスクラッチパッドに保存

    Args:
        state: 現在のグラフ状態
        config: LangGraph設定
        store: メモリストア
    """
    namespace = ("scratchpad", config["configurable"]["thread_id"])
    store.put(
        namespace=namespace,
        key="current_progress",
        value={
            "completed_steps": state["completed_steps"],
            "key_decisions": state["key_decisions"],
            "pending_tasks": state["pending_tasks"],
        },
    )
```

#### Memory（メモリ）

セッションをまたぐ長期記憶。短期記憶（セッション内）と長期記憶（セッション間）に分類される。

**長期記憶の実装パターン**:

```python
from langgraph.store.memory import InMemoryStore

class AgentLongTermMemory:
    """エージェントの長期記憶管理

    セッション間で知識を保持し、反省と定期的合成で
    記憶の質を維持する。
    """

    def __init__(self):
        self.store = InMemoryStore()

    def reflect_and_store(
        self,
        session_history: list[dict],
        user_id: str,
    ) -> None:
        """セッション終了時に反省して記憶を保存

        Args:
            session_history: セッションの全履歴
            user_id: ユーザー識別子
        """
        # LLMで反省を生成
        reflection = llm.invoke(
            f"このセッションから学んだ重要な知見を3つ抽出: "
            f"{session_history}"
        )

        namespace = ("long_term", user_id)
        self.store.put(
            namespace=namespace,
            key=f"reflection_{datetime.now().isoformat()}",
            value={"content": reflection, "source": "session_reflection"},
        )

    def periodic_synthesis(self, user_id: str) -> str:
        """定期的にメモリを統合・合成

        Args:
            user_id: ユーザー識別子

        Returns:
            合成された知識の要約
        """
        namespace = ("long_term", user_id)
        all_memories = self.store.search(namespace=namespace)

        synthesis = llm.invoke(
            f"以下の記憶を統合し、重要なパターンを抽出: "
            f"{all_memories}"
        )

        # 合成結果を保存し、元の記憶を圧縮
        self.store.put(
            namespace=namespace,
            key="synthesis_latest",
            value={"content": synthesis, "source": "periodic_synthesis"},
        )

        return synthesis
```

**プロダクション事例**: ChatGPT、Cursor、Windsurfは、インタラクションからメモリを自動生成する。

### Select戦略: 動的な検索と選択

必要な情報だけをコンテキストに動的にロードする戦略。

#### RAG応用: ツール選択の最適化

LangGraphのBigtoolは、RAGをツール説明に適用し、大量のツールから関連ツールを選択する：

$$
\text{SelectedTools} = \text{TopK}\left(\text{sim}(\mathbf{e}_q, \mathbf{e}_{t_i}) \mid t_i \in \mathcal{T}\right)
$$

ここで$\mathbf{e}_q$はクエリの埋め込み、$\mathbf{e}_{t_i}$はツール$t_i$の説明の埋め込み、$\mathcal{T}$は全ツール集合。

この手法でツール選択精度が**3倍に向上**することが報告されている。

#### コードエージェントRAG

Windsurfのアプローチは、複数の検索手法を組み合わせる：

1. **AST解析**: 構文木からシンボル（関数名、クラス名）を抽出
2. **セマンティックチャンキング**: 意味的な境界でコードを分割
3. **埋め込み検索**: ベクトル類似度による候補取得
4. **grep/ファイル検索**: 正確なパターンマッチング
5. **知識グラフ検索**: エンティティ間の関係を辿る
6. **リランキング**: 候補を統合的にスコアリング

```python
class CodeAgentRAG:
    """コードエージェント向けの多段階RAG

    複数の検索手法を組み合わせ、コードベースから
    最も関連性の高いコンテキストを取得する。
    """

    def retrieve(self, query: str, top_k: int = 10) -> list[CodeChunk]:
        """多段階検索でコードチャンクを取得

        Args:
            query: 検索クエリ
            top_k: 返す結果数

        Returns:
            関連度の高いコードチャンクのリスト
        """
        # Stage 1: 複数ソースから候補取得
        candidates = []
        candidates.extend(self.ast_search(query))
        candidates.extend(self.embedding_search(query))
        candidates.extend(self.grep_search(query))
        candidates.extend(self.graph_search(query))

        # Stage 2: 重複除去
        unique = deduplicate(candidates)

        # Stage 3: リランキング
        scored = self.reranker.score(query, unique)
        scored.sort(key=lambda x: x.score, reverse=True)

        return scored[:top_k]
```

### Compress戦略: コンテキストの圧縮

#### 要約（Summarization）

LangChainは複数の要約パターンを提示する：

**再帰的要約**: 長い履歴を段階的に要約する
**階層的要約**: タスクフェーズごとに独立に要約する
**自動Compact**: コンテキスト使用率95%で自動トリガー

```python
from langgraph.graph import StateGraph

def auto_compact_node(state):
    """コンテキスト使用率に基づく自動圧縮ノード

    LangGraphのノードとして実装し、
    使用率が95%を超えたら圧縮を実行する。
    """
    messages = state["messages"]
    usage = estimate_tokens(messages) / MAX_TOKENS

    if usage > 0.95:
        # 古いメッセージを要約
        old_messages = messages[:-10]  # 直近10件は保持
        recent_messages = messages[-10:]

        summary = llm.invoke(
            f"以下の会話を要約してください。"
            f"アーキテクチャ決定と未解決の問題を保持: "
            f"{old_messages}"
        )

        return {
            "messages": [
                {"role": "system", "content": f"Previous context summary: {summary}"},
                *recent_messages,
            ]
        }

    return state  # 閾値未満ならそのまま
```

#### トリミング

**ハードコードヒューリスティクス**: 古いメッセージを機械的に削除
**学習済みプルーナー**: Provenceのようなモデルで、QAタスクに必要な情報を保持しながら不要部分を削除

#### ツール出力の後処理

特定のツール出力を圧縮する専用ノードを配置：

```python
def compress_tool_output(state):
    """ツール出力を圧縮するLangGraphノード

    大きなツール出力（ファイル読み取り、検索結果等）を
    タスク関連の情報のみに圧縮する。
    """
    last_message = state["messages"][-1]

    if (hasattr(last_message, "tool_output")
            and len(last_message.content) > 3000):
        compressed = llm.invoke(
            f"以下のツール出力から、現在のタスクに関連する"
            f"情報のみを抽出: {last_message.content}"
        )
        last_message.content = compressed

    return state
```

### Isolate戦略: コンテキストの分離

#### マルチエージェントアーキテクチャ

各サブエージェントが専門のコンテキストウィンドウで動作する：

```python
from langgraph.prebuilt import create_react_agent

# メインエージェント（コーディネーター）
main_agent = create_react_agent(
    model=llm,
    tools=[delegate_to_research, delegate_to_code, delegate_to_test],
    system_prompt="タスクを分解し、専門エージェントに委譲する",
)

# 研究エージェント（クリーンなコンテキスト）
research_agent = create_react_agent(
    model=llm,
    tools=[web_search, read_doc],
    system_prompt="技術的な調査を行い、結果を要約する",
)

# コーディングエージェント（クリーンなコンテキスト）
code_agent = create_react_agent(
    model=llm,
    tools=[read_file, write_file, run_test],
    system_prompt="コードの実装・修正を行う",
)
```

**トレードオフ**: マルチエージェントはトークン使用量が**最大15倍**に増加する可能性がある（Anthropic研究）。しかし、各エージェントのコンテキストはクリーンで、Context Rotの影響を受けにくい。

#### 環境ベースの分離

**CodeAgent パターン**: ツール呼び出しをサンドボックス（E2B、Pyodide）で実行し、状態をオブジェクトとして保持する。LLMのトークン化を回避し、バイナリアセット（画像、音声）も効率的に扱える。

```python
# E2Bサンドボックスでの実行
from e2b_code_interpreter import Sandbox

sandbox = Sandbox()

# コードの実行結果は状態オブジェクトとして保持
# LLMのコンテキストには結果の要約のみ挿入
result = sandbox.run_code("""
import pandas as pd
df = pd.read_csv('data.csv')
print(df.describe())
""")

# 要約のみコンテキストに追加
context_update = f"DataFrameの統計情報: {result.output[:500]}"
```

#### State Schemaの分離

LangGraphの状態スキーマで、LLMに公開するフィールドを選択的に制御する：

```python
from langgraph.graph import StateGraph
from typing import Annotated

class AgentState(TypedDict):
    # LLMに公開するフィールド
    messages: Annotated[list, add_messages]
    current_task: str

    # LLMに公開しないフィールド（内部状態）
    _token_count: int
    _tool_call_history: list[dict]
    _compression_state: dict
```

## パフォーマンス最適化（Performance）

### LangSmithによるオブザーバビリティ

LangChainは「**最適化の前にオブザーバビリティを確立する**」ことを強調する。

**計測すべきメトリクス**:
1. **トークン使用量**: 各ステップでの入力/出力トークン数
2. **軌跡長**: タスク完了までのターン数
3. **ツール呼び出し頻度**: 各ツールの使用パターン
4. **コンテキスト使用率**: 各ステップでのコンテキスト占有率

### 最適化のフロー

$$
\text{Observe} \rightarrow \text{Measure} \rightarrow \text{Identify Bottleneck} \rightarrow \text{Apply Strategy} \rightarrow \text{Evaluate Impact}
$$

1. LangSmithでトレースを収集
2. トークン使用パターンを分析
3. ボトルネックを特定（どのツール出力が最もトークンを消費しているか）
4. 適切な戦略を適用（Compress? Isolate? Select?）
5. 性能影響を評価（精度低下がないか）

## 運用での学び（Production Lessons）

### プロダクション事例からの教訓

1. **Cursor**: AST + セマンティック検索 + grep のハイブリッドRAGで、コードベースの関連コンテキストを効率的に取得
2. **Windsurf**: 6種類の検索手法を組み合わせたコードエージェントRAG
3. **ChatGPT Memory**: ユーザーインタラクションから自動的にメモリを生成
4. **Anthropic Multi-Agent**: サブエージェントで探索、メインで統合。15倍のトークン増でも整合性維持

### よくある間違い

| 間違い | 正しいアプローチ |
|--------|--------------|
| 全データをプリロード | JIT（Just-in-Time）で動的取得 |
| 単一の巨大コンテキスト | マルチエージェントで分離 |
| 固定長トランケーション | 意味ベースの圧縮 |
| オブザーバビリティなしの最適化 | まずLangSmithで計測 |

## 学術研究との関連（Academic Connection）

### 引用された学術研究

- **MemGPT** (Packer et al., 2023): Write戦略のOS風メモリ管理の学術的基盤
- **RAG** (Lewis et al., 2020): Select戦略の学術的基盤。LangChainのRAGパイプラインの原型
- **Provence** (2024): Compress戦略の学習済みプルーナー。QAタスクでの情報選択
- **Generative Agents** (Park et al., 2023): Write戦略の反省メカニズム

### LangGraphが実現する学術→実装の橋渡し

| 学術概念 | LangGraph実装 |
|---------|-------------|
| 短期メモリ | `Checkpointer`（スレッドスコープ） |
| 長期メモリ | `InMemoryStore` / `InMemoryVectorStore` |
| RAG | `similarity_search()` |
| マルチエージェント | Supervisor / Swarm ライブラリ |
| 構造化ノート | `store.put()` / `store.search()` |
| コンテキスト圧縮 | メッセージリスト要約/トリミングユーティリティ |
| サンドボックス実行 | E2B統合 |

## まとめと実践への示唆

### 核心メッセージ

LangChainのガイドは、Context Engineeringを4つの戦略（Write / Select / Compress / Isolate）に整理し、各戦略のLangGraph実装コンポーネントとの対応を明確にした。特に重要なのは、**オブザーバビリティを前提とした段階的最適化**のアプローチと、**コンテキスト失敗モード**（Poisoning / Distraction / Confusion / Clash）への対処である。

### 実践への適用ポイント

1. **LangSmithで計測を開始**: 最適化の前にベースラインを確立する
2. **Write戦略から始める**: `InMemoryStore`でスクラッチパッドとメモリを実装する
3. **Select戦略でRAGを活用**: 大規模なツール集合にはBigtoolのセマンティック検索を導入する
4. **Compress戦略を段階的に適用**: 自動Compact（95%閾値）とツール出力後処理を組み合わせる
5. **Isolate戦略は必要に応じて**: トークンコストの15倍増を許容できる場合にマルチエージェントを導入する

### Zenn記事との対応

Zenn記事で紹介した4戦略のフレームワークは、このLangChainブログに基づいている。本記事では各戦略の**LangGraph実装パターン**と**プロダクション事例**を深掘りし、実際にコードを書く際の具体的な実装方法を提示した。

## 参考文献

- **Blog URL**: [https://blog.langchain.com/context-engineering-for-agents/](https://blog.langchain.com/context-engineering-for-agents/)
- **Related**: [LangChain - Context Management for Deep Agents](https://blog.langchain.com/context-management-for-deepagents/)
- **Related**: [LangChain - The Rise of Context Engineering](https://blog.langchain.com/the-rise-of-context-engineering/)
- **LangGraph Docs**: [LangGraph Memory](https://langchain-ai.github.io/langgraph/concepts/memory/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/e918777f3aa87c](https://zenn.dev/0h_n0/articles/e918777f3aa87c)
