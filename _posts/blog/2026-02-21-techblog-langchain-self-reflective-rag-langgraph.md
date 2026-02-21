---
layout: post
title: "ブログ解説: LangChain Self-Reflective RAG with LangGraph — 条件付きエッジで実現する自己修正型検索生成"
description: "LangChainブログのSelf-Reflective RAG実装をLangGraphの状態グラフ設計から詳細解説し、Document Grading・Query Rewriting・Answer Generationの実装パターンを分析します"
categories: [blog, techblog]
tags: [LangGraph, RAG, Self-RAG, Corrective-RAG, LangChain, state-graph]
date: 2026-02-21 13:00:00 +0900
source_type: techblog
source_url: https://blog.langchain.com/agentic-rag-with-langgraph/
zenn_article: 32bc8fd091100d
zenn_url: https://zenn.dev/0h_n0/articles/32bc8fd091100d
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

LangChainのAnkush Golaが2024年2月に公開した「Self-Reflective RAG with LangGraph」は、RAGパイプラインに**自己反省ループ**を組み込む実装パターンを解説したテックブログです。LangGraphの状態グラフ（StateGraph）と条件付きエッジ（conditional edges）を活用し、Document Grading → Query Rewriting → Answer Generationの3段階フローを実装しています。Corrective RAGとSelf-RAGの概念をファインチューニング不要でLangGraph上に再現する実践的なアプローチが特徴です。

この記事は [Zenn記事: LangGraph×Claude Sonnet 4.6エージェント型RAGの精度評価と最適化](https://zenn.dev/0h_n0/articles/32bc8fd091100d) の深掘りです。

## 情報源

- **ブログURL**: [https://blog.langchain.com/agentic-rag-with-langgraph/](https://blog.langchain.com/agentic-rag-with-langgraph/)
- **著者**: Ankush Gola（LangChain共同創業者）
- **公開日**: 2024年2月7日（更新: 2024年2月15日）
- **関連ドキュメント**: [LangGraph Agentic RAG Tutorial](https://docs.langchain.com/oss/python/langgraph/agentic-rag)

## 技術的背景（Technical Background）

### RAGの3つの認知アーキテクチャ

ブログでは、RAGパイプラインを3つの認知アーキテクチャに分類しています。

**Chain（チェイン型）**: 最も単純なパターンで、検索→生成の固定パイプラインです。LLMは検索結果に基づいて回答を生成するだけで、検索結果の品質を評価しません。

**Routing（ルーティング型）**: LLMがクエリの種類に応じて異なるリトリーバー（ベクトルDB、SQL、Web検索等）を選択します。検索先の動的選択は行いますが、結果の評価ループはありません。

**State Machine（状態機械型）**: **ループとフィードバック**をネイティブにサポートするアーキテクチャです。検索結果の品質評価、クエリの書き換え、再検索、ハルシネーション検証を、条件付きの状態遷移として表現します。

Zenn記事のLangGraph実装はこの第3のState Machine型に該当し、ブログで提案されているパターンを実際のプロダクション環境に適用したものです。

### なぜLangGraphか

ブログがLangGraphを選択した理由は、**条件付きエッジ（conditional edges）によるフロー制御**にあります。

従来のLangChain（LCEL: LangChain Expression Language）はDAG（有向非巡回グラフ）のみをサポートしており、ループ構造を表現できませんでした。RAGの自己修正には「検索→評価→不十分なら再検索」のループが不可欠であり、LangGraphの`add_conditional_edges()`がこれを実現します。

## 実装アーキテクチャ（Implementation Architecture）

### 状態グラフの全体設計

```
                    START
                      │
                      ▼
         ┌─ generate_query_or_respond ──┐
         │            │                  │
         │   (tool_call?)               │
         │    yes ↓        no ──────────┘──→ END
         │            │
         │   retrieve (ToolNode)
         │            │
         │            ▼
         │   grade_documents
         │    │              │
         │  relevant?    not relevant?
         │    ↓              ↓
         │  generate    rewrite_question
         │  _answer          │
         │    │              │
         │    ▼              └────→ generate_query_or_respond
         │   END                     (ループバック)
         └──────────────────────────────────────────────┘
```

### GraphState定義

ブログの公式実装では`MessagesState`を使用し、メッセージリストで全状態を管理します。

```python
from langgraph.graph import MessagesState, StateGraph, START, END

# MessagesStateはmessagesキーを持つTypedDict
# messages: list[BaseMessage]
workflow = StateGraph(MessagesState)
```

Zenn記事の`GraphState`は、`MessagesState`を拡張して`retry_count`、`grade_score`、`is_hallucination`等の追加フィールドを定義しています。これはプロダクション環境でのデバッグ・モニタリングに必要な設計拡張です。

### ノード実装の詳細

#### 1. generate_query_or_respond（エントリポイント）

LLMにリトリーバーツールをバインドし、クエリに応じて検索を実行するか直接回答するかを判断します。

```python
from langchain.chat_models import init_chat_model

response_model = init_chat_model("gpt-4.1", temperature=0)

def generate_query_or_respond(state: MessagesState):
    """LLMがツール呼び出しの要否を判断するノード"""
    response = (
        response_model
        .bind_tools([retrieve_blog_posts])
        .invoke(state["messages"])
    )
    return {"messages": [response]}
```

**設計のポイント**: `bind_tools()`によりLLMがツール呼び出しのJSON構造を生成し、`tools_condition`がこれを検出してルーティングを行います。この設計はOpenAI Function Callingの仕組みに依存しています。

Zenn記事では、Claude Sonnet 4.6のAdaptive Thinkingを活用してこの判断を行っており、Function Callingではなくeffortパラメータによる推論深度制御を採用しています。

#### 2. retrieve（検索実行）

```python
from langchain.tools import tool

@tool
def retrieve_blog_posts(query: str) -> str:
    """検索ツールの定義。LangGraphのToolNodeが自動的に実行"""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])
```

LangGraphの`ToolNode`がツール呼び出しを自動的にディスパッチし、結果を`ToolMessage`としてstateに追加します。

#### 3. grade_documents（文書評価 — 条件付きエッジ）

検索文書の関連性を二値分類し、結果に応じてフローを分岐させる**条件付きエッジ関数**です。

```python
from pydantic import BaseModel, Field
from typing import Literal

class GradeDocuments(BaseModel):
    """検索文書の関連性スコア"""
    binary_score: str = Field(
        description="'yes' if relevant, or 'no' if not relevant"
    )

grader_model = init_chat_model("gpt-4.1", temperature=0)

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document "
    "to a user question."
)

def grade_documents(
    state: MessagesState
) -> Literal["generate_answer", "rewrite_question"]:
    """文書の関連性を評価し、フローを分岐"""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    response = (
        grader_model
        .with_structured_output(GradeDocuments)
        .invoke([{
            "role": "user",
            "content": GRADE_PROMPT.format(
                question=question, context=context
            )
        }])
    )

    if response.binary_score == "yes":
        return "generate_answer"   # → 回答生成へ
    else:
        return "rewrite_question"  # → クエリ書き換えへ
```

**Zenn記事との対応**: Zenn記事のDocument Gradingノードはこの`grade_documents`関数の拡張版です。Zenn記事では二値分類に加えてスコアリング（0.0-1.0）を行い、閾値ベースの判定で精度を向上させています。

#### 4. rewrite_question（クエリ書き換え）

検索結果が不十分な場合に、元のクエリを意味的に書き換えて再検索を試みます。

```python
from langchain.messages import HumanMessage

REWRITE_PROMPT = (
    "Look at the input and try to reason about "
    "the underlying semantic intent / meaning."
)

def rewrite_question(state: MessagesState):
    """セマンティック意図を推論してクエリを書き換え"""
    question = state["messages"][0].content
    response = response_model.invoke([{
        "role": "user",
        "content": REWRITE_PROMPT.format(question=question)
    }])
    return {"messages": [HumanMessage(content=response.content)]}
```

書き換え後のクエリは`HumanMessage`として追加され、`generate_query_or_respond`ノードに戻ることで**再検索ループ**が形成されます。

#### 5. generate_answer（回答生成）

```python
GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question."
)

def generate_answer(state: MessagesState):
    """検索コンテキストに基づいて回答を生成"""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    response = response_model.invoke([{
        "role": "user",
        "content": GENERATE_PROMPT.format(
            question=question, context=context
        )
    }])
    return {"messages": [response]}
```

### グラフの組み立て

```python
from langgraph.prebuilt import ToolNode, tools_condition

workflow = StateGraph(MessagesState)

# ノード登録
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retrieve_blog_posts]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

# エッジ定義
workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {"tools": "retrieve", END: END}
)
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()
```

**条件付きエッジの2つの使い方**:
1. `tools_condition`: LLMのツール呼び出し有無を検出するビルトイン関数（ToolNode向け）
2. `grade_documents`: カスタムロジックで次のノードを決定する関数（文字列を返す）

## パフォーマンス最適化（Performance Optimization）

### ブログで示された最適化パターン

**Tavily Search APIによるWeb検索フォールバック**: Corrective RAGパターンでは、ベクトルDB検索が不十分な場合にTavily Search APIでWeb検索にフォールバックします。Zenn記事では同様のフォールバック機能をLangGraphの条件付きエッジで実装しています。

**Structured Output by Pydantic**: `with_structured_output(GradeDocuments)`により、LLMの出力をPydanticモデルに型安全にパースします。これによりグレーディング結果の信頼性が向上し、パース失敗のハンドリングが不要になります。

**LangSmithによるトレーシング**: LangSmithを統合することで、各ノードの実行時間、LLM呼び出し回数、トークン使用量を可視化できます。Zenn記事ではDeepEvalとRAGASによる定量評価を行っていますが、LangSmithは開発時のデバッグ・最適化ツールとして相補的に機能します。

### Zenn記事での拡張

ブログの基本実装に対して、Zenn記事は以下の拡張を行っています。

| 観点 | ブログ実装 | Zenn記事の拡張 |
|------|----------|--------------|
| LLM | GPT-4.1 | Claude Sonnet 4.6 + Adaptive Thinking |
| 文書評価 | 二値分類（yes/no） | スコアリング（0.0-1.0） + 閾値 |
| ハルシネーション検証 | なし | Hallucination Checkノード |
| 再試行制御 | 暗黙的ループ | `retry_count`による明示的制限 |
| 評価フレームワーク | なし | RAGAS + DeepEval（Faithfulness 0.91達成） |
| 推論深度制御 | なし | effortパラメータ最適化（medium/high） |

## 運用での学び（Operational Insights）

### 条件付きエッジ設計のベストプラクティス

ブログの実装から得られる設計指針:

1. **条件付きエッジは単一責務にする**: `grade_documents`は文書評価のみ、`tools_condition`はツール呼び出し検出のみ。複合条件を1つのエッジに詰め込まない
2. **ループには終了条件を明示する**: ブログ実装ではrewrite→re-retrieve→grade→rewriteの無限ループが理論上可能。Zenn記事の`retry_count`は本番運用に必須の安全弁
3. **Structured Outputで型安全性を確保**: Pydantic `BaseModel`による出力型定義は、LLM応答のパースエラーを防止

### LangGraphの限界と対策

ブログでは言及されていませんが、実運用で注意すべき点:

- **ステートの肥大化**: `MessagesState`のメッセージリストはループのたびに増大。長い会話ではコンテキストウィンドウの上限に注意
- **LLM呼び出し回数**: Document Grading + Answer Generation + (必要に応じてRewrite + 再検索)で、1クエリあたり最低3回のLLM呼び出しが発生
- **評価の不安定性**: 二値分類（yes/no）はLLMの温度パラメータに影響されやすい。temperature=0でも確率的な変動が残る

## 学術研究との関連（Academic Context）

### Corrective RAG (CRAG) との関係

ブログのDocument Gradingパターンは、Yan et al. (2024) のCorrective RAG（arXiv:2401.15884）の設計思想をLangGraph上で簡略化したものです。オリジナルのCRAGは3段階評価（Correct/Incorrect/Ambiguous）ですが、ブログ実装は二値分類に簡略化しています。

### Self-RAG との関係

ブログタイトルの「Self-Reflective RAG」は、Asai et al. (2023) のSelf-RAG（arXiv:2310.11511）の反省メカニズムをファインチューニング不要で近似するアプローチです。Self-RAGの4つの反省トークン（`[Retrieve]`, `[IsREL]`, `[IsSUP]`, `[IsUSE]`）を、LangGraphの条件付きエッジとStructured Outputで代替しています。

| Self-RAGのトークン | ブログの対応実装 | Zenn記事の対応実装 |
|-----------------|--------------|-----------------|
| `[Retrieve]` | `tools_condition` | LangGraphのルーティング |
| `[IsREL]` | `grade_documents` | Document Grading（スコアリング） |
| `[IsSUP]` | なし | Hallucination Check |
| `[IsUSE]` | なし | RAGAS Answer Relevancy |

### RAGLAB との関係

RAGLAB（arXiv:2408.15712）はブログと同じSelf-RAG/CRAGアルゴリズムを統一フレームワークで再現していますが、目的が異なります。RAGLABは公平な定量比較を目的とする研究フレームワークであり、ブログの実装はプロダクションアプリケーション構築を目的としています。

## まとめ

LangChainのSelf-Reflective RAGブログは、LangGraphの**条件付きエッジ**を活用した自己修正型RAGパイプラインの実装パターンを確立しました。Document Grading → Query Rewriting → Answer Generationの3段階フローは、Zenn記事のLangGraph実装の直接的な基盤となっています。

Zenn記事はこのブログの基本設計を拡張し、Claude Sonnet 4.6のAdaptive Thinking、Hallucination Check、RAGAS/DeepEvalによる定量評価を追加することで、Faithfulness 0.91・ハルシネーション率2.3%のプロダクション品質を達成しています。ブログが提供する「設計パターン」と、Zenn記事が提供する「本番運用品質の実装」は、Self-Reflective RAGの理論と実践を橋渡しする相補的な関係にあります。

## 参考文献

- **ブログ**: [https://blog.langchain.com/agentic-rag-with-langgraph/](https://blog.langchain.com/agentic-rag-with-langgraph/)
- **LangGraph Tutorial**: [https://docs.langchain.com/oss/python/langgraph/agentic-rag](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/32bc8fd091100d](https://zenn.dev/0h_n0/articles/32bc8fd091100d)
