---
layout: post
title: "arXiv論文解説: LLM-based Multi-Agent Systems — マルチエージェントアーキテクチャの体系的サーベイ"
description: "LLMベースマルチエージェントシステムの通信・アーキテクチャ・能力拡張を体系的に整理したサーベイ論文の解説"
categories: [blog, paper, arxiv]
tags: [multi-agent, LLM, survey, architecture, communication, agent]
date: 2026-06-14 12:00:00 +0900
source_type: arxiv
arxiv_id: "2402.01680"
source_url: https://arxiv.org/abs/2402.01680
zenn_article: fa2c321db68933
zenn_url: https://zenn.dev/0h_n0/articles/fa2c321db68933
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## 論文概要（Abstract）

本記事は [arXiv:2402.01680「LLM-based Multi-Agent Systems: Progress and Challenges」](https://arxiv.org/abs/2402.01680) の解説記事です。

著者ら（Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, Xiangliang Zhang）は、LLMベースのマルチエージェントシステム（MAS）に関する体系的サーベイを行い、エージェントの**プロファイリング**、**通信パターン**、**能力拡張**、**アプリケーション**の4つの軸で既存研究を分類・整理しています。

本論文は200以上の関連研究を対象とし、マルチエージェントシステムの設計パターンを体系化した包括的なレビューです。LangGraphのようなフレームワークで実際にマルチエージェントシステムを構築する際の設計判断の基盤となる知見を提供しています。

この記事は [Zenn記事: LangGraph v1.2でステートマシン設計――5つの分岐パターンと本番運用](https://zenn.dev/0h_n0/articles/fa2c321db68933) の深掘りです。

## 情報源

- **論文タイトル**: LLM-based Multi-Agent Systems: Progress and Challenges
- **arXiv ID**: 2402.01680
- **URL**: [https://arxiv.org/abs/2402.01680](https://arxiv.org/abs/2402.01680)
- **著者**: Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, Xiangliang Zhang
- **投稿日**: 2024年2月

## 技術的詳細（Technical Details）

### マルチエージェントシステムの分類体系

著者らは、LLMベースMASの構成要素を以下の4つの軸で体系化しています。

```mermaid
graph TD
    MAS[LLM-based MAS] --> A[Agent Profiling]
    MAS --> B[Communication]
    MAS --> C[Capability Enhancement]
    MAS --> D[Applications]
    A --> A1[Pre-defined]
    A --> A2[Model-generated]
    A --> A3[Data-derived]
    B --> B1[Structure]
    B --> B2[Content]
    B --> B3[Protocol]
    C --> C1[Reasoning]
    C --> C2[Memory]
    C --> C3[Tool Use]
```

### 1. エージェントプロファイリング（Agent Profiling）

プロファイリングとは、各エージェントに役割・性格・専門知識を割り当てる設計プロセスです。著者らは3つのアプローチを整理しています。

**Pre-defined（事前定義型）**: 人間が明示的にエージェントの役割を定義する。ChatDevの「CEO」「CTO」「Developer」のような固定的な役割割り当てが代表例。LangGraphでは、各ノードのシステムプロンプトに対応します。

```python
from langgraph.graph import StateGraph

def architect_node(state: AgentState) -> AgentState:
    """事前定義型: システムプロンプトで役割を固定"""
    response = llm.invoke([
        SystemMessage(content="あなたはソフトウェアアーキテクトです。設計に関する意見のみ述べてください。"),
        HumanMessage(content=state.task),
    ])
    return {"outputs": [response.content]}
```

**Model-generated（モデル生成型）**: LLM自身がタスクに応じてエージェントの役割を動的に生成する。AutoAgents（IJCAI 2024）が代表例。LangGraphでは、ルーター関数でエージェントのプロンプトを動的に構成する設計に対応します。

**Data-derived（データ駆動型）**: 実際のデータ（ユーザー行動ログ、人口統計データなど）からエージェントのプロファイルを構築する。社会シミュレーション（Generative Agents, Stanford 2023）が代表例。

### 2. 通信パターン（Communication）

著者らは通信パターンを**構造**・**内容**・**プロトコル**の3つの観点で分類しています。

#### 通信構造（Communication Structure）

**Layered（階層型）**: エージェントが階層構造を持ち、上位から下位への指示と下位から上位への報告で通信する。LangGraphのサブグラフ構成（パターン4）がこの構造に対応します。

```mermaid
graph TD
    Manager[Manager Agent] --> W1[Worker A]
    Manager --> W2[Worker B]
    Manager --> W3[Worker C]
    W1 --> Manager
    W2 --> Manager
    W3 --> Manager
```

**Decentralized（分散型）**: エージェント間に上下関係がなく、自由に通信する。AutoGenのGroupChatが代表例。LangGraphでは、全ノードが相互にエッジを持つ完全グラフに対応しますが、実装の複雑さが増大します。

**Shared Message Pool（共有メッセージプール型）**: 全エージェントが共有のメッセージ空間を通じて通信する。MetaGPTが代表例。LangGraphでは、`State`のメッセージリスト（`Annotated[list[str], operator.add]`）がこの設計に対応します。

**Ordered（順序型）**: エージェントの通信順序が固定されたパイプライン構造。ChatDevの基本構造が代表例。LangGraphでは、`add_edge`で線形にノードを接続する設計がこれに該当します。

| 通信構造 | 代表例 | LangGraphパターン | 適用場面 |
|----------|--------|------------------|----------|
| Layered | CAMEL | サブグラフ（パターン4） | 複雑なタスク分解 |
| Decentralized | AutoGen GroupChat | 条件分岐（パターン1） | 創造的議論 |
| Shared Pool | MetaGPT | State reducer | 情報共有重視 |
| Ordered | ChatDev | 線形エッジ | 定型ワークフロー |

#### 通信内容（Communication Content）

著者らは通信内容を以下のように分類しています。

- **Natural Language（自然言語）**: 最も一般的。人間が解釈しやすいが、曖昧性を含む。
- **Structured Data（構造化データ）**: JSON、コードなどの形式的な通信。LangGraphの`State`（TypedDictやPydanticモデル）がこれに該当。精密だが柔軟性が低い。
- **Mixed（混合型）**: 自然言語と構造化データの組み合わせ。実務では最も実用的。

#### 通信プロトコル（Communication Protocol）

著者らは以下のプロトコルを整理しています。

- **Debate（議論型）**: エージェント間で意見を交換し、合意を形成する。
- **Cooperation（協調型）**: タスクを分割し、各エージェントが分担して処理する。
- **Instruction（指示型）**: 上位エージェントが下位エージェントに指示を出す。

### 3. 能力拡張（Capability Enhancement）

著者らは、マルチエージェントシステムの能力拡張を**推論**・**メモリ**・**ツール使用**の3つの観点で分析しています。

**推論の拡張**: Chain-of-Thought（CoT）、Tree-of-Thought（ToT）、Graph-of-Thought（GoT）など、構造化された推論をマルチエージェントに組み込む手法。複数エージェントが独立にCoTを実行し、結果を統合する「Multi-Agent Debate」が注目されています。

**メモリの拡張**: 短期メモリ（現在のタスクの文脈）と長期メモリ（過去のタスクの経験）を組み合わせる設計。LangGraphでは、`MemorySaver`や`PostgresSaver`を用いたチェックポイント機能がこの役割を担います。Zenn記事で紹介されている`PostgresSaver`/`AsyncPostgresSaver`は、まさにこの長期メモリの実装です。

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

checkpointer = AsyncPostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/langgraph"
)
graph = builder.compile(checkpointer=checkpointer)
```

**ツール使用の拡張**: API呼び出し、コード実行、Web検索などの外部ツールとの連携。マルチエージェント環境では、各エージェントが異なるツールセットを持ち、必要に応じて他エージェントのツールを間接的に利用するパターンが報告されています。

### 4. アプリケーション領域

著者らは、LLMベースMASの応用領域を以下のように分類しています。

- **ソフトウェア開発**: ChatDev、MetaGPT、AutoGen
- **科学研究**: 仮説生成、実験設計の自動化
- **社会シミュレーション**: Generative Agents（Stanford, 2023）
- **ゲーム**: Diplomacy、Minecraftでの協調プレイ
- **教育**: 複数エージェントによる個別指導

## 実験結果・分析（Analysis）

著者らはサーベイを通じて以下の知見を報告しています。

**スケーラビリティの課題**: エージェント数が増加すると、通信コスト（トークン消費量）が$O(n^2)$で増大する。著者らは「エージェント数の増加は必ずしも性能向上に直結しない」と指摘しており、適切なエージェント数の選択が重要であると述べています。

この知見はLangGraphの設計においても示唆的です。Send API（パターン3）で大量の並列ブランチを生成する場合、各ブランチの出力を集約するreducerのコストとトークン消費量を考慮する必要があります。

**トポロジーの選択**: 著者らは、タスクの性質に応じて最適なトポロジーが異なると分析しています。

- **創造的タスク**: 分散型（Decentralized）が有効。多様な視点が統合される
- **分析的タスク**: 階層型（Layered）が有効。体系的な分解と統合が可能
- **定型タスク**: 順序型（Ordered）が有効。再現性と予測可能性が高い

**エージェント間の「社会的行動」**: 著者らは、LLMベースエージェントが人間の社会的行動（同調バイアス、権威への服従、集団思考）を模倣する傾向があると指摘しています。これはマルチエージェントディベートにおいて「全員が同じ（誤った）結論に収束する」リスクをもたらします。

## 実運用への応用（Practical Applications）

サーベイの知見をLangGraphでのシステム設計に活かす方法を考察します。

**1. トポロジー選択のガイドライン**:

```python
def select_topology(task_type: str) -> str:
    """サーベイの知見に基づくトポロジー選択"""
    mapping = {
        "creative": "decentralized",
        "analytical": "layered",
        "routine": "ordered",
        "complex": "layered_with_debate",
    }
    return mapping.get(task_type, "ordered")
```

LangGraphでは、`StateGraph`のエッジ構成を変えることでこれらのトポロジーを実装できます。Zenn記事のパターン1（条件分岐）は分散型に近く、パターン4（サブグラフ）は階層型に対応します。

**2. 通信コストの制御**:

著者らの$O(n^2)$通信コストの分析は、LangGraphのState設計に直接影響します。`State`にメッセージ履歴を全件保持する設計は、エージェント数とラウンド数に応じてチェックポイントサイズが急増します。`messages`フィールドに直近$k$件のみを保持するreducerの設計が推奨されます。

**3. 同調バイアスの緩和**:

LangGraphでマルチエージェントディベートを実装する場合、各エージェントに「他のエージェントの意見に反論する」ロールを明示的に割り当てることで、集団思考のリスクを低減できます。Zenn記事のパターン5（Human-in-the-Loop）で`interrupt()`を挟むことも、バイアス緩和の手段の1つです。

## 関連研究（Related Work）

- **ChatDev (Qian et al., 2023)**: ソフトウェア開発タスクに特化したマルチエージェントフレームワーク。Ordered型トポロジーの代表例。
- **MetaGPT (Hong et al., 2023)**: Shared Message Pool型の通信構造を採用し、構造化された中間成果物（SRS、設計書）でエージェント間を連携させる。
- **CAMEL (Li et al., 2023)**: Inception Promptingによる2エージェント間の自律的対話を実現。Layered型の単純な実装。
- **Generative Agents (Park et al., 2023)**: 25体のエージェントが社会シミュレーションを行い、emergent social behaviorsを観測した画期的な研究。
- **AutoGen (Wu et al., 2023)**: Microsoftによる汎用マルチエージェント会話フレームワーク。GroupChatやNestedChatなど複数のトポロジーをサポート。

## まとめと今後の展望

本サーベイ論文は、LLMベースマルチエージェントシステムの設計空間を**プロファイリング・通信・能力拡張・応用**の4軸で体系化した包括的レビューです。

**主要な知見**: エージェントの通信構造（Layered / Decentralized / Shared Pool / Ordered）の選択がシステム性能に大きく影響し、タスク特性に応じた適切な選択が求められます。

**LangGraphへの示唆**: LangGraphのStateGraph設計は、本サーベイの通信構造分類と直接対応します。`add_edge`（Ordered）、`add_conditional_edges`（Decentralized）、`State`のreducer（Shared Pool）、サブグラフ（Layered）という4つの構成要素を、タスク特性に応じて組み合わせることが設計の要点です。

**今後の課題**: 著者らは、エージェント間の同調バイアス問題、$O(n^2)$通信コストのスケーラビリティ課題、および動的なトポロジー変更（Puppeteerのようなアプローチ）を今後の研究方向として挙げています。

## 参考文献

- **arXiv URL**: [https://arxiv.org/abs/2402.01680](https://arxiv.org/abs/2402.01680)
- **Related Zenn article**: [LangGraph v1.2でステートマシン設計――5つの分岐パターンと本番運用](https://zenn.dev/0h_n0/articles/fa2c321db68933)
