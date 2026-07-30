---
layout: post
title: "論文解説: Route Before Retrieve — LLMの潜在的ルーティング能力を活性化するRAG vs ロングコンテキスト選択"
description: "Self-Routeを超えるPre-Routeフレームワークで、構造化推論によりRAGとロングコンテキストを動的に選択する手法の詳細解説"
categories: [blog, paper, arxiv]
tags: [LLM, RAG, long-context, routing, context-engineering, retrieval]
date: 2026-07-31 09:00:00 +0900
source_type: arxiv
arxiv_id: "2605.10235"
source_url: https://arxiv.org/abs/2605.10235
zenn_article: cfc6a5ad9e22fd
zenn_url: https://zenn.dev/0h_n0/articles/cfc6a5ad9e22fd
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## 論文概要

本記事は [Route Before Retrieve: Activating Latent Routing Abilities of LLMs for RAG vs. Long-Context Selection](https://arxiv.org/abs/2605.10235) の解説記事です。

現代のLLMは128Kトークン以上のコンテキストウィンドウをサポートしているが、与えられたクエリに対してRAG（Retrieval-Augmented Generation）を用いるかロングコンテキスト（LC）処理を用いるかの選択は依然として重要な課題である。本論文では、回答前に構造化推論を行うプロアクティブなルーティングフレームワーク「Pre-Route」を提案している。軽量メタデータ（文書タイプ、長さ、先頭スニペット等）を用いたタスク分析・カバレッジ推定・情報要求予測により、説明可能かつコスト効率の高いルーティング判断を実現する。LaRA（in-domain）およびLongBench-v2（OOD）での評価において、Pre-RouteはAlways-RAG、Always-LC、Self-Routeの各ベースラインを上回る性能を達成したと著者らは報告している。

この記事は [Zenn記事: LLMコンテキストエンジニアリング実践：圧縮・ルーティングで1Mトークンを制御](https://zenn.dev/0h_n0/articles/cfc6a5ad9e22fd) の深掘りです。

## 情報源

- **arXiv ID**: 2605.10235
- **URL**: [https://arxiv.org/abs/2605.10235](https://arxiv.org/abs/2605.10235)
- **著者**: Yiwen Chen, Kuan Li, Fuzhen Zhuang et al.
- **発表年**: 2026
- **分野**: cs.CL（Computation and Language）

## 背景と動機

LLMのコンテキストウィンドウが急速に拡大する中で、ナレッジベース型のQAシステムには2つの戦略が存在する。RAGは検索で関連する文書断片のみを取得するため計算コストが低いが、検索品質に制約される。一方LCは文書全体をコンテキストに投入して包括的な推論を可能にするが、コストが高く、位置バイアス（position sensitivity）の影響を受けやすい。

### Self-Routeの限界

先行研究であるSelf-Route（Li et al., EMNLP 2024）は、まずRAGで回答を試み、モデルが「回答不可能」と自己判断した場合にLCへフォールバックする手法である。Gemini-1.5-Proで65%、GPT-4で39%のコスト削減を達成したと報告されている。しかし著者らは、Self-Routeには以下の構造的な問題があると指摘している。

- **受動的（Passive）**: RAGが失敗して初めてLCに切り替わるため、RAGの不要な実行オーバーヘッドが常に発生する
- **非効率（Inefficient）**: 全クエリに対してRAGを先に試行するため、LC向きのクエリでも検索・再ランキングの計算コストが無駄になる
- **解釈困難（Hard to interpret）**: 「回答不可能」という自己判断に基づくフォールバックであり、なぜLCが必要かの説明が得られない

### ルーティング問題の重要性

著者らは、ルーティング判断の質がシステム全体のコスト効率に直結すると論じている。LC処理の計算コストはRAGの数十倍にのぼるため、不要なLC呼び出しを抑制するだけで総コストを大幅に削減できる。逆に、LC向きのクエリをRAGで処理すると回答品質が低下する。このトレードオフを事前に（proactively）判断する能力が求められている。

## 主要な貢献

著者らは以下の4点を主要な貢献として挙げている。

- **Pre-Routeフレームワークの提案**: 回答前に構造化推論を行い、軽量メタデータのみでRAG/LCの二値判断を行うプロアクティブなルーティング手法。6段階の推論ステップと決定規則から構成される。

- **LLMの潜在的ルーティング能力の発見**: Best-of-N分析により、LLMは構造化ガイドラインなしでもルーティング能力を潜在的に保持していることを示し、構造化プロンプトがこの能力を安定的に引き出す「キャリブレータ」として機能することを実証している。

- **線形プローブによる表現空間の分析**: 凍結した最終層埋め込みに対する線形プローブ実験により、構造化プロンプトがルーティング判断の表現空間における分離性を向上させることを定量的に示している。

- **小型モデルへの蒸留**: 大規模モデル（Qwen3-235B-A22B、DeepSeek-R1）の推論構造をQwen3-1.7Bに蒸留し、ルーティング精度を維持しつつ大幅なコスト削減を実現している。

## 技術的詳細

### Pre-Routeフレームワークのアーキテクチャ

Pre-Routeは「回答前にルーティング判断を完了する」という設計思想に基づいている。以下のダイアグラムにフレームワーク全体の流れを示す。

```mermaid
flowchart TD
    A[ユーザクエリ] --> B[軽量メタデータ収集]
    B --> C[構造化推論プロセス]
    C --> D{ルーティング判断}
    D -->|RAG| E[検索 + 再ランキング]
    D -->|LC| F[全文コンテキスト投入]
    E --> G[回答生成]
    F --> G

    subgraph メタデータ
        B1[文書タイプ]
        B2[文書長]
        B3[先頭スニペット]
        B4[タスクタイプ]
        B5[RAG構成情報]
        B6[回答モデル仕様]
    end

    B --> メタデータ

    subgraph 構造化推論 6ステップ
        S1["Step 1: タスク・文書特性分析"]
        S2["Step 2: 情報分布パターン判断"]
        S3["Step 3: コンテキストウィンドウ充足性"]
        S4["Step 4: 検索実現可能性評価"]
        S5["Step 5: モデル能力の考慮"]
        S6["Step 6: 効率トレードオフ"]
        S1 --> S2 --> S3 --> S4 --> S5 --> S6
    end

    C --> 構造化推論 6ステップ
```

### 構造化推論プロセス

Pre-Routeの中核は、4つの意思決定次元を6つの逐次ステップで具体化する構造化推論チェーンである。各ステップの役割を以下に詳述する。

**Step 1: タスク・文書特性分析（Task & Document Characterization）**

クエリのタスクタイプ（要約、QA、情報抽出等）と文書タイプ（技術文書、法律文書、対話記録等）が、RAGとLCそれぞれの適合性にどう影響するかを分析する。たとえば、文書全体の要約タスクはLC向き、特定の事実検索はRAG向きと判断される。

**Step 2: 情報分布パターン判断（Distribution Pattern Judgment）**

回答に必要な情報が文書中にどのように分布しているかを評価する。情報が欠損している場合、断片的に分散している場合、局所的に集中している場合のそれぞれでRAG/LCの適合性が異なる。

**Step 3: コンテキストウィンドウ充足性（Context-Window Feasibility）**

検索で取得する断片のみで十分な情報が得られるか、それとも文書全体のコンテキストが必要かを判断する。RAGのtop-kチャンクがクエリの情報要求を満たすかどうかの推定が行われる。

**Step 4: 検索実現可能性評価（Retrieval Feasibility）**

RAGパイプラインの検索・再ランキング構成（再ランクサイズ、チャンクサイズ等）の制約を考慮し、検索が十分な品質で機能するかを評価する。

**Step 5: モデル能力の考慮（Model Capability Consideration）**

回答に用いるLLMの具体的な特性（コンテキスト長の上限、位置バイアスの程度、特定タスクでの強み）を考慮した判断調整を行う。

**Step 6: 効率トレードオフ（Efficiency Trade-off）**

性能と計算コストのバランスを評価する。LC処理のコストはRAGの数十倍であるため、性能差が僅少な場合はRAGが選好される。

### 軽量メタデータの設計

Pre-Routeが判断に用いるメタデータは以下の6項目であり、文書全体の読解を必要としない点が特徴である。

| メタデータ項目 | 内容 | 取得コスト |
|:---|:---|:---|
| ユーザクエリ | 質問文そのもの | なし |
| タスクタイプ | QA、要約、抽出等 | クエリから推定 |
| 文書タイプ・タイトル | 技術文書、法律文書等 | メタデータから取得 |
| 文書長 | トークン数 | 即時計算 |
| 先頭スニペット | 文書の最初の数百トークン | 部分読み込み |
| RAG構成情報 | 再ランクサイズ、チャンク設定 | システム設定 |

著者らはTable 6のメタデータロバストネス実験で、先頭スニペット（Head-only）のみでもQAスコア3.42、ルーティング精度0.68を達成し、Self-Routeのベースライン（QAスコア3.36、精度0.52）を上回ることを報告している。これは、文書全体を処理せずとも有効なルーティング判断が可能であることを示している。

### Best-of-N分析による潜在的ルーティング能力の実証

著者らは構造化ガイドラインの効果を検証するため、3つのプロンプトパラダイム（直接回答、制約なしCoT、Pre-Routeガイドライン）に対してBest-of-Nサンプリング実験を実施している（論文Figure 2より）。Qwen3-235B-A22Bを回答モデルとした結果は以下の通りである。

| プロンプト方式 | N=1 | N=2 | N=4 | N=8 |
|:---|---:|---:|---:|---:|
| 直接回答（Answer Directly） | 0.53 | 0.68 | 0.80 | 0.87 |
| 制約なしCoT（Unconstrained CoT） | 0.58 | 0.72 | 0.82 | 0.85 |
| Pre-Routeガイドライン | 0.70 | 0.80 | 0.83 | 0.87 |

直接回答方式ではN=1の精度が0.53と低いが、N=8では0.87に到達する。この大きな傾きは、LLMが正しいルーティング判断を行う能力を潜在的に保持しているものの、確率的にしかアクセスできないことを意味する。Pre-Routeガイドライン方式ではN=1で0.70を達成しており、N=8の上限（0.87）に近い性能を単一サンプルで引き出している。

著者らはこの結果について、構造化ガイドラインは「新たな知識を注入するのではなく、明確な推論の足場を提供することで、モデルの潜在的なルーティング能力を確実に引き出し方向づけるキャリブレータかつスタビライザとして機能する」と述べている。

### 線形プローブによる表現空間分析

著者らは行動レベルの分析に加え、LLMの内部表現がルーティング判断をどのように符号化しているかを線形プローブで分析している。凍結した最終トークンの最終層埋め込みに対して線形分類器を訓練し、以下の4つのターゲットの予測精度を測定している。

**予測ターゲット**:
- **Ideal Label**: タスク最適な判断（QA性能を最大化するRAG/LC選択）
- **Model's Route Choice**: モデル自身のルーティング出力
- **Document Type**: 文書タイプの7クラス分類
- **Task Type**: タスクタイプの4クラス分類

線形プローブの精度が高いほど、当該情報がモデルの表現空間において線形分離可能な形で符号化されていることを意味する。

Qwen3モデルでの結果は以下の通りである（論文Table 1より）。

| モデル / 設定 | Ideal精度 | Route精度 | 文書タイプ | タスクタイプ |
|:---|---:|---:|---:|---:|
| Qwen3-1.7B Pre-Route蒸留 | 0.639 | 0.799 | 0.396 | 0.410 |
| Qwen3-1.7B Pre-Routeプロンプト | 0.625 | 0.764 | 0.333 | 0.417 |
| Qwen3-8B 直接回答 | 0.549 | 0.660 | 0.299 | 0.257 |
| Qwen3-1.7B 制約なしCoT | 0.396 | 0.576 | 0.347 | 0.264 |

この結果から以下の知見が得られている。

1. **構造化プロンプトが表現空間の分離性を向上**: 制約なしCoTでのIdeal精度0.396に対し、Pre-Routeプロンプトでは0.625に向上している。これは構造化推論により、最適なルーティング判断がモデルの表現空間でより線形分離可能な形で符号化されることを示す。

2. **蒸留がさらに分離性を強化**: 蒸留後のモデルではIdeal精度が0.639に到達し、教師モデルの推論パターンが表現レベルで内在化されていることを示唆している。

3. **浅い手がかりに依存しない判断**: 文書タイプ精度（0.33-0.40）およびタスクタイプ精度（0.26-0.42）が相対的に低い値にとどまっている。これはルーティング判断が文書タイプやタスクタイプという表層的な属性のショートカットに依存していないことを意味する。

数式で表現すると、線形プローブは凍結された最終層埋め込み $\mathbf{h} \in \mathbb{R}^d$ に対して線形分類器 $f(\mathbf{h}) = \mathbf{W}\mathbf{h} + \mathbf{b}$ を訓練するものであり、ここで $\mathbf{W} \in \mathbb{R}^{C \times d}$、$\mathbf{b} \in \mathbb{R}^C$ はそれぞれ重み行列とバイアス、$C$ はクラス数である。

### モデル蒸留の手法

Pre-Routeの構造化推論を本番環境で効率的に実行するため、著者らは大規模モデルの推論構造を小型モデルに蒸留する手法を提案している。

#### 教師-生徒構成

- **教師モデル**: Qwen3-235B-A22B、DeepSeek-R1
- **生徒モデル**: Qwen3-1.7B

#### 2段階訓練アプローチ

**Stage 1: 棄却サンプリング（Rejection Sampling）**

教師モデルの出力のうち、判断がIdeal Labelと一致するもののみをフィルタリングする。

$$
\mathcal{D}_{\text{filtered}} = \{(m_i, T_i, y_i) \mid y_i = \hat{y}_{\text{ideal},i}\}
$$

ここで、
- $m_i$: メタデータ入力
- $T_i$: 推論チェーン（構造化推論の出力テキスト）
- $y_i$: 教師モデルのルーティング判断
- $\hat{y}_{\text{ideal},i}$: Ideal Label（タスク最適な正解）

この棄却サンプリングにより、正しい判断に至った推論過程のみが訓練データに含まれる。

**Stage 2: パスSFT（Path Supervised Fine-Tuning）**

フィルタリングされたデータに対して教師あり微調整を行う。

$$
\mathcal{L}_{\text{SFT}}(\theta_S) = -\mathbb{E}_{(m, T, y) \sim \mathcal{D}_{\text{filtered}}} \left[ \log \pi_S(T, y \mid m) \right]
$$

ここで、
- $\theta_S$: 生徒モデルのパラメータ
- $\pi_S(T, y \mid m)$: 生徒モデルがメタデータ $m$ から推論チェーン $T$ とルーティング判断 $y$ を生成する確率

この損失関数は「何を答えるか」だけでなく「どのように推論するか」を転移する。推論チェーン $T$ を含めて最適化することで、生徒モデルは教師モデルの構造化推論パターンを学習する。

#### Ideal Labelの定義

ルーティングの正解ラベルは以下のように定義される。

$$
\hat{y}_{\text{ideal}} =
\begin{cases}
\text{LC} & \text{if } U(\text{LC}; q, D) > U(\text{RAG}; q, D) \\
\text{RAG} & \text{otherwise}
\end{cases}
$$

ここで $U(\cdot; q, D)$ はクエリ $q$ と文書集合 $D$ に対するQA性能を表す効用関数である。LCはRAGに対して性能上の優位性がある場合にのみ選択され、同等の場合はコストの低いRAGがデフォルトとなる。

#### データ構築戦略

著者らは訓練データの構築に際して以下の工夫を行っている。

- Qwen3-235BとDeepSeek-R1の両方から推論チェーンを生成し多様性を確保
- 異なるスケールの回答モデルに対して一貫したIdeal Labelを付与
- 文書タイプ、難易度、コンテキスト長で層化した70/10/20のtrain/validation/test分割
- 教師の判断がIdeal Labelと一致する例のみを訓練に使用

## アルゴリズム

以下にPre-Routeパイプラインの実装イメージをPythonコードで示す。

```python
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RouteDecision(Enum):
    """ルーティング判断の二値選択"""
    RAG = "rag"
    LC = "long_context"


@dataclass
class DocumentMetadata:
    """ルーティング判断に用いる軽量メタデータ

    文書全体の読解を必要とせず、
    メタデータのみでルーティング判断を行う。
    """
    query: str
    doc_type: str          # 例: "technical", "legal", "dialogue"
    doc_length_tokens: int
    head_snippet: str      # 文書先頭の数百トークン
    task_type: str         # 例: "qa", "summarization", "extraction"
    rag_rerank_size: int   # RAGパイプラインの再ランクサイズ
    answer_model: str      # 回答に用いるモデル名


@dataclass
class RoutingResult:
    """ルーティング判断の結果と根拠"""
    decision: RouteDecision
    reasoning_chain: str
    confidence: float


def build_structured_prompt(metadata: DocumentMetadata) -> str:
    """構造化推論プロンプトを構築する

    6段階の推論ステップをガイドラインとして埋め込み、
    LLMの潜在的ルーティング能力を引き出す。

    Args:
        metadata: 軽量メタデータ

    Returns:
        構造化推論を促すプロンプト文字列
    """
    return f"""You are a routing decision expert. Analyze the following query
and document metadata to decide whether RAG or Long-Context (LC) is optimal.

## Input Metadata
- Query: {metadata.query}
- Document Type: {metadata.doc_type}
- Document Length: {metadata.doc_length_tokens} tokens
- Head Snippet: {metadata.head_snippet[:500]}
- Task Type: {metadata.task_type}
- RAG Rerank Size: {metadata.rag_rerank_size}
- Answer Model: {metadata.answer_model}

## Structured Reasoning Guidelines

Follow these 6 steps sequentially:

Step 1 - Task & Document Characterization:
Analyze how the task type and document type affect retrieval vs. LC needs.

Step 2 - Distribution Pattern Judgment:
Assess whether relevant content is missing, fragmented, or dispersed.

Step 3 - Context-Window Feasibility:
Determine if RAG top-k chunks suffice or full context is required.

Step 4 - Retrieval Feasibility:
Evaluate retrieval quality constraints given the RAG configuration.

Step 5 - Model Capability Consideration:
Account for the answer model's strengths and context length limits.

Step 6 - Efficiency Trade-off:
Balance expected performance gain against computational cost.
LC should be chosen ONLY when it provides a clear performance advantage.

## Decision Rules
- Default to RAG when performance difference is marginal
- Choose LC only when information needs span the entire document
- Consider the cost ratio: LC is typically 10-50x more expensive than RAG

Output your reasoning for each step, then provide your final decision.
"""


def pre_route(
    metadata: DocumentMetadata,
    router_model: Any,
) -> RoutingResult:
    """Pre-Routeによるルーティング判断を実行する

    Args:
        metadata: 文書メタデータ
        router_model: ルーティング判断を行うLLM

    Returns:
        ルーティング判断結果
    """
    prompt = build_structured_prompt(metadata)
    response = router_model.generate(prompt)

    decision = _parse_decision(response)
    return RoutingResult(
        decision=decision,
        reasoning_chain=response,
        confidence=_estimate_confidence(response),
    )


def answer_with_routing(
    query: str,
    documents: list[str],
    metadata: DocumentMetadata,
    router_model: Any,
    answer_model: Any,
    retriever: Any,
) -> str:
    """Pre-Routeパイプライン全体を実行する

    Args:
        query: ユーザクエリ
        documents: 文書集合
        metadata: 軽量メタデータ
        router_model: ルーターLLM（蒸留済み小型モデル推奨）
        answer_model: 回答生成LLM
        retriever: RAGパイプラインの検索エンジン

    Returns:
        生成された回答文字列
    """
    # Step 1: Pre-Route判断（回答前にルーティングを完了）
    routing = pre_route(metadata, router_model)

    # Step 2: 判断に基づいてコンテキストを構築
    if routing.decision == RouteDecision.RAG:
        # RAG: 検索 + 再ランキングで関連チャンクのみ取得
        chunks = retriever.search_and_rerank(
            query=query,
            documents=documents,
            top_k=metadata.rag_rerank_size,
        )
        context = "\n\n".join(chunks)
    else:
        # LC: 文書全体をコンテキストに投入
        context = "\n\n".join(documents)

    # Step 3: 回答生成
    answer = answer_model.generate(
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    return answer


def _parse_decision(response: str) -> RouteDecision:
    """LLM出力からルーティング判断を抽出する"""
    lower = response.lower()
    if "long_context" in lower or "lc" in lower.split()[-5:]:
        return RouteDecision.LC
    return RouteDecision.RAG


def _estimate_confidence(response: str) -> float:
    """推論チェーンから信頼度を推定する（簡易実装）"""
    # 本番環境ではlogitsベースの信頼度推定を推奨
    return 0.8
```

## 実装のポイント

Pre-Routeを実システムに組み込む際の注意点を以下にまとめる。

**メタデータの前処理**: 文書長（トークン数）の計算にはtiktokenやsentencepieceのトークナイザを用いる。先頭スニペットの長さは500トークン程度が推奨される。著者らのメタデータロバストネス実験（論文Table 6より）では、先頭スニペットのみ（Head-only）でもQAスコア3.42、ルーティング精度0.68を達成しており、完全なメタデータが得られない環境でも有効に機能する。

**蒸留モデルの選択**: 著者らはQwen3-1.7Bへの蒸留で、Qwen3-235Bをルーターとして用いた場合とほぼ同等のルーティング精度を達成している。蒸留モデルのルーティングコストは$0.00016/判断であり、Qwen3-235BのSelf-Routeコスト$0.00076と比較して約1/5に削減されている（論文Table 2より）。

**決定規則の重要性**: 消去法実験（論文Table 5より）では、決定規則（Decision Rules）を除去するとLC選択率が20.7%から45.3%に急増し、ルーティング精度が0.68から0.57に低下している。決定規則はLCの過剰選択を防ぐ上で不可欠な構成要素である。

**再ランク構成への頑健性**: Figure 4の実験では、再ランクサイズを5、7、10と変化させてもPre-Route（蒸留版）のルーティング精度は安定しており、RAGパイプラインの構成変更に対する頑健性が確認されている。

## Production Deployment Guide

Pre-Routeベースのルーティングシステムをプロダクション環境にデプロイするためのAWS構成を、トラフィック量別に提示する。以下のコスト試算は2026年7月時点のAWS ap-northeast-1（東京）リージョンの概算値であり、実際のコストはトラフィックパターンやバースト使用量により変動する。最新料金はAWS料金計算ツールで確認を推奨する。

### AWS実装パターン（コスト最適化重視）

#### Small構成（~100 req/日）: Lambda + Bedrock

| サービス | 用途 | 月額概算 |
|:---|:---|---:|
| Lambda | ルーター関数 + オーケストレーション | $5 |
| Bedrock (Claude Haiku) | ルーティング判断 | $10-30 |
| Bedrock (Claude Sonnet) | RAG/LC回答生成 | $30-80 |
| DynamoDB | ルーティング判断キャッシュ | $5 |
| S3 | 文書ストレージ | $5 |
| CloudWatch | ログ・監視 | $5 |
| **合計** | | **$60-130** |

Lambda関数がPre-Routeのメタデータ収集と構造化プロンプト構築を担当し、Bedrockのルーターモデル（小型モデル推奨）でルーティング判断を行う。判断結果に基づきRAGまたはLCパスを選択して回答生成モデルを呼び出す。DynamoDBにルーティング判断をキャッシュすることで、同一文書への再クエリ時のコストを削減する。

#### Medium構成（~1000 req/日）: ECS Fargate + Bedrock + ElastiCache

| サービス | 用途 | 月額概算 |
|:---|:---|---:|
| ECS Fargate (2 task) | ルーターAPI + RAGパイプライン | $100-150 |
| Bedrock | ルーティング + 回答生成 | $200-400 |
| ElastiCache (Redis) | ルーティングキャッシュ + 埋め込みキャッシュ | $50-80 |
| OpenSearch Serverless | ベクトル検索（RAGパス） | $50-100 |
| ALB | ロードバランシング | $30 |
| CloudWatch | 監視・アラーム | $10 |
| **合計** | | **$440-770** |

ECS Fargateでルーターサービスを常駐させ、ElastiCacheでルーティング判断と埋め込みベクトルをキャッシュする。OpenSearch ServerlessがRAGパスのベクトル検索を担当する。

#### Large構成（10000+ req/日）: EKS + GPU推論

| サービス | 用途 | 月額概算 |
|:---|:---|---:|
| EKS + Karpenter | コンテナオーケストレーション | $200 |
| GPU Spot Instances (g5.xlarge) | 蒸留ルーターモデル推論 | $300-600 |
| Bedrock / SageMaker | 回答生成 | $800-2,000 |
| OpenSearch Serverless | ベクトル検索 | $200-400 |
| ElastiCache (Redis Cluster) | 分散キャッシュ | $150-250 |
| ALB + WAF | ロードバランシング + セキュリティ | $80 |
| CloudWatch + X-Ray | 監視・トレーシング | $50 |
| **合計** | | **$1,780-3,580** |

蒸留済みQwen3-1.7BモデルをGPU Spot Instancesでホスティングし、ルーティング判断をオンプレミス推論で処理する。Karpenterによる自動スケーリングでSpot Instancesを優先配分し、コストを最適化する。

**コスト削減テクニック**:
- Spot Instancesの活用でGPU推論コストを最大90%削減
- Reserved Instances（1年コミット）でCompute Savings Plansにより最大72%削減
- Bedrock Batch APIの使用でLLM推論コストを50%削減
- Prompt Cachingの有効化でルーティングプロンプトのコストを30-90%削減
- ルーティング判断のキャッシュにより同一パターンのクエリでルーターコストをゼロに

### Terraformインフラコード

#### Small構成（Serverless）

```hcl
# Small構成: Lambda + Bedrock + DynamoDB
# Pre-Routeルーティングシステム（~100 req/日向け）

terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}

provider "aws" {
  region = "ap-northeast-1"
}

# --- IAM: 最小権限の原則 ---
resource "aws_iam_role" "pre_route_lambda" {
  name = "pre-route-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "pre_route_policy" {
  name = "pre-route-policy"
  role = aws_iam_role.pre_route_lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = "arn:aws:bedrock:ap-northeast-1::foundation-model/*"
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:Query"
        ]
        Resource = aws_dynamodb_table.routing_cache.arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        # X-Ray トレーシング
        Effect   = "Allow"
        Action   = ["xray:PutTraceSegments", "xray:PutTelemetryRecords"]
        Resource = "*"
      }
    ]
  })
}

# --- Lambda: Pre-Routeルーター ---
resource "aws_lambda_function" "pre_route_router" {
  function_name = "pre-route-router"
  runtime       = "python3.12"
  handler       = "handler.lambda_handler"
  role          = aws_iam_role.pre_route_lambda.arn
  timeout       = 60  # 構造化推論に十分な時間
  memory_size   = 512 # メタデータ処理に最適化

  # コスト監視: メモリ512MBで$0.0000083/秒（東京）
  environment {
    variables = {
      ROUTER_MODEL_ID   = "anthropic.claude-3-haiku-20240307-v1:0"
      ANSWER_MODEL_ID   = "anthropic.claude-sonnet-4-20250514-v1:0"
      CACHE_TABLE_NAME  = aws_dynamodb_table.routing_cache.name
      ROUTING_CACHE_TTL = "3600" # 1時間キャッシュ
    }
  }

  tracing_config {
    mode = "Active" # X-Ray有効化
  }

  filename         = "lambda_package.zip"
  source_code_hash = filebase64sha256("lambda_package.zip")
}

# --- DynamoDB: ルーティング判断キャッシュ ---
resource "aws_dynamodb_table" "routing_cache" {
  name         = "pre-route-cache"
  billing_mode = "PAY_PER_REQUEST" # On-Demand: 低トラフィックに最適
  hash_key     = "cache_key"

  attribute {
    name = "cache_key"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }

  # KMS暗号化
  server_side_encryption {
    enabled = true
  }
}

# --- CloudWatch: コスト監視アラーム ---
resource "aws_cloudwatch_metric_alarm" "lambda_cost_alarm" {
  alarm_name          = "pre-route-lambda-invocations-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Invocations"
  namespace           = "AWS/Lambda"
  period              = 86400 # 24時間
  statistic           = "Sum"
  threshold           = 500   # 想定の5倍でアラート
  alarm_description   = "Lambda invocations exceeded expected daily volume"

  dimensions = {
    FunctionName = aws_lambda_function.pre_route_router.function_name
  }
}
```

#### Large構成（Container）

```hcl
# Large構成: EKS + Karpenter + Spot Instances
# Pre-Routeルーティングシステム（10000+ req/日向け）

# --- EKSクラスタ ---
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.24"

  cluster_name    = "pre-route-cluster"
  cluster_version = "1.31"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # コスト最適化: コントロールプレーンのみ（$0.10/時間）
  cluster_endpoint_public_access = false

  eks_managed_node_groups = {
    # システムノード（Spot優先）
    system = {
      instance_types = ["m7i.large", "m6i.large"]
      capacity_type  = "SPOT"
      min_size       = 2
      max_size       = 4
      desired_size   = 2
    }
  }
}

# --- Karpenter: GPU Spot自動スケーリング ---
resource "kubectl_manifest" "karpenter_gpu_provisioner" {
  yaml_body = yamlencode({
    apiVersion = "karpenter.sh/v1"
    kind       = "NodePool"
    metadata   = { name = "gpu-spot-routing" }
    spec = {
      template = {
        spec = {
          requirements = [
            { key = "karpenter.sh/capacity-type", operator = "In", values = ["spot"] },
            { key = "node.kubernetes.io/instance-type", operator = "In",
              values = ["g5.xlarge", "g5.2xlarge"] },
          ]
          nodeClassRef = { name = "default" }
        }
      }
      limits   = { cpu = "64", memory = "256Gi" }
      disruption = {
        consolidationPolicy = "WhenEmptyOrUnderutilized"
        consolidateAfter    = "30s"
      }
    }
  })
}

# --- Secrets Manager: モデル設定 ---
resource "aws_secretsmanager_secret" "model_config" {
  name = "pre-route/model-config"
}

resource "aws_secretsmanager_secret_version" "model_config" {
  secret_id = aws_secretsmanager_secret.model_config.id
  secret_string = jsonencode({
    router_model_path   = "s3://models/qwen3-1.7b-pre-route-distilled/"
    answer_model_id     = "anthropic.claude-sonnet-4-20250514-v1:0"
    rag_rerank_size     = 7
    routing_cache_ttl   = 3600
  })
}

# --- AWS Budgets: 予算アラート ---
resource "aws_budgets_budget" "pre_route_monthly" {
  name         = "pre-route-monthly-budget"
  budget_type  = "COST"
  limit_amount = "4000"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = ["ops-team@example.com"]
  }

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 100
    threshold_type            = "PERCENTAGE"
    notification_type         = "FORECASTED"
    subscriber_email_addresses = ["ops-team@example.com"]
  }
}
```

### 運用・監視設定

**CloudWatch Logs Insights クエリ**:

```
# ルーティング判断の分布とコスト異常検知
fields @timestamp, route_decision, input_tokens, output_tokens, latency_ms
| stats count() as total,
        sum(case route_decision = 'lc' then 1 else 0 end) as lc_count,
        avg(latency_ms) as avg_latency,
        pct(latency_ms, 95) as p95_latency,
        sum(input_tokens + output_tokens) as total_tokens
  by bin(1h) as hour
| sort hour desc
```

```
# LC選択率の急変検知（異常なルーティング偏りの検出）
fields @timestamp, route_decision
| stats count() as total,
        sum(case route_decision = 'lc' then 1 else 0 end) * 100.0 / count() as lc_rate
  by bin(1h)
| filter lc_rate > 50
| sort @timestamp desc
```

**CloudWatch アラーム設定コード（Python）**:

```python
import boto3

cloudwatch = boto3.client("cloudwatch", region_name="ap-northeast-1")

# Bedrockトークン使用量スパイク検知
cloudwatch.put_metric_alarm(
    AlarmName="pre-route-bedrock-token-spike",
    MetricName="InputTokenCount",
    Namespace="AWS/Bedrock",
    Statistic="Sum",
    Period=3600,
    EvaluationPeriods=1,
    Threshold=500000,  # 1時間あたり50万トークン
    ComparisonOperator="GreaterThanThreshold",
    AlarmActions=["arn:aws:sns:ap-northeast-1:ACCOUNT:ops-alerts"],
)

# LC選択率異常検知（カスタムメトリクス）
cloudwatch.put_metric_alarm(
    AlarmName="pre-route-lc-rate-high",
    MetricName="LCSelectionRate",
    Namespace="PreRoute/Routing",
    Statistic="Average",
    Period=3600,
    EvaluationPeriods=2,
    Threshold=0.5,  # LC選択率50%超でアラート
    ComparisonOperator="GreaterThanThreshold",
    AlarmActions=["arn:aws:sns:ap-northeast-1:ACCOUNT:ops-alerts"],
)
```

**X-Ray トレーシング設定（Python）**:

```python
from aws_xray_sdk.core import xray_recorder, patch_all

# boto3の自動計装
patch_all()

@xray_recorder.capture("pre_route_decision")
def route_and_answer(query: str, metadata: dict) -> dict:
    """Pre-Routeパイプライン全体をトレーシングする"""
    # ルーティング判断をアノテーション付きで記録
    subsegment = xray_recorder.current_subsegment()
    subsegment.put_annotation("task_type", metadata.get("task_type", "unknown"))
    subsegment.put_annotation("doc_length", metadata.get("doc_length_tokens", 0))

    # ルーティング判断
    decision = pre_route(metadata)
    subsegment.put_annotation("route_decision", decision.value)
    subsegment.put_metadata("reasoning", decision.reasoning_chain[:500])

    # 判断に基づく回答生成
    answer = generate_answer(query, decision)
    return {"answer": answer, "route": decision.value}
```

**Cost Explorer自動レポート（Python）**:

```python
import boto3
from datetime import datetime, timedelta

ce = boto3.client("ce", region_name="us-east-1")
sns = boto3.client("sns", region_name="ap-northeast-1")

def daily_cost_report() -> None:
    """日次コストレポートを生成しSNS通知する"""
    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    result = ce.get_cost_and_usage(
        TimePeriod={"Start": start, "End": end},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
        Filter={
            "Tags": {
                "Key": "Project",
                "Values": ["pre-route"],
            }
        },
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )

    total = sum(
        float(g["Metrics"]["UnblendedCost"]["Amount"])
        for r in result["ResultsByTime"]
        for g in r["Groups"]
    )

    if total > 100:
        sns.publish(
            TopicArn="arn:aws:sns:ap-northeast-1:ACCOUNT:cost-alerts",
            Subject=f"Pre-Route Daily Cost Alert: ${total:.2f}",
            Message=f"Daily cost exceeded $100 threshold: ${total:.2f}",
        )
```

### コスト最適化チェックリスト

**アーキテクチャ選択**:
- [ ] トラフィック量に応じた構成を選択（~100 req/日: Serverless、~1000: Hybrid、10000+: Container）
- [ ] ルーターとアンサーモデルのAPIコール分離

**リソース最適化**:
- [ ] GPU推論にSpot Instances優先（最大90%削減）
- [ ] Reserved Instances / Savings Plans検討（1年コミットで最大72%削減）
- [ ] Lambda: メモリサイズ最適化（512MB推奨、Power Tuningで検証）
- [ ] ECS/EKS: Karpenterでアイドル時スケールダウン（consolidateAfter: 30s）
- [ ] 不要なNAT Gatewayの排除（VPCエンドポイント使用）

**LLMコスト削減**:
- [ ] ルーターに蒸留済み小型モデル使用（1.7Bクラス、コスト1/5）
- [ ] Bedrock Batch API使用（非同期処理で50%削減）
- [ ] Prompt Caching有効化（構造化プロンプトのシステム部分をキャッシュ、30-90%削減）
- [ ] ルーティング判断のDynamoDB/Redisキャッシュ（同一パターンのクエリでコストゼロ）
- [ ] トークン数制限の設定（max_tokensの適正化）

**監視・アラート**:
- [ ] AWS Budgets設定（月次予算の80%/100%でアラート）
- [ ] CloudWatch アラーム（Bedrockトークン使用量、Lambda実行時間）
- [ ] LC選択率の異常検知アラーム（50%超過でアラート）
- [ ] Cost Anomaly Detection有効化
- [ ] 日次コストレポートのSNS通知

**リソース管理**:
- [ ] 未使用リソースの定期削除（未使用ECRイメージ、古いLambdaバージョン）
- [ ] タグ戦略の徹底（Project, Environment, CostCenter）
- [ ] S3ライフサイクルポリシー（ログの90日後Glacier移行）
- [ ] 開発環境の夜間停止（EventBridgeスケジュール）
- [ ] CloudTrail / AWS Config有効化（監査証跡）

## 実験結果

### LaRAベンチマーク（In-Domain評価）

LaRAはRAG/LCルーティングの評価に特化したベンチマークで、0-4の段階評価でQAスコアを測定する。以下に主要な結果を示す（論文Table 3より）。

**Qwen3-235B-A22B回答モデルでの比較**:

| ルーター手法 | QAスコア | LC選択率(%) | ルーティング精度 |
|:---|---:|---:|---:|
| Always-LC | 3.51 | 100.0 | 0.40 |
| Always-RAG | 3.33 | 0.0 | 0.60 |
| Self-Route | 3.34 | 33.9 | 0.52 |
| Pre-Route (DeepSeek-R1) | 3.37 | 10.8 | 0.68 |
| Pre-Route (Qwen3-235B) | 3.40 | 18.3 | 0.68 |
| Pre-Route (蒸留1.7B) | 3.44 | 26.0 | 0.67 |

Always-LCが最高のQAスコア3.51を達成しているが、LC選択率は100%であり最大のコストがかかる。Pre-Route（蒸留1.7B）はQAスコア3.44を達成しつつLC選択率を26.0%に抑えており、Self-Route（QA 3.34、LC率33.9%）と比較してQAスコアで+0.10、LC選択率で-7.9ポイントの改善を同時に実現している。

**小型回答モデルでの効果（Qwen3-1.7B回答モデル）**:

| ルーター手法 | QAスコア | LC選択率(%) | ルーティング精度 |
|:---|---:|---:|---:|
| Always-LC | 2.29 | 100.0 | 0.18 |
| Self-Route | 2.40 | 31.7 | 0.48 |
| Pre-Route (DeepSeek-R1) | 2.88 | 1.8 | 0.83 |
| Pre-Route (蒸留1.7B) | 2.89 | 3.9 | 0.83 |

小型モデルではLC処理の性能が低い（Always-LC: QA 2.29）ため、RAG寄りのルーティングが有効となる。Pre-Route（蒸留1.7B）はQAスコア2.89を達成し、Self-Route（2.40）に対して+0.49の改善を示している。LC選択率はわずか3.9%であり、ほぼすべてのクエリがRAGに正しくルーティングされている。ルーティング精度は0.83とSelf-Routeの0.48を大幅に上回っている。

著者らは全結果の統計的有意性をp<0.01で確認しており、Cohen's d効果量は0.19-0.26と報告している。

### LongBench-v2ベンチマーク（OOD評価）

LongBench-v2はバイナリ多肢選択形式の評価ベンチマークであり、LaRAとは異なる評価プロトコルを用いたout-of-distribution（OOD）テストとなる（論文Table 4より）。

**Qwen3-235B-A22B回答モデルでの比較**:

| ルーター手法 | QAスコア | LC選択率(%) | ルーティング精度 |
|:---|---:|---:|---:|
| Always-LC | 0.52 | 100.0 | 0.52 |
| Always-RAG | 0.45 | 0.0 | 0.48 |
| Self-Route | 0.50 | 46.6 | 0.55 |
| Pre-Route (蒸留1.7B) | 0.50 | 28.8 | 0.61 |

OOD設定においてもPre-Route（蒸留1.7B）はSelf-Routeと同等のQAスコア0.50を維持しつつ、LC選択率を46.6%から28.8%に削減している。ルーティング精度は0.55から0.61に改善している。

**小型回答モデルでの効果（Qwen3-1.7B回答モデル、LongBench-v2）**:

| ルーター手法 | QAスコア | LC選択率(%) | ルーティング精度 |
|:---|---:|---:|---:|
| Self-Route | 0.34 | 31.1 | 0.68 |
| Pre-Route (蒸留1.7B) | 0.34 | 6.8 | 0.84 |

小型モデルではLC選択率の削減がとくに顕著であり（31.1%→6.8%）、ルーティング精度も0.68から0.84に向上している。

### コスト分析

著者らはルーティング判断1回あたりのコストを定量化している（論文Table 2より）。

| 手法 | モデル | 入力トークン | 出力トークン | 合計コスト |
|:---|:---|---:|---:|---:|
| Self-Route | Qwen3-235B | 2,600 | 27 | $0.00076 |
| Pre-Route | Qwen3-235B | 1,205 | 648 | $0.00107 |
| Pre-Route | Qwen3-1.7B（蒸留） | 1,205 | 670 | $0.00016 |

Self-RouteはQwen3-235Bを用いて$0.00076/判断であるのに対し、Pre-Routeの蒸留1.7Bモデルは$0.00016/判断と約1/5のコストで動作する。Pre-Routeの大規模モデル版（$0.00107）はSelf-Routeより高コストだが、これはルーティングコストが全体コストの4%未満であるため、LC選択率の削減によるトータルコスト削減で十分に相殺されると著者らは論じている。

### 消去法実験

構造化推論の各コンポーネントの寄与を検証する消去法実験の結果を示す（論文Table 5より、Qwen3-235B回答モデル）。

| 除去した要素 | QAスコア | LC選択率(%) | ルーティング精度 |
|:---|---:|---:|---:|
| なし（Full Pre-Route） | 3.38 | 20.7 | 0.68 |
| 決定規則を除去 | 3.38 | 45.3 | 0.57 |
| 振り返りを除去 | 3.33 | 20.8 | 0.65 |
| Step 1を除去 | 3.33 | 10.1 | 0.68 |
| Step 2を除去 | 3.37 | 27.0 | 0.66 |
| Step 4を除去 | 3.31 | 17.2 | 0.66 |

決定規則の除去はLC選択率を20.7%から45.3%に急増させ、ルーティング精度を0.68から0.57に低下させている。これは決定規則がLCの過剰選択を抑制する上で最も重要な構成要素であることを示している。Step 1-2はLCの必要性の誤判断を防ぎ、Step 3-6はシナリオ間の誤配分を防止する役割を担っていると著者らは分析している。

## 実運用への応用

Pre-Routeフレームワークは以下のような実運用シナリオで効果を発揮する。

**企業ナレッジベースQAシステム**: 社内文書（技術仕様書、契約書、マニュアル等）に対するQAシステムにおいて、Pre-Routeは文書タイプと質問の性質から最適な処理パスを事前に選択する。技術仕様の特定セクションに関する質問はRAGで効率的に処理し、契約書の全体的な整合性確認はLCに振り分けるといった判断が自動化される。

**マルチドキュメント推論**: 複数文書にまたがる推論が必要な場合と、単一文書の局所的な情報検索で十分な場合を判別する。著者らの実験では、情報分布パターン判断（Step 2）がこの判別に寄与していることが消去法実験で示されている。

**コスト制約のあるAPI提供サービス**: LLM APIの呼び出しコストが課題となるSaaSプロダクトにおいて、蒸留済みルーターモデル（$0.00016/判断）を用いることで、LC呼び出しの不要な発生を抑制しトータルコストを最適化する。

**レイテンシへの配慮**: ルーティング判断はメタデータのみを入力とするため、蒸留済み1.7Bモデルであれば推論時間は数十ミリ秒程度と推定される。一方、Self-RouteはRAGパイプライン全体を先に実行する必要があるため、LC向きクエリでは検索・再ランキングの処理時間がオーバーヘッドとなる。

## 関連研究

- **Self-Route（Li et al., EMNLP 2024）**: RAGで回答を試み、「回答不可能」と判断した場合にLCへフォールバックする手法。Gemini-1.5-Proで65%、GPT-4で39%のコスト削減を達成したと報告されている。Pre-Routeはこの受動的アプローチの限界を克服し、事前判断によるプロアクティブなルーティングを実現している。

- **Adaptive-RAG（Jeong et al., 2024）**: T5-Largeベースの分類器でクエリの複雑度を3クラス（simple/moderate/complex）に分類し、異なる検索戦略にルーティングする手法。RAG内部の戦略選択に特化しているのに対し、Pre-RouteはRAGとLCの二値選択という上位レベルの判断を行う点で相補的である。

- **Corrective RAG（CRAG, 2024）**: 検索結果の品質を評価し、低品質の場合にWeb検索へフォールバックする自己修正型RAGフレームワーク。RAGパイプライン内部の品質向上に焦点を当てており、Pre-Routeの「RAGかLCか」という判断とは異なるレイヤーの問題を扱っている。

- **RAGRouter（2025）**: 複数のRAGモデルへのクエリルーティングを学習する手法。RAG内部のモデル選択を扱っており、Pre-Routeとは相補的な関係にある。

## まとめと今後の展望

本論文の主要な成果は、LLMが潜在的にRAG/LCのルーティング能力を保持していることを示し、構造化プロンプトでこれを安定的に引き出すPre-Routeフレームワークを提案した点にある。蒸留済み1.7Bモデルにより、ルーティングコストを$0.00016/判断に抑えながら、Self-Routeを上回るルーティング精度（LaRAで0.67 vs 0.52）を達成している。

実務的な示唆として、本手法は既存のRAGパイプラインに最小限の変更で組み込める。メタデータ収集とルーターモデルの呼び出しを回答生成の前段に追加するだけで、LC呼び出しの最適化による大幅なコスト削減が見込まれる。

今後の研究方向としては、RAG/LCの二値選択からハイブリッド戦略（部分的LC + 部分的RAG）への拡張、マルチモーダル文書への対応、ストリーミング環境での動的ルーティングなどが考えられる。また、線形プローブ分析で示された表現空間の構造を活用し、ルーティング判断の解釈性をさらに高める研究も期待される。

## 参考文献

- **arXiv**: [https://arxiv.org/abs/2605.10235](https://arxiv.org/abs/2605.10235)
- **Self-Route (Li et al., 2024)**: [https://aclanthology.org/2024.emnlp-industry.66/](https://aclanthology.org/2024.emnlp-industry.66/)
- **Adaptive-RAG (Jeong et al., 2024)**: Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity
- **CRAG (Yan et al., 2024)**: [https://openreview.net/forum?id=JnWJbrnaUE](https://openreview.net/forum?id=JnWJbrnaUE)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/cfc6a5ad9e22fd](https://zenn.dev/0h_n0/articles/cfc6a5ad9e22fd)
