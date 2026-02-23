---
layout: post
title: "NeurIPS 2024論文解説: HippoRAG — 海馬モデルに基づくLLMの長期記憶型RAGアーキテクチャ"
description: "人間の海馬の記憶形成メカニズムをRAGに適用し、知識グラフとベクトル検索を統合した長期記憶型検索拡張生成の詳細解説"
categories: [blog, paper, conference]
tags: [RAG, memory, knowledge-graph, LLM, langgraph, NeurIPS]
date: 2026-02-23 11:00:00 +0900
source_type: conference
conference: NeurIPS 2024
source_url: https://arxiv.org/abs/2405.14831
zenn_article: 3901eb498f526c
zenn_url: https://zenn.dev/0h_n0/articles/3901eb498f526c
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [NeurIPS 2024で採択された HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831) の解説記事です。

## 論文概要（Abstract）

著者ら（Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su, Ohio State University, 2024年5月）は、人間の海馬（hippocampus）の記憶形成メカニズムにインスパイアされたRAGフレームワーク「HippoRAG」を提案している。従来のRAGが独立したチャンクをベクトル検索で取得するのに対し、HippoRAGはLLMを新皮質（neocortex）、知識グラフ（KG）を海馬インデックスに見立て、パターン分離（pattern separation）とパターン完成（pattern completion）の2段階で関連情報を連鎖的に検索する。マルチホップ推論が必要なQAタスクにおいて、標準RAGやIRCoTと比較して大幅な精度向上を達成したと報告されている。

この記事は [Zenn記事: LangGraph Store APIで実装するマルチエージェントRAGの共有メモリと長期記憶](https://zenn.dev/0h_n0/articles/3901eb498f526c) の深掘りです。

## 情報源

- **会議名**: NeurIPS 2024（Neural Information Processing Systems）
- **年**: 2024
- **URL**: [https://arxiv.org/abs/2405.14831](https://arxiv.org/abs/2405.14831)
- **著者**: Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su
- **公開実装**: [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)

## カンファレンス情報

**NeurIPS 2024について**:
NeurIPS（Neural Information Processing Systems）は機械学習・人工知能分野の最高峰会議の1つであり、2024年は15,671件の投稿から3,604件が採択された（採択率約23%）。HippoRAGはメインカンファレンスの論文として採択されている。

## 技術的詳細（Technical Details）

### 神経科学的背景: 海馬の記憶形成モデル

HippoRAGの設計は、CLS理論（Complementary Learning Systems theory）に基づいている。人間の記憶システムは以下の2つのサブシステムで構成される。

1. **新皮質（Neocortex）**: 汎用的な知識表現。ゆっくりと統合される長期記憶。LLMのパラメータ知識に対応
2. **海馬（Hippocampus）**: 新しいエピソードの高速エンコーディング。パターン分離（類似経験の区別）とパターン完成（部分的な手がかりから完全な記憶を復元）を担当

### HippoRAGの3コンポーネントアーキテクチャ

**1. LLM（新皮質モデル）**: OpenKIE（Open Knowledge Information Extraction）

LLMを使って入力文書からトリプル（主語, 述語, 目的語）を抽出し、知識グラフを構築する。

$$
\text{KG} = \{(s, p, o) \mid (s, p, o) = \text{LLM}_{\text{OpenKIE}}(d), d \in \mathcal{D}\}
$$

ここで、
- $\mathcal{D}$: 文書コーパス
- $s$: 主語エンティティ
- $p$: 述語（関係）
- $o$: 目的語エンティティ
- $\text{LLM}_{\text{OpenKIE}}$: トリプル抽出用LLM

**2. 海馬インデックス**: 知識グラフ + ベクトルインデックス

抽出されたトリプルから知識グラフを構築し、各エンティティノードにembeddingベクトルを付与する。

```python
from typing import TypedDict

class KGNode(TypedDict):
    """知識グラフのノード"""
    entity: str           # エンティティ名
    embedding: list[float] # ベクトル表現（1024次元）
    passages: list[str]    # 関連する元文書のID
    neighbors: list[str]   # 隣接エンティティ

class KGEdge(TypedDict):
    """知識グラフのエッジ"""
    source: str    # 主語エンティティ
    relation: str  # 述語
    target: str    # 目的語エンティティ
    passage_id: str # 元文書ID
```

**3. パターン分離 + パターン完成による検索**

検索は2段階で行われる。

**ステップ1: パターン分離（Pattern Separation）**

クエリからエンティティを抽出し、知識グラフ上で最も関連性の高いノードを特定する。

$$
\text{query\_entities} = \text{LLM}_{\text{NER}}(q)
$$

$$
\text{seed\_nodes} = \arg\max_{v \in \text{KG}} \text{cos\_sim}(\text{emb}(q_e), \text{emb}(v)) \quad \forall q_e \in \text{query\_entities}
$$

ここで、
- $q$: ユーザークエリ
- $q_e$: クエリから抽出されたエンティティ
- $\text{emb}(\cdot)$: embeddingベクトル
- $\text{cos\_sim}$: コサイン類似度

**ステップ2: パターン完成（Pattern Completion）**

Seed nodesからPersonalized PageRank（PPR）アルゴリズムでグラフを探索し、関連ノードを連鎖的に発見する。

$$
\mathbf{r} = \alpha \cdot \mathbf{s} + (1 - \alpha) \cdot \mathbf{A}^T \mathbf{r}
$$

ここで、
- $\mathbf{r}$: ノードのランキングスコアベクトル
- $\mathbf{s}$: seed nodesの初期スコアベクトル（seed nodesのみ1、他は0）
- $\mathbf{A}$: 知識グラフの正規化隣接行列
- $\alpha$: テレポート確率（デフォルト0.5）

PPRの上位ノードに関連付けられた元文書（パッセージ）を検索結果として返す。

```python
import numpy as np
from scipy.sparse import csr_matrix

def personalized_pagerank(
    adjacency: csr_matrix,
    seed_scores: np.ndarray,
    alpha: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Personalized PageRankの計算

    Args:
        adjacency: 正規化隣接行列 (n x n)
        seed_scores: seed nodesの初期スコア (n,)
        alpha: テレポート確率
        max_iter: 最大反復回数
        tol: 収束判定の閾値

    Returns:
        各ノードのランキングスコア (n,)
    """
    n = adjacency.shape[0]
    rank = seed_scores.copy()

    for _ in range(max_iter):
        prev_rank = rank.copy()
        rank = alpha * seed_scores + (1 - alpha) * adjacency.T.dot(rank)
        if np.linalg.norm(rank - prev_rank, 1) < tol:
            break

    return rank
```

### LangGraph Store APIとの統合パターン

HippoRAGの知識グラフ型メモリは、LangGraph Store APIの名前空間設計と以下のように統合できる。

| HippoRAGコンポーネント | LangGraph Store対応 | 用途 |
|---------------------|-------------------|------|
| KGノード（エンティティ） | `store.put(("rag", "kg", "entities"), entity_id, ...)` | エンティティのベクトル表現を格納 |
| KGエッジ（関係） | `store.put(("rag", "kg", "relations"), edge_id, ...)` | トリプル関係を格納 |
| パッセージ参照 | `store.put(("rag", "shared", "knowledge"), passage_id, ...)` | 元文書チャンクのキャッシュ |
| PPRスコア | Checkpointerの中間状態 | セッション内の検索スコア |

## 実装のポイント（Implementation）

### 知識グラフ構築のプロンプト設計

著者らはOpenKIEのプロンプトとして、以下の構造を使用している。

```python
OPENKIE_PROMPT = """
Given the following passage, extract all knowledge triples
in the format (subject, predicate, object).

Rules:
- subject and object must be named entities or noun phrases
- predicate must be a verb or preposition
- each triple must be atomic (one fact per triple)
- output JSON array format

Passage: {passage}

Output:
"""

async def extract_triples(
    passage: str,
    llm: ChatAnthropic,
) -> list[tuple[str, str, str]]:
    """文書からトリプルを抽出

    Args:
        passage: 入力テキスト
        llm: LLMクライアント

    Returns:
        (subject, predicate, object) のリスト
    """
    response = await llm.ainvoke(
        OPENKIE_PROMPT.format(passage=passage)
    )
    triples = json.loads(response.content)
    return [(t["subject"], t["predicate"], t["object"]) for t in triples]
```

### pgvectorでのエンティティインデックス

HippoRAGのエンティティembeddingはpgvectorで効率的に検索できる。

```sql
-- エンティティテーブル
CREATE TABLE kg_entities (
    id SERIAL PRIMARY KEY,
    entity_name TEXT NOT NULL,
    embedding vector(1024),
    passage_ids TEXT[] NOT NULL
);

-- HNSWインデックス
CREATE INDEX ON kg_entities
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- パターン分離: クエリエンティティに最も近いKGノードを検索
SELECT entity_name, passage_ids,
       1 - (embedding <=> $1::vector) AS similarity
FROM kg_entities
ORDER BY embedding <=> $1::vector
LIMIT 10;
```

### 実装上の注意点

1. **トリプル抽出の品質**: OpenKIEの精度はLLMの能力に依存する。著者らはGPT-4を使用しているが、Claude Sonnet 4.6でも同等の精度が得られると推測される。ただし、ドメイン固有の用語に対してはFew-shotプロンプトが有効
2. **PPRの計算コスト**: グラフノード数が100万を超える場合、PPRの収束に数秒を要する。近似PPR（例: ISTA-based PPR）の採用を検討すべき
3. **知識グラフの更新**: 新文書追加時にトリプルの再抽出と既存グラフとのマージが必要。エンティティの同定（entity resolution）が未解決の課題

## 実験結果（Results）

著者らは、マルチホップQAの3つのベンチマークで評価を行っている。

| 手法 | MuSiQue (F1) | 2WikiMQA (F1) | HotpotQA (F1) |
|------|-------------|--------------|--------------|
| Standard RAG | 20.8 | 39.5 | 40.1 |
| IRCoT | 26.5 | 55.6 | 58.5 |
| HippoRAG | **37.1** | **68.9** | **63.4** |

論文Table 1より、HippoRAGはMuSiQueでStandard RAGと比較してF1を20.8→37.1に改善（+78%）したと報告されている。特にマルチホップ推論（複数文書を連鎖的に辿る必要がある質問）で大きな改善が見られる。

**分析**: 著者らは、パターン完成（PPR）による連鎖的な情報探索がマルチホップ推論に有効であると分析している。Standard RAGは各チャンクを独立に検索するため、「AがBに関連し、BがCに関連する」という連鎖を発見できない。HippoRAGは知識グラフ上のパス探索でこの連鎖を自然に辿れる。

## 実運用への応用（Practical Applications）

Zenn記事で解説したLangGraph Store APIのマルチエージェントRAGに、HippoRAGの知識グラフベースメモリを統合することで、以下の改善が期待できる。

1. **クロスエージェントの知識連鎖**: Query AnalyzerがエンティティAを発見し、RetrieverがKG上でAからBへの関係を辿り、SynthesizerがBに関連する文書を活用する
2. **メモリのグラフ構造化**: Store APIに格納するメモリをトリプル形式で構造化することで、名前空間を跨いだ関係性検索が可能になる
3. **共有メモリの品質向上**: 成功パターンをトリプルとして知識グラフに蓄積し、PPRで関連パターンを連鎖的に発見

**制約**: HippoRAGの知識グラフ構築にはLLM呼び出しが必要であり、文書数が多い場合のインデキシングコストが高い。リアルタイム更新が必要な用途には不向きである。

## 関連研究（Related Work）

- **GraphRAG**（Microsoft, 2024）: コミュニティ検出ベースのグラフRAG。HippoRAGとの違いは、GraphRAGがグラフ全体の要約を生成するのに対し、HippoRAGはPPRで局所的な探索を行う点
- **RAPTOR**（Sarthi et al., 2024）: 階層的クラスタリングによるRAG。チャンク間の関係を木構造で表現するが、知識グラフのような柔軟な関係表現はできない
- **LangGraph Store API**: pgvectorによるベクトル検索を提供するが、グラフ構造の検索は未サポート。HippoRAGのPPRメカニズムを外部グラフDB（Neo4j）と組み合わせて補完可能

## まとめと今後の展望

HippoRAGは、海馬の記憶形成メカニズム（パターン分離 + パターン完成）をRAGに適用することで、マルチホップ推論の精度を大幅に向上させた。知識グラフ + ベクトルインデックスのハイブリッド設計は、LangGraph Store APIの名前空間ベースメモリ管理を補完する有力なアプローチである。

今後の研究方向として、リアルタイムの知識グラフ更新メカニズム、大規模グラフでの効率的なPPR計算、およびマルチエージェント間でのグラフメモリ共有プロトコルが挙げられる。

## 参考文献

- **Conference URL**: [https://arxiv.org/abs/2405.14831](https://arxiv.org/abs/2405.14831)
- **NeurIPS 2024 Proceedings**: [https://proceedings.neurips.cc/paper_files/paper/2024/](https://proceedings.neurips.cc/paper_files/paper/2024/)
- **Code**: [https://github.com/OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/3901eb498f526c](https://zenn.dev/0h_n0/articles/3901eb498f526c)
