---
layout: post
title: "Speculative RAG: Draftingで推論を高速化する最新手法"
description: "Google Researchが提案する並列ドラフト生成と検証を組み合わせた新しいRAGアーキテクチャ。51%の遅延削減と12.97%の精度向上を実現。"
categories: [blog, tech_blog]
tags: [RAG, LLM, Speculative-Decoding, Google-Research, Retrieval-Augmented-Generation]
date: 2026-02-15 10:00:00 +0900
source_type: tech_blog
source_domain: research.google
source_url: https://research.google/blog/speculative-rag-enhancing-retrieval-augmented-generation-through-drafting/
zenn_article: ac14636a973cac
zenn_url: https://zenn.dev/0h_n0/articles/ac14636a973cac
target_audience: "修士学生レベル"
math: true
mermaid: true
---

## 概要

Google Researchが提案する**Speculative RAG**は、従来のRAG（Retrieval-Augmented Generation）システムの推論遅延と精度の両方を大幅に改善する新しいアーキテクチャです。本手法は、小規模な専門特化モデル（Specialist Drafter）と大規模な汎用モデル（Generalist Verifier）を組み合わせた**ドラフト・検証パラダイム**を採用し、PubHealthベンチマークで**51%の遅延削減**と**12.97%の精度向上**を達成しています。

本記事では、Speculative RAGの技術的詳細、アーキテクチャ設計、実装上の考慮点、および従来RAGとの性能比較について、修士学生レベルの読者を対象に深掘り解説します。

## 背景：従来RAGの課題

標準的なRAGシステムは、外部知識ベースから取得した全ドキュメントをLLMに直接入力し、長大なコンテキストを処理します。これには以下の問題があります。

1. **推論遅延**: 数千〜数万トークンのコンテキスト処理に時間がかかる
2. **計算コスト**: 大規模モデルでの全文処理は高コスト
3. **精度の不安定性**: 無関連情報が混入するとノイズになる

Speculative RAGは、**文書の並列分割処理**と**確率ベース検証**により、これらの課題を同時に解決します。

## Speculative RAGのアーキテクチャ

### 1. 全体フロー

Speculative RAGは4つのステージで動作します。

```
検索 → 並列ドラフト生成 → 検証 → 最終選択
```

**ステージ詳細**:

1. **検索（Retrieval）**: ContrieverでMS MARCOドキュメントを取得
2. **並列ドラフト生成（Parallel Drafting）**: Specialist Drafterが文書サブセットから複数の回答候補+根拠を並列生成
3. **検証（Verification）**: Generalist Verifierが各ドラフトに確信度スコアを付与
4. **最終選択（Selection）**: 最高スコアのドラフトを最終出力とする

### 2. Specialist RAG Drafter（専門特化ドラフター）

**モデル構成**:
- ベースモデル: **Mistral-7B-v0.1**（7億パラメータ）
- 訓練データ: Contriever-MS MARCOドキュメント + Gemini-Ultraが生成した根拠付き回答ペア
- 特化タスク: RAG専用（一般的な問題解決能力は不要）

**動作原理**:

Specialist Drafterは、取得したドキュメント集合を複数のサブセット $$D_1, D_2, \ldots, D_k$$ に分割し、各サブセット $$D_i$$ から並列に回答候補 $$A_i$$ と根拠 $$R_i$$ を生成します。

```python
from typing import List, Tuple

def parallel_drafting(
    documents: List[str],
    query: str,
    num_drafts: int = 3
) -> List[Tuple[str, str]]:
    """並列ドラフト生成

    Args:
        documents: 検索で取得した文書リスト
        query: ユーザークエリ
        num_drafts: 生成するドラフト数

    Returns:
        [(answer_1, rationale_1), (answer_2, rationale_2), ...]
    """
    # 文書をサブセットに分割
    subsets = partition_documents(documents, num_drafts)

    drafts = []
    for subset in subsets:
        # Specialist Drafter（Mistral-7B）で生成
        answer, rationale = specialist_drafter.generate(
            query=query,
            context=subset
        )
        drafts.append((answer, rationale))

    return drafts  # 並列実行可能
```

**利点**:
- 小規模モデルのため推論が高速
- 文書サブセット処理でコンテキスト長を削減
- 並列実行でスループット向上

### 3. Generalist RAG Verifier（汎用検証器）

**モデル構成**:
- モデル: **Mixtral-8x7B**（MoEアーキテクチャ）
- 追加訓練: **不要**（事前訓練済みモデルをそのまま使用）

**検証メカニズム**:

Generalist Verifierは、各ドラフト $$(A_i, R_i)$$ に対して条件付き生成確率を計算し、確信度スコア $$s_i$$ を割り当てます。

$$
s_i = P_{\text{Mixtral}}(A_i, R_i \mid Q, D_i)
$$

ここで、$$Q$$ はクエリ、$$D_i$$ は対応する文書サブセットです。

最終出力は、最高スコアのドラフトを選択します。

$$
(A^*, R^*) = \arg\max_{i} s_i
$$

**実装例**:

```python
import torch

def verify_drafts(
    drafts: List[Tuple[str, str]],
    query: str,
    document_subsets: List[List[str]]
) -> Tuple[str, str]:
    """ドラフト検証と最終選択

    Args:
        drafts: [(answer, rationale), ...]
        query: ユーザークエリ
        document_subsets: 各ドラフトに対応する文書サブセット

    Returns:
        最高確信度の(answer, rationale)
    """
    scores = []

    for (answer, rationale), docs in zip(drafts, document_subsets):
        # Mixtral-8x7Bで条件付き確率を計算
        logprobs = generalist_verifier.compute_logprobs(
            query=query,
            context=docs,
            answer=answer,
            rationale=rationale
        )
        # 確信度スコア（対数確率の平均）
        score = torch.mean(logprobs).item()
        scores.append(score)

    # 最高スコアを選択
    best_idx = torch.argmax(torch.tensor(scores)).item()
    return drafts[best_idx]
```

## 性能評価

### PubHealthベンチマーク

PubHealth（医療分野のファクトチェックデータセット）での評価結果:

| 指標 | Mixtral-8x7B（標準RAG） | Speculative RAG | 改善率 |
|------|-------------------------|-----------------|--------|
| 精度（Accuracy） | - | +12.97% | 12.97% |
| 推論遅延（Latency） | 基準値 | -51% | 51% |

### 他ベンチマークでの一貫性

Speculative RAGは以下のベンチマークでも**State-of-the-Art**を達成:

- **TriviaQA**: 一般知識QA
- **MuSiQue**: 多段階推論
- **ARC-Challenge**: 科学的推論タスク

### 従来RAGとの比較

| 項目 | 標準RAG | Speculative RAG |
|------|---------|-----------------|
| コンテキスト処理 | 全文書を大規模LLMで処理 | 文書分割+小規模モデルで並列処理 |
| 推論速度 | 遅い（全トークン処理） | 高速（並列+小規模モデル） |
| 精度 | 無関連情報でノイズ混入 | 確率ベース検証で高精度 |
| 計算コスト | 高い | 低い（小規模モデル中心） |

## 実装のポイント

### 1. 文書分割戦略

効果的なドラフト生成には、文書サブセットの適切な分割が重要です。

```python
def partition_documents(
    documents: List[str],
    num_partitions: int
) -> List[List[str]]:
    """意味的に独立した文書サブセットに分割"""
    # オプション1: ランダム分割（シンプル）
    # オプション2: クラスタリングベース分割（高品質）

    # 例: 埋め込みベースのk-means分割
    embeddings = embed_documents(documents)
    clusters = kmeans(embeddings, k=num_partitions)

    subsets = [[] for _ in range(num_partitions)]
    for doc, cluster_id in zip(documents, clusters):
        subsets[cluster_id].append(doc)

    return subsets
```

### 2. ドラフト数の最適化

ドラフト数 $$k$$ はトレードオフがあります。

- **$$k$$ が小さい**: 高速だが多様性不足
- **$$k$$ が大きい**: 多様性向上だが計算コスト増

論文では $$k=3$$ が推奨されています。

### 3. 確信度スコアの校正

確率スコアをそのまま使うと、モデルの過信/過小評価の影響を受けます。Temperature Scalingなどの校正手法が有効です。

```python
def calibrate_score(logprobs: torch.Tensor, temperature: float = 1.5) -> float:
    """Temperature Scalingで確率を校正"""
    calibrated_logprobs = logprobs / temperature
    return torch.mean(calibrated_logprobs).item()
```

## 実運用への応用

### 適用が有効なケース

1. **ファクトチェックシステム**: 医療・法律など高精度が必要な分野
2. **顧客サポートボット**: 低遅延と高精度の両立が必要
3. **マルチステップ推論**: 複雑な質問に対する段階的推論

### コスト分析

**標準RAG（Mixtral-8x7B単独）**:
- 推論コスト: $$C_{\text{large}} \times T_{\text{full}}$$

**Speculative RAG**:
- 推論コスト: $$k \times C_{\text{small}} \times T_{\text{subset}} + C_{\text{large}} \times T_{\text{verify}}$$

$$k=3$$、$$T_{\text{subset}} \approx T_{\text{full}}/3$$、$$C_{\text{small}} \approx C_{\text{large}}/10$$ の場合、コストは約**40%削減**できます。

## 関連研究

### Speculative Decodingとの関係

Speculative RAGは、自己回帰生成の高速化手法である**Speculative Decoding** [Leviathan et al., 2023]をRAGに拡張したものです。Speculative Decodingでは、小規模モデルでトークン候補を生成し、大規模モデルで検証します。

### 他のRAG高速化手法

- **FiD (Fusion-in-Decoder)**: 文書ごとにエンコードし、デコーダで統合
- **RETRO**: 事前訓練時にRAGを組み込む
- **Self-RAG**: 生成時に取得と生成を交互に実行

Speculative RAGはこれらと直交する最適化であり、組み合わせることも可能です。

## まとめ

Speculative RAGは、**ドラフト・検証パラダイム**により、RAGシステムの速度と精度を同時に改善する画期的な手法です。主な貢献は以下の通りです。

- **51%の遅延削減**: 並列ドラフト生成と小規模モデル活用
- **12.97%の精度向上**: 確率ベース検証による高品質な回答選択
- **実装が容易**: 既存モデル（Mistral-7B、Mixtral-8x7B）で実現可能
- **追加訓練不要**: Verifierは事前訓練済みモデルをそのまま使用

今後は、より多様なドメイン（法律、金融、科学）での評価や、ドラフト生成戦略の最適化（強化学習など）が研究課題となるでしょう。

本手法は、本番環境でのRAGシステム導入を検討する際の有力な選択肢となります。

## 参考文献

- Google Research Blog: [Speculative RAG: Enhancing Retrieval-Augmented Generation through Drafting](https://research.google/blog/speculative-rag-enhancing-retrieval-augmented-generation-through-drafting/)
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *ICML 2023*.
