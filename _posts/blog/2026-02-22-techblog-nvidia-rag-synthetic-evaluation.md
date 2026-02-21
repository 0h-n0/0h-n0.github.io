---
layout: post
title: "NVIDIA解説: 合成データによるRAGパイプライン評価・最適化 — NeMo Curator実践ガイド"
description: "NVIDIA NeMo CuratorとNIMで合成QAデータを生成し、RAGパイプラインの検索精度を自動評価・改善する手法を解説"
categories: [blog, tech_blog]
tags: [NVIDIA, NeMo, RAG, evaluation, synthetic-data, embedding, fine-tuning, langgraph]
date: 2026-02-22 12:00:00 +0900
source_type: tech_blog
source_domain: developer.nvidia.com
source_url: https://developer.nvidia.com/blog/evaluating-and-enhancing-rag-pipeline-performance-using-synthetic-data/
zenn_article: f15c5b29dc16ed
zenn_url: https://zenn.dev/0h_n0/articles/f15c5b29dc16ed
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

NVIDIA Technical Blogの本記事は、**合成データを用いたRAGパイプラインの自動評価・最適化**フレームワークを解説しています。NVIDIA NeMo CuratorとNIM microservicesを活用し、(1) 高品質な合成QAペアの自動生成、(2) 埋め込みモデルの定量評価、(3) Hard-Negative Miningによるファインチューニング、の3段階パイプラインを提供します。人手アノテーション比で**Recall@5の偏差が4.57%以内**という高精度な合成データ品質を実証しています。

この記事は [Zenn記事: LangGraph Agentic RAGの本番運用設計：マルチソースルーティングと評価駆動リランキング](https://zenn.dev/0h_n0/articles/f15c5b29dc16ed) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://developer.nvidia.com/blog/evaluating-and-enhancing-rag-pipeline-performance-using-synthetic-data/](https://developer.nvidia.com/blog/evaluating-and-enhancing-rag-pipeline-performance-using-synthetic-data/)
- **組織**: NVIDIA（Deep Learning / Product Marketing）
- **著者**: Vinay Raman（Senior Deep Learning Scientist）, Nirmal Kumar Juluru（Product Marketing Manager）
- **発表日**: 2025年4月7日

## 技術的背景（Technical Background）

RAGシステムの品質評価には**大量のアノテーション付きQAペア**が必要ですが、手動作成は時間とコストがかかります。特にドメイン固有の文書（社内マニュアル、技術仕様書等）では、既存のベンチマークデータセット（FiQA、Natural Questions等）がそのまま使えません。

Zenn記事では、RAGASメトリクス（Faithfulness, Context Precision等）を用いた**A/B評価パイプライン**を紹介しましたが、評価の前提となる**テストデータセットの作成**が実務上のボトルネックでした。NVIDIAのアプローチは、LLMを使って合成QAペアを自動生成し、このボトルネックを解消します。

### なぜ合成データが有効か

既存のベンチマークデータセットは以下の問題を抱えています。

1. **ドメイン不一致**: FiQAやNQは汎用ドメイン。社内文書の専門用語やコンテキストをカバーしない
2. **評価バイアス**: 公開ベンチマークで高スコアのモデルが、自社データで低スコアになることがある
3. **スケーラビリティ**: 手動アノテーションは100問作成に数日かかる

合成データ生成は、これらの問題を解決しつつ、**Recall@5で人手アノテーションとの偏差4.57%以内**の精度を達成しています。

## 実装アーキテクチャ（Architecture）

### 3段階パイプライン

```
ソース文書
     |
     v
[Stage 1: QAペア生成] ← NVIDIA NIM (Llama-3.1-70B)
     |
     v
[Stage 2: 品質フィルタリング]
     ├── 埋め込みモデルジャッジ（コサイン類似度閾値）
     └── 回答可能性フィルタ（LLM-as-judge）
     |
     v
[Stage 3: Hard-Negative Mining]
     ├── Top-K選択
     ├── 閾値ベース選択
     └── Positive-Aware Mining
     |
     v
合成評価データセット + ファインチューニングデータ
```

### Stage 1: QAペア生成

NVIDIA NIM上のLlama-3.1-70B-Instructを使用し、ソース文書から質問-回答ペアを生成します。

```python
from nemo_curator.synthetic import QAGenerator

def generate_qa_pairs(
    documents: list[str],
    model_id: str = "meta/llama-3.1-70b-instruct",
    max_pairs_per_doc: int = 5
) -> list[dict]:
    """NeMo Curatorで合成QAペアを生成

    品質基準:
    - クエリ独立性: 文書を見なくても理解できる質問
    - 現実性: 実際のユーザーが尋ねそうな質問
    - 多様性: 同一文書から異なるタイプの質問を生成
    - 文脈関連性: 文書内容に基づいた回答可能な質問
    """
    generator = QAGenerator(
        model_id=model_id,
        quality_criteria=[
            "query_independence",
            "realism",
            "diversity",
            "contextual_relevance"
        ]
    )
    return generator.generate(documents, max_pairs=max_pairs_per_doc)
```

### Stage 2: 品質フィルタリング

生成されたQAペアの品質を2段階でフィルタリングします。

**埋め込みモデルジャッジ**: 質問のコサイン類似度で難易度を制御します。

$$
\text{difficulty}(q, D) = 1 - \max_{d \in D} \cos(\text{embed}(q), \text{embed}(d))
$$

類似度が高い（=簡単すぎる）質問や低すぎる（=回答不能）質問をフィルタリングします。閾値はパーセンタイルベースで設定し、**70パーセンタイルが最も低偏差**（人手アノテーションとの差が最小）でした。

**回答可能性フィルタ（LLM-as-judge）**:

Llama-3.1-70b-Instructを使用し、生成された質問がソース文書から回答可能かを判定します。このフィルタの精度は**Precision 94%、Recall 90%**です。

```python
def filter_answerable(
    qa_pairs: list[dict],
    source_docs: list[str],
    llm_judge: str = "meta/llama-3.1-70b-instruct"
) -> list[dict]:
    """回答可能性フィルタ: LLM-as-judgeで判定"""
    filtered = []
    for pair in qa_pairs:
        prompt = f"""Given the source document below, determine if the
following question can be answered solely from this document.

Source: {pair['source_doc']}
Question: {pair['question']}

Answer 'YES' if answerable, 'NO' otherwise."""

        response = llm_invoke(llm_judge, prompt)
        if "YES" in response.upper():
            filtered.append(pair)
    return filtered
```

### Stage 3: Hard-Negative Mining

ファインチューニング用に、検索モデルが間違えやすい「難しい負例」を自動生成します。

**3つのマイニング手法**:

1. **Top-K選択**: 質問に最も類似度が高い非正解文書をK個選択

$$
\text{HN}_{\text{topk}}(q) = \arg\text{top-}K_{d \in D \setminus D^+} \cos(\text{embed}(q), \text{embed}(d))
$$

2. **閾値ベース選択**: 類似度が指定範囲内の文書を選択

$$
\text{HN}_{\text{thresh}}(q) = \{d \in D \setminus D^+ \mid \alpha \leq \cos(\text{embed}(q), \text{embed}(d)) \leq \beta\}
$$

3. **Positive-Aware Mining**: 正例の類似度スコアを基準に、95%の値を負例閾値に設定

$$
\text{threshold}_{\text{neg}} = 0.95 \times \cos(\text{embed}(q), \text{embed}(d^+))
$$

```python
def mine_hard_negatives(
    query: str,
    positive_docs: list[str],
    corpus: list[str],
    embedding_model,
    method: str = "positive_aware",
    k: int = 5
) -> list[str]:
    """Hard-Negative Mining: 3手法から選択"""
    q_emb = embedding_model.encode(query)
    pos_embs = [embedding_model.encode(d) for d in positive_docs]
    all_embs = [embedding_model.encode(d) for d in corpus]

    # 全文書のスコア計算
    scores = [cosine_similarity(q_emb, e) for e in all_embs]

    if method == "topk":
        # 正例を除いて上位K件
        candidates = [
            (d, s) for d, s in zip(corpus, scores)
            if d not in positive_docs
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:k]]

    elif method == "positive_aware":
        # 正例スコアの95%を閾値に
        pos_score = max(
            cosine_similarity(q_emb, pe) for pe in pos_embs
        )
        threshold = 0.95 * pos_score
        candidates = [
            (d, s) for d, s in zip(corpus, scores)
            if d not in positive_docs and s <= threshold and s > 0.3
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:k]]
```

## パフォーマンス最適化（Performance）

### 合成データの品質検証

FiQAデータセットでの比較結果です。

| 評価指標 | 手動アノテーション | 合成データ | 偏差 |
|---------|-----------------|----------|------|
| Recall@5 | 0.782 | 0.748 | 4.57% |
| Recall@5（最適フィルタ） | 0.782 | 0.753 | 3.71% |

**相関係数: 0.91** — 合成データと手動アノテーションのモデルランキングが高い一致度を示します。

### 埋め込みモデル別の偏差

| 埋め込みモデル | 平均偏差 |
|-------------|---------|
| nv-embedqa-e5-v5 | 6.98% |
| nv-embedqa-mistral-7b-v2 | 6.02% |
| intfloat/e5-large-unsupervised | 6.43% |

**nv-embedqa-mistral-7b-v2が最も低偏差**。ドメイン固有のデータで事前学習されたモデルが安定した結果を出す傾向にあります。

### フィルタ閾値の影響

| パーセンタイル閾値 | Recall@5偏差 |
|----------------|------------|
| 50th | 5.65% |
| 60th | 5.12% |
| **70th** | **4.57%** |
| 80th | 4.89% |
| 90th | 5.21% |

**70パーセンタイルが最適**。厳しすぎるフィルタは有用な難問を除去し、緩すぎるフィルタはノイズを含みます。

## 運用での学び（Production Lessons）

### 評価データセットの更新戦略

文書の追加・更新に伴い、評価データセットも定期的に再生成する必要があります。

1. **トリガー**: 文書コーパスの10%以上が更新された場合
2. **差分生成**: 新規文書のみからQAペアを追加生成
3. **品質チェック**: 既存QAペアの回答可能性を再検証

### RAGASとの統合

NVIDIAの合成データ生成パイプラインは、Zenn記事のRAGAS評価パイプラインとシームレスに統合できます。

```python
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness
from ragas.dataset_schema import SingleTurnSample
from ragas import EvaluationDataset

def evaluate_with_synthetic_data(
    rag_pipeline: callable,
    synthetic_qa_pairs: list[dict]
) -> dict:
    """合成データでRAGパイプラインをRAGAS評価"""
    samples = []
    for pair in synthetic_qa_pairs:
        result = rag_pipeline(pair["question"])
        samples.append(SingleTurnSample(
            user_input=pair["question"],
            response=result["answer"],
            retrieved_contexts=result["contexts"],
            reference=pair["answer"],
        ))
    dataset = EvaluationDataset(samples=samples)
    return evaluate(
        dataset=dataset,
        metrics=[context_precision, faithfulness],
    )
```

### 実務での推奨ワークフロー

1. **合成データ生成（Stage 1-2）**: NeMo Curatorで100-500件のQAペアを生成
2. **ベースライン評価**: RAGASで現行パイプラインのスコアを計測
3. **Hard-Negative Mining（Stage 3）**: ファインチューニングデータを生成
4. **埋め込みモデル改善**: Hard-Negativeでファインチューニング
5. **A/B評価**: 改善前後のRAGASスコアを比較
6. **継続監視**: LangSmithで本番メトリクスを追跡

## 学術研究との関連（Academic Connection）

本ブログの手法は以下の学術研究に基づいています。

- **HyDE** (Gao et al., 2022): 仮想文書生成による検索改善。NVIDIAの合成QA生成は同様のLLM活用パターン
- **ARES** (Saad-Falcon et al., 2024): LLM-as-judgeによるRAG自動評価。NVIDIAの回答可能性フィルタと同系統
- **Contriever** (Izacard et al., 2022): 対照学習による密検索モデル。Hard-Negative Miningの理論的基盤

## まとめと実践への示唆

NVIDIAの合成データ評価フレームワークは、RAGパイプラインの**継続的品質管理**を実現する実践的なツールです。人手アノテーションとの偏差4.57%以内の精度で合成QAペアを生成でき、RAGASメトリクスとの組み合わせにより、Zenn記事の評価駆動リランキングを**データ生成段階から自動化**できます。

特に、Positive-Aware Hard-Negative Miningは、埋め込みモデルのファインチューニングにおいて「正例の95%類似度」を負例閾値にするという実用的なヒューリスティクスを提供しており、ドメイン固有のRAGパイプライン改善に直接適用できます。

## 参考文献

- **Blog URL**: [https://developer.nvidia.com/blog/evaluating-and-enhancing-rag-pipeline-performance-using-synthetic-data/](https://developer.nvidia.com/blog/evaluating-and-enhancing-rag-pipeline-performance-using-synthetic-data/)
- **NeMo Curator**: [https://github.com/NVIDIA/NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/f15c5b29dc16ed](https://zenn.dev/0h_n0/articles/f15c5b29dc16ed)
