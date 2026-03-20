---
layout: post
title: "ICLR 2025論文解説: MMTEB — Massive Multilingual Text Embedding Benchmark"
description: "MTEBを250言語・500超データセットに拡張した多言語埋め込みベンチマークMMTEBの設計と評価結果を解説"
categories: [blog, paper, conference]
tags: [embedding, benchmark, multilingual, MTEB, MMTEB, evaluation, ICLR]
date: 2026-03-21 11:00:00 +0900
source_type: conference
conference: ICLR 2025
source_url: https://arxiv.org/abs/2502.13595
zenn_article: 1798f7e5c5fd69
zenn_url: https://zenn.dev/0h_n0/articles/1798f7e5c5fd69
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [arXiv:2502.13595 (MMTEB: Massive Multilingual Text Embedding Benchmark)](https://arxiv.org/abs/2502.13595) の解説記事です。本論文はICLR 2025に採択されています。

## 論文概要（Abstract）

MMTEB（Massive Multilingual Text Embedding Benchmark）は、英語中心であったMTEBを**250言語以上・500超データセット**に拡張した大規模多言語テキスト埋め込みベンチマークである。著者ら（Enevoldsen, Chung, Muennighoff et al., 50名超の共著者によるコミュニティ論文）は、分類・クラスタリング・検索・再ランキング・文類似度・要約・対訳マイニングの7タスクカテゴリを統合し、合計10億件超のアノテーション済みサンプルを含むベンチマークを構築した。

この記事は [Zenn記事: Embeddingモデルの本番評価パイプライン構築](https://zenn.dev/0h_n0/articles/1798f7e5c5fd69) の深掘りです。

## 情報源

- **会議名**: ICLR 2025（International Conference on Learning Representations 2025）
- **年**: 2025
- **URL**: https://arxiv.org/abs/2502.13595
- **著者**: Kenneth Enevoldsen, Isaac Chung, Niklas Muennighoff et al.（50名超）
- **発表形式**: Poster

## カンファレンス情報

**ICLRについて**: ICLRは機械学習・深層学習分野の最高峰会議の一つであり、表現学習（Representation Learning）に焦点を当てた研究コミュニティである。採択率は通常25-30%程度であり、高い競争率を持つ。MMTEBは2025年のICLRに採択された。

## 技術的詳細（Technical Details）

### MTEBからの拡張

| 項目 | MTEB (2022) | MMTEB (2025) |
|------|-------------|--------------|
| 主対象言語 | 英語中心 | **250言語以上** |
| データセット数 | 56 | **500超** |
| タスクカテゴリ | 7 | 7（多言語に拡張） |
| アノテーション数 | — | **10億件超** |
| 低リソース言語 | ほぼ未対応 | **体系的に収録** |
| データ品質検証 | 限定的 | **ラベルエラー・漏洩修正済み** |
| 軽量版 | なし | **MMTEB-Lite提供** |

### タスクカテゴリと評価指標

```mermaid
graph LR
    MMTEB[MMTEB<br/>500+ datasets<br/>250+ languages] --> CLS[Classification<br/>Accuracy]
    MMTEB --> CLU[Clustering<br/>V-measure]
    MMTEB --> RET[Retrieval<br/>nDCG@10]
    MMTEB --> RER[Reranking<br/>MAP]
    MMTEB --> STS[STS<br/>Spearman]
    MMTEB --> SUM[Summarization<br/>Spearman]
    MMTEB --> BIT[BitextMining<br/>F1]
```

| タスクカテゴリ | 主指標 | 説明 |
|-------------|-------|------|
| Classification | Accuracy | テキスト分類の正解率 |
| Clustering | V-measure | クラスタリング品質（均質性と完全性の調和平均） |
| Retrieval | nDCG@10 | 検索結果上位10件のランク品質 |
| Reranking | MAP | 再ランキング後の平均適合率 |
| STS | Spearman相関 | 文対文類似度とコサイン類似度の相関 |
| Summarization | Spearman相関 | 要約と原文の類似度相関 |
| BitextMining | F1 | 対訳ペア検出のF1スコア |

### 言語カバレッジの設計

著者らは、従来のMTEBがヨーロッパ言語に偏重していた問題を解決するため、以下の言語ファミリーを体系的にカバーしている。

- **インド・ヨーロッパ語族**: 英語、フランス語、ドイツ語、ヒンディー語等
- **シナ・チベット語族**: 中国語（簡体字・繁体字）
- **アフロ・アジア語族**: アラビア語、ヘブライ語
- **ニジェール・コンゴ語族**: スワヒリ語、ヨルバ語、ウォロフ語
- **オーストロネシア語族**: インドネシア語、タガログ語
- **日本語**: JMTEBとの統合データを含む

### スコア集約方法

多言語ベンチマークでは、言語間のタスク難易度の違いが問題となる。著者らは**言語均等重み付け集約**（equal-weight per language）を採用し、リソースの多い言語がスコアを支配しないよう設計している。

$$
\text{MMTEB\_Score} = \frac{1}{|L|} \sum_{l \in L} \frac{1}{|T_l|} \sum_{t \in T_l} \text{score}(t, l)
$$

ここで、
- $L$: 評価対象言語の集合
- $T_l$: 言語 $l$ で利用可能なタスクの集合
- $\text{score}(t, l)$: タスク $t$、言語 $l$ でのモデルスコア

### MMTEB-Lite: 軽量評価サブセット

全500超データセットの評価には相当な計算資源が必要であるため、著者らは代表的タスクのサブセットであるMMTEB-Liteを提供している。

**設計方針**: フル評価との相関を最大化しつつ、計算コストを削減する。著者らの報告によると、MMTEB-LiteスコアとフルMMTEBスコアのPearson相関は**0.99以上**である。

### データ品質改善

著者らは既存MTEBタスクの品質問題を特定・修正している。

1. **ラベルエラーの修正**: 一部データセットのアノテーション誤りを手動で修正
2. **データ漏洩の検出**: n-gram照合によりテストデータが訓練データに含まれるケースを特定
3. **テストセット汚染チェックプロトコルの提供**: モデル評価の公正性を担保する手続きを標準化

## 実験結果（Results）

### モデル別総合スコア

著者らは40超の埋め込みモデルをMMTEBで評価している。以下は論文で報告された上位モデルのスコアである。

| モデル | MMTEB全言語平均 | 特記事項 |
|-------|---------------|---------|
| multilingual-e5-large-instruct | 約65前後 | Microsoft、多言語トップクラス |
| bge-m3 | 約63-65 | BAAI、多機能埋め込み |
| e5-mistral-7b-instruct | 高（英語特化） | 多言語では性能低下 |
| text-embedding-3-large | 高（英語） | OpenAI、多言語は中程度 |

（論文Table 1-3より概要。正確な数値はarXiv原文を参照。）

### タスク別主要結果

**Retrieval（検索タスク、nDCG@10）**:
- 英語タスク: 最高モデルでnDCG@10 = 54-60程度
- 多言語タスク: 英語比5-15ポイント低下が一般的
- MIRACLベンチマーク（18言語）では多言語専用モデルが英語特化モデルを上回る傾向

**STS（文類似度）**:
- Spearman相関: 英語で0.88-0.92、多言語で0.75-0.85の範囲

**BitextMining（対訳検出）**:
- 高リソース言語（欧州語）: F1 > 90%
- 低リソース言語: F1 = 50-70%

### モデルサイズと多言語性能の関係

著者らの分析によると、パラメータ数と性能には強い正の相関があるが、多言語タスクでは**専用の多言語訓練データの有無がモデルサイズ差より重要**な場合があると報告されている。

### 高リソース言語 vs 低リソース言語

最良モデルであっても、低リソース言語では高リソース言語に対して**10-20ポイント以上の性能差**が存在する。これは、訓練データの量と質に起因する構造的な課題である。

## 実装のポイント（Implementation）

### MMTEB-Liteの活用

全データセットの実行は計算コストが高いため、実務ではMMTEB-Liteの活用が推奨される。

```python
# MMTEBの実行例（Python 3.11+）
# 必要パッケージ: pip install mteb sentence-transformers
import mteb
from sentence_transformers import SentenceTransformer


def evaluate_model_mmteb_lite(
    model_name: str,
    languages: list[str] | None = None,
) -> dict:
    """MMTEB-Liteでモデルを評価する。

    Args:
        model_name: HuggingFace Hub上のモデル名
        languages: 評価対象言語リスト（Noneで全言語）

    Returns:
        タスク別・言語別スコア辞書
    """
    model = SentenceTransformer(model_name)

    # MMTEB-Liteのタスクを取得
    tasks = mteb.get_tasks(
        task_types=["Retrieval", "Classification", "STS"],
        languages=languages or ["eng", "jpn", "zho"],
    )

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder="results/")
    return results
```

### 日本語評価での注意点

MMTEBは日本語タスク（JMTEB統合データ含む）をサポートしているが、以下の点に注意が必要である。

- JMTEBの28データセットがMMTEBに統合されている
- 日本語固有の形態素解析（MeCab等）を必要とするタスクがある
- 日本語Retrievalタスクはドメインが限定的（ニュース・Wikipedia中心）

### モデル選定への活用

Zenn記事で解説されている評価パイプラインの「ベンチマークでのフィルタリング」段階で、MMTEBは以下の形で活用できる。

1. **MMTEB-Liteで候補モデルを5-10に絞る**: 対象言語のRetrievalスコアでフィルタリング
2. **JMTEBスコアを補助指標として参照**: 日本語固有の性能を確認
3. **最終選定は自社データでのオフライン評価**: ベンチマークはフィルタリングの第一段階として使用

## 実運用への応用（Practical Applications）

### 多言語RAGシステムでのモデル選定

多言語対応が求められるRAGシステムでは、英語のMTEBスコアだけでなくMMTEBの言語別スコアを参照すべきである。著者らの結果は、英語で高性能なモデルが他言語で必ずしも優れないことを示している。

### ドメイン固有評価との組み合わせ

MMTEBはウェブ・ニュース・Wikipedia等の一般ドメインが中心であるため、医療・法律・金融等の専門ドメインでの評価は自前で構築する必要がある。MMTEBは初期フィルタリングに使い、最終判断は自社データでの実測で行うという2段階アプローチが推奨される。

## 関連研究（Related Work）

- **MTEB (arXiv:2210.07316)**: MMTEBの前身。56タスク・8カテゴリの英語中心ベンチマーク
- **MIRACL (arXiv:2210.09984)**: 18言語の多言語検索ベンチマーク。MMTEBに統合されている
- **JMTEB**: 日本語特化の28データセットベンチマーク。MMTEBに統合済み
- **RTEB (Hugging Face, 2025)**: 非公開データ含む検索特化ベンチマーク。テストデータ汚染への対策を重視

## まとめと今後の展望

MMTEBは、テキスト埋め込みモデルの多言語評価における新たな標準として位置付けられる。250言語以上のカバレッジ、データ品質の改善、MMTEB-Liteによる計算コスト削減は、実務者にとって大きな価値がある。

著者らが今後の課題として挙げている点は、ドメイン特化タスクの追加、低リソース言語のさらなる収録、および動的評価（定期更新型テストセット）の導入である。

## 参考文献

- **Conference URL**: [https://arxiv.org/abs/2502.13595](https://arxiv.org/abs/2502.13595)
- **ICLR 2025**: [https://iclr.cc/virtual/2025/poster/27651](https://iclr.cc/virtual/2025/poster/27651)
- **MTEB GitHub**: [https://github.com/embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb)
- **MTEB Leaderboard**: [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/1798f7e5c5fd69](https://zenn.dev/0h_n0/articles/1798f7e5c5fd69)
