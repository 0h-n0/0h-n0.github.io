---
layout: post
title: "論文解説: HaloScope — ラベルなしLLM生成文からのHallucination検出"
description: "NeurIPS 2024 Spotlight論文。LLMの潜在表現空間にHallucination部分空間を発見し、ラベルなしデータのみで高精度な検出を実現する手法を解説"
categories: [blog, paper, arxiv]
tags: [hallucination, LLM, validation, guardrails, deepeval]
date: 2026-02-17 09:00:00 +0900
source_type: arxiv
arxiv_id: "2409.17504"
source_url: https://arxiv.org/abs/2409.17504
zenn_article: f1eab19b1726e1
zenn_url: https://zenn.dev/0h_n0/articles/f1eab19b1726e1
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## 論文概要（Abstract）

HaloScopeは、ラベルなしのLLM生成文を活用してHallucination（幻覚）を検出するフレームワークである。LLMの活性化空間においてHallucinationに関連する部分空間を特定し、その部分空間への射影強度で真偽を判定する。NeurIPS 2024でSpotlight論文として採択された。

この記事は [Zenn記事: LLM出力検証フレームワーク実践：本番運用で99%精度を実現する3層戦略](https://zenn.dev/0h_n0/articles/f1eab19b1726e1) の深掘りです。

## 情報源

- **arXiv ID**: 2409.17504
- **URL**: [https://arxiv.org/abs/2409.17504](https://arxiv.org/abs/2409.17504)
- **著者**: Xuefeng Du, Chaowei Xiao, Yixuan Li
- **発表年**: 2024
- **分野**: cs.LG, cs.CL
- **コード**: [https://github.com/deeplearning-wisc/haloscope](https://github.com/deeplearning-wisc/haloscope)

## 背景と動機（Background & Motivation）

LLMが生成する文のうち、もっともらしく見えるが事実と異なる「Hallucination」は、本番運用における最大の課題の1つである。従来のHallucination検出手法は、(1)複数回のサンプリングによる一貫性チェック（SelfCheckGPTなど）や(2)外部知識ベースとの照合（Retrieval-based手法）が主流であったが、いずれも計算コストが高い、または外部リソースへの依存が大きいという問題を抱えていた。

HaloScopeは「LLM内部の潜在表現にはHallucinationの手がかりが既に含まれている」という仮説に基づき、**ラベル付きデータを一切使わず**にHallucinationを検出する新しいアプローチを提案する。

## 主要な貢献（Key Contributions）

- **貢献1**: LLMの活性化空間にHallucination部分空間が存在することを実験的に示し、その部分空間への射影によりHallucinationスコアを算出する手法を提案
- **貢献2**: ラベルなしデータから真偽を推定する自動メンバーシップ推定スコア（Automated Membership Estimation Score）を導入し、教師なし学習でHallucination検出器を訓練
- **貢献3**: 既存手法と比較して45x〜450xの高速化を達成しつつ、検出性能も向上

## 技術的詳細（Technical Details）

### Hallucination部分空間の発見

HaloScopeの核心は、LLMの中間層の活性化ベクトルにおいて、Hallucinationに対応する**部分空間**が存在するという発見である。

具体的には、LLMの$l$層目の隠れ状態$\mathbf{h}^{(l)} \in \mathbb{R}^d$に対して、主成分分析（PCA）を適用し、上位$k$個の主成分で張られる部分空間$\mathcal{S}_k$を構築する。

$$
\mathbf{P}_k = \sum_{i=1}^{k} \mathbf{v}_i \mathbf{v}_i^T
$$

ここで、
- $\mathbf{v}_i$: $i$番目の主成分ベクトル
- $\mathbf{P}_k$: 部分空間$\mathcal{S}_k$への射影行列
- $k$: 部分空間の次元数（ハイパーパラメータ）

### Hallucinationスコアの算出

入力文$x$に対するLLMの生成文$y$の活性化ベクトル$\mathbf{h}$について、Hallucinationスコア$s$は以下で定義される：

$$
s(\mathbf{h}) = \frac{\|\mathbf{P}_k \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
$$

スコアが高いほど、活性化ベクトルがHallucination部分空間に強く射影されており、Hallucinationの可能性が高いと判定される。

### 自動メンバーシップ推定

ラベルなしデータから真偽を推定するために、以下のステップを実行する：

1. **ラベルなし生成文の収集**: LLMに多様な質問を与え、生成文を収集
2. **活性化ベクトルの抽出**: 各生成文の中間層活性化を取得
3. **2クラスタリング**: GMM（Gaussian Mixture Model）で活性化ベクトルを2クラスタに分離
4. **メンバーシップスコア**: 各クラスタへの所属確率をスコアとして使用

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

def haloscope_score(
    hidden_states: np.ndarray,  # (n_samples, hidden_dim)
    n_components: int = 10,
) -> np.ndarray:
    """HaloScopeスコアを算出する

    Args:
        hidden_states: LLMの中間層活性化ベクトル
        n_components: PCAの主成分数

    Returns:
        各サンプルのHallucinationスコア
    """
    # PCAで部分空間を特定
    pca = PCA(n_components=n_components)
    pca.fit(hidden_states)

    # 射影行列による射影
    projected = pca.transform(hidden_states)
    proj_norms = np.linalg.norm(projected, axis=1)
    orig_norms = np.linalg.norm(hidden_states, axis=1)

    scores = proj_norms / (orig_norms + 1e-8)
    return scores
```

## 実装のポイント（Implementation）

- **レイヤー選択**: 中間層（全体の50-70%付近）が最も判別力が高い。最終層は分類タスクに最適化されすぎており、Hallucination検出には不向き
- **主成分数$k$**: 5-20程度が推奨。データセットにより最適値は異なるが、$k=10$が一般的に良好
- **ホワイトボックスアクセス**: 本手法はLLMの内部状態へのアクセスが必要。API経由のブラックボックスLLM（GPT-4等）には直接適用不可
- **計算効率**: 推論時は射影行列の乗算のみ。SelfCheckGPTの複数回サンプリングと比較して45x〜450xの高速化

## 実験結果（Results）

| データセット | SelfCheckGPT | RefChecker | HaloScope | 高速化 |
|-------------|-------------|-----------|-----------|--------|
| TruthfulQA | 72.3% AUROC | 74.1% | **78.5%** | 450x |
| HaluEval | 68.9% | 71.2% | **76.8%** | 180x |
| PHD | 65.4% | 67.8% | **73.2%** | 45x |

**分析**:
- 全データセットで既存手法を上回るAUROCを達成
- 特にTruthfulQAでは6%以上の改善
- 計算コストは既存手法の1/45〜1/450

## 実運用への応用（Practical Applications）

Zenn記事で紹介されたSemantic Validation層（DeepEvalによるHallucination検出）に、HaloScopeのアプローチを組み込むことで、**低レイテンシかつ高精度なHallucination検出**が可能になる。

具体的な適用シナリオ：
- **リアルタイム検証**: API応答の都度、活性化ベクトルを分析してHallucinationリスクを算出
- **バッチ検証**: 定期的にLLM出力を収集し、Hallucination傾向の変化（drift）を監視
- **コスト削減**: 従来の複数回サンプリング手法を置き換えることで、LLM呼び出し回数を大幅削減

## 関連研究（Related Work）

- **SelfCheckGPT** (Manakul et al., 2023): 複数回サンプリングによる一貫性チェック。計算コストが高い
- **RefChecker** (Hu et al., 2024): 参照ベースのFact-checking。外部知識ベースが必要
- **INSIDE** (Chen et al., 2024): 内部表現の固有次元に基づくHallucination検出。HaloScopeと相補的

## まとめと今後の展望

HaloScopeは、LLMの内部表現空間にHallucination固有の部分空間が存在するという重要な発見に基づき、ラベルなしデータのみで高速かつ高精度なHallucination検出を実現した。本番運用におけるSemantic Validation層の高速化・低コスト化に直結する研究であり、Guardrails AIやDeepEvalとの統合が今後の方向性として期待される。

## 参考文献

- **arXiv**: [https://arxiv.org/abs/2409.17504](https://arxiv.org/abs/2409.17504)
- **Code**: [https://github.com/deeplearning-wisc/haloscope](https://github.com/deeplearning-wisc/haloscope)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/f1eab19b1726e1](https://zenn.dev/0h_n0/articles/f1eab19b1726e1)
