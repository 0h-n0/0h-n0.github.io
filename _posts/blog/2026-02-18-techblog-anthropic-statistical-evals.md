---
layout: post
title: "Anthropic解説: A Statistical Approach to Model Evaluations — LLM評価に統計的厳密性を導入する5つの提言"
description: "Anthropicが提唱するLLM評価の統計的手法を詳細解説。CLTによるSEM計算、クラスター標準誤差、ペア差分析、検出力分析まで、A/Bテスト設計に直結する実践的フレームワーク"
categories: [blog, tech_blog]
tags: [LLM, evaluation, statistics, A/B testing, Anthropic, confidence interval, hypothesis testing]
date: 2026-02-18 21:10:00 +0900
source_type: tech_blog
source_domain: anthropic.com
source_url: https://www.anthropic.com/research/statistical-approach-to-model-evals
zenn_article: d86aba5cf2c154
zenn_url: https://zenn.dev/0h_n0/articles/d86aba5cf2c154
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

Anthropicが2024年11月に公開した研究ブログ「A Statistical Approach to Model Evaluations」は、LLM評価における統計的厳密性の欠如を指摘し、5つの具体的な改善提言を行っています。核心的な問いは「モデル間の能力差は本物か、それとも質問の運で偶然そう見えただけか？」です。クラスター標準誤差がnaive計算の3倍以上になりうることを示し、ペア差分析と検出力分析を組み合わせた実践的なフレームワークを提案しています。

この記事は [Zenn記事: LLMのA/Bテスト戦略：プロンプト改善サイクルを3倍速にする実践ガイド](https://zenn.dev/0h_n0/articles/d86aba5cf2c154) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（Anthropic Research）
- **URL**: [https://www.anthropic.com/research/statistical-approach-to-model-evals](https://www.anthropic.com/research/statistical-approach-to-model-evals)
- **組織**: Anthropic
- **発表日**: 2024年11月
- **関連論文**: arXiv:2411.00640 "Adding Error Bars to Evals"

## 技術的背景（Technical Background）

### なぜ統計的厳密性が必要か

LLMの評価結果（eval scores）は通常、テストセット上の正解率として報告されます。しかし、このスコアには2つのランダム性が存在します：

1. **質問の選択（sampling noise）**: テストセットの質問は全可能な質問の部分集合であり、異なる質問セットでは異なるスコアが得られる
2. **モデルの応答の確率的性質**: 同じ質問に対してもモデルは異なる回答を生成しうる（特にChain-of-Thought推論時）

これらのランダム性を無視すると、存在しない能力差を「発見」してしまう偽陽性のリスクがあります。Anthropicの分析では、人気ベンチマークにおけるクラスター標準誤差がnaive標準誤差の3倍以上になるケースが確認されており、従来の報告方法では「統計的に有意でない差を有意と誤認する」ケースが頻発していました。

### 学術研究との関連

この問題意識は統計学の古典的な「多重比較問題」と「クラスター化データの分析」に根差しています。LLMの評価ベンチマークでは、同じソーステキストに基づく複数の質問（例：読解問題で同一文書から複数問出題）が独立でないため、標本の独立同分布（i.i.d.）仮定が崩れます。

## 5つの提言（Recommendations）

### 提言1: 中心極限定理（CLT）に基づくSEMの報告

**核心**: 観測平均ではなく「理論的平均」に対する推定として扱い、Standard Error of the Mean（SEM）を必ず付記すべき。

$$\text{SEM} = \frac{s}{\sqrt{n}}$$

ここで $s$ は標本標準偏差、$n$ はサンプル数です。

95%信頼区間は以下で計算されます：

$$\text{CI}_{95\%} = \bar{x} \pm 1.96 \times \text{SEM}$$

**実装例**:

```python
import numpy as np
from scipy import stats

def compute_eval_score_with_ci(
    scores: list[float],
    confidence: float = 0.95,
) -> dict:
    """LLM評価スコアに信頼区間を付与する.

    Args:
        scores: 各質問でのスコア（0 or 1、または連続値）
        confidence: 信頼水準（デフォルト95%）

    Returns:
        平均スコア、SEM、信頼区間を含む辞書
    """
    n = len(scores)
    mean = np.mean(scores)
    sem = stats.sem(scores)
    z = stats.norm.ppf((1 + confidence) / 2)
    ci_lower = mean - z * sem
    ci_upper = mean + z * sem
    return {
        "mean": round(mean, 4),
        "sem": round(sem, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n": n,
    }
```

### 提言2: クラスター標準誤差の使用

**核心**: 多くのベンチマークでは質問が独立でない（同一文書から複数問出題等）。naive SEMは真の不確実性を過小評価する。

**Anthropicの発見**: 人気ベンチマークにおけるクラスター標準誤差はnaive標準誤差の **3倍以上** になるケースが確認されました。これは、独立性仮定に基づくnaive SEMを使うと、実際には有意差がないモデル間の差を誤って「有意」と判定してしまうことを意味します。

**クラスター標準誤差の計算**:

$$\text{SE}_{\text{cluster}} = \sqrt{\frac{M}{M-1} \cdot \frac{1}{N^2} \sum_{m=1}^{M} \left( \sum_{i \in C_m} (x_i - \bar{x}) \right)^2}$$

ここで $M$ はクラスター数、$N$ は総サンプル数、$C_m$ は $m$ 番目のクラスターに属するサンプルのインデックス集合です。

**実装例**:

```python
def compute_clustered_se(
    scores: list[float],
    cluster_ids: list[int],
) -> float:
    """クラスター標準誤差を計算する.

    Args:
        scores: 各質問でのスコア
        cluster_ids: 各質問が属するクラスターのID

    Returns:
        クラスター標準誤差
    """
    scores_arr = np.array(scores)
    clusters = np.array(cluster_ids)
    overall_mean = np.mean(scores_arr)
    unique_clusters = np.unique(clusters)
    M = len(unique_clusters)
    N = len(scores_arr)

    cluster_residual_sums = []
    for c in unique_clusters:
        mask = clusters == c
        cluster_sum = np.sum(scores_arr[mask] - overall_mean)
        cluster_residual_sums.append(cluster_sum ** 2)

    variance = (M / (M - 1)) * (1 / N**2) * np.sum(cluster_residual_sums)
    return float(np.sqrt(variance))
```

### 提言3: 質問内のバリアンス削減

**核心**: モデルのスコアは「平均スコア（underlying skill）」と「ランダム成分（deviation）」に分解できる。ランダム成分を減らすことで、より少ないサンプルで有意差を検出できる。

**Chain-of-Thought（CoT）プロンプトの場合**: 同じ質問に対して複数回サンプリングし、平均を取ることでランダム性を削減できます。例えばInspectフレームワークでは `epochs` パラメータで繰り返し回数を設定可能です。

**決定的回答の場合**: next-tokenの確率を直接使用し、バイナリ（正解/不正解）ではなく連続値としてスコアリングすることで、ランダム性を完全に排除できます。

```python
def reduce_variance_with_resampling(
    model,
    question: str,
    n_samples: int = 5,
) -> float:
    """CoT推論の質問内バリアンスを削減する.

    Args:
        model: 評価対象モデル
        question: 評価質問
        n_samples: リサンプリング回数

    Returns:
        平均化されたスコア（0.0〜1.0）
    """
    scores = []
    for _ in range(n_samples):
        response = model.generate(question, temperature=0.7)
        is_correct = evaluate_correctness(response, question)
        scores.append(float(is_correct))
    return np.mean(scores)
```

### 提言4: ペア差分析（Paired Differences Analysis）

**核心**: 2つのモデルを比較する際、独立な2標本t検定ではなく、同一質問でのスコア差（paired difference）を分析すべき。

フロンティアモデル同士は同じ質問を正解/不正解にする傾向があり（相関係数0.3〜0.7）、この相関を活用することで「無料で」バリアンスを削減できます。

**数式**: モデルAとBの $i$ 番目の質問でのスコア差を $d_i = s_i^A - s_i^B$ とすると、差の平均と標準誤差は：

$$\bar{d} = \frac{1}{n} \sum_{i=1}^{n} d_i, \quad \text{SE}_d = \frac{s_d}{\sqrt{n}}$$

ここで $s_d$ は差分 $\{d_i\}$ の標準偏差です。

```python
def paired_model_comparison(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> dict:
    """ペア差分析によるモデル比較.

    Args:
        scores_a: モデルAの各質問でのスコア
        scores_b: モデルBの各質問でのスコア
        alpha: 有意水準

    Returns:
        比較結果の辞書
    """
    differences = np.array(scores_b) - np.array(scores_a)
    mean_diff = np.mean(differences)
    se_diff = stats.sem(differences)
    t_stat = mean_diff / se_diff
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(differences) - 1))
    ci_lower = mean_diff - 1.96 * se_diff
    ci_upper = mean_diff + 1.96 * se_diff

    return {
        "mean_difference": round(mean_diff, 4),
        "se_difference": round(se_diff, 4),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "ci_95": (round(ci_lower, 4), round(ci_upper, 4)),
        "significant": p_value < alpha,
        "recommendation": (
            "B is significantly better" if p_value < alpha and mean_diff > 0
            else "A is significantly better" if p_value < alpha and mean_diff < 0
            else "No significant difference"
        ),
    }
```

### 提言5: 検出力分析（Power Analysis）

**核心**: 実験を始める前に、必要なサンプル数を計算すべき。検出力分析は「観測数」「統計的検出力（power）」「偽陽性率」「効果量（effect size）」の4変数を結びつけます。

$$n = \left( \frac{z_{\alpha/2} + z_\beta}{\delta / \sigma} \right)^2$$

ここで $z_{\alpha/2}$ は有意水準に対応するz値、$z_\beta$ は検出力に対応するz値、$\delta$ は検出したい最小効果量、$\sigma$ はスコアの標準偏差です。

```python
from statsmodels.stats.power import TTestIndPower

def compute_required_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """必要サンプル数を検出力分析で算出する.

    Args:
        effect_size: 検出したい効果量（Cohen's d）
        alpha: 有意水準（偽陽性率）
        power: 検出力（1 - 偽陰性率）

    Returns:
        1群あたりの必要サンプル数
    """
    analysis = TTestIndPower()
    n = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )
    return int(np.ceil(n))

# 例: 3%ポイントの差を検出したい場合
# (σ≈0.5と仮定すると effect_size = 0.03/0.5 = 0.06)
n = compute_required_sample_size(effect_size=0.06)
# n ≈ 4,350 サンプル/群
```

## パフォーマンス最適化（Performance）

### 評価コストの最適化戦略

Anthropicの提言を組み合わせることで、同じ統計的検出力をより少ないAPIコールで達成できます。

| 最適化手法 | バリアンス削減率 | 追加コスト |
|-----------|---------------|----------|
| ペア差分析（提言4） | 30〜70% | 0%（同一質問を使うだけ） |
| CoTリサンプリング×5（提言3） | 80%（5回平均で$1/\sqrt{5}$） | 5倍のAPIコール |
| クラスター標準誤差（提言2） | 正確な推定値を提供 | 計算コストのみ |

**推奨ワークフロー**:
1. まず検出力分析（提言5）で必要サンプル数を決定
2. ペア差分析（提言4）で「無料の」バリアンス削減を活用
3. それでもサンプル数が不足する場合はCoTリサンプリング（提言3）を追加
4. 最終結果にはクラスター標準誤差（提言2）を使用
5. 全スコアにSEMと95% CIを付記（提言1）

## 運用での学び（Production Lessons）

### Zenn記事との関連

[Zenn記事](https://zenn.dev/0h_n0/articles/d86aba5cf2c154)で紹介されている`check_significance`関数（`scipy.stats.ttest_ind`を使用）は、本ブログの提言4「ペア差分析」に対応します。ただしZenn記事の実装は2標本t検定であり、より検出力の高いペアt検定（`scipy.stats.ttest_rel`）に変更することが推奨されます。

また、Zenn記事で「各バリアント最低50サンプル」と述べていますが、Anthropicの検出力分析に基づくと、3%ポイントの差を検出するには数千サンプルが必要です。50サンプルで検出可能なのは10%ポイント以上の大きな差のみであり、この点をプロダクション判断に反映すべきです。

### モニタリング戦略

- **SEM付きダッシュボード**: 評価スコアを常にSEM付きで表示し、信頼区間が重なるモデルは「統計的に区別不能」とラベリングする
- **逐次検定の注意**: 途中経過を何度も見て「有意差が出たら停止」すると偽陽性率が上昇する。事前に決めた検出力で実験を完了させるべき
- **クラスター情報の保持**: 評価セットのメタデータ（同一文書から何問出題しているか等）を保持し、クラスター標準誤差の計算に使用する

## 学術研究との関連（Academic Connection）

本ブログの内容は以下の統計学的背景に基づいています：

- **中心極限定理（CLT）**: 標本平均の分布が正規分布に近似することの活用
- **クラスター堅牢標準誤差**: 計量経済学でパネルデータ分析に広く使われる手法（Cameron & Miller, 2015）
- **検出力分析**: Cohen (1988) の効果量とサンプルサイズの関係の応用
- **ペアt検定**: 対応のある2標本の比較で効率的な差の検出を実現

Anthropic独自の貢献は、これらの確立された統計手法をLLM評価の文脈に具体的に適用し、「クラスター標準誤差がnaive SEMの3倍以上になりうる」という実証的発見を提示したことです。

## まとめと実践への示唆

Anthropicの5つの提言は、LLMのA/Bテストを設計する際の統計的基盤を提供します。最も重要な実務的示唆は以下の3点です：

1. **SEMを常に報告する**: 評価スコアの数字だけでは、その差が統計的に意味のあるものか判断できない
2. **ペア差分析を標準にする**: 同じ質問セットで比較することで、無料でバリアンスを30〜70%削減できる
3. **検出力分析で実験を設計する**: 「何サンプル必要か」を事前に計算してから実験を開始する

これらの原則は、Zenn記事で紹介されているLangfuseベースのA/Bテスト基盤と組み合わせることで、統計的に信頼できるプロンプト改善サイクルを実現します。

## 参考文献

- **Blog URL**: [https://www.anthropic.com/research/statistical-approach-to-model-evals](https://www.anthropic.com/research/statistical-approach-to-model-evals)
- **Related Paper**: arXiv:2411.00640 "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations"
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/d86aba5cf2c154](https://zenn.dev/0h_n0/articles/d86aba5cf2c154)
