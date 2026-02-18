---
layout: post
title: "論文解説: State of What Art? — マルチプロンプトLLM評価の必要性"
description: "プロンプトの違いでLLMの性能が最大10ポイント変動する問題を25モデル×6タスクで実証し、マルチプロンプト評価手法を提案"
categories: [blog, paper, arxiv]
tags: [LLM, evaluation, prompt sensitivity, benchmark, A/B testing]
date: 2026-02-18 09:00:00 +0900
source_type: arxiv
arxiv_id: "2401.00595"
source_url: https://arxiv.org/abs/2401.00595
zenn_article: d86aba5cf2c154
zenn_url: https://zenn.dev/0h_n0/articles/d86aba5cf2c154
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## 論文概要（Abstract）

LLMの評価は通常、単一のプロンプトテンプレートで行われる。しかし「プロンプト感受性」により、テンプレートを変えるだけでベンチマークスコアが最大10ポイント変動し、モデルランキングが大きく入れ替わることが25モデル×6タスクの大規模実験で明らかになった。本論文はこの問題を定量化し、多様なプロンプトセットを用いたマルチプロンプト評価手法を提案する。

この記事は [Zenn記事: LLMのA/Bテスト戦略：プロンプト改善サイクルを3倍速にする実践ガイド](https://zenn.dev/0h_n0/articles/d86aba5cf2c154) の深掘りです。

## 情報源

- **arXiv ID**: 2401.00595
- **URL**: [https://arxiv.org/abs/2401.00595](https://arxiv.org/abs/2401.00595)
- **著者**: Ori Yoran, Moran Geva, Jonnie Sheridan, Mor Geva, Jonathan Berant
- **発表年**: 2024（TACL 2024に採択）
- **分野**: cs.CL

## 背景と動機（Background & Motivation）

LLMの評価では、MMLUやHumanEvalなどのベンチマークで1つのプロンプトテンプレートを使ってスコアを測定し、リーダーボードで序列化するのが標準的な手法である。しかし、LLMはプロンプトの微細な変更（語順、指示の詳細度、Few-shotの有無など）に対して極めて敏感であることが知られている。

この問題は、あるモデルが「最良」と判定されるかどうかが、評価者が選んだプロンプトテンプレート1つに依存することを意味する。つまり、現在のLLMリーダーボードの順位は、選ばれたプロンプトテンプレートの偶然の産物である可能性がある。本論文はこの問題を初めて体系的かつ大規模に定量化し、解決策を提案した。

## 主要な貢献（Key Contributions）

- **プロンプト感受性の大規模定量化**: 25モデル×6タスク×10-20プロンプトの体系的実験で、性能分散が最大10ポイントに達することを実証
- **プロンプト品質のモデル間一貫性の発見**: 「良いプロンプト」はモデルを問わず良い（Spearman相関 > 0.8）ことを示し、プロンプト品質がタスク固有の性質であることを証明
- **プロンプト品質の予測可能性**: 単純な特徴量（長さ、Few-shot有無など）でプロンプトの良し悪しを最大96%の精度で予測可能
- **マルチプロンプト評価手法の提案**: ランキング安定性を0.65-0.82から0.88-0.95に向上させる実用的な手法を提供

## 技術的詳細（Technical Details）

### 実験設計

25のLLM（Llama 2の7B/13B/70B、Mistral 7B、GPT-3.5-turbo、GPT-4、Claude 2、Gemini Proなど）を、6タスクでそれぞれ10〜20種のプロンプトテンプレートを用いて評価した。

| タスク | データセット | 評価指標 | LLM-as-a-Judge |
|--------|-------------|----------|----------------|
| 読解 | CoQA | F1 | No |
| オープンQA | TriviaQA | F1 | No |
| 常識推論 | Winogrande | Accuracy | No |
| 自然言語推論 | ANLI | Accuracy | No |
| 指示追従 | AlpacaEval | Win Rate | Yes |
| 要約 | SummEval | 人間評価相関 | Yes |

### プロンプト間の性能分散

各タスクにおけるプロンプト間の最大性能差は以下の通り：

| タスク | 最大性能差 |
|--------|-----------|
| CoQA | 8.5ポイント（F1） |
| TriviaQA | 6.2ポイント（F1） |
| Winogrande | **10.1ポイント**（Accuracy） |
| ANLI | 9.3ポイント（Accuracy） |
| AlpacaEval | 8.7ポイント（Win Rate） |
| SummEval | 7.4ポイント |

この差はモデルのメジャーバージョン間の性能差に匹敵する。つまり「GPT-3.5からGPT-4への改善幅」と同程度の差が、プロンプトを変えるだけで生じうる。

### ランキングの不安定性

単一プロンプトでのモデル間Spearman順位相関は $\rho = 0.65 \sim 0.82$ にとどまる。あるプロンプトで1位のモデルが、別のプロンプトでは5位以下に転落するケースが確認された。

### モデルサイズとプロンプト感受性

直感に反して、大型モデルほどプロンプト感受性が低いという相関は見られなかった。スケーリングだけではプロンプト感受性は解消されない。

### プロンプト品質の予測

プロンプトの「良し悪し」を中央値以上/以下のバイナリ分類問題として定式化し、以下の特徴量で予測する：

**構造的特徴量**: プロンプト長（トークン数）、指示文の文数、Few-shotの有無と数、出力フォーマット指定の有無

**言語的特徴量**: 平均文長、語彙の豊富さ、特定キーワード（"step by step"、"briefly"など）の有無

ロジスティック回帰と勾配ブースティングによる予測精度：

| タスク | 予測精度 |
|--------|---------|
| CoQA | 89% |
| TriviaQA | 91% |
| Winogrande | **96%** |
| ANLI | 87% |
| AlpacaEval | 84% |
| SummEval | 88% |

重要な発見として、同一タスク・異なるモデルへの汎化では精度低下が約3%にとどまる一方、異なるタスクへの汎化では約15%低下する。これは「プロンプト品質はタスク固有の構造を持つ」ことの強力な証拠である。

## 実装のポイント（Implementation）

### マルチプロンプト評価の5ステップアルゴリズム

```python
from sklearn.cluster import KMeans
import numpy as np

def select_evaluation_prompts(
    prompt_pool: list[str],
    calibration_scores: np.ndarray,  # shape: (n_prompts, n_calibration_models)
    n_select: int = 7,
) -> list[int]:
    """多様かつ代表的な評価プロンプトセットを選択する.

    Args:
        prompt_pool: 候補プロンプト一覧 (20-30種)
        calibration_scores: 3-5モデルでの各プロンプトの性能スコア
        n_select: 選択するプロンプト数 (推奨: 5-10)

    Returns:
        選択されたプロンプトのインデックス
    """
    # Step 3: 性能パターンでクラスタリング
    kmeans = KMeans(n_clusters=n_select, random_state=42)
    clusters = kmeans.fit_predict(calibration_scores)

    # Step 4-5: 各クラスターからクラスター中心に最も近いプロンプトを選択
    selected = []
    for c in range(n_select):
        members = np.where(clusters == c)[0]
        center = kmeans.cluster_centers_[c]
        distances = np.linalg.norm(calibration_scores[members] - center, axis=1)
        selected.append(members[np.argmin(distances)])

    return selected
```

### 報告フォーマット

単一の数値ではなく、プロンプト間の平均と分散を報告する：

```python
def report_multi_prompt_results(
    scores: dict[str, list[float]],
) -> dict[str, dict]:
    """マルチプロンプト評価結果を報告用に集約する.

    Args:
        scores: {model_name: [prompt1_score, prompt2_score, ...]}

    Returns:
        {model_name: {"mean": float, "std": float, "min": float, "max": float}}
    """
    results = {}
    for model, s in scores.items():
        arr = np.array(s)
        results[model] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return dict(sorted(results.items(), key=lambda x: -x[1]["mean"]))
```

## 実験結果（Results）

マルチプロンプト評価の導入効果：

| 指標 | 単一プロンプト | マルチプロンプト |
|------|-------------|---------------|
| 性能推定の分散 | ベースライン | **40-60%削減** |
| ランキングの順位相関 | $\rho = 0.65 \sim 0.82$ | $\rho = 0.88 \sim 0.95$ |
| モデル間の性能差推定 | 不安定 | 信頼性向上 |

5〜10個のプロンプトで十分な安定性が得られるため、評価コストの増加は許容範囲内である。

## 実運用への応用（Practical Applications）

**A/Bテストへの直接的な含意**: Zenn記事で紹介したLLM-as-a-Judgeによる自動評価パイプラインにおいて、ジャッジプロンプト自体も感受性を持つことが本論文で示されている。ジャッジプロンプトを1つに固定すると、特定のバリアントに有利なバイアスが生じうる。複数のジャッジプロンプトで評価し、平均をとることで信頼性を向上できる。

**プロンプト改善サイクルの設計**: 本論文の知見を踏まえると、プロンプトのA/Bテストでは「テストケース」自体を複数のプロンプトテンプレートで構成すべきである。これにより「特定のプロンプトでたまたま勝った」という偽陽性を防げる。

**ゴールデンデータセット設計**: 評価用ゴールデンデータセットを構築する際、同一タスクに対して複数のプロンプト表現を用意し、それぞれの結果を集約して最終スコアとする設計が推奨される。

## 関連研究（Related Work）

- **HELM（Liang et al., 2022）**: 多面的LLM評価フレームワーク。単一プロンプトだがメトリクスの多様性を重視
- **Chatbot Arena（Zheng et al., 2024）**: 人間の選好投票による評価。プロンプト感受性とは直交する問題を解決
- **APE（Zhou et al., 2022）**: 自動プロンプト最適化。最良プロンプトの探索が目的で、評価の頑健性とは異なるゴール

## まとめと今後の展望

本論文は「LLMの性能はプロンプト1つで最大10ポイント変動する」という事実を体系的に示し、単一プロンプト評価の危うさを明確にした。マルチプロンプト評価への移行は、LLM評価の信頼性向上に不可欠である。特にA/Bテストやプロンプト改善サイクルを実施する際には、評価プロンプト自体の感受性を考慮した設計が必要である。今後は、命令追従や安全性評価といったLLM-as-a-Judgeタスクでのマルチプロンプト評価の拡張が期待される。

## 参考文献

- **arXiv**: [https://arxiv.org/abs/2401.00595](https://arxiv.org/abs/2401.00595)
- **TACL 2024**: [https://aclanthology.org/2024.tacl-1.52/](https://aclanthology.org/2024.tacl-1.52/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/d86aba5cf2c154](https://zenn.dev/0h_n0/articles/d86aba5cf2c154)
