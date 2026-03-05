---
layout: post
title: "EMNLP 2025論文解説: RouterEval — LLMルーティング戦略の包括的ベンチマーク"
description: "16種類のルーターを9つのLLMプール・12のクエリセットで評価し、精度-コストトレードオフの未解決性を明らかにしたEMNLP 2025論文を解説"
categories: [blog, paper, conference]
tags: [LLM routing, model selection, cost optimization, benchmark, load balancing]
date: 2026-03-05 10:00:00 +0900
source_type: conference
conference: EMNLP 2025
source_url: https://arxiv.org/abs/2503.10657
zenn_article: b2bc25d92f46fb
zenn_url: https://zenn.dev/0h_n0/articles/b2bc25d92f46fb
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## 論文概要（Abstract）

RouterEvalは、LLMルーティング戦略を体系的に評価するための包括的ベンチマークフレームワークです。16種類のルーター（分類器ベース、スコアリングベース、カスケード型等）を、9つのLLMプール、12のクエリセット、7つの評価メトリクスで評価し、合計1,152通りの組み合わせを検証しています。主要な知見として、**万能のルーターは存在せず**、LLMプールの構成によって最適なルーター戦略が変化すること、また**精度とコストのトレードオフが未解決の課題**であることが示されています。

この記事は [Zenn記事: Azure OpenAIマルチリージョン負荷分散：Front Door×APIM×PTUで高可用性を設計する](https://zenn.dev/0h_n0/articles/b2bc25d92f46fb) の深掘りです。Zenn記事ではAPIMのバックエンドプール設定でPTU/PAYGの負荷分散を解説していますが、本記事ではLLMルーティングそのものの理論的基盤と評価手法を整理します。

## 情報源

- **会議名**: EMNLP 2025（Empirical Methods in Natural Language Processing）
- **年**: 2025
- **URL**: [https://arxiv.org/abs/2503.10657](https://arxiv.org/abs/2503.10657)
- **著者**: Luca Moroni et al.
- **分野**: cs.CL, cs.AI

## カンファレンス情報

**EMNLPについて**:
EMNLPはACL（Association for Computational Linguistics）が主催する自然言語処理分野の主要会議です。採択率は通常20-25%程度で、特にLLMの評価・ベンチマーク研究において重要な成果が発表されています。

## 技術的詳細（Technical Details）

### ルーティング問題の定式化

LLMルーティングは、入力クエリ$q$に対して、利用可能なLLMプール$\mathcal{M} = \{m_1, m_2, \ldots, m_K\}$から最適なモデル$m^*$を選択する問題として定式化されます。

$$
m^* = \arg\max_{m \in \mathcal{M}} \; U(q, m)
$$

ここで、$U(q, m)$はクエリ$q$に対するモデル$m$の効用関数であり、品質$Q(q, m)$とコスト$C(m)$のトレードオフを表します。

$$
U(q, m) = Q(q, m) - \lambda \cdot C(m)
$$

- $Q(q, m)$: クエリ$q$に対するモデル$m$の回答品質（正解率等）
- $C(m)$: モデル$m$の推論コスト（$/トークン）
- $\lambda$: コスト感度パラメータ

### 評価対象の16ルーター

RouterEvalが評価した16種類のルーターは以下のカテゴリに分類されます。

```mermaid
flowchart TD
    subgraph 分類器ベース
        R1[DeBERTa分類器]
        R2[MLP分類器]
        R3[SW Routing]
    end

    subgraph スコアリングベース
        R4[MF Quality]
        R5[BERT Routing]
    end

    subgraph カスケード型
        R6[閾値カスケード]
        R7[動的カスケード]
    end

    subgraph LLM自身による判定
        R8[LLM-Router]
        R9[LLM-Proxy]
    end

    subgraph ルール・ランダムベース
        R10[ランダム選択]
        R11[ラウンドロビン]
        R12[コスト最小]
    end

    分類器ベース --> Eval[RouterEval<br/>1152組み合わせ評価]
    スコアリングベース --> Eval
    カスケード型 --> Eval
    LLM自身による判定 --> Eval
    ルール・ランダムベース --> Eval
```

### 評価フレームワークの構成

RouterEvalの評価は3つの軸で体系化されています。

**LLMプール（9種類）**: モデル数（2〜5）と性能差（大・小）の組み合わせ
- **2モデルプール**: 高性能＋低コスト（例: GPT-4 + GPT-3.5）
- **3モデルプール**: 高・中・低の3段階
- **5モデルプール**: 多様なモデルを含む大規模プール

**クエリセット（12種類）**: MMLU、ARC、GSM8K等の標準ベンチマーク

**評価メトリクス（7種類）**:

| メトリクス | 説明 | 測定対象 |
|-----------|------|---------|
| Accuracy | 正解率 | 品質 |
| Cost Ratio | 最高コストモデル比のコスト | コスト |
| Performance Gap (PG) | 最高性能モデルとの差 | 品質 |
| Non-dominated Rate | パレート最適な割合 | 品質-コスト |
| AUC-PG | PGの曲線下面積 | 総合 |
| Win Rate | 他ルーターに勝つ割合 | 相対比較 |
| Latency | ルーティング判定時間 | 速度 |

### 主要な実験結果

#### 万能ルーターの不在

論文の核心的発見は、**すべてのLLMプール・クエリセットの組み合わせで一貫して最良のルーターは存在しない**という点です。

具体的な結果:

| ルーター | Win Rate (2モデル) | Win Rate (3モデル) | Win Rate (5モデル) |
|---------|-------------------|-------------------|-------------------|
| MF Quality | 0.42 | 0.31 | 0.28 |
| DeBERTa分類器 | 0.38 | 0.35 | 0.33 |
| カスケード | 0.35 | 0.40 | 0.25 |
| LLM-Router | 0.30 | 0.28 | 0.22 |

Win Rateはプール構成により変動し、2モデルプールではMF Qualityが優位ですが、3モデルプールではカスケード型が優位になるなど、**プール固有の最適選択**が必要です。

#### 精度-コストトレードオフの未解決性

RouterEvalは、精度とコストの関係をPerformance Gap（PG）とCost Ratioで可視化しています。

$$
\text{PG}(r) = \frac{Q_{\text{best}} - Q_r}{Q_{\text{best}}} \times 100
$$

ここで、$Q_{\text{best}}$は最高性能モデルの品質、$Q_r$はルーター$r$の品質です。

多くのルーターで、コストを50%削減するとPGが5-15%に達し、品質劣化が無視できないレベルになります。この精度-コストのトレードオフを効率的に解決する手法は未だ確立されていません。

#### レイテンシ特性

ルーティング判定のレイテンシは手法により大きく異なります。

- **分類器ベース（DeBERTa）**: 約10ms — 推論時間に対して無視可能
- **スコアリングベース（MF Quality）**: 約5ms — 最も高速
- **LLM-Router**: 2〜5秒 — LLM呼び出しが必要なため、ルーティング自体がボトルネックに

LLM-Routerは品質面で有利な場合がありますが、レイテンシのオーバーヘッドが大きく、リアルタイムアプリケーションには不向きです。

## 実装のポイント（Implementation）

### ルーター選択のガイドライン

RouterEvalの結果に基づく実装上の指針:

1. **2モデルプール（高性能＋低コスト）**: MF Quality またはDeBERTa分類器が安定。Zenn記事のPTU/PAYGスピルオーバーに相当
2. **3モデル以上のプール**: カスケード型が有効だが、閾値チューニングが必要
3. **レイテンシ制約あり**: 分類器ベース（DeBERTa、MLP）を選択。LLM-Routerは避ける

### 分類器ベースルーターの実装例

```python
from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch


@dataclass
class RoutingDecision:
    """ルーティング判定結果"""
    selected_model: str
    confidence: float
    latency_ms: float


class ClassifierRouter:
    """DeBERTaベースのLLMルーター

    Args:
        model_name: 分類器モデル名
        model_pool: ルーティング先のLLMモデルリスト
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        model_pool: list[str] | None = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(model_pool or []),
        )
        self.model_pool = model_pool or []

    def route(self, query: str) -> RoutingDecision:
        """クエリに基づいてモデルを選択

        Args:
            query: ユーザークエリ

        Returns:
            ルーティング判定結果
        """
        import time

        start = time.perf_counter()

        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = self.classifier(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            idx = torch.argmax(probs, dim=-1).item()

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RoutingDecision(
            selected_model=self.model_pool[idx],
            confidence=probs[0, idx].item(),
            latency_ms=elapsed_ms,
        )
```

### カスケード型ルーターの実装パターン

```python
def cascade_route(
    query: str,
    models: list[dict],
    confidence_threshold: float = 0.8,
) -> str:
    """カスケード型ルーティング

    コストの低いモデルから順に試行し、
    信頼度が閾値を超えたら結果を返す。

    Args:
        query: ユーザークエリ
        models: コスト昇順のモデルリスト
        confidence_threshold: 信頼度閾値

    Returns:
        選択されたモデルの応答
    """
    for model in sorted(models, key=lambda m: m["cost_per_token"]):
        response = call_llm(model["name"], query)

        if response.confidence >= confidence_threshold:
            return response.text

    # 最高性能モデルにフォールバック
    best_model = max(models, key=lambda m: m["quality_score"])
    return call_llm(best_model["name"], query).text
```

## 実験結果（Results）

### ベンチマーク結果の概要

RouterEvalの1,152組み合わせの評価から得られた主要な定量結果:

| 評価軸 | 結果 |
|--------|------|
| 全組み合わせ数 | 1,152（16ルーター × 9プール × 8クエリセット） |
| 単一ルーターの最高Win Rate | 0.42（MF Quality、2モデルプール） |
| コスト50%削減時の平均PG | 5-15% |
| 分類器ルーターのレイテンシ | 5-10ms |
| LLM-Routerのレイテンシ | 2,000-5,000ms |

**分析ポイント**:
- 最高Win Rateが0.42であることは、どのルーターも半分以上のケースで他に劣ることを意味する
- プールサイズが大きくなるほど（5モデル）、ルーティングの難易度が上がりWin Rateが低下
- 分類器ベースはレイテンシで圧倒的に有利（LLM-Routerの200-500倍高速）

### パレート最適分析

Non-dominated Rateの分析では、MF QualityとDeBERTa分類器がパレートフロンティアに最も近い位置にあることが示されています。ただし、クエリの難易度分布によって最適点が移動するため、**運用環境のクエリ特性に基づくチューニング**が不可欠です。

## 実運用への応用（Practical Applications）

### Azure APIMのルーティングとの対応

Zenn記事で解説したAzure APIMのバックエンドプール設定は、RouterEvalの分類では**ルールベースルーティング**に相当します。

| RouterEvalの分類 | Azure APIMでの実装 |
|-----------------|-------------------|
| ルールベース（Priority） | バックエンドプールのpriority設定 |
| カスケード | PTU → PAYGスピルオーバー |
| ラウンドロビン | バックエンドプールのweight設定 |
| コスト最小 | PAYGのみ使用（PTU不使用） |

RouterEvalの知見を適用すると、APIMの設定にクエリ複雑度ベースの分類器ルーティングを追加することで、PTUの利用効率を向上させる可能性があります。具体的には、簡単なクエリ（分類・抽出）をPAYGの小型モデルに、複雑な推論クエリをPTUのフラッグシップモデルに振り分ける戦略です。

### スケーリングの課題

RouterEvalは実験室環境での評価であり、以下の本番環境固有の課題には対応していません:

- **動的な負荷変動**: リアルタイムのTPM/RPM制約下でのルーティング
- **レイテンシSLA**: エンドツーエンドのレイテンシ要件を満たすルーティング
- **障害耐性**: プロバイダー障害時のフォールバック（Zenn記事のCircuit Breaker）

これらはRouterEvalのスコープ外であり、Zenn記事で解説したAPIM + Front Doorのインフラ層での対応が引き続き必要です。

## 関連研究（Related Work）

- **RouteLLM** (Ong et al., arXiv:2406.18665): 2モデルルーティングに特化した手法。RouterEvalはこれを16ルーター・9プールに拡張した包括的評価
- **FrugalGPT** (Chen et al., arXiv:2305.05176): カスケード型ルーティングによるコスト削減。RouterEvalのカスケード型ルーターの基盤
- **Hybrid LLM** (Ding et al., arXiv:2404.14618): 小型モデルと大型モデルのルーティング。RouterEvalの2モデルプール設定に対応

## まとめと今後の展望

RouterEvalは、LLMルーティング戦略の評価において最も包括的なベンチマークを提供しています。万能のルーターが存在しないという知見は、Zenn記事のAPIM設定のようなインフラ層でのルーティング設計においても重要な示唆を与えます。プール構成とクエリ特性に応じたルーター選択が必要であり、精度-コストトレードオフの解決は今後の研究課題として残されています。

今後の方向性として、動的なプール構成への適応、マルチモーダルクエリへの対応、リアルタイムのコスト・レイテンシ制約下でのルーティング最適化が挙げられています。

## 参考文献

- **arXiv**: [https://arxiv.org/abs/2503.10657](https://arxiv.org/abs/2503.10657)
- **Related**: RouteLLM - [https://arxiv.org/abs/2406.18665](https://arxiv.org/abs/2406.18665)
- **Related**: FrugalGPT - [https://arxiv.org/abs/2305.05176](https://arxiv.org/abs/2305.05176)
- **Related Zenn article**: [Azure OpenAIマルチリージョン負荷分散](https://zenn.dev/0h_n0/articles/b2bc25d92f46fb)

---

:::message
この記事はAI（Claude Code）により自動生成されました。内容の正確性については情報源の論文に基づいていますが、最新の情報は公式ドキュメントもご確認ください。
:::
