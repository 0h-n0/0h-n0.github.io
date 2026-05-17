---
layout: post
title: "ICML 2025論文解説: Auditing Prompt Caching in Language Model APIs — プロバイダのキャッシュ実装を統計的に暴く"
description: "17のLLM APIプロバイダのプロンプトキャッシュ実装をタイミング攻撃で監査し、セキュリティリスクを明らかにしたICML 2025採択論文の解説"
categories: [blog, paper, conference]
tags: [prompt-caching, security, side-channel, llm-api, icml]
date: 2026-05-18 13:00:00 +0900
source_type: conference
conference: ICML 2025
source_url: https://arxiv.org/abs/2502.07776
zenn_article: 37e71fbb85e1a6
zenn_url: https://zenn.dev/0h_n0/articles/37e71fbb85e1a6
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Auditing Prompt Caching in Language Model APIs (ICML 2025)](https://arxiv.org/abs/2502.07776) の解説記事です。

## 論文概要（Abstract）

本論文は、プロンプトキャッシュが広く導入されたLLM APIにおいて、キャッシュの存在がタイミングサイドチャネルを生み出し、ユーザー間のプライバシーリスクに繋がる可能性を実証的に検証した研究である。著者らは17のLLM APIプロバイダを対象に統計的監査を実施し、そのうち8プロバイダでグローバル（ユーザー間共有）なキャッシュが検出されたと報告している。さらに、このタイミング差を利用して「OpenAIのembeddingモデルがdecoder-only Transformerである」という、従来公開されていなかったアーキテクチャ情報を推定した事例も示されている。

この記事は [Zenn記事: エージェントのプロンプトキャッシュ設計 — ツール定義と思考トークンを壊さない実装](https://zenn.dev/0h_n0/articles/37e71fbb85e1a6) の深掘りです。

## 情報源

- **会議名**: ICML 2025（International Conference on Machine Learning）
- **年**: 2025
- **URL**: https://arxiv.org/abs/2502.07776
- **著者**: Chenchen Gu, Xiang Lisa Li, Rohith Kuditipudi, Percy Liang, Tatsunori Hashimoto
- **分野**: cs.CL, cs.CR, cs.LG

## カンファレンス情報

**ICMLについて**: ICML（International Conference on Machine Learning）は機械学習分野のトップカンファレンスの1つであり、NeurIPS・ICLRと並ぶ三大会議として位置づけられている。採択率は通常25-30%程度で、厳密な査読プロセスを経て採択される。本論文は2025年2月に投稿され、ICML 2025に採択された。セキュリティと機械学習の交差領域という新規性が評価されたものと推察される。

## 技術的詳細（Technical Details）

### タイミングサイドチャネルの原理

プロンプトキャッシュはプレフィックスのKV計算を省略するため、キャッシュヒット時の応答時間がミス時よりも短くなる。この応答時間差（タイミング差）が統計的に検出可能なサイドチャネルとなる。

キャッシュヒット時のTTFT（Time to First Token）を$T_{\text{hit}}$、ミス時を$T_{\text{miss}}$とすると：

$$
T_{\text{miss}} - T_{\text{hit}} = \frac{n \cdot C_{\text{prefill}}}{F_{\text{GPU}}} - T_{\text{cache\_read}}
$$

ここで、
- $n$: プレフィックスのトークン数
- $C_{\text{prefill}}$: 1トークンあたりのprefill FLOPs
- $F_{\text{GPU}}$: GPUの演算性能（FLOP/s）
- $T_{\text{cache\_read}}$: キャッシュからのKV読み込み時間

大規模モデル（100Bパラメータ級）で10,000トークンのプレフィックスの場合、タイミング差は数百ミリ秒〜数秒に達する。これは統計的検定で十分に検出可能な差である。

### 統計的監査手法

著者らは以下の手順で各プロバイダのキャッシュ実装を監査している：

**ステップ1: ベースライン測定**

同一プロンプトを2回連続で送信し、1回目（確実にミス）と2回目（ヒットの可能性）のTTFTを記録する。

**ステップ2: 仮説検定**

帰無仮説$H_0$: 「キャッシュは存在しない（1回目と2回目のTTFTに有意差なし）」に対して、対応のあるt検定を適用する。

$$
t = \frac{\bar{d}}{s_d / \sqrt{n}}
$$

ここで、
- $\bar{d}$: 各ペアのTTFT差の平均
- $s_d$: TTFT差の標準偏差
- $n$: 測定ペア数

**ステップ3: グローバル vs ローカルキャッシュの判定**

異なるAPIキーから同一プロンプトを送信し、2つ目のキーでキャッシュヒットが検出された場合、グローバルキャッシュ（ユーザー間共有）と判定する。

```python
import time
from anthropic import Anthropic

def audit_cache(client: Anthropic, prompt: str, n_trials: int = 30) -> dict:
    """キャッシュ存在の統計的監査
    
    注意: この監査はAPIの利用規約に従って実施すること
    """
    ttft_first: list[float] = []
    ttft_second: list[float] = []

    for _ in range(n_trials):
        # 1回目: 必ずキャッシュミス（ユニークなプレフィックスを付加）
        unique_prefix = f"audit_{time.time_ns()}_"
        start = time.perf_counter()
        resp1 = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1,
            messages=[{"role": "user", "content": unique_prefix + prompt}],
        )
        ttft_first.append(time.perf_counter() - start)

        # 2回目: 同一プロンプト（キャッシュヒットの可能性）
        start = time.perf_counter()
        resp2 = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1,
            messages=[{"role": "user", "content": unique_prefix + prompt}],
        )
        ttft_second.append(time.perf_counter() - start)

        time.sleep(1)  # レート制限対策

    return {
        "mean_first": sum(ttft_first) / n_trials,
        "mean_second": sum(ttft_second) / n_trials,
        "speedup_ratio": sum(ttft_first) / sum(ttft_second),
    }
```

### アーキテクチャ推定への応用

著者らの最も注目すべき発見は、タイミング情報からモデルのアーキテクチャを推定できる点である。具体的には、OpenAIのembeddingモデルが「decoder-only Transformer」であることを以下のロジックで推定している：

1. embeddingリクエストのTTFTがプロンプト長に対して線形に増加する → prefill計算が存在
2. 同一プロンプトの2回目リクエストでTTFTが短縮される → KVキャッシュが存在
3. KVキャッシュはSelf-Attention（decoder-only）でのみ意味がある → decoder-onlyアーキテクチャ

この推定は、embeddingモデルが一般的にBERT系（encoder-only、bidirectional attention）と仮定されていた既存の認識と矛盾し、OpenAIのアーキテクチャ選択に関する新たな知見を提供している。

## 実験結果（Results）

### プロバイダ別の監査結果

著者らが2024年9月〜10月に実施した監査の結果：

| プロバイダ | キャッシュ検出 | キャッシュタイプ | タイミング差 |
|:---:|:---:|:---:|:---:|
| OpenAI | 検出 | グローバル | 有意（p < 0.01） |
| Anthropic | 検出 | ユーザー別 | 有意（p < 0.01） |
| Google (Gemini) | 検出 | グローバル | 有意（p < 0.01） |
| DeepSeek | 検出 | グローバル | 有意（p < 0.01） |
| Moonshot (Kimi) | 検出 | グローバル | 有意（p < 0.01） |
| 他3プロバイダ | 検出 | 各種 | 有意 |
| 残り9プロバイダ | 非検出 | N/A | 非有意 |

17プロバイダ中8プロバイダでキャッシュが検出され、うち少なくとも5プロバイダでユーザー間共有（グローバル）キャッシュが確認された。

### セキュリティインパクト

グローバルキャッシュが存在するプロバイダでは、以下の攻撃が理論上可能である：

1. **プロンプト存在推定**: 特定のプロンプトが他のユーザーによって送信されたか否かを判定
2. **トラフィックパターン推定**: 特定トピックに関するリクエスト頻度の推定
3. **アーキテクチャ推定**: 前述のembeddingモデルの例

ただし著者らは、「実用的な攻撃には大量のリクエストと統計的分析が必要であり、即座にプライバシー侵害に繋がるわけではない」と述べている。

## 実装のポイント（Implementation）

### エージェント開発者への示唆

本論文の知見は、プロンプトキャッシュを活用するエージェント開発者にとって以下の設計上の注意点を提起する：

1. **キャッシュ共有の認識**: グローバルキャッシュを持つプロバイダでは、自分のプロンプトのKVが他ユーザーと共有される可能性がある。機密性の高いシステムプロンプトを使用する場合はこの点を認識すべき。

2. **タイミング情報の漏洩**: キャッシュヒット/ミスの応答時間差がクライアントから観測可能であるため、自社のエージェントのプロンプト構成がタイミング分析で推定されるリスクがある。

3. **プロバイダ選択の判断材料**: プライバシーを重視する場合、ユーザー別キャッシュ（Anthropicの方式）を採用するプロバイダを選択する判断材料となる。

### 防御策

```python
def add_timing_noise(base_response_time: float, noise_std: float = 0.05) -> float:
    """レスポンス時間にノイズを追加し、タイミングサイドチャネルを緩和

    注意: これはクライアント側の緩和策であり、プロバイダ側の対策が本質的
    """
    import random
    noise = random.gauss(0, noise_std)
    return max(0, base_response_time + noise)
```

## 実運用への応用（Practical Applications）

### プロンプトキャッシュ設計への影響

本論文の知見は、Zenn記事で解説されている「キャッシュブレークポイントの最適配置」に対して、セキュリティの観点を追加する：

- **システムプロンプトの機密性**: ツール定義やシステムプロンプトに企業固有の知識（営業ノウハウ、内部ルール等）を含む場合、グローバルキャッシュを通じて他ユーザーがその存在を推定できるリスクがある
- **キャッシュTTLの意味**: TTLは単なるコスト最適化パラメータではなく、キャッシュの残存時間がタイミング攻撃のウィンドウを決定するセキュリティパラメータでもある
- **ユーザー分離**: プライバシー要件の高いシステムでは、ユーザーごとにキャッシュを分離するプロバイダ（Anthropic）を選択するか、独自推論サーバを運用する

### キャッシュ監査の実装

本論文の手法を応用し、自社のLLM推論サーバでキャッシュが意図通りに動作しているかを監査するテストスイートを構築できる：

```python
import statistics
from scipy import stats


def test_cache_isolation(
    client_a,
    client_b,
    test_prompt: str,
    n_trials: int = 50,
) -> dict:
    """2つのクライアント間でキャッシュが分離されているか検証
    
    帰無仮説: client_aのリクエスト後、client_bのTTFTに変化なし
    """
    # client_b のベースラインTTFT
    baseline_ttfts = [measure_ttft(client_b, test_prompt) for _ in range(n_trials)]

    # client_a でプロンプトを送信（キャッシュ生成）
    _ = client_a.messages.create(
        model="claude-sonnet-4-6", max_tokens=1,
        messages=[{"role": "user", "content": test_prompt}],
    )

    # client_b のTTFTを再測定
    post_ttfts = [measure_ttft(client_b, test_prompt) for _ in range(n_trials)]

    t_stat, p_value = stats.ttest_ind(baseline_ttfts, post_ttfts)

    return {
        "baseline_mean": statistics.mean(baseline_ttfts),
        "post_mean": statistics.mean(post_ttfts),
        "t_statistic": t_stat,
        "p_value": p_value,
        "cache_shared": p_value < 0.01,  # 有意水準1%
    }
```

## 関連研究（Related Work）

- **"Don't Break the Cache" (2601.06007)**: プロンプトキャッシュの「効果」に焦点を当てた研究。本論文はキャッシュの「リスク」を評価する補完的研究
- **CacheSolidarity (2603.10726)**: マルチテナント環境でのプレフィックスキャッシュ・サイドチャネル防御手法。本論文の発見を受けた防御側研究
- **Marconi (2411.19379)**: ハイブリッドLLM向けプレフィックスキャッシュシステム。キャッシュ効率とプライバシーのトレードオフを考慮

## まとめと今後の展望

本論文は、プロンプトキャッシュがもたらすセキュリティリスクを初めて体系的に実証した研究として重要である。17プロバイダ中8プロバイダでキャッシュが検出され、うち5プロバイダ以上でユーザー間共有キャッシュが確認されたことは、キャッシュ設計において「効率」と「プライバシー」の両立が今後の重要課題であることを示している。エージェント開発者は、キャッシュ最適化を追求する際にこのセキュリティ次元も考慮に入れるべきである。今後は、差分プライバシーやノイズ注入によるサイドチャネル緩和手法の研究が期待される。

## 参考文献

- **arXiv**: https://arxiv.org/abs/2502.07776
- **Conference**: ICML 2025
- **CacheSolidarity**: https://arxiv.org/abs/2603.10726
- **Related Zenn article**: https://zenn.dev/0h_n0/articles/37e71fbb85e1a6
