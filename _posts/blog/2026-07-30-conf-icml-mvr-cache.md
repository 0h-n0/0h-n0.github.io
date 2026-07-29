---
layout: post
title: "ICML 2026論文解説: MVR-cache - マルチベクトル検索によるセマンティックキャッシュ最適化"
description: "プロンプトをセグメント分解しマルチベクトル検索で類似度判定を高精度化、キャッシュヒット率を最大37%向上させたICML 2026採択論文を解説"
categories: [blog, paper, conference]
tags: [semantic-cache, multi-vector-retrieval, reinforcement-learning, llm, vectordb, cache, python, rag]
date: 2026-07-30 09:00:00 +0900
source_type: conference
conference: "ICML 2026"
arxiv_id: "2605.24914"
source_url: https://arxiv.org/abs/2605.24914
zenn_article: 1707bd6149514c
zenn_url: https://zenn.dev/0h_n0/articles/1707bd6149514c
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## 論文概要（Abstract）

本記事は [MVR-cache: Optimizing Semantic Caching via Multi-Vector Retrieval and Learned Prompt Segmentation](https://arxiv.org/abs/2605.24914) の解説記事です。

MVR-cacheは、LLMのセマンティックキャッシュにおける類似度判定を高精度化するフレームワークである。従来の単一ベクトルによるコサイン類似度比較では、プロンプト中の微妙な意味の違いを見落とす問題があった。著者らは、プロンプトを学習可能なセグメンテーションモデルで分割し、マルチベクトル検索（MaxSim）で細粒度の類似度を計算する手法を提案している。セグメンテーションの最適化にはREINFORCE法を用い、正確性保証を維持しながらキャッシュヒット率を最大37%向上させたと報告されている。

この記事は [Zenn記事: セマンティックキャッシュの類似度閾値チューニングとベクトルDB別性能比較](https://zenn.dev/0h_n0/articles/1707bd6149514c) の深掘りです。

## 情報源

- **会議名**: ICML 2026（International Conference on Machine Learning）
- **arXiv ID**: 2605.24914
- **URL**: [https://arxiv.org/abs/2605.24914](https://arxiv.org/abs/2605.24914)
- **著者**: Ali Noshad, Zishan Zheng, Yinjun Wu
- **発表年**: 2026

## カンファレンス情報

ICML（International Conference on Machine Learning）は、機械学習分野における最高峰の国際会議の1つである。ICML 2026の採択率は26.6%（6,352件/23,918件投稿）であり、前年のICML 2025（12,107件投稿）から投稿数がほぼ倍増している。Oral採択率は0.7%（168件）、Spotlight Poster採択率は2.2%（536件）と報告されている。MVR-cacheがこの競争率の高い会議に採択された点は、セマンティックキャッシュの学術的・実務的重要性を示している。

## 背景と動機

セマンティックキャッシュは、LLM APIのコスト削減とレイテンシ低減のための技術である。新しいプロンプトが過去のプロンプトと意味的に類似していれば、LLMを再呼び出しせずにキャッシュ済みの応答を返す。しかし、従来手法は単一のエンベディングベクトル間のコサイン類似度で類似性を判定しており、この方法には根本的な限界がある。

著者らは論文中で、映画レビュー分類タスクの例を挙げている。「この映画のポジティブなレビューを書いて」と「この映画のネガティブなレビューを書いて」という2つのプロンプトは、単一ベクトル表現では高い類似度を示す。しかし、期待されるLLM応答は正反対であり、キャッシュヒットとして処理すると誤った応答を返してしまう。この問題は、単一ベクトルがプロンプト全体を1つの点に圧縮するため、局所的な意味の違い（「ポジティブ」vs「ネガティブ」）を捉えきれないことに起因する。

vCache（ICLR 2026）はプロンプトごとに動的な閾値を学習することでこの問題を緩和したが、類似度計算自体は依然として単一ベクトルのコサイン類似度に依存していた。MVR-cacheはこの限界を、類似度計算そのものをマルチベクトル化することで根本的に解決しようとする試みである。

## 主要な貢献

著者らは以下の3点を主要な貢献として挙げている。

- **マルチベクトル検索（MVR）によるセマンティックキャッシュ**: プロンプトを複数のセグメントに分割し、ColBERT由来のMaxSimスコアで細粒度の類似度を計算する枠組みを提案。単一ベクトルでは区別できないプロンプト間の微妙な差異を捉えることが可能になる。

- **学習可能なプロンプトセグメンテーション**: BERTエンコーダ、MLP、LSTM、Attentionを組み合わせたポインターネットワークで、プロンプトの最適な分割位置を学習する。セグメンテーションは離散的な組合せ最適化であるため、REINFORCE法を用いて訓練する。GPUメモリ使用量は500-600 MBに抑えられている。

- **理論的保証付きの訓練目的関数**: MLE損失の最小化がユーザ指定の誤り率$\delta$の下でキャッシュヒット率を最大化することを定理として証明（Theorem 3.3）。vCacheの正確性保証を継承しつつ、キャッシュヒット率の最適性を理論的に裏付けている。

## 技術的詳細（Technical Details）

### セマンティックキャッシュの定式化

セマンティックキャッシュの判定問題は以下のように定式化される。新しいプロンプト$x$に対して、キャッシュ内の最近傍プロンプト$x_j$との類似度$s$がある閾値$t$を超えるかどうかで、キャッシュヒットかミスかを判定する。

vCacheはこの閾値$t$をプロンプトごとに動的に推定するが、類似度$s$の計算は単一ベクトルのコサイン類似度に限られていた。MVR-cacheは類似度計算そのものを改良する。

### MaxSimスコア

ColBERTのlate interactionに着想を得て、プロンプトを複数のセグメントに分割し、セグメント単位で類似度を計算する。MaxSimスコアは以下のように定義される。

$$
\text{MaxSim}(x, x_j) = \sum_{t} \max_{s} \text{sim}(\mathcal{E}(x^{(t)}), \mathcal{E}(x_j^{(s)}))
$$

ここで、
- $x^{(t)}$: プロンプト$x$の$t$番目のセグメント
- $x_j^{(s)}$: キャッシュ内プロンプト$x_j$の$s$番目のセグメント
- $\mathcal{E}(\cdot)$: セグメントのエンベディング関数（BGEモデル）
- $\text{sim}(\cdot, \cdot)$: コサイン類似度

直感的には、クエリ側の各セグメントに対して、キャッシュ側で最も類似するセグメントを見つけ、その類似度の総和を取る。これにより「ポジティブレビューを書いて」と「ネガティブレビューを書いて」のような局所的な意味の違いを検出できる。

### 対称MaxSim（SMaxSim）

MaxSimは非対称であるため、著者らは対称版のSMaxSimを定義している。

$$
\text{SMaxSim}_{\Theta}(x_i, x_j) := \frac{1}{2} \left[ \frac{\text{MaxSim}(x_i, x_j)}{|x_i|} + \frac{\text{MaxSim}(x_j, x_i)}{|x_j|} \right]
$$

ここで、
- $|x_i|$, $|x_j|$: それぞれのプロンプトのセグメント数
- $\Theta$: セグメンテーションモデルのパラメータ

セグメント数で正規化することで、異なる長さのプロンプト間でも公平な比較が可能になる。

### ポインターネットワークによるセグメンテーション

プロンプトの分割位置を決定するために、4つのコンポーネントからなるポインターネットワークを用いる。

```mermaid
flowchart LR
    A[プロンプトトークン列] --> B["BERT Encoder (Theta_1)"]
    B --> C["MLP (Theta_2)"]
    C --> D["LSTM (Theta_3)"]
    D --> E["Attention (Theta_4)"]
    E --> F[セグメント境界]
```

- **$\Theta_1$（BERTエンコーダ）**: 入力トークン列をコンテキスト化された埋め込みに変換
- **$\Theta_2$（単層MLP）**: 埋め込みをポインターネットワークの状態空間に射影
- **$\Theta_3$（単層LSTM）**: 状態を集約し、セグメント境界の文脈を捉える
- **$\Theta_4$（Attention層）**: 学習可能なパラメータ$v$, $W_1$, $W_2$を用いて分割位置を選択

Attention層は以下のように候補位置のスコアを計算する。

$$
u_{1j} = v^T \tanh(W_1 h_j + W_2 d_1), \quad j \in [1, \ldots, L]
$$

$$
a_{1j} = \text{softmax}(u_{1j}) \cdot \mathbf{I}(j \in \mathcal{P}_x)
$$

ここで、
- $h_j$: LSTM出力の$j$番目の隠れ状態
- $d_1$: デコーダの状態ベクトル
- $\mathcal{P}_x$: 句読点やスペースなどの候補分割位置の集合
- $\mathbf{I}(\cdot)$: 指示関数（候補位置でないトークンをマスク）

候補分割位置$\mathcal{P}_x$は句読点やスペースなど自然な区切りに限定されており、単語の途中で分割されることを防いでいる。選択された位置から再帰的にセグメント境界を決定していく。

### REINFORCE法による訓練

セグメンテーションは離散的な決定であるため、勾配を直接計算できない。著者らはREINFORCE法を用いて、キャッシュ判定精度を報酬としたポリシー勾配で最適化する。

$$
R = \sum_{nn_{\Theta}(x_j) = x_i} -\ell_{\text{BCE}}(\mathcal{L}(\text{SMaxSim}_{\Theta}(x_i, x_j); t_i, \gamma_i), c_j)
$$

ここで、
- $nn_{\Theta}(x_j)$: セグメンテーション$\Theta$の下での$x_j$の最近傍
- $\ell_{\text{BCE}}$: バイナリクロスエントロピー損失
- $\mathcal{L}(\cdot; t_i, \gamma_i)$: SMaxSimスコアからキャッシュヒット確率への変換（ロジスティック関数、閾値$t_i$、スケール$\gamma_i$）
- $c_j$: $x_j$がキャッシュヒットであるべきかのラベル（1または0）

報酬$R$が大きいほど、セグメンテーションがキャッシュ判定の精度向上に寄与していることを意味する。

### 理論的保証（Theorem 3.3）

著者らはTheorem 3.3において、以下の理論的保証を証明している。類似度スコアの正規分布仮定（Assumption 3.1）と均衡なクラス事前確率$\Pr(c=1) = \Pr(c=0) = 0.5$（Assumption 3.2）の下で、MLE損失（Equation 3）を最小化するセグメンテーションモデルは、任意のユーザ指定誤り率$\delta$の下でキャッシュヒット率を最大化する。

$$
(\hat{t}, \hat{\gamma}) = \arg\min_{t, \gamma} \sum_{(s_i, c_i) \in \mathcal{O}(nn(x))} \ell_{\text{BCE}}(\mathcal{L}(s_i; t, \gamma), c_i)
$$

ここで、
- $\mathcal{O}(nn(x))$: プロンプト$x$の最近傍に関する観測データ
- $s_i$: 類似度スコア
- $c_i$: 正解ラベル

この定理により、MVR-cacheはvCacheの正確性保証（誤り率$\leq \delta$）を継承しつつ、キャッシュヒット率の最適性が理論的に保証される。

## アルゴリズム

以下はMVR-cacheの中核であるMaxSimスコア計算とSMaxSimスコア計算のPython実装例である。

```python
import numpy as np
from numpy.typing import NDArray


def maxsim_score(
    query_segments: list[NDArray[np.float32]],
    cache_segments: list[NDArray[np.float32]],
) -> float:
    """MaxSimスコアを計算する。

    クエリ側の各セグメント埋め込みに対して、キャッシュ側で
    最もコサイン類似度が高いセグメント埋め込みを見つけ、
    その類似度の総和を返す。

    Args:
        query_segments: クエリプロンプトのセグメント埋め込みリスト。
            各要素は shape (dim,) のベクトル。
        cache_segments: キャッシュプロンプトのセグメント埋め込みリスト。
            各要素は shape (dim,) のベクトル。

    Returns:
        MaxSimスコア（float）。値が大きいほど類似度が高い。
    """
    total: float = 0.0
    for q_emb in query_segments:
        max_sim: float = -1.0
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        for c_emb in cache_segments:
            c_norm = c_emb / (np.linalg.norm(c_emb) + 1e-9)
            sim = float(np.dot(q_norm, c_norm))
            if sim > max_sim:
                max_sim = sim
        total += max_sim
    return total


def smaxsim_score(
    segments_i: list[NDArray[np.float32]],
    segments_j: list[NDArray[np.float32]],
) -> float:
    """対称MaxSim（SMaxSim）スコアを計算する。

    双方向のMaxSimを正規化して平均を取ることで、
    セグメント数の異なるプロンプト間でも公平に比較する。

    Args:
        segments_i: プロンプトiのセグメント埋め込みリスト。
        segments_j: プロンプトjのセグメント埋め込みリスト。

    Returns:
        SMaxSimスコア（0-1の範囲）。
    """
    n_i = len(segments_i)
    n_j = len(segments_j)
    if n_i == 0 or n_j == 0:
        return 0.0

    forward = maxsim_score(segments_i, segments_j) / n_i
    backward = maxsim_score(segments_j, segments_i) / n_j
    return 0.5 * (forward + backward)


def cache_lookup(
    query_segments: list[NDArray[np.float32]],
    cache_entries: list[dict],
    delta: float = 0.01,
) -> dict | None:
    """MVR-cacheのキャッシュルックアップを実行する。

    キャッシュ内の各エントリとSMaxSimスコアを計算し、
    プロンプト固有の閾値を超えた場合にキャッシュヒットを返す。

    Args:
        query_segments: クエリプロンプトのセグメント埋め込みリスト。
        cache_entries: キャッシュエントリのリスト。各エントリは
            {"segments": list[NDArray], "threshold": float,
             "response": str} の辞書。
        delta: ユーザ指定の最大誤り率。

    Returns:
        キャッシュヒットした場合はエントリ辞書、ミスの場合はNone。
    """
    best_score: float = -1.0
    best_entry: dict | None = None

    for entry in cache_entries:
        score = smaxsim_score(query_segments, entry["segments"])
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is not None and best_score >= best_entry["threshold"]:
        return best_entry

    return None
```

## 実装のポイント

MVR-cacheの実装において、著者らが強調している技術的なポイントは以下の通りである。

**エンベディングモデルの選択**: セグメント単位のエンベディングにはBGEモデルが使用されている。プロンプト全体ではなくセグメント（文や句の単位）を入力とするため、短いテキストに対して安定した埋め込みを生成できるモデルが求められる。

**候補分割位置の制限**: セグメンテーションの探索空間を削減するため、分割候補を句読点やスペースなどの自然な区切り位置に限定している。これにより、単語の途中で不自然に分割されることを防ぎ、各セグメントが意味的にまとまった単位となることを保証している。

**近傍の定期更新**: セグメンテーションモデルのパラメータ$\Theta$が更新されると、各プロンプトのセグメント埋め込みも変化するため、最近傍関係も変化する。著者らは$K$ステップごとに最近傍を再計算する方針を採用しており、計算コストと精度のトレードオフを制御している。

**弱教師ありラベリング**: 訓練に必要なラベル（2つのプロンプトのLLM応答が同等かどうか）の生成にGPT-4o-miniをプロキシとして使用し、GPT-4による完全なラベリングと比較して80.4%のコスト削減を実現している。著者らの報告によると、プロキシラベルとGPT-4ラベルの一致率は97.1%である。

**軽量な推論コスト**: セグメンテーションモデル全体のGPUメモリ使用量は500-600 MBであり、LLM推論と比較して十分に軽量である。キャッシュ判定のオーバーヘッドがLLM呼び出しのコスト削減を上回らないよう設計されている。

## 実験結果

著者らは4つのデータセットで評価を行っている。

**データセットの概要**:

| データセット | プロンプト数 | タスク種別 |
|:---|---:|:---|
| SemCacheClassification | 45K | 短文分類 |
| SemCacheSearchQueries | 150K | Web検索クエリ（ORCAS由来） |
| PromptBench | 38K | 質問応答（摂動SQUAD-V2） |
| QNLI | 29K | 質問応答 |

訓練には各データセットから3Kのラベル付きプロンプトを使用している。

**エンドツーエンドレイテンシ（分単位、論文Table 1より）**:

| データセット | vCache | ColBERT | POQD | MVR-cache |
|:---|---:|---:|---:|---:|
| SemCacheClassification | 408.49 | 501.46 | 971.51 | **383.32** |
| SemCacheSearchQueries | 6361.52 | 6521.89 | 6990.08 | **6345.61** |
| PromptBench | 1870.57 | 2294.38 | 2945.20 | **1866.58** |
| QNLI | 1536.00 | 1626.37 | 2648.80 | **1504.43** |

全データセットにおいてMVR-cacheが最小のエンドツーエンドレイテンシを達成している。SemCacheClassificationではvCacheの408.49分に対し383.32分と約6%の削減、POQDの971.51分に対しては約60%の削減である。

**キャッシュヒット率の改善**: 著者らは、SemCacheClassificationにおいてvCacheに対し約9%のキャッシュヒット率改善を報告している。この9%の改善は約4,100回のLLM呼び出し削減に相当すると述べている。SemCacheSearchQueriesではalways-cacheプロトコル（全クエリをキャッシュ対象とする設定）において最大37%のキャッシュヒット率改善を達成している。

**誤り率の遵守**: 論文のFigure 4-5において、MVR-cacheがユーザ指定の誤り率$\delta$を全データセットで遵守していることが確認されている。これは、キャッシュヒット率の向上が正確性を犠牲にしたものではないことを示している。

**データセット間の汎化**: 論文のFigure 6において、あるデータセットで訓練されたセグメンテーションモデルが、未見のQNLIデータセットに対しても性能向上を示すことが報告されている。

## 実運用への応用

MVR-cacheの実運用への応用は、関連するZenn記事「[セマンティックキャッシュの類似度閾値チューニングとベクトルDB別性能比較](https://zenn.dev/0h_n0/articles/1707bd6149514c)」で議論されている3層キャッシュアーキテクチャと直接関連する。

Zenn記事ではLayer 2（セマンティックキャッシュ層）の閾値チューニングが詳しく解説されているが、MVR-cacheはこのLayer 2の類似度計算を根本的に改善する手法である。単一ベクトルのコサイン類似度をSMaxSimに置き換えることで、閾値設定の困難さを緩和できる可能性がある。Zenn記事が指摘する「顧客向けアプリケーションでは0.92-0.95、FAQ系チャットボットでは0.80-0.85」という用途別閾値の問題に対して、MVR-cacheはプロンプトごとに動的な閾値を推定するため、用途ごとの手動調整の必要性が低減されると考えられる。

一方で、実運用における課題も存在する。マルチベクトル化によりベクトルDBに格納するベクトル数が増加するため、ストレージコストとインデックス構築コストが上昇する。著者らが報告しているセグメンテーションモデルのGPUメモリ500-600 MBは推論時のみの数値であり、訓練には3Kのラベル付きデータとGPT-4o-miniによるラベリングパイプラインが必要となる。Zenn記事で比較されているpgvector、Qdrant、Redis VSSといったベクトルDBがマルチベクトル検索（MaxSim演算）をネイティブにサポートしているかどうかも、採用判断の重要な要素となる。QdrantはColBERTスタイルのマルチベクトル検索をサポートしており、MVR-cacheとの親和性が高い。

## 関連研究

- **vCache**（ICLR 2026）: セマンティックキャッシュにおける最初の検証済み正確性保証を提供した研究である。プロンプトごとの動的閾値推定をオンライン学習で実現したが、類似度計算は単一ベクトルのコサイン類似度に依存している。MVR-cacheはvCacheの閾値推定メカニズムを継承しつつ、類似度計算をマルチベクトル化することで改善している。

- **ColBERT / ColBERTv2**: 情報検索分野におけるlate interaction手法であり、文書をトークンレベルの複数ベクトルで表現しMaxSimで類似度を計算する。MVR-cacheのMaxSimスコアはColBERTに直接由来する。ただし、ColBERTはトークン単位で分割するのに対し、MVR-cacheは学習されたセグメント（句や文の単位）で分割する点が異なる。

- **POQD**（Performance-Oriented Query Decomposer）: マルチベクトル検索のためのクエリ分解手法である。MVR-cacheとの比較実験では、全データセットにおいてMVR-cacheがレイテンシ・キャッシュヒット率の両面で上回っている（論文Table 1より）。

## まとめ

MVR-cacheは、セマンティックキャッシュの類似度判定をマルチベクトル検索で高精度化するフレームワークである。著者らは、学習可能なセグメンテーション、MaxSimスコア、REINFORCE訓練、理論的保証の4つの要素を統合し、正確性を犠牲にせずにキャッシュヒット率を改善したと報告している。3Kのラベル付きデータと500-600 MBのGPUメモリで動作する点は、実運用への導入障壁を下げている。

今後の研究方向としては、マルチターン対話におけるセグメンテーション戦略、ストリーミング環境でのオンライン適応、ベクトルDB側でのMaxSimネイティブサポートの拡充が考えられる。セマンティックキャッシュは依然としてLLMのコスト最適化における重要技術であり、MVR-cacheのアプローチは単一ベクトルの限界を超える有望な方向性を示している。

## 参考文献

- **arXiv**: [https://arxiv.org/abs/2605.24914](https://arxiv.org/abs/2605.24914)
- **vCache**: [https://arxiv.org/abs/2502.03771](https://arxiv.org/abs/2502.03771)
- **ColBERTv2**: [https://arxiv.org/abs/2112.01488](https://arxiv.org/abs/2112.01488)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/1707bd6149514c](https://zenn.dev/0h_n0/articles/1707bd6149514c)
