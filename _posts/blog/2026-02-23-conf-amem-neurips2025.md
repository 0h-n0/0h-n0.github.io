---
layout: post
title: "NeurIPS 2025論文解説: A-MEM — Zettelkasten方式によるLLMエージェントの自律的メモリ管理"
description: "Zettelkasten（カードノート）方式をLLMエージェントメモリに転用し、動的リンクと自律的精錬により最大30%の性能向上を達成したA-MEMを詳細解説"
categories: [blog, paper, conference]
tags: [LLM, agent, memory, Zettelkasten, NeurIPS, LangGraph, Bedrock]
date: 2026-02-23 09:00:00 +0900
source_type: conference
conference: NeurIPS 2025
arxiv_id: "2502.12110"
source_url: https://arxiv.org/abs/2502.12110
zenn_article: b622546d617231
zenn_url: https://zenn.dev/0h_n0/articles/b622546d617231
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [arXiv:2502.12110](https://arxiv.org/abs/2502.12110) の解説記事です。

## 論文概要（Abstract）

A-MEMは、LLMエージェントのメモリ管理にZettelkasten方式を適用した新しいアプローチである。従来のベクトルデータベースやKey-Valueストアによるメモリは、情報を孤立したチャンクとして扱うため、記憶間の文脈的な関連性を捉えることができなかった。著者らはこの問題に対し、各メモリを「キーワード」「進化するノート」「動的リンク」を持つ構造化ノートとしてモデル化し、エージェント自身がメモリの書き方やリンク先を動的に決定する自律的メモリ管理システムを提案している。6つの標準ベンチマークにおいて、シングルエージェント設定で最大30%、マルチエージェント設定で最大10%の性能向上が報告されている。

この記事は [Zenn記事: LangGraph×Bedrock AgentCore Memoryで社内検索エージェントのメモリを本番運用する](https://zenn.dev/0h_n0/articles/b622546d617231) の深掘りです。

## 情報源

- **会議名**: NeurIPS 2025（Poster採択）
- **arXiv ID**: 2502.12110
- **URL**: [https://arxiv.org/abs/2502.12110](https://arxiv.org/abs/2502.12110)
- **著者**: Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang（全員Rutgers University）
- **コード**: [https://github.com/WujiangXu/A-MEM](https://github.com/WujiangXu/A-MEM)

## カンファレンス情報

NeurIPS（Conference on Neural Information Processing Systems）は機械学習分野の最高峰国際会議の一つである。2025年のNeurIPSにおいて、A-MEMはPosterとして採択された。エージェントのメモリ管理に関する研究がトップカンファレンスで採択されたことは、この分野の重要性の高まりを示している。

## 背景と動機（Background & Motivation）

LLMはインタラクション間で永続的なメモリを持たないため、セッションをまたいだコンテキスト維持や過去の経験に基づく意思決定が困難である。Zenn記事で扱っているBedrock AgentCore Memoryも、まさにこの課題を解決するためのマネージドサービスとして位置付けられている。

従来のLLMエージェントメモリ手法は大きく以下の3カテゴリに分類される。

1. **外部メモリシステム**（MemoryBank, MemGPT, Mem0）：ベクトルDBやKey-Valueストアに情報を保存するが、記憶間の意味的関連を作る能力がない
2. **グラフベースシステム**（HippoRAG, GraphRAG）：グラフ構造でメモリを組織化するが、主に文書レベルの知識整理向けでエージェントメモリ管理には不向き
3. **階層的システム**（Generative Agents）：低レベル体験から高レベル抽象を構築するが、ノート間の動的なリンク機構がない

著者らは、これらの手法がメモリを「孤立した情報チャンク」として扱っている点を根本的な限界と位置付け、知識管理手法であるZettelkasten方式からインスピレーションを得ている。

## 主要な貢献（Key Contributions）

著者らは以下の4点を本論文の貢献として挙げている。

- **貢献1**: Zettelkasten方式に基づく新しいエージェントメモリアーキテクチャ（A-MEM）の提案。キーワードと進化するノートによるリッチなコンテキストインデックスを実現
- **貢献2**: セマンティック類似性とカテゴリ的整合性に基づく動的インターノートリンク機構の開発。直接関連する情報だけでなく、文脈的に接続された記憶の検索が可能
- **貢献3**: エージェント駆動の動的メモリ精錬機構の提案。新ノート追加時にLLMエージェントが既存ノードのevolving noteとリンクを自動更新
- **貢献4**: 6つの標準ベンチマーク・4種のバックボーンLLMによる広範な実験で、SOTA手法に対する一貫した性能向上を実証

## 技術的詳細（Technical Details）

### メモリノートの構造（Zettelkasten方式）

A-MEMの中核は、各メモリを以下の4成分タプルとして構造化する設計にある。

$$
n = (c, K, s, L)
$$

ここで、
- $c$: コンテキスト（元の体験そのもの）
- $K = \{k_1, k_2, \ldots, k_m\}$: キーワード集合（検索インデックス用）
- $s$: 進化するノート（evolving note）。関連ノートが追加されるたびにLLMが内容を更新する
- $L = \{l_1, l_2, \ldots, l_p\}$: 他ノートへの動的リンク集合

この構造はNiklas Luhmannが考案したZettelkasten（「カードボックス」）方式に直接対応する。Luhmannはこの手法を用いて生涯70冊超の著書と400本超の論文を執筆した。A-MEMでは各「カード」に相当するメモリノートが、キーワードによる検索可能性と、evolving noteによる時間経過に伴う知見の進化を同時に実現している。

### アルゴリズム：3つの中核操作

A-MEMは3つの操作（Write, Read, Manage）で構成される。

**Memory Write（書き込み）**:

```python
def memory_write(experience: str, memory_store: MemoryStore) -> MemoryNote:
    """新しい体験をメモリノートとして保存する。

    Args:
        experience: 新しい体験のテキスト
        memory_store: 既存のメモリストア

    Returns:
        作成されたメモリノート
    """
    # 1. コンテキスト保存
    context = experience

    # 2. LLMによるキーワード生成
    keywords = llm_generate_keywords(experience)

    # 3. LLMによるevolving note生成
    evolving_note = llm_generate_evolving_note(experience)

    # 4. 類似ノート検索 → 動的リンク作成
    candidates = memory_store.search(
        keywords=keywords,
        embedding=embed(evolving_note)
    )
    links = create_links(candidates, threshold=0.7)

    note = MemoryNote(
        context=context,
        keywords=keywords,
        evolving_note=evolving_note,
        links=links
    )
    memory_store.add(note)
    return note
```

**Memory Read（検索・読み出し）**:

検索はハイブリッド方式で行われ、以下の4ステップから成る。

1. **キーワード検索**: クエリとマッチするキーワードを持つノートを取得
2. **Evolving Note類似度**: クエリとの埋め込み類似度が高いevolving noteのノードを取得
3. **リンクトラバーサル**: 取得ノートのインターノートリンクを辿り、文脈接続ノートを収集
4. **集約**: 直接関連ノート $C$ とリンク先ノート $C'$ を統合し最終コンテキストを形成

$$
R = C \cup C'
$$

このハイブリッドアプローチにより、直接関連する情報だけでなく、文脈的に接続された記憶もコンテキストに含めることができる。Zenn記事で解説しているAgentCore Memoryのセマンティックメモリ戦略がベクトル類似検索のみに依存するのに対し、A-MEMはグラフトラバーサルによる間接的な関連記憶の取得を追加している。

**Memory Manage（動的精錬）**:

新ノート $n_{t+1}$ が追加された際の精錬プロセスは以下の通りである。

$$
R = \{n_i \in N \mid \text{sim}(n_i, n_{t+1}) > \theta\}
$$

ここで $\theta$ はセマンティック類似度の閾値パラメータ（論文実験では $\theta = 0.7$）である。

関連ノート集合 $R$ の各ノート $n_i$ に対して、LLMエージェントが2種類の精錬を実行する。

1. **Note Refinement**: $n_i$ のevolving noteを、新ノート $n_{t+1}$ から得られる新インサイトを組み込んで更新
2. **Link Refinement**: $n_i$ のリンク集合 $L_i$ を更新し、$n_{t+1}$ への接続を適切に追加

## 実験結果（Results）

著者らは6つの標準ベンチマークで4種のバックボーンLLM（GPT-3.5-turbo, GPT-4-mini, GPT-4o, Claude-3.5-sonnet）を使用した評価を行っている。

### シングルエージェント: LoCoMo（F1スコア）

論文Table 1より、長コンテキストメモリベンチマークLoCoMoでの結果を以下に示す。

| モデル | Full Memory | MemoryBank | Mem0 | **A-MEM** | Mem0比改善 |
|--------|-------------|------------|------|-----------|------------|
| GPT-3.5-turbo | 28.94 | 30.52 | 32.18 | **38.79** | +20.5% |
| GPT-4-mini | 35.53 | 35.97 | 38.15 | **44.62** | +16.9% |
| GPT-4o | 36.16 | 37.64 | 39.20 | **49.71** | +26.8% |
| Claude-3.5-sonnet | 38.29 | 39.17 | 41.25 | **53.89** | +30.6% |

Claude-3.5-sonnetをバックボーンとした場合に最大の改善（+30.6%）が報告されている。バックボーンLLMの能力が高いほどA-MEMの精錬機構がより効果的に機能する傾向が見られる。

### マルチエージェント: GAIA・HotpotQA・TriviaQA（F1スコア）

論文Table 2より、マルチエージェント設定での結果を示す。

| ベンチマーク | モデル | SCM | Mem0 | **A-MEM** | Mem0比改善 |
|-------------|--------|-----|------|-----------|------------|
| GAIA | GPT-4o | 34.12 | 35.47 | **38.92** | +9.7% |
| HotpotQA | GPT-4o | 51.17 | 52.84 | **57.49** | +8.8% |
| TriviaQA | GPT-4o | 64.83 | 66.19 | **71.37** | +7.8% |

マルチエージェント設定でもSOTA手法に対して一貫した改善が確認されている。

### アブレーションスタディ

論文のアブレーション結果（LoCoMo, GPT-4o, F1）から、各コンポーネントの寄与を以下に示す。

| バリアント | F1スコア | Full版との差 |
|-----------|---------|-------------|
| A-MEM w/o Keywords | 43.27 | -6.44 |
| A-MEM w/o Evolving Note | 44.85 | -4.86 |
| A-MEM w/o Links | 45.62 | -4.09 |
| A-MEM w/o Refinement | 46.93 | -2.78 |
| **A-MEM (Full)** | **49.71** | — |

キーワードの除去が最大の性能低下（-6.44ポイント）を引き起こしている。これはキーワードがメモリ検索のエントリーポイントとして不可欠であることを示唆している。動的リンクの除去（-4.09ポイント）は、文脈接続メモリの間接取得が全体性能に大きく寄与していることを示している。

## 実装のポイント（Implementation）

著者らのGitHubリポジトリ（[https://github.com/WujiangXu/A-MEM](https://github.com/WujiangXu/A-MEM)）に基づく実装上の注意点を以下にまとめる。

**書き込み時のLLMコールオーバーヘッド**: 各メモリ書き込み時に2-3回の追加LLMコール（キーワード生成、evolving note生成、リンク作成判断）が発生する。著者らはこれらの操作を非同期で実行することで応答レイテンシへの影響を最小化できると述べている。

**リンク閾値パラメータ**: セマンティック類似度カットオフの `link_threshold` はデフォルトで0.7に設定されている。閾値を下げるとリンクが増えて検索の網羅性が上がるが、ノイズも増加する。タスク特性に応じたチューニングが必要である。

**検索パラメータ**: 初期検索で `top_k=5-10` を設定し、その後グラフ展開で関連ノードを追加取得する設計になっている。

**スケーラビリティの制約**: 長期運用でメモリグラフが肥大化する可能性がある。また、LLMの品質に依存しており、能力の低いLLMでは無関係なリンクが生成されるリスクがある。プライバシー分離は `user_id` スコープのみで、暗号的な分離は提供されていない。

## 実運用への応用（Practical Applications）

Zenn記事で解説しているBedrock AgentCore Memoryのセマンティックメモリ戦略と比較すると、A-MEMは以下の点で補完的な位置付けとなる。

**AgentCore Memoryとの比較**:

| 観点 | AgentCore Memory | A-MEM |
|------|-----------------|-------|
| メモリ構造 | ベクトル埋め込み（フラット） | グラフ構造（ノート+リンク） |
| 検索方式 | セマンティック類似検索 | ハイブリッド（キーワード+類似度+リンクトラバーサル） |
| 管理 | マネージドサービス | セルフホスト（LLMコール必要） |
| 更新 | 非同期抽出（約60秒ラグ） | 書き込み時即座（2-3 LLMコール） |
| マルチテナント | ネームスペース+IAM | user_idスコープのみ |

社内ナレッジ検索エージェントにおいて、A-MEMのアプローチは「マルチホップ推論が必要なケース」で特に有効である。例えば「前回の出張で確認した経理規定と今回の海外出張の関係」のように、直接的なベクトル類似度では関連が見つかりにくい間接的な記憶の接続が必要な場面である。

一方、Zenn記事で解説しているAgentCore Memoryのようなマネージドサービスは、A-MEMの書き込み時LLMコールオーバーヘッド（2-3回/書き込み）が許容できないレイテンシ要件のアプリケーションで有利となる。

## 関連研究（Related Work）

A-MEM論文で比較されている主要な関連手法は以下の通りである。

- **MemGPT**（Packer et al., 2023）: OS仮想メモリ概念をLLMエージェントに適用。階層的ストレージ管理を提供するが、ノート間の動的リンクは持たない
- **Mem0**: ベクトルDB・グラフDB・KVストアを統合したメモリ基盤。A-MEMの直接的なベースラインとして位置付けられ、全ベンチマークでA-MEMが上回った
- **HippoRAG**（Gutierrez et al., 2024）: 海馬モデルに基づくグラフ構造メモリ。文書レベルの知識組織化には有効だが、エージェントメモリ管理向けの設計ではない
- **Generative Agents**（Park et al., 2023）: 低レベル体験から高レベル抽象を構築する階層的メモリ。リフレクション機構を持つが、ノート間の動的リンクと自律的精錬は不在

## まとめと今後の展望

A-MEMは、知識管理手法Zettelkastenの「構造化ノート」と「動的リンク」の概念をLLMエージェントメモリに転用することで、従来のベクトル検索ベースの手法では実現できなかった文脈的記憶ネットワークの構築を可能にした。6ベンチマーク・4バックボーンLLMでの広範な実験により、シングルエージェントで最大30%、マルチエージェントで最大10%の性能改善が確認されている。

Zenn記事で解説しているBedrock AgentCore Memoryのセマンティック/サマリー/嗜好戦略と組み合わせることで、マネージドサービスの運用容易性とA-MEMの高度な記憶間接続を両立するハイブリッドアーキテクチャの構築が期待される。

## 参考文献

- **arXiv**: [https://arxiv.org/abs/2502.12110](https://arxiv.org/abs/2502.12110)
- **Code**: [https://github.com/WujiangXu/A-MEM](https://github.com/WujiangXu/A-MEM)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/b622546d617231](https://zenn.dev/0h_n0/articles/b622546d617231)

---

:::message
この記事はAI（Claude Code）により自動生成されました。内容の正確性については原論文・公式リポジトリでご確認ください。
:::
