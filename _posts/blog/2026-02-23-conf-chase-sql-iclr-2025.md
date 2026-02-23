---
layout: post
title: "ICLR 2025論文解説: CHASE-SQL — マルチパス推論と選好最適化によるText-to-SQL"
description: "CHASE-SQLの3つのSQL生成パス（分割統治・CoT・合成例）とLLM-as-Judge候補選択を解説。BIRDベンチ73.0%達成の手法を詳述"
categories: [blog, paper, conference]
tags: [Text-to-SQL, LLM, multi-agent, BIRD, Spider, langgraph]
date: 2026-02-23 10:00:00 +0900
source_type: conference
conference: "ICLR 2025"
source_url: https://arxiv.org/abs/2410.01943
zenn_article: 58dc3076d2ffba
zenn_url: https://zenn.dev/0h_n0/articles/58dc3076d2ffba
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [arXiv:2410.01943 "CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL"](https://arxiv.org/abs/2410.01943)（ICLR 2025採択）の解説記事です。

## 論文概要（Abstract）

Text-to-SQLタスクにおいて、LLMの性能を引き出すための新しいフレームワークCHASE-SQL（Chain-of-thought And Selection for Enhanced SQL）を提案する。本手法はテスト時の計算量（test-time compute）をマルチエージェントモデリングに活用し、(1) 分割統治によるクエリ分解、(2) クエリ実行計画に基づくChain-of-Thought推論、(3) インスタンス固有の合成例生成の3つのSQL生成パスから候補を生成し、選好最適化されたバイナリ選択LLMで最終候補を選択する。

この記事は [Zenn記事: LangGraph×Claude Sonnet 4.6でSQL統合Agentic RAGを実装する](https://zenn.dev/0h_n0/articles/58dc3076d2ffba) の深掘りです。

## 情報源

- **会議名**: ICLR 2025（International Conference on Learning Representations）
- **年**: 2025
- **URL**: [https://arxiv.org/abs/2410.01943](https://arxiv.org/abs/2410.01943)
- **著者**: Mohammadreza Pourreza, Hailong Li, Ruoxi Sun, Yeounoh Chung, Shayan Talaei, Gaurav Tarlok Kakkar, Yu Gan, Amin Saberi, Fatma Ozcan, Sercan O. Arik
- **発表形式**: Conference Paper

## カンファレンス情報

**ICLRについて**:
ICLRは機械学習・深層学習分野における最高峰の国際会議の1つであり、表現学習に焦点を当てた研究が集まる。CHASE-SQLがICLR 2025に採択されたことは、Text-to-SQLタスクにおけるマルチパス推論アプローチの学術的重要性を示している。

## 技術的詳細（Technical Details）

### 全体アーキテクチャ

CHASE-SQLは3つのSQL生成パスを並列に実行し、最終的にLLM-as-Judgeで最適な候補を選択するパイプラインである。

```
ユーザークエリ + DBスキーマ
        ↓
┌───────────────────────────────────────────┐
│  Path 1: 分割統治（Divide-and-Conquer）     │
│  Path 2: CoT推論（Chain-of-Thought）        │
│  Path 3: 合成例生成（Instance-aware Few-shot）│
└───────────────────────────────────────────┘
        ↓ 候補SQL群
  LLM-as-Judge（選好最適化済み）
        ↓
     最終SQL
```

### Path 1: 分割統治（Divide-and-Conquer）

複雑なクエリをサブクエリに分解し、各サブクエリのSQLを生成してから結合する。

$$
\text{SQL}_{\text{final}} = \text{Compose}\left(\{\text{SQL}_i\}_{i=1}^{k}\right)
$$

ここで、
- $k$: 分解されたサブクエリの数
- $\text{SQL}_i$: 各サブクエリに対応するSQL
- $\text{Compose}$: サブクエリSQLの結合関数（UNION, JOIN, サブクエリのネスト等）

著者らは、1回のLLM呼び出しでクエリ分解とサブクエリSQL生成を同時に行うことで、APIコールを最小化していると述べている。

### Path 2: Chain-of-Thought推論

データベースエンジンのクエリ実行計画（Execution Plan）をヒントとして活用する。

$$
P(\text{SQL} \mid q, S, E) > P(\text{SQL} \mid q, S)
$$

ここで、
- $q$: 自然言語クエリ
- $S$: DBスキーマ
- $E$: クエリ実行計画のヒント

著者らによれば、実行計画をCoTのガイドとして使用することで、JOINの順序やインデックスの活用などデータベース固有の最適化知識をLLMに注入できる。

### Path 3: インスタンス固有の合成例生成

テスト時に入力クエリに類似した合成例（Synthetic Few-shot Examples）を動的に生成する。

$$
\mathcal{E}_{\text{syn}} = \text{Generate}(q, S, \mathcal{E}_{\text{train}})
$$

ここで、
- $\mathcal{E}_{\text{syn}}$: 合成されたfew-shot例の集合
- $\mathcal{E}_{\text{train}}$: 訓練データから抽出した例のプール

従来のfew-shotプロンプティングでは固定例を使用するが、CHASE-SQLでは入力クエリの特徴（使用テーブル、JOIN構造、集約関数の有無等）に基づいて類似例を動的に合成する。

### LLM-as-Judge候補選択

3つのパスから生成された候補SQLを、選好最適化（Preference Optimization）でファインチューニングされたバイナリ選択LLMが評価する。

$$
\text{SQL}^* = \arg\max_{c \in \mathcal{C}} \sum_{c' \in \mathcal{C} \setminus \{c\}} \mathbb{1}[\text{Judge}(c, c') = c]
$$

ここで、
- $\mathcal{C}$: 候補SQLの集合
- $\text{Judge}(c, c')$: 2つの候補を比較し優れた方を返す関数
- $\mathbb{1}[\cdot]$: 指示関数

ペアワイズ比較を繰り返し、最多勝利の候補を最終SQLとして選択する。著者らは、この方式が単純なスコアリングよりも頑健であると報告している。

### アルゴリズム: マルチパスSQL生成

```python
from dataclasses import dataclass

@dataclass
class SQLCandidate:
    """SQL候補"""
    sql: str
    path: str  # "divide_conquer" | "cot" | "few_shot"
    confidence: float

async def chase_sql_pipeline(
    query: str,
    schema: str,
    execution_plan_hint: str | None = None,
) -> str:
    """CHASE-SQLパイプライン

    Args:
        query: 自然言語クエリ
        schema: DBスキーマ情報
        execution_plan_hint: クエリ実行計画のヒント

    Returns:
        最終SQL文字列
    """
    # Step 1: 3パス並列でSQL候補生成
    candidates: list[SQLCandidate] = []

    # Path 1: 分割統治
    dc_sql = await divide_and_conquer_generate(query, schema)
    candidates.append(SQLCandidate(sql=dc_sql, path="divide_conquer", confidence=0.0))

    # Path 2: CoT推論（実行計画ヒント付き）
    cot_sql = await cot_with_execution_plan(query, schema, execution_plan_hint)
    candidates.append(SQLCandidate(sql=cot_sql, path="cot", confidence=0.0))

    # Path 3: 合成例ベースfew-shot
    fs_sql = await instance_aware_few_shot(query, schema)
    candidates.append(SQLCandidate(sql=fs_sql, path="few_shot", confidence=0.0))

    # Step 2: LLM-as-Judge ペアワイズ比較
    best_sql = await pairwise_judge_selection(candidates)

    return best_sql
```

## 実験結果（Results）

著者らが報告したBIRD・Spiderベンチマークでの実行精度（Execution Accuracy）は以下の通りである（論文Table 1, Table 2より）:

| ベンチマーク | データセット | 実行精度 | 備考 |
|------------|-----------|---------|------|
| BIRD dev | 12,751 QA pairs | 73.01% | 論文投稿時点でリーダーボード1位 |
| BIRD test | 12,751 QA pairs | 73.0% | 公式テストセット |
| Spider dev | 10,181 QA pairs | 87.6% | 複雑なSQL生成タスク |

**アブレーション実験**（論文Table 3より）:

| 構成 | BIRD dev精度 |
|------|-----------|
| Path 1のみ（分割統治） | 69.5% |
| Path 2のみ（CoT） | 70.2% |
| Path 3のみ（合成例） | 68.8% |
| 3パス + ランダム選択 | 71.4% |
| 3パス + LLM-as-Judge | 73.01% |

このアブレーション結果から、個々のパスの精度は68-70%程度だが、3パスの多様な候補生成とLLM-as-Judgeによる候補選択を組み合わせることで73%まで向上していることが読み取れる。著者らは、多様性のある候補生成がLLM-as-Judgeの性能を最大化する鍵であると述べている。

**制約と限界**:
- 3パス並列実行 + LLM-as-Judgeのペアワイズ比較により、単一パスの3-5倍のAPIコストが発生する
- スキーマが100テーブル以上の大規模DBでのスケーラビリティは未検証
- Gemini Pro / GPT-4oに最適化されており、小規模LLMへの移植は精度劣化の可能性がある

## 実装のポイント（Implementation）

### Zenn記事のSQL生成への応用

Zenn記事で実装した`SQLDatabaseToolkit`による単一パスSQL生成は、CHASE-SQLの観点からはPath 2（CoT推論）に最も近い。本番環境でSQL生成精度を向上させるには、以下のアプローチが考えられる:

1. **複数パスの導入**: Zenn記事のSQL検索ノードを3つのサブノードに分割し、異なるプロンプト戦略でSQL候補を生成する
2. **SQL検証の強化**: `sql_db_query_checker`に加え、実行結果の妥当性チェック（空結果の場合は再生成）を追加する
3. **スキーマリンキングの精度向上**: `include_tables`で対象テーブルを制限するだけでなく、クエリに関連するカラムも動的にフィルタリングする

### コストとレイテンシのトレードオフ

| アプローチ | APIコール数 | レイテンシ | SQL精度（推定） |
|-----------|-----------|----------|---------------|
| 単一パス（Zenn記事） | 4回 | ~800ms | 60-65% |
| 2パス + ランダム | 8回 | ~1.6s | 68-70% |
| 3パス + LLM-as-Judge | 12+回 | ~3s | 73%+ |

## 実運用への応用（Practical Applications）

CHASE-SQLのマルチパス戦略は、Agentic RAGのSQL検索ノード品質を向上させる直接的な手法である。ただし、コスト増加を考慮し、以下のような段階的導入が著者らの手法から示唆される:

1. **Phase 1**: 単一パス（現状のZenn記事構成）でベースラインを確立
2. **Phase 2**: クエリ複雑度に応じて1パスまたは3パスを動的に切り替える
3. **Phase 3**: ペアワイズ比較の代わりに軽量なヒューリスティック（SQL構文の複雑さ、実行結果の行数等）で候補選択

## まとめ

CHASE-SQLは、テスト時のマルチパス推論と選好最適化された候補選択を組み合わせることで、BIRDベンチマーク73.0%というText-to-SQLの高い実行精度を達成した手法である。Zenn記事のSQL統合Agentic RAGにおけるSQL生成品質向上の参考として、段階的なマルチパス導入を検討する価値がある。

## 参考文献

- **Conference URL (ICLR 2025)**: [https://openreview.net/forum?id=CvGqMD5OtX](https://openreview.net/forum?id=CvGqMD5OtX)
- **arXiv**: [https://arxiv.org/abs/2410.01943](https://arxiv.org/abs/2410.01943)
- **BIRD Benchmark**: [https://bird-bench.github.io/](https://bird-bench.github.io/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/58dc3076d2ffba](https://zenn.dev/0h_n0/articles/58dc3076d2ffba)

---

:::message
本記事はAI（Claude Code）により自動生成された、ICLR 2025採択論文の解説記事です。論文の内容を正確に伝えることを目指していますが、解釈に誤りがある可能性があります。必ず原論文をご確認ください。
:::
