---
layout: post
title: "COLING 2025論文解説: MAC-SQL — マルチエージェント協調によるText-to-SQLフレームワーク"
description: "MAC-SQLの3エージェント構成（Selector/Decomposer/Refiner）を解説。大規模DBスキーマ対応とSQL自己修正の実装パターンを詳述"
categories: [blog, paper, conference]
tags: [Text-to-SQL, multi-agent, LLM, BIRD, Spider, sql, langgraph]
date: 2026-02-23 13:00:00 +0900
source_type: conference
conference: "COLING 2025"
source_url: https://aclanthology.org/2025.coling-main.36/
zenn_article: 58dc3076d2ffba
zenn_url: https://zenn.dev/0h_n0/articles/58dc3076d2ffba
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [COLING 2025 "MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL"](https://aclanthology.org/2025.coling-main.36/) の解説記事です。

## 論文概要（Abstract）

Text-to-SQLタスクにおいて、大規模データベーススキーマ（数十〜数百テーブル）への対応と複雑なクエリの正確なSQL生成は依然として困難な課題である。MAC-SQL（Multi-Agent Collaborative SQL）は、この課題をSelector、Decomposer、Refinerの3つの専門エージェントの協調で解決するフレームワークを提案している。各エージェントが責務を明確に分担することで、モノリシックなSQL生成アプローチと比較して精度と解釈性の両方を向上させている。

この記事は [Zenn記事: LangGraph×Claude Sonnet 4.6でSQL統合Agentic RAGを実装する](https://zenn.dev/0h_n0/articles/58dc3076d2ffba) の深掘りです。

## 情報源

- **会議名**: COLING 2025（International Conference on Computational Linguistics）
- **年**: 2025
- **URL**: [https://aclanthology.org/2025.coling-main.36/](https://aclanthology.org/2025.coling-main.36/)
- **著者**: Bing Wang, Changyou Ren, Jian Yang, Xinnian Liang, Jiaqi Bai, LinZheng Chai, Zhao Yan, Qian-Wen Zhang et al.

## カンファレンス情報

**COLINGについて**:
COLINGは計算言語学分野で最も歴史のある国際会議の1つであり、自然言語処理（NLP）の幅広いテーマを扱う。2025年大会でText-to-SQL関連の論文が採択されたことは、NLコミュニティにおけるSQL生成タスクの重要性を反映している。

## 技術的詳細（Technical Details）

### 3エージェント協調アーキテクチャ

MAC-SQLは、SQL生成プロセスを3つの独立したエージェントに分割する:

```
自然言語クエリ + 大規模DBスキーマ
            ↓
┌─────────────────────────────────┐
│  Agent 1: Selector               │
│  → 大規模スキーマから関連テーブル・ │
│    カラムを選択（スキーマプルーニング）│
└─────────────┬───────────────────┘
              ↓ 縮小されたスキーマ
┌─────────────────────────────────┐
│  Agent 2: Decomposer             │
│  → 複雑なクエリをサブクエリに分解  │
│  → 各サブクエリのSQL生成          │
│  → 最終SQLの結合                 │
└─────────────┬───────────────────┘
              ↓ 候補SQL
┌─────────────────────────────────┐
│  Agent 3: Refiner                │
│  → SQL実行・エラー検出            │
│  → エラー時は修正SQL生成          │
│  → 結果の妥当性チェック           │
└─────────────┬───────────────────┘
              ↓
          最終SQL + 実行結果
```

### Agent 1: Selector（スキーマプルーニング）

大規模データベース（100+テーブル）から、クエリに関連するテーブルとカラムのみを選択する。

$$
S' = \text{Select}(q, S) \quad \text{where} \quad |S'| \ll |S|
$$

ここで、
- $S$: 元のフルスキーマ（テーブル数: $|S|$）
- $S'$: プルーニング後のサブスキーマ
- $q$: 自然言語クエリ

**Selectorの重要性**: 著者らは、LLMのコンテキストウィンドウに入りきらない大規模スキーマをそのまま渡すと、(1) テーブル名の取り違え、(2) 不要なJOINの生成、(3) トークンコストの増大が発生すると述べている。Selectorによるスキーマプルーニングは、これらの問題を軽減する。

**Zenn記事との対応**: Zenn記事の`SQLDatabase`で設定する`include_tables`パラメータは、手動でのスキーマプルーニングに相当する。MAC-SQLのSelectorはこれを**LLMベースで自動化**する点が異なる。

```python
from typing import TypedDict

class SelectorState(TypedDict):
    """Selectorの状態"""
    query: str
    full_schema: dict
    selected_tables: list[str]
    selected_columns: dict[str, list[str]]

async def selector_agent(
    state: SelectorState,
    llm_client: object,
) -> SelectorState:
    """スキーマプルーニングを行うSelectorエージェント

    Args:
        state: 現在の状態（クエリとフルスキーマ）
        llm_client: LLMクライアント

    Returns:
        プルーニング後のスキーマ情報を含む更新された状態
    """
    prompt = f"""以下のデータベーススキーマから、
ユーザークエリに回答するために必要な
テーブルとカラムのみを選択してください。

クエリ: {state['query']}

スキーマ:
{format_schema(state['full_schema'])}

出力形式:
- テーブル名: [必要なカラム名のリスト]
"""

    response = await llm_client.generate(prompt)
    selected = parse_selection(response)

    return {
        **state,
        "selected_tables": selected["tables"],
        "selected_columns": selected["columns"],
    }
```

### Agent 2: Decomposer（クエリ分解）

複雑な自然言語クエリをより単純なサブクエリに分解し、各サブクエリのSQLを生成してから最終SQLに結合する。

$$
q \xrightarrow{\text{Decompose}} \{q_1, q_2, \ldots, q_k\} \xrightarrow{\text{Generate}} \{\text{SQL}_1, \ldots, \text{SQL}_k\} \xrightarrow{\text{Compose}} \text{SQL}_{\text{final}}
$$

著者らによれば、このアプローチは以下の利点がある:
1. **各サブクエリが単純になる**: 単一テーブル操作やシンプルなJOINに分解されるため、LLMの生成精度が向上
2. **デバッグ容易性**: どのサブクエリの変換で失敗したかを特定しやすい
3. **再利用性**: 同じサブクエリパターンを異なる複合クエリで再利用可能

### Agent 3: Refiner（SQL検証・修正）

生成されたSQLを実際にデータベースで実行し、エラーが発生した場合は修正SQLを生成する自己修正ループを担当する。

$$
\text{SQL}_{t+1} = \text{Refine}(\text{SQL}_t, \text{Error}_t, S')
$$

ここで、
- $\text{SQL}_t$: 時刻$t$のSQL候補
- $\text{Error}_t$: 実行エラーメッセージ
- $S'$: サブスキーマ

著者らは、Refinerの自己修正ループを最大3回に制限している。これは、AWSブログで推奨されているMAX_RETRIES=3と同じ設計判断であり、無限ループの防止とコスト制御のバランスを取った値である。

```python
async def refiner_agent(
    sql: str,
    schema: dict,
    db_connection: object,
    max_retries: int = 3,
) -> tuple[str, str | None]:
    """SQL検証・修正エージェント

    Args:
        sql: 候補SQL
        schema: サブスキーマ
        db_connection: DB接続
        max_retries: 最大リトライ回数

    Returns:
        (最終SQL, 実行結果 or None)
    """
    for attempt in range(max_retries):
        try:
            result = await db_connection.execute(sql)

            # 結果の妥当性チェック
            if is_empty_result(result):
                # 空結果の場合、WHERE句の緩和を試行
                sql = await relax_where_clause(sql, schema)
                continue

            return sql, result
        except SyntaxError as e:
            sql = await fix_syntax_error(sql, str(e), schema)
        except ColumnNotFoundError as e:
            sql = await fix_column_reference(sql, str(e), schema)
        except Exception as e:
            sql = await regenerate_with_error(sql, str(e), schema)

    return sql, None  # max_retries到達
```

### LangGraph StateGraphとの対応

MAC-SQLの3エージェント構成は、LangGraphのStateGraphで自然に実装できる。Zenn記事のアーキテクチャとの対応関係は以下の通り:

| MAC-SQL | Zenn記事のSQL検索ノード | LangGraph実装 |
|---------|---------------------|--------------|
| Selector | `include_tables`設定 | 新規ノード追加 |
| Decomposer | `sql_db_schema` + SQL生成 | 既存ノードの拡張 |
| Refiner | `sql_db_query_checker` | 既存ノードの拡張 |

```python
from langgraph.graph import StateGraph, END

# MAC-SQL風のStateGraph構築
graph = StateGraph(MACSSQLState)

graph.add_node("selector", selector_agent)
graph.add_node("decomposer", decomposer_agent)
graph.add_node("refiner", refiner_agent)
graph.add_node("generate_answer", generate_answer_node)

graph.set_entry_point("selector")
graph.add_edge("selector", "decomposer")
graph.add_edge("decomposer", "refiner")
graph.add_edge("refiner", "generate_answer")
graph.add_edge("generate_answer", END)

app = graph.compile()
```

## 実験結果（Results）

著者らが報告したBIRD・Spiderベンチマークでの結果（論文の実験セクションより）:

| ベンチマーク | 単一LLM (GPT-4) | MAC-SQL (GPT-4) | 改善幅 |
|------------|----------------|-----------------|-------|
| BIRD dev | 著者らの報告によりベースライン | 改善達成 | スキーマプルーニングと自己修正による精度向上 |
| Spider dev | ベースライン | 改善達成 | クエリ分解による複雑クエリ対応力の向上 |

**各エージェントの貢献分析**（著者らのアブレーション実験より）:

| 構成 | 効果 |
|------|------|
| Selectorなし（フルスキーマ） | 大規模DB（50+テーブル）で精度が顕著に低下 |
| Decomposerなし（直接生成） | 複雑なJOIN/サブクエリで精度低下 |
| Refinerなし（検証なし） | 構文エラー・実行時エラーが増加 |
| 3エージェント全体 | 各コンポーネントの相乗効果で最高精度 |

著者らは、特にSelector（スキーマプルーニング）の効果が大きく、50テーブル以上のデータベースではSelectorの有無で精度が大きく変化すると報告している。

**制約と限界**:
- 3エージェント分のLLM APIコールが必要（単一パスの約3倍のコスト）
- エージェント間の通信オーバーヘッドによるレイテンシ増加
- Selectorのスキーマプルーニング精度がパイプライン全体のボトルネックになりうる

## 実装のポイント（Implementation）

### 段階的な導入戦略

MAC-SQLの3エージェントを一度に導入するのではなく、効果の大きいコンポーネントから段階的に導入することが推奨される:

1. **Phase 1: Refiner（自己修正）の導入** — 最小の変更で精度向上が見込める。Zenn記事の`sql_db_query_checker`をRefinerに拡張し、実行エラー時の再生成ループを追加する

2. **Phase 2: Selector（スキーマプルーニング）の導入** — テーブル数が10を超えるDBで有効。LLMによるテーブル・カラム選択ノードを追加する

3. **Phase 3: Decomposer（クエリ分解）の導入** — 複雑な複合クエリ（3テーブル以上のJOIN等）が頻繁に発生する場合に有効

### コスト最適化

著者らの手法を実運用に適用する際のコスト最適化:

- **クエリ複雑度に基づく動的切替**: 単純なクエリ（単一テーブル）は直接SQL生成、複雑なクエリ（マルチテーブルJOIN）のみMAC-SQLパイプラインを適用
- **Selectorのキャッシュ**: 同一DBスキーマに対するSelector結果をキャッシュし、同様のクエリパターンでの再計算を回避
- **軽量LLMの活用**: SelectorとRefinerには軽量モデル（Haiku等）を使用し、Decomposerのみ高性能モデル（Sonnet等）を使用

## 実運用への応用（Practical Applications）

MAC-SQLのマルチエージェント構成は、Zenn記事のSQL統合Agentic RAGにおけるSQL検索品質を向上させる具体的な設計パターンを提供している:

1. **大規模企業DB対応**: 100+テーブルの企業データベースでは、Selectorによるスキーマプルーニングが必須となる
2. **複合クエリ対応**: 「先月の各部署の売上合計と前月比」のような複合クエリは、Decomposerによる分解が有効
3. **運用信頼性**: Refinerの自己修正ループにより、SQL実行エラーの自動回復が可能になる

## まとめ

MAC-SQLは、Text-to-SQLをSelector（スキーマプルーニング）・Decomposer（クエリ分解）・Refiner（SQL検証・修正）の3エージェントに分割することで、精度と解釈性を同時に向上させるフレームワークである。各エージェントはLangGraphのノードとして自然に実装でき、Zenn記事のSQL統合Agentic RAGの拡張として段階的に導入可能である。

## 参考文献

- **Conference URL**: [https://aclanthology.org/2025.coling-main.36/](https://aclanthology.org/2025.coling-main.36/)
- **BIRD Benchmark**: [https://bird-bench.github.io/](https://bird-bench.github.io/)
- **Spider Benchmark**: [https://yale-lily.github.io/spider](https://yale-lily.github.io/spider)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/58dc3076d2ffba](https://zenn.dev/0h_n0/articles/58dc3076d2ffba)

---

:::message
本記事はAI（Claude Code）により自動生成された、COLING 2025採択論文の解説記事です。論文の内容を正確に伝えることを目指していますが、解釈に誤りがある可能性があります。必ず原論文をご確認ください。
:::
