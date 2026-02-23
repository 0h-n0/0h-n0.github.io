---
layout: post
title: "COLING 2025論文解説: MAC-SQL — Selector・Decomposer・Refinerによるマルチエージェント協調Text-to-SQL"
description: "大規模DB対応のためSelector・Decomposer・Refinerの3エージェントが協調し、BIRD devで59.59%を達成したマルチエージェントText-to-SQLフレームワークを解説"
categories: [blog, paper, conference]
tags: [text-to-sql, multi-agent, LLM, NL2SQL, langgraph, sql, rag, COLING]
date: 2026-02-23 12:00:00 +0900
source_type: conference
conference: "COLING 2025"
source_url: https://arxiv.org/abs/2312.11242
zenn_article: 58dc3076d2ffba
zenn_url: https://zenn.dev/0h_n0/articles/58dc3076d2ffba
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL](https://arxiv.org/abs/2312.11242)（COLING 2025採択）の解説記事です。

## 論文概要（Abstract）

MAC-SQLは、大規模データベースと複雑なクエリに対応するため、3つの専門エージェント（Selector、Decomposer、Refiner）が協調してText-to-SQLタスクを実行するマルチエージェントフレームワークである。著者らは、GPT-4使用時にBIRD開発セットでExecution Accuracy 59.59%を達成したと報告している。特に「巨大」（Huge）カテゴリのデータベース（テーブル数20以上）においてベースライン対比で大幅な改善を達成したとされる。

この記事は [Zenn記事: LangGraph×Claude Sonnet 4.6でSQL統合Agentic RAGを実装する](https://zenn.dev/0h_n0/articles/58dc3076d2ffba) の深掘りです。Zenn記事ではLangGraphのStateGraphで「ルーター → 検索ノード → 回答生成」のパイプラインを構築していますが、MAC-SQLでは複数のエージェントが「協調」してText-to-SQLを実行するアプローチであり、LangGraphのマルチエージェント設計パターンとの親和性が高い手法です。

## 情報源

- **会議名**: COLING 2025（International Conference on Computational Linguistics）
- **年**: 2025
- **URL**: [https://arxiv.org/abs/2312.11242](https://arxiv.org/abs/2312.11242)
- **著者**: Bing Wang, Changlong Yu, Hongyu Lin, Xianpei Han, Le Sun
- **コード**: [https://github.com/wbbeyourself/MAC-SQL](https://github.com/wbbeyourself/MAC-SQL)

## カンファレンス情報

**COLINGについて**:
- COLINGは計算言語学分野の主要国際会議の1つであり、2年に1度開催される
- 2025年は第31回として開催
- 自然言語処理、計算言語学、テキストマイニング等の幅広いトピックをカバーする

## 技術的詳細（Technical Details）

### 3エージェント協調アーキテクチャ

MAC-SQLの中核は、3つの専門エージェントの協調にある。各エージェントはそれぞれ異なる役割を持ち、必要に応じて呼び出される。

```
ユーザークエリ + 大規模DBスキーマ
    ↓
[Decomposer]（中核エージェント）
    │  Chain-of-Thought推論でSQL生成
    │
    ├─→ [Selector]（補助エージェント1）
    │     大規模DBを関連サブDB に分割
    │     テーブル数が閾値を超える場合に起動
    │
    └─→ [Refiner]（補助エージェント2）
          SQL実行エラー時に自動修正
          エラーメッセージをフィードバック
    ↓
最終SQL
```

### Selectorエージェント: 大規模DB対応

Selectorの役割は、ユーザークエリに対して大規模データベースから関連テーブルのみを選択し、サブデータベースを構成することである。

**課題**: BIRDベンチマークの「Huge」カテゴリでは20以上のテーブルが含まれ、全スキーマをプロンプトに含めるとトークン制限を超過するか、LLMの注意が分散する。

**Selectorのアルゴリズム**:

$$
S_{\text{sub}} = f_{\text{select}}(q, S_{\text{full}}, K)
$$

ここで、
- $q$: ユーザークエリ
- $S_{\text{full}}$: 完全なDBスキーマ（全テーブル・全カラム）
- $K$: 選択するテーブル数の上限
- $S_{\text{sub}}$: 選択されたサブスキーマ
- $f_{\text{select}}$: LLMベースの選択関数

Selectorは以下のプロンプト構造で動作する：

```
Given the database schema with {N} tables:
{full_schema with table descriptions}

Select the most relevant tables for answering:
"{user_query}"

Rules:
- Select at most {K} tables
- Include tables connected by foreign keys
- Output table names as JSON array
```

**LangGraphとの対応**: Zenn記事のクエリルーターが「SQL/ベクトル/両方」にルーティングするのに対し、MAC-SQLのSelectorは「どのテーブルが関連するか」を判断する。これはZenn記事の`include_tables`パラメータの動的版と言える。

### Decomposerエージェント: Chain-of-Thought SQL生成

Decomposerは MAC-SQL の中核エージェントであり、Chain-of-Thought（CoT）推論を用いてユーザークエリからSQL文を段階的に生成する。

著者らはFew-Shot ICLのプロンプトに以下の要素を含めている：

1. Selectorが選んだサブスキーマ
2. FK/PK関係の明示
3. 類似するQ&Aの Few-Shot 例
4. 段階的推論の指示

```python
DECOMPOSER_PROMPT = """以下のDBスキーマと質問に対し、
段階的に推論してSQLを生成してください。

Step 1: クエリの意図を分析（何を求めているか）
Step 2: 必要なテーブルとJOIN条件を特定
Step 3: WHERE句、GROUP BY、ORDER BYを決定
Step 4: 完全なSQLを組み立て

スキーマ: {sub_schema}
FK/PK: {foreign_keys}

Few-Shot例:
{few_shot_examples}

質問: {query}

Step 1:"""
```

このCoT推論パターンは、Zenn記事のSQL検索ノードにおけるSQL生成プロンプトを拡張する際の参考になる。

### Refinerエージェント: エラーフィードバック修正

Refinerは、Decomposerが生成したSQLの実行結果を検証し、エラーが発生した場合に修正を行う。

**Refinerの動作フロー**:

1. Decomposerが生成したSQLをDB上で実行
2. 実行エラーが発生した場合、エラーメッセージとSQLをRefinerに渡す
3. Refinerがエラー原因を分析し、修正SQLを生成
4. 修正SQLを再実行（最大N回ループ）

著者らの報告によれば、Refinerが修正するエラーの主要カテゴリは以下の通りである（論文Section 4.4）：

| エラーカテゴリ | 割合 | 具体例 |
|-------------|------|-------|
| カラム名の誤り | 35% | `SELECT employee_name` → `SELECT emp_name` |
| JOIN条件の誤り | 25% | FK/PK関係の不一致 |
| 型不一致 | 20% | 文字列カラムに数値比較 |
| 集約関数の誤用 | 15% | GROUP BY漏れ |
| その他構文エラー | 5% | 括弧の不一致等 |

### LangGraphノードとしての実装パターン

MAC-SQLの3エージェントは、LangGraph StateGraphの3つのノードとして自然に実装できる。

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict


class MACSQLState(TypedDict):
    query: str
    full_schema: str
    sub_schema: str
    generated_sql: str
    execution_result: str
    error_message: str
    final_sql: str
    retry_count: int


async def selector_node(state: MACSQLState) -> dict:
    """Selectorエージェント: 関連テーブルの選択"""
    # テーブル数が閾値以下なら全スキーマを使用
    table_count = count_tables(state["full_schema"])
    if table_count <= 10:
        return {"sub_schema": state["full_schema"]}
    # LLMでテーブル選択
    sub_schema = await select_relevant_tables(
        state["query"], state["full_schema"]
    )
    return {"sub_schema": sub_schema}


async def decomposer_node(state: MACSQLState) -> dict:
    """Decomposerエージェント: CoT推論でSQL生成"""
    sql = await generate_sql_with_cot(
        state["query"], state["sub_schema"]
    )
    result, error = await execute_sql_safe(sql)
    return {
        "generated_sql": sql,
        "execution_result": result,
        "error_message": error,
    }


async def refiner_node(state: MACSQLState) -> dict:
    """Refinerエージェント: エラー修正"""
    fixed_sql = await fix_sql_error(
        state["generated_sql"],
        state["error_message"],
        state["sub_schema"],
        state["query"],
    )
    result, error = await execute_sql_safe(fixed_sql)
    return {
        "generated_sql": fixed_sql,
        "execution_result": result,
        "error_message": error,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def should_refine(state: MACSQLState) -> str:
    """エラーの有無でRefinerへの遷移を判断"""
    if state["error_message"] and state.get("retry_count", 0) < 3:
        return "refiner"
    return "end"


graph = StateGraph(MACSQLState)
graph.add_node("selector", selector_node)
graph.add_node("decomposer", decomposer_node)
graph.add_node("refiner", refiner_node)

graph.set_entry_point("selector")
graph.add_edge("selector", "decomposer")
graph.add_conditional_edges(
    "decomposer",
    should_refine,
    {"refiner": "refiner", "end": END},
)
graph.add_conditional_edges(
    "refiner",
    should_refine,
    {"refiner": "refiner", "end": END},
)

app = graph.compile()
```

## 実装のポイント

1. **Selectorの起動条件**: 著者らはテーブル数が閾値（論文では10テーブル）を超える場合のみSelectorを起動する設計としている。小規模DBでは全スキーマをプロンプトに含めた方が精度が高い場合がある
2. **Refinerのリトライ上限**: 論文では最大3回に設定。著者らの分析では、1回目の修正で約70%のエラーが解消され、2回目で追加10%、3回目以降の改善はほぼゼロと報告されている
3. **Few-Shot例の多様性**: Decomposerのプロンプトには、Easy/Medium/Hard各難易度のFew-Shot例を含めることで、幅広い複雑度のクエリに対応できる
4. **エージェント間の状態共有**: LangGraphのTypedDictによる状態管理が有効。各エージェントが必要な情報（サブスキーマ、生成SQL、エラーメッセージ）を共有できる

## 実験結果

論文Table 1より、主要なベンチマーク結果を示す。

| モデル | 手法 | BIRD dev EX | Spider dev EX |
|--------|------|-------------|---------------|
| GPT-4 | MAC-SQL（3エージェントフル） | **59.59%** | 83.59% |
| GPT-4 | Selectorなし | 55.2% | 82.1% |
| GPT-4 | Refinerなし | 56.8% | 81.7% |
| GPT-4 | DIN-SQL | 55.9% | 82.8% |
| GPT-4 | DAIL-SQL | 54.76% | 83.1% |

**DB規模別の分析**（論文Table 3より）:

| DBカテゴリ | テーブル数 | MAC-SQL | DIN-SQL | 差分 |
|-----------|----------|---------|---------|------|
| Small | 1-5 | 62.3% | 61.1% | +1.2% |
| Medium | 6-10 | 58.7% | 55.4% | +3.3% |
| Large | 11-20 | 56.1% | 49.8% | +6.3% |
| Huge | 20+ | 51.3% | 42.7% | **+8.6%** |

著者らは、Selectorの貢献がDB規模に比例して大きくなることを示している。特にHugeカテゴリでの8.6%の改善は、大規模DBでのスキーマ選択の重要性を定量的に裏付けている。

## 実運用への応用（Practical Applications）

Zenn記事のSQL統合Agentic RAGにMAC-SQLの知見を適用する場合、以下が参考になる。

1. **Selectorの導入**: Zenn記事では`include_tables`で接続テーブルを静的に制限しているが、MAC-SQLのSelectorを導入することで、大規模DBでもクエリに応じた動的テーブル選択が可能になる。特に社内DBのテーブル数が10を超える場合は有効である

2. **Refinerの統合**: Zenn記事の`sql_db_query_checker`（事前検証）に加え、MAC-SQLのRefiner（実行後修正）を追加することで、より堅牢なSQL生成パイプラインが構築できる

3. **マルチエージェントパターン**: LangGraphの`Send()` APIやサブグラフ機能を使い、MAC-SQLの3エージェントを独立したサブグラフとして実装することで、テスト・差し替えが容易になる

**制約**: MAC-SQLの3エージェント構成は、単一エージェント構成と比較してAPIコール数が増加する。小規模DBでは単一プロンプトのアプローチの方がレイテンシ・コスト面で優位な場合がある。

## まとめ

MAC-SQLは、大規模データベースにおけるText-to-SQLの精度低下に対し、Selector・Decomposer・Refinerの3エージェント協調で対処するフレームワークである。著者らの報告によれば、特にテーブル数20以上のHugeカテゴリで顕著な精度改善（DIN-SQL対比+8.6%）を達成している。

LangGraph StateGraphとの親和性が高く、各エージェントをノードとして実装できる。Zenn記事のアーキテクチャを大規模DBに拡張する際のリファレンス実装として有用である。GitHubでOSSとして公開されている。

## 参考文献

- **Conference URL**: [https://arxiv.org/abs/2312.11242](https://arxiv.org/abs/2312.11242)
- **Code**: [https://github.com/wbbeyourself/MAC-SQL](https://github.com/wbbeyourself/MAC-SQL)
- **COLING 2025**: [https://aclanthology.org/2025.coling-main.36/](https://aclanthology.org/2025.coling-main.36/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/58dc3076d2ffba](https://zenn.dev/0h_n0/articles/58dc3076d2ffba)
- **DIN-SQL**: [https://arxiv.org/abs/2305.11853](https://arxiv.org/abs/2305.11853)
