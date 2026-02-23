---
layout: post
title: "COLING 2025論文解説: MAC-SQL — マルチエージェント協調によるText-to-SQL"
description: "分解・検索・修正の3エージェント協調でBIRDベンチマーク59.59% EXを達成したMAC-SQLフレームワークの技術詳細解説。"
categories: [blog, paper, conference]
tags: [text-to-sql, multi-agent, llm, sql, langgraph, code-llama]
date: 2026-02-23 10:00:00 +0900
source_type: conference
conference: "COLING 2025"
source_url: https://aclanthology.org/2025.coling-main.36/
zenn_article: 58dc3076d2ffba
zenn_url: https://zenn.dev/0h_n0/articles/58dc3076d2ffba
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [COLING 2025 MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL](https://aclanthology.org/2025.coling-main.36/) の解説記事です。

## 論文概要（Abstract）

MAC-SQL（Multi-Agent Collaborative Framework for Text-to-SQL）は、LLMベースのText-to-SQLシステムにおいて、大規模データベースや複雑なクエリで性能が低下する問題に対し、マルチエージェント協調フレームワークを提案する。著者らは、SQL生成を担うコアの分解エージェントと、外部ツール・モデルを活用する2つの補助エージェントを組み合わせることで、BIRDベンチマークにおいてGPT-4使用時に59.59%の実行精度（EX）を達成し、論文発表時点のSoTAを報告している。

この記事は [Zenn記事: LangGraph×Claude Sonnet 4.6でSQL統合Agentic RAGを実装する](https://zenn.dev/0h_n0/articles/58dc3076d2ffba) の深掘りです。

## 情報源

- **会議名**: COLING 2025（31st International Conference on Computational Linguistics）
- **年**: 2025年1月（アブダビ, UAE）
- **URL**: [https://aclanthology.org/2025.coling-main.36/](https://aclanthology.org/2025.coling-main.36/)
- **著者**: Bing Wang, Changyu Ren, Jian Yang, Xinnian Liang, Jiaqi Bai, LinZheng Chai, Zhao Yan, Qian-Wen Zhang, Di Yin, Xing Sun, Zhoujun Li
- **ページ**: 540–557

## カンファレンス情報

**COLINGについて**:
- COLINGは計算言語学分野の国際会議であり、1965年から続く歴史ある学会
- 2025年はアブダビ（UAE）で開催
- NLP・自然言語処理コミュニティの主要会議の一つ

## 背景と動機

LLMベースのText-to-SQLシステムは、単純なクエリに対しては高い精度を示すが、以下のケースで性能が大幅に低下することが著者らにより指摘されている。

1. **大規模（"huge"）データベース**: テーブル数・カラム数が多いDBでは、全スキーマをLLMのコンテキストに収めることが困難
2. **複雑なクエリ**: 多段推論（multi-step reasoning）が必要なクエリでは、単一のLLM呼び出しでは不十分
3. **外部ツール・モデルとの協調**: 従来手法の多くがLLM単体での処理に閉じており、外部リソースの活用が不十分

著者らは、これらの課題に対してマルチエージェントアプローチが有効であるという仮説のもと、3つのエージェントが協調してText-to-SQLを実行するフレームワークMAC-SQLを設計している。

## 技術的詳細（Technical Details）

### MAC-SQLのマルチエージェントアーキテクチャ

MAC-SQLは以下の3エージェントで構成される。

```
自然言語質問 (NLQ) + データベース情報
       │
       ▼
┌────────────────────────────────────┐
│  Selector Agent（補助エージェント1） │
│  - DBスキーマの絞り込み              │
│  - 関連テーブル/カラムの選択          │
└──────────────┬─────────────────────┘
               │ 絞り込み済みスキーマ
               ▼
┌────────────────────────────────────┐
│  Decomposer Agent（コアエージェント）│
│  - Few-shot Chain-of-Thought       │
│  - サブ問題への分解                  │
│  - SQL生成                          │
└──────────────┬─────────────────────┘
               │ 生成SQL
               ▼
┌────────────────────────────────────┐
│  Refiner Agent（補助エージェント2）  │
│  - SQL実行・結果検証                 │
│  - エラー修正（自己修正ループ）       │
└──────────────┬─────────────────────┘
               │
               ▼
           最終SQL
```

### Selector Agent（スキーマ選択）

大規模DBに対してNLQを処理する前段として、データベースのスキーマ情報を絞り込むエージェントである。

**処理手順**:
1. NLQからキーワード・エンティティを抽出
2. DB内の全テーブル・カラムのメタ情報とのマッチングを実行
3. 関連度の高いテーブル・カラムのみを選択
4. 絞り込み済みスキーマをDecomposer Agentに渡す

著者らは、この絞り込みにより「huge」データベース（テーブル数が多い）での性能低下を抑制できると述べている。

### Decomposer Agent（SQL生成の中核）

Few-shot Chain-of-Thought（CoT）プロンプティングを用いてSQL生成を行うコアエージェントである。

**Few-shot CoTの構成**:

```
[System Prompt]
あなたはText-to-SQLの専門家です。
以下の手順でSQLを生成してください：
1. 質問を小さなサブ問題に分解
2. 各サブ問題に対応するSQL部品を生成
3. SQL部品を組み合わせて最終クエリを構築

[Few-shot Example 1]
Question: 「過去3ヶ月で最も売上が高い商品カテゴリの平均単価は？」
思考過程:
- サブ問題1: 過去3ヶ月の売上データを取得
- サブ問題2: カテゴリ別の合計売上を計算
- サブ問題3: 最高売上カテゴリを特定
- サブ問題4: そのカテゴリの平均単価を計算

SQL:
SELECT AVG(p.price)
FROM products p
JOIN sales s ON p.id = s.product_id
WHERE s.sale_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
  AND p.category = (
    SELECT p2.category
    FROM products p2
    JOIN sales s2 ON p2.id = s2.product_id
    WHERE s2.sale_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
    GROUP BY p2.category
    ORDER BY SUM(s2.amount) DESC
    LIMIT 1
  );

[Actual Question]
Question: {user_query}
Schema: {selected_schema}
```

### Refiner Agent（SQL修正）

生成されたSQLを実際にDBに対して実行し、エラーや異常結果を検出した場合に修正を行うエージェントである。

```python
def refine_sql(
    question: str,
    schema: str,
    generated_sql: str,
    db_connection,
    max_retries: int = 3
) -> str:
    """Refiner Agentの修正ループ

    Args:
        question: 元の自然言語質問
        schema: 使用スキーマ
        generated_sql: Decomposer Agentが生成したSQL
        db_connection: DB接続オブジェクト
        max_retries: 最大修正回数

    Returns:
        修正済みSQL
    """
    current_sql = generated_sql

    for attempt in range(max_retries):
        try:
            result = db_connection.execute(current_sql)
            if result and len(result) > 0:
                return current_sql
        except Exception as e:
            error_msg = str(e)

        # LLMに修正を依頼
        correction_prompt = f"""
SQL実行時にエラーが発生しました。
元の質問: {question}
スキーマ: {schema}
現在のSQL: {current_sql}
エラー内容: {error_msg}

修正したSQLを出力してください。"""

        current_sql = llm.invoke(correction_prompt)

    return current_sql
```

### SQL-Llama: GPT-4性能のオープンソース蒸留

著者らはMAC-SQLの設計をGPT-4で検証した後、Code Llama 7Bをベースにファインチューニングした**SQL-Llama**モデルを構築している。

**蒸留プロセス**:
1. BIRDトレーニングセットに対してGPT-4で各エージェントタスクの出力を生成
2. 生成データをCode Llama 7Bのファインチューニング用データとして整形
3. 各エージェントのタスク（スキーマ選択、SQL分解・生成、エラー修正）を統合的に学習
4. SQL-LlamaとしてBIRDベンチマークで評価

## 実験結果（Results）

### BIRDベンチマーク メイン結果

BIRDベンチマークでの実行精度（EX）の比較（論文Table 2相当）：

| 手法 | EX (%) | 使用モデル |
|------|--------|-----------|
| GPT-4 (ベースライン) | 46.35 | GPT-4 |
| SQL-Llama (蒸留版) | 43.94 | Code Llama 7B ファインチューニング |
| DIN-SQL | 55.9 | GPT-4 |
| DAIL-SQL | 57.41 | GPT-4 |
| **MAC-SQL** | **59.59** | GPT-4 |

著者らは、MAC-SQL + GPT-4構成がBIRDベンチマークで59.59% EXを達成し、論文発表時点でのSoTAを報告している。

### SQL-Llamaの性能分析

SQL-Llama（7Bパラメータ）は、GPT-4単体のベースライン（46.35%）に近い43.94%を達成している。これは、7Bという小規模モデルでもマルチエージェント協調のタスク設計をファインチューニングで学習できることを示唆している。ただし、MAC-SQL + GPT-4（59.59%）とのギャップは依然として大きく、著者らはモデルサイズの拡大や追加データでの改善を今後の課題として挙げている。

### データベース規模別の分析

著者らは、データベースの「規模」（テーブル数・カラム数）に応じた精度変化を分析している。

- **小規模DB（テーブル数5以下）**: MAC-SQL と単体LLMの差が小さい
- **中規模DB（テーブル数6-15）**: MAC-SQLが顕著に優位
- **大規模DB（テーブル数16以上）**: MAC-SQLの優位性が最大。Selector Agentによるスキーマ絞り込みの効果が最も発揮される

## 実装のポイント（Implementation）

### Zenn記事のLangGraphとの関連

[Zenn記事](https://zenn.dev/0h_n0/articles/58dc3076d2ffba)ではLangGraphのStateGraphで「ルーター → SQL検索 → ベクトル検索 → 回答生成」のパイプラインを構築しているが、MAC-SQLの3エージェント協調パターンは、このSQL検索ノード内部のアーキテクチャとして適用可能である。

**具体的な対応関係**:

| MAC-SQL Agent | LangGraph ノード対応 |
|--------------|---------------------|
| Selector Agent | `schema_selection_node` — テーブル/カラムを動的選択 |
| Decomposer Agent | `sql_generation_node` — CoT + Few-shotでSQL生成 |
| Refiner Agent | `sql_correction_node` — 実行結果に基づく自己修正 |

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

# MAC-SQLの3エージェントをノードとして実装
graph.add_node("schema_selector", schema_selection_node)
graph.add_node("sql_decomposer", sql_generation_node)
graph.add_node("sql_refiner", sql_correction_node)

# パイプライン接続
graph.set_entry_point("schema_selector")
graph.add_edge("schema_selector", "sql_decomposer")
graph.add_edge("sql_decomposer", "sql_refiner")

# 修正成功/失敗の条件分岐
graph.add_conditional_edges(
    "sql_refiner",
    check_sql_result,
    {
        "success": END,
        "retry": "sql_decomposer",
    },
)
```

### 実装時の注意点

- **エージェント間の通信コスト**: 3回のLLM呼び出しが必須。レイテンシが問題になる場合、Selector AgentとDecomposer Agentの処理を統合してLLM呼び出しを2回に減らす設計も考えられる
- **ファインチューニングデータの品質**: SQL-Llamaの性能はGPT-4の出力品質に依存。GPT-4の出力にエラーが含まれる場合、蒸留モデルもそのエラーを学習するリスクがある
- **拡張性**: 著者らはフレームワークが「新機能やツールを追加しやすい拡張可能なアーキテクチャ」であると述べている。LangGraphのノード追加と同様のモジュラリティがある

## 実運用への応用（Practical Applications）

MAC-SQLのマルチエージェントパターンは、LangGraphベースのAgentic RAGシステムにおいて以下の形で適用可能である。

- **大規模社内DB検索**: テーブル数が多い基幹システムDBでの自然言語検索に適している。Selector Agentがスキーマを絞り込むことで、LLMのコンテキストウィンドウ制約を回避できる
- **SQL品質の保証**: Refiner Agentによる自己修正ループは、本番環境でのSQL実行エラーを低減する。特に、構文エラーや結合条件の誤りを自動修正できる点が実用的
- **モデルコストの最適化**: SQL-Llamaのような蒸留モデルを使えば、GPT-4のAPIコストを大幅に削減可能。ただし精度とのトレードオフがある

## 関連研究（Related Work）

- **DIN-SQL** (Pourreza & Rafiei, NeurIPS 2023): GPT-4によるIn-context分解。MAC-SQLの前駆的研究
- **CHESS** (Talaei et al., 2024): RAGベーススキーマ選択 + 自己修正パイプライン。MAC-SQLと相補的アプローチ
- **Self-Refine** (Madaan et al., NeurIPS 2023): 自己フィードバック反復改善。Refiner Agentの設計に影響
- **AgentCoder** (2024): コード生成におけるマルチエージェント協調。MAC-SQLはText-to-SQL特化

## まとめ

MAC-SQLは、Text-to-SQLにおけるマルチエージェント協調の有効性を実証した研究である。著者らは、Selector Agent（スキーマ選択）、Decomposer Agent（SQL生成）、Refiner Agent（SQL修正）の3エージェント協調により、単体LLMでは困難な大規模DB・複雑クエリへの対応を実現している。SQL-Llamaによるオープンソース蒸留モデルの提案も、実用化に向けた重要な貢献である。LangGraphのStateGraphにMAC-SQLのエージェントパターンを組み込むことで、Zenn記事のSQL検索ノードの精度向上が期待される。

## 参考文献

- **Conference URL**: [https://aclanthology.org/2025.coling-main.36/](https://aclanthology.org/2025.coling-main.36/)
- **arXiv**: [https://arxiv.org/abs/2402.12926](https://arxiv.org/abs/2402.12926)
- **BIRD Benchmark**: [https://bird-bench.github.io/](https://bird-bench.github.io/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/58dc3076d2ffba](https://zenn.dev/0h_n0/articles/58dc3076d2ffba)
