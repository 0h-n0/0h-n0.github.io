---
layout: post
title: "CIDR 2025論文解説: AOP — DAGベース並列実行によるLLMパイプライン自動オーケストレーション"
description: "複雑なクエリに対するLLMパイプラインの自動生成・DAGリライト・並列実行最適化を解説"
categories: [blog, paper, conference]
tags: [LLM, pipeline-orchestration, DAG, parallel-execution, CIDR]
date: 2026-02-23 11:00:00 +0900
source_type: conference
conference: "CIDR 2025"
source_url: https://vldb.org/cidrdb/2025/aop-automated-and-interactive-llm-pipeline-orchestration-for-answering-complex-queries.html
zenn_article: a5be5c172a5a99
zenn_url: https://zenn.dev/0h_n0/articles/a5be5c172a5a99
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [AOP: Automated and Interactive LLM Pipeline Orchestration for Answering Complex Queries (CIDR 2025)](https://vldb.org/cidrdb/2025/aop-automated-and-interactive-llm-pipeline-orchestration-for-answering-complex-queries.html) の解説記事です。

## 論文概要（Abstract）

AOP（Automated and Interactive LLM Pipeline Orchestration）は、複雑なクエリに対してLLMパイプラインを自動的に生成・最適化するシステムである。著者ら（清華大学のJiayi Wang、Guoliang Li）は、Pipeline Generatorで複数のチェーン形式パイプラインを生成し、Pipeline Rewriterで独立したオペレータを特定してDAG（有向非巡回グラフ）構造にリライトすることで並列実行を実現する。著者らの実験では、複雑なテストセットにおいて回答精度を45%向上させたと報告されている。

この記事は [Zenn記事: LangChain LCEL実践ガイド：LLMチェーンのレイテンシを50%削減する最適化手法](https://zenn.dev/0h_n0/articles/a5be5c172a5a99) の深掘りです。Zenn記事ではLCELのパイプ演算子による線形パイプラインと`RunnableParallel`による並列化を解説していますが、本記事ではパイプライン構造の自動最適化という学術的アプローチを深掘りします。

## 情報源

- **会議名**: CIDR 2025（Conference on Innovative Data Systems Research）
- **年**: 2025
- **URL**: [https://vldb.org/cidrdb/2025/aop-automated-and-interactive-llm-pipeline-orchestration-for-answering-complex-queries.html](https://vldb.org/cidrdb/2025/aop-automated-and-interactive-llm-pipeline-orchestration-for-answering-complex-queries.html)
- **著者**: Jiayi Wang, Guoliang Li（清華大学）

## カンファレンス情報

**CIDRについて**:
- CIDRはデータシステム研究のトップカンファレンスの一つで、VLDB Endowmentが主催する
- 革新的なデータシステムの設計・実装に関する研究を採択対象とする
- AOPはLLMパイプラインの自動最適化という新しいカテゴリの研究として採択された

## 技術的詳細（Technical Details）

### システムアーキテクチャ

AOPは以下の3つの主要コンポーネントで構成される：

1. **Pipeline Generator**: 自然言語クエリからLLMパイプラインを自動生成
2. **Pipeline Rewriter**: チェーン形式パイプラインをDAG構造にリライトし並列実行を可能にする
3. **Interactive Executor**: 実行中にユーザーフィードバックを受け付け、パイプラインを動的に修正

### オペレータ体系

AOPでは、パイプラインの構成要素を2種類のオペレータに分類している：

**プログラム型オペレータ（Pre-programmed Operators）**:
- データベース検索、ファイル操作、APIコールなど
- 決定的な処理で結果が一意に定まる
- 実行時間が予測可能

**セマンティック型オペレータ（Semantic Operators）**:
- LLMプロンプトによって実行される
- テキスト要約、分類、推論、バリデーションなど
- 結果が非決定的で実行時間が変動する

```python
from abc import ABC, abstractmethod
from typing import Any

class Operator(ABC):
    """AOPオペレータの基底クラス"""

    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """オペレータの実行"""
        ...

    @abstractmethod
    def get_dependencies(self) -> list[str]:
        """このオペレータが依存する他オペレータのIDリスト"""
        ...

class SemanticOperator(Operator):
    """LLMで実行されるセマンティックオペレータ

    Args:
        prompt_template: LLMに渡すプロンプトテンプレート
        model: 使用するLLMモデル名
    """

    def __init__(self, prompt_template: str, model: str = "gpt-4o"):
        self.prompt_template = prompt_template
        self.model = model

    def execute(self, input_data: Any) -> Any:
        prompt = self.prompt_template.format(**input_data)
        return llm_call(self.model, prompt)

class ProgrammedOperator(Operator):
    """プログラムで実行される決定的オペレータ

    Args:
        func: 実行する関数
    """

    def __init__(self, func):
        self.func = func

    def execute(self, input_data: Any) -> Any:
        return self.func(input_data)
```

### Pipeline Generator: パイプライン自動生成

Pipeline Generatorは、入力されたクエリを分析し、必要なオペレータとその接続関係を含む実行パイプラインを自動生成する。生成されるパイプラインは初期段階ではチェーン形式（線形）である。

著者らの論文によると、生成プロセスは以下のステップで構成される：

1. **クエリ分解**: 複雑なクエリをサブタスクに分解
2. **オペレータ選択**: 各サブタスクに対応するオペレータを事前定義されたカタログから選択
3. **依存関係推定**: サブタスク間のデータ依存関係を推定
4. **パイプライン構成**: オペレータをチェーン形式に配列

### Pipeline Rewriter: DAG構造への変換

Pipeline Rewriterは、AOPの最も重要な貢献である。チェーン形式のパイプラインを分析し、独立したオペレータを特定してDAG構造にリライトする。

**依存関係分析のアルゴリズム**:

$$
\text{Independent}(O_i, O_j) \iff \neg(\text{DataDep}(O_i, O_j) \lor \text{DataDep}(O_j, O_i))
$$

ここで、
- $O_i, O_j$: パイプライン内のオペレータ
- $\text{DataDep}(O_i, O_j)$: $O_j$ が $O_i$ の出力に依存する関係
- $\text{Independent}(O_i, O_j)$: 2つのオペレータが独立（並列実行可能）

**DAGリライトの擬似コード**:

```python
from collections import defaultdict

def rewrite_chain_to_dag(
    chain: list[Operator],
) -> dict[str, list[str]]:
    """チェーン形式パイプラインをDAGにリライト

    Args:
        chain: 線形に配列されたオペレータのリスト

    Returns:
        DAGの隣接リスト（オペレータID → 依存先IDリスト）
    """
    dag = defaultdict(list)
    n = len(chain)

    for i in range(n):
        actual_deps = []
        for j in range(i):
            if has_data_dependency(chain[j], chain[i]):
                actual_deps.append(chain[j].id)
        dag[chain[i].id] = actual_deps

    return dict(dag)

def execute_dag(
    dag: dict[str, list[str]],
    operators: dict[str, Operator],
) -> dict[str, Any]:
    """DAGのトポロジカル順序で並列実行

    同じレイヤー（依存関係が満たされたオペレータ群）を並列実行する。
    """
    import concurrent.futures

    results = {}
    completed = set()

    while len(completed) < len(dag):
        ready = [
            op_id for op_id, deps in dag.items()
            if op_id not in completed
            and all(d in completed for d in deps)
        ]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    operators[op_id].execute,
                    {d: results[d] for d in dag[op_id]},
                ): op_id
                for op_id in ready
            }
            for future in concurrent.futures.as_completed(futures):
                op_id = futures[future]
                results[op_id] = future.result()
                completed.add(op_id)

    return results
```

### LCELとの対比

AOPのPipeline Rewriterの動作は、LCELの`RunnableParallel`と本質的に同じ最適化を自動的に行うものとして理解できる。

| 側面 | LCEL | AOP |
|------|------|-----|
| **並列化の指定方法** | 開発者が明示的に`RunnableParallel`を使用 | システムが自動的にDAGを構築 |
| **最適化のタイミング** | 設計時（静的） | 実行時（動的） |
| **対象パイプライン** | 線形+手動分岐 | 任意のチェーン→DAG自動変換 |
| **依存関係分析** | 開発者が判断 | データフロー分析で自動推定 |

LCELで`RunnableParallel`を使って手動で並列化する箇所を、AOPは自動的に特定・変換する。これはRDBMSのクエリオプティマイザが開発者の書いたSQLを最適な実行計画に変換するのと同様のアプローチである。

### Prefetching最適化

著者らの論文では、DAGリライトに加えてPrefetching最適化も提案されている。これは、オペレータの実行中に次に実行される可能性の高いオペレータの入力データを事前取得する手法である。

$$
T_{\text{total}} = \sum_{l=1}^{L} \max_{O_i \in \text{Layer}_l} T(O_i) - T_{\text{prefetch}}
$$

ここで、
- $L$: DAGのレイヤー数
- $\text{Layer}_l$: レイヤー $l$ に属するオペレータの集合
- $T(O_i)$: オペレータ $O_i$ の実行時間
- $T_{\text{prefetch}}$: プリフェッチにより節約された時間

## 実装のポイント（Implementation）

### LCEL開発者への示唆

AOPの研究から、LCEL開発者が学べる実践的な知見：

1. **依存関係の明示化**: LCELのチェーンを設計する際、各ステージの入出力依存関係を明確に把握し、独立したステージを`RunnableParallel`で並列化する
2. **パイプラインのレイヤー構造**: DAGのレイヤー実行と同様に、LCELチェーンも「並列実行可能なグループ」と「逐次実行が必要なグループ」に分けて設計する
3. **プリフェッチの応用**: `RunnableParallel`の1つのブランチでデータを事前取得し、他のブランチの結果と合流させるパターンが有効

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

# AOPのDAGレイヤー実行をLCELで表現
layer_1 = RunnableParallel({
    "search_results": RunnableLambda(search_database),
    "faq_results": RunnableLambda(search_faq),
    "user_context": RunnableLambda(get_user_context),
})

layer_2 = RunnableParallel({
    "summarized": summarize_chain,
    "classified": classify_chain,
})

pipeline = layer_1 | merge_results | layer_2 | final_answer_chain
```

## 実験結果（Results）

著者らの論文で報告された主要な実験結果：

- **回答精度**: 複雑なテストセットにおいて、ベースライン（単一LLMコール）と比較して回答精度を45%向上
- **並列実行効果**: DAGリライトにより、チェーン形式と比較してレイテンシを大幅に削減（独立オペレータの数に比例）
- **対象データ**: 構造化データ、半構造化データ、非構造化データの混合クエリに対応

**分析ポイント**: 精度向上はパイプラインの自動生成による適切なオペレータ選択に起因し、レイテンシ削減はDAGリライトによる並列実行に起因すると著者らは分析している。

## 実運用への応用（Practical Applications）

AOPの手法は、以下のシナリオで実用的である：

1. **データレイク上のクエリ処理**: 複数のデータソース（RDB、ドキュメントDB、API）にまたがるクエリの自動パイプライン構築
2. **RAGパイプラインの自動最適化**: マルチソースRAGでの検索ステージの自動並列化
3. **BI（ビジネスインテリジェンス）ツール**: 自然言語でのデータ分析クエリを自動的に最適な処理パイプラインに変換

**制約と限界**: AOPはデータレイクのクエリ処理に特化しており、汎用的なLLMエージェントの構築には追加の研究が必要である。また、セマンティックオペレータの依存関係分析はLLMの推論に依存するため、誤検出のリスクがある。

## 関連研究（Related Work）

- **LangChain LCEL**: 開発者が手動でパイプラインを構築する宣言的フレームワーク。AOPは同様のパイプラインを自動的に生成・最適化する
- **WorkflowLLM** (2025): 106,763サンプルの大規模データセットでLLMのワークフロー生成能力を強化。AOPとは異なり、LLM自体のワークフロー生成能力をファインチューニングで改善するアプローチ
- **Prompt2DAG** (2025): LLMベースのデータエンリッチメントパイプライン生成。AOPと同様のDAG構造を使用するが、プロンプトからのコード生成に焦点を当てている

## まとめと今後の展望

AOPは、LLMパイプラインの自動生成とDAGリライトによる並列実行最適化を実現した研究である。著者らの提案するPipeline Rewriterの手法は、LCELの`RunnableParallel`を手動で設計する際の理論的基盤を提供する。データベースのクエリオプティマイザがSQLを最適な実行計画に変換するのと同様に、将来的にはLLMパイプラインフレームワークにもこのような自動最適化が組み込まれることが期待される。

## 参考文献

- **Conference URL**: [https://vldb.org/cidrdb/2025/aop-automated-and-interactive-llm-pipeline-orchestration-for-answering-complex-queries.html](https://vldb.org/cidrdb/2025/aop-automated-and-interactive-llm-pipeline-orchestration-for-answering-complex-queries.html)
- **PDF**: [https://vldb.org/cidrdb/papers/2025/p32-wang.pdf](https://vldb.org/cidrdb/papers/2025/p32-wang.pdf)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/a5be5c172a5a99](https://zenn.dev/0h_n0/articles/a5be5c172a5a99)
