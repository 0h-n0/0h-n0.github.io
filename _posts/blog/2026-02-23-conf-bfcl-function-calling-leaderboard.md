---
layout: post
title: "カンファレンス論文解説: BFCL — Berkeley Function Calling Leaderboardによるツール呼び出し能力の標準ベンチマーク"
description: "ICML 2025採択。AST評価・Executable評価・Multi-turn Agentic評価の3軸でLLMのFunction Calling能力を体系的に測定するベンチマーク設計と評価手法を解説。"
categories: [blog, paper, conference]
tags: [function-calling, tool-use, benchmark, evaluation, llm, agent, python]
date: 2026-02-23 15:00:00 +0900
source_type: conference
conference: "ICML 2025"
source_url: https://proceedings.mlr.press/v267/patil25a.html
zenn_article: b2d1df91e5f5de
zenn_url: https://zenn.dev/0h_n0/articles/b2d1df91e5f5de
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [ICML 2025](https://proceedings.mlr.press/v267/patil25a.html) に採択された論文の解説記事です。

## 論文概要（Abstract）

Patil, Mao, Yan, Ji, Suresh, Stoica, Gonzalez (2025) は、LLMのFunction Calling（ツール呼び出し）能力を体系的に評価するベンチマーク **BFCL（Berkeley Function Calling Leaderboard）** を提案している。関数呼び出しの正当性判定の困難さと、多様な実世界関数の収集という2つの課題に対し、**抽象構文木（AST）ベースの評価手法**と、専門家・ユーザー貢献による関数収集を組み合わせたベンチマーク設計を確立した。

この記事は [Zenn記事: Function Calling×Structured Outputs実装入門：3社APIで型安全なツール連携を構築する](https://zenn.dev/0h_n0/articles/b2d1df91e5f5de) の深掘りです。

## 情報源

- **論文タイトル**: The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models
- **カンファレンス**: ICML 2025（PMLR 267:48371-48392）
- **URL**: [https://proceedings.mlr.press/v267/patil25a.html](https://proceedings.mlr.press/v267/patil25a.html)
- **著者**: Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, Joseph E. Gonzalez
- **所属**: UC Berkeley Sky Computing Lab
- **発表年**: 2025

## カンファレンス情報

**ICML 2025**（International Conference on Machine Learning）は機械学習分野のトップカンファレンスの1つである。BFCLはPoster発表として採択された。Function Callingの評価基準がトップカンファレンスで標準化されたことは、この分野の成熟を示している。

## 背景と動機（Background & Motivation）

Zenn記事ではOpenAI・Claude・GeminiのFunction Calling APIを用いた型安全なツール呼び出しパターンを解説しているが、各モデルのFunction Calling能力をどう比較すればよいかという問題は解決されていなかった。

従来のFunction Calling評価には以下の課題があった：

1. **正当性判定の困難さ**: 同じ意図を持つ関数呼び出しでも、引数の形式（`"Tokyo"` vs `"tokyo"` vs `"Tokyo, Japan"`）が異なりうる
2. **多様性の欠如**: 既存ベンチマーク（API-Bank, ToolBench等）は限定的なドメインのAPI集合に依存
3. **実行依存の制約**: 関数を実際に実行して検証する手法はスケーラビリティに欠ける
4. **マルチターン評価の不在**: 複数回の関数呼び出しが必要なAgenticシナリオの標準評価が存在しなかった

## 主要な貢献（Key Contributions）

- **貢献1**: AST（抽象構文木）ベースの自動評価手法を提案し、関数呼び出しの構造的正当性を実行不要で検証可能にした
- **貢献2**: 7カテゴリ・2,000問以上のベンチマークデータセットを構築。専門家キュレーションとユーザー貢献の組み合わせにより多様性を確保
- **貢献3**: Single-turnからMulti-turn Agenticまでの段階的評価フレームワークを設計し、LLMの「ツール使いこなし度」を多面的に測定

## 技術的詳細（Technical Details）

### 評価カテゴリ

BFCLは以下の7カテゴリで評価を行う：

| カテゴリ | 説明 | データ数 |
|---------|------|---------|
| **Simple** | 1関数・1呼び出し | 400 (AST) + 100 (Exec) |
| **Multiple** | 複数関数候補から1つ選択 | 400 (AST) + 100 (Exec) |
| **Parallel** | 同時に複数関数を呼び出し | 400 (AST) + 100 (Exec) |
| **Parallel Multiple** | 複数候補×並列呼び出し | 400 (AST) + 100 (Exec) |
| **Relevance Detection** | 呼び出すべき関数がない場合の棄却 | 250 |
| **Multi-turn (v3)** | 状態を持つ複数ターンの対話 | 複数シナリオ |
| **Live (v2)** | ユーザー貢献のリアルタイムデータ | 1,335 |

Zenn記事で紹介しているFunction Callingの基本パターン（単一ツール呼び出し）はSimpleカテゴリに対応し、Parallel・Parallel Multipleは複数ツールの同時呼び出しパターンに対応する。

### AST評価手法

BFCLの中核技術であるAST評価は、モデル出力の関数呼び出しを構文木に変換して構造的に比較する：

$$
\text{AST\_Match}(y, \hat{y}) = \begin{cases}
1 & \text{if } \text{func}(y) = \text{func}(\hat{y}) \land \text{args\_match}(y, \hat{y}) \\
0 & \text{otherwise}
\end{cases}
$$

ここで $y$ はモデル出力、$\hat{y}$ は正解、$\text{func}(\cdot)$ は関数名の抽出、$\text{args\_match}$ は引数の構造的一致判定である。

**引数一致の判定ロジック**:

```python
from ast import parse, dump
from typing import Any

def ast_match(
    model_output: str,
    ground_truth: list[str],
) -> bool:
    """AST評価: モデル出力と正解を構文木レベルで比較

    Args:
        model_output: モデルが生成した関数呼び出し文字列
        ground_truth: 正解として許容される関数呼び出しのリスト

    Returns:
        構造的に一致する場合True
    """
    try:
        model_ast = parse(model_output, mode="eval")
    except SyntaxError:
        return False

    model_func = extract_function_name(model_ast)
    model_args = extract_arguments(model_ast)

    for gt in ground_truth:
        gt_ast = parse(gt, mode="eval")
        gt_func = extract_function_name(gt_ast)
        gt_args = extract_arguments(gt_ast)

        if model_func == gt_func and check_args_match(model_args, gt_args):
            return True

    return False


def check_args_match(
    model_args: dict[str, Any],
    gt_args: dict[str, Any],
) -> bool:
    """引数の構造的一致チェック

    型変換（"3" vs 3）やデフォルト値の省略も考慮する。

    Args:
        model_args: モデルが生成した引数
        gt_args: 正解の引数

    Returns:
        引数が構造的に一致する場合True
    """
    for key, gt_value in gt_args.items():
        if key not in model_args:
            return False  # 必須引数の欠落
        model_value = model_args[key]
        # 型の柔軟なマッチング
        if not type_flexible_match(model_value, gt_value):
            return False
    return True
```

AST評価の利点は、関数を実際に実行せずに正当性を判定できることであり、数千の関数にスケールできる点にある。

### Executable評価

AST評価を補完するため、一部のテストケースでは実際に関数を実行して結果を検証する：

$$
\text{Exec\_Match}(y) = \mathbb{1}\left[\text{execute}(y) = \text{expected\_result}\right]
$$

各カテゴリ100問がExecutable評価用に設計されており、AST評価では捕捉できない実行時の正当性（API応答の妥当性等）を検証する。

### Relevance Detection（棄却能力）

提供された関数がユーザーの質問に対して不適切な場合、モデルが「呼び出すべき関数がない」と判断できるかを評価する。この能力はZenn記事のFunction Calling実装でも重要であり、不要なAPI呼び出しを防ぐ安全機構として機能する。

### Multi-turn Agentic評価（BFCL v3）

v3で追加されたMulti-turn評価では、状態を持つ複数ターンの対話における関数呼び出しの正当性を検証する：

1. **状態ベース評価**: 各ターン終了後のシステム状態が期待通りか確認
2. **応答ベース評価**: モデルの応答内容が前のターンの結果を正しく反映しているか確認

$$
\text{Multi-turn Score} = \frac{1}{|T|} \sum_{t=1}^{|T|} \mathbb{1}\left[\text{state\_check}(t) \land \text{response\_check}(t)\right]
$$

全ターンで両方のチェックをパスした場合のみ正解となるため、ターン数が増えるほど難易度が上がる。

## 実装のポイント（Implementation）

**ベンチマーク実行環境**: BFCLはPythonベースの評価フレームワークとして公開されている。各モデルのAPI呼び出し結果を統一フォーマットで収集し、AST/Exec評価を適用する。

```python
# BFCL評価の概念的フロー
from typing import Any

# テストケース定義
test_case: dict[str, Any] = {
    "id": "simple_001",
    "category": "simple",
    "question": "Get the weather for Tokyo",
    "function": [{
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    }],
    "ground_truth": [
        'get_weather(city="Tokyo")',
        'get_weather(city="Tokyo", unit="celsius")',
    ],
}
```

**Zenn記事との対応**: Zenn記事で解説しているPydanticベースのスキーマ定義は、BFCLの`function`フィールド（JSON Schema形式）と同じ構造である。Zenn記事のInstructorライブラリが生成するJSON Schemaは、BFCLの評価に直接利用可能な形式となっている。

**注意点**:
- BFCLはPython関数呼び出し形式を基本としており、Java・JavaScript形式の評価も一部含む
- Live（v2）データはユーザー貢献のため、品質にばらつきがある
- Multi-turn評価は状態管理のオーバーヘッドが大きく、実行コストが高い

## 実験結果（Results）

著者らが報告する主要な実験結果（論文Table 2, Figure 3より）：

| モデル | Simple | Multiple | Parallel | P. Multiple | Relevance | Overall |
|--------|--------|----------|----------|-------------|-----------|---------|
| GPT-4-turbo | 90.5% | 88.0% | 82.0% | 78.5% | 83.6% | 86.5% |
| Claude 3.5 Sonnet | 89.0% | 86.5% | 79.5% | 76.0% | 85.2% | 84.8% |
| GPT-4o | 91.2% | 89.5% | 84.0% | 80.0% | 82.0% | 87.2% |
| Gorilla-OpenFunctions-v2 | 87.5% | 84.0% | 78.0% | 72.0% | 78.0% | 82.5% |

- 著者らの報告によると、Single-turnタスクでは最先端モデルが85%以上のスコアを達成している
- Parallel Multiple（複数候補×並列呼び出し）は全モデルで最も難易度が高いカテゴリである
- Multi-turnシナリオでは、全モデルのスコアがSingle-turnから大幅に低下すると報告されており、長期的な状態管理と動的意思決定が依然として課題であることが示されている

## 実運用への応用（Practical Applications）

Zenn記事のFunction Calling実装と関連する実運用上のポイント：

- **モデル選択の指針**: BFCLスコアを参照することで、OpenAI・Claude・Geminiの中から用途に最適なモデルを選択できる。Zenn記事で紹介している3社統一パターンの実装時に、各社モデルの得意・不得意を定量的に把握できる
- **Parallel Function Calling**: Zenn記事のFunction Callingは主にSingle呼び出しを扱っているが、BFCLのParallelカテゴリはOpenAIの`parallel_tool_calls`パラメータに対応する機能の評価である
- **Relevance Detection**: 不要なツール呼び出しを防ぐ能力はコスト最適化に直結する。Zenn記事のInstructorベースの実装に棄却ロジックを追加する際の評価基準となる
- **AST評価の応用**: 自社Function Callingシステムのテスト自動化にBFCLのAST評価手法を転用できる

## 関連研究（Related Work）

- **ToolBench / ToolLLM (Qin et al., 2023)**: 16,000+ APIでの学習フレームワーク。BFCLが「評価」に特化するのに対し、ToolLLMは「学習」に焦点を当てる。評価手法としてはToolEval（LLM-as-Judge）を用いるが、BFCLのAST評価はLLM依存なしに自動評価できる点で再現性が高い
- **API-Bank (Li et al., 2023)**: API呼び出しのベンチマーク。BFCLと比較して関数の多様性が限定的
- **JSONSchemaBench (Geng et al., 2025)**: JSON Schema準拠の出力評価ベンチマーク。BFCLが関数呼び出しの「選択と引数生成」を評価するのに対し、JSONSchemaBenchは「スキーマ準拠の構造化出力」を評価する。両者は相補的な関係にある

## まとめと今後の展望

BFCLは、LLMのFunction Calling能力を「Simple→Multiple→Parallel→Multi-turn」と段階的に評価するフレームワークを確立した。AST評価によるスケーラブルな自動検証と、Multi-turn Agenticシナリオの導入により、ツール呼び出し能力の包括的な測定が可能になっている。

Zenn記事で解説されているFunction Calling実装を本番環境に投入する際、BFCLの評価カテゴリに沿ったテスト設計を行うことで、モデル間の性能差を定量的に把握し、最適なモデル選択とフォールバック戦略を構築できる。

## 参考文献

- **ICML 2025**: [https://proceedings.mlr.press/v267/patil25a.html](https://proceedings.mlr.press/v267/patil25a.html)
- **Leaderboard**: [https://gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html)
- **Code**: [https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- **Dataset**: [HuggingFace: gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/b2d1df91e5f5de](https://zenn.dev/0h_n0/articles/b2d1df91e5f5de)
