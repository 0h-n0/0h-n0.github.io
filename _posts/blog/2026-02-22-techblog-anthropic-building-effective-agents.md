---
layout: post
title: "Anthropic解説: Building Effective Agents — 5つの構成パターンとエージェント設計の実践原則"
description: "Anthropicが公開したLLMエージェント設計ガイド。5つのワークフローパターン、ツール設計原則、コーディングエージェントでの成功パターンを解説"
categories: [blog, tech_blog]
tags: [LLM-agent, anthropic, design-patterns, claudesonnet, codereview, python, agent]
date: 2026-02-22 23:40:00 +0900
source_type: tech_blog
source_domain: anthropic.com
source_url: https://www.anthropic.com/research/building-effective-agents
zenn_article: a41a3cb117cc46
zenn_url: https://zenn.dev/0h_n0/articles/a41a3cb117cc46
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Building Effective Agents (Anthropic Research)](https://www.anthropic.com/research/building-effective-agents) の解説記事です。

## ブログ概要（Summary）

Anthropicが公開した本ガイドは、多数のチームとの協業経験に基づき、LLMエージェント設計の実践的な原則とパターンを体系化したものである。著者らは、エージェントシステムの構成要素を「拡張されたLLM（Augmented LLM）」と定義し、5つのワークフローパターン（プロンプトチェーン、ルーティング、並列化、オーケストレータ-ワーカー、評価者-最適化者）を紹介している。コーディングエージェントが特に効果的なユースケースとして挙げられており、コードの検証可能性が反復的改善を可能にすると述べられている。

この記事は [Zenn記事: Claude Sonnet 4.6の1Mコンテキストで大規模コードレビューエージェントを構築する](https://zenn.dev/0h_n0/articles/a41a3cb117cc46) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://www.anthropic.com/research/building-effective-agents](https://www.anthropic.com/research/building-effective-agents)
- **組織**: Anthropic
- **発表日**: 2024年12月

## 技術的背景（Technical Background）

LLMエージェントの開発において、多くのチームが複雑なフレームワークやアーキテクチャを採用する傾向がある。しかし、Anthropicが多数の実運用プロジェクトを支援した経験からは、「最も成功した実装は、シンプルで構成可能なパターンを使用している」ことが明らかになっている。

このガイドが書かれた背景には、エージェントフレームワーク（LangChain、LlamaIndex等）の急速な普及と、それに伴う過度な抽象化の問題がある。フレームワークの内部動作を理解せずに使用することで、デバッグが困難になり、予期しない動作が発生するケースが報告されている。Anthropicは、API直接利用を出発点とし、必要に応じて段階的に複雑さを追加するアプローチを推奨している。

## 実装アーキテクチャ（Architecture）

### 基本構成要素: 拡張されたLLM（Augmented LLM）

すべてのエージェントシステムの基本構成要素は、検索（Retrieval）、ツール（Tools）、メモリ（Memory）で拡張されたLLMである。Anthropicはこれを「Augmented LLM」と呼んでいる。

```
┌─────────────────────────────┐
│      Augmented LLM          │
│  ┌─────────┐ ┌─────────┐  │
│  │Retrieval │ │  Tools  │  │
│  └─────────┘ └─────────┘  │
│  ┌─────────┐ ┌─────────┐  │
│  │ Memory  │ │   LLM   │  │
│  └─────────┘ └─────────┘  │
└─────────────────────────────┘
```

Model Context Protocol（MCP）を使用することで、サードパーティツールとの統合を標準化された方法で実現できる。

### 5つのワークフローパターン

**パターン1: プロンプトチェーン（Prompt Chaining）**

タスクを逐次的なステップに分解し、各LLM呼び出しが前のステップの出力を処理する。レイテンシと精度のトレードオフが明確なパターンである。

```python
def prompt_chain_review(code: str) -> dict:
    """プロンプトチェーンによるコードレビュー

    ステップ1: コード構造の分析
    ステップ2: 問題の検出
    ステップ3: 改善提案の生成
    """
    # Step 1: 構造分析
    structure = llm_call(
        "このコードの構造を分析してください: " + code
    )

    # Step 2: 問題検出（構造分析結果を利用）
    issues = llm_call(
        f"以下の構造分析に基づき、問題を検出してください:\n"
        f"構造: {structure}\n"
        f"コード: {code}"
    )

    # Step 3: 改善提案（問題検出結果を利用）
    suggestions = llm_call(
        f"以下の問題に対する改善提案を生成してください:\n"
        f"問題: {issues}"
    )

    return {"structure": structure, "issues": issues, "suggestions": suggestions}
```

**パターン2: ルーティング（Routing）**

入力を分類し、専門化されたハンドラに振り分ける。レビュー観点（セキュリティ、パフォーマンス、設計等）ごとに特化したプロンプトを使い分ける場合に有効である。

```python
def route_review(code_diff: str) -> str:
    """レビュー観点に基づくルーティング"""
    # 入力を分類
    category = llm_call(
        "このコード変更のカテゴリを判定してください: "
        "security / performance / design / style\n"
        f"Diff: {code_diff}"
    )

    # カテゴリ別の専門プロンプトでレビュー
    prompts = {
        "security": "セキュリティ観点で詳細にレビューしてください...",
        "performance": "パフォーマンス観点で詳細にレビューしてください...",
        "design": "設計観点で詳細にレビューしてください...",
        "style": "コーディングスタイル観点でレビューしてください...",
    }

    return llm_call(prompts.get(category, prompts["design"]) + code_diff)
```

**パターン3: 並列化（Parallelization）**

複数のLLM呼び出しを同時に実行する。2つのバリアントがある。

- **セクショニング**: 独立したサブタスクを並列実行（例: 複数ファイルの独立レビュー）
- **投票**: 同一タスクを複数回実行し、多様な出力を得る（例: 複数の観点でのレビュー）

**パターン4: オーケストレータ-ワーカー（Orchestrator-Workers）**

中央のLLMがタスクを動的に分解し、ワーカーLLMに委譲する。並列化との違いは、「サブタスクが事前定義されず、オーケストレータが入力に基づいて動的に決定する」点である。

Zenn記事のコードレビューエージェントは、このパターンの応用と見なすことができる。レビューエージェントがリポジトリ全体を分析し、問題箇所を特定した上で、各問題の詳細分析をワーカー（追加のLLM呼び出し）に委譲する設計が可能である。

**パターン5: 評価者-最適化者（Evaluator-Optimizer）**

一方のLLMが応答を生成し、もう一方のLLMが反復的にフィードバックを提供するループ構造。「LLMの応答が人間のフィードバックにより明確に改善できる場合」に効果的と述べられている。

コードレビューでは、初回レビュー結果に対して別のLLMが「このレビューは具体的か」「行動可能な提案になっているか」を評価し、不十分な場合に再生成を要求する、といった使い方ができる。

### ワークフロー vs エージェント

Anthropicは以下の使い分けを推奨している。

| 特性 | ワークフロー | エージェント |
|------|------------|------------|
| **フロー制御** | 事前定義されたコードパス | LLMが動的に決定 |
| **適用場面** | 手順が明確なタスク | オープンエンドの問題 |
| **コスト** | 予測可能 | 変動（ターン数依存） |
| **エラー伝播** | 限定的 | 累積リスクあり |
| **適切な例** | CI/CDパイプラインレビュー | リポジトリ全体の探索的レビュー |

### 3つの設計原則

1. **シンプルさ（Simplicity）**: 最小限のコンポーネントで構成する。フレームワークの抽象化に頼る前にAPI直接利用を検討する
2. **透明性（Transparency）**: 計画ステップを明示的に表示する。エージェントの意思決定過程をユーザーに見せる
3. **強力なACI（Agent-Computer Interface）**: ツール定義にはプロンプトエンジニアリングと同等の注意を払う

## パフォーマンス最適化（Performance）

### ツール設計のベストプラクティス

Anthropicが強調しているのは、「ツール定義はプロンプト本体と同等のエンジニアリング投資が必要」という点である。具体的な推奨事項を以下に示す。

```python
# 推奨: 十分な説明を含むツール定義
GOOD_TOOL = {
    "name": "search_code",
    "description": (
        "リポジトリ内のソースコードを正規表現パターンで検索します。"
        "検索対象はファイル内容のみで、ファイル名検索にはlist_filesを使用してください。"
        "大文字小文字を区別します。"
    ),
    "parameters": {
        "pattern": {
            "type": "string",
            "description": "検索する正規表現パターン（例: 'def\\s+\\w+\\('）",
        },
        "file_glob": {
            "type": "string",
            "description": "検索対象ファイルのglobパターン（例: '**/*.py'）",
        },
    },
}

# 非推奨: 説明が不十分なツール定義
BAD_TOOL = {
    "name": "search",
    "description": "コードを検索",
    "parameters": {
        "q": {"type": "string", "description": "クエリ"},
    },
}
```

**ポカヨケ原則（Poka-yoke）**: ツール設計においてミスを防止する仕組みを組み込む。例えば、ファイルパスのバリデーション、危険な操作の確認プロンプト、パラメータのデフォルト値設定などである。

### コーディングエージェントの成功パターン

Anthropicは、コーディングエージェントが特に効果的なユースケースであると述べている。その理由は以下の2点である。

1. **検証可能性**: コードはテスト実行により客観的に検証できるため、エージェントの反復的改善ループが機能しやすい
2. **明確な成功指標**: テストの合格/不合格が明確な成功基準を提供する

これはZenn記事のコードレビューエージェントの設計方針と直接的に関連する。レビュー結果をJSON構造化出力で返し、その後のパイプライン（GitHub PRコメント投稿、CI統合）で自動処理する設計は、Anthropicの推奨するパターンに合致している。

## 運用での学び（Production Lessons）

### フレームワーク利用の注意点

Anthropicは、フレームワーク（LangChain等）の使用を否定していないが、「内部動作を理解していない状態での利用」を強く警告している。具体的な問題として以下が挙げられている。

- フレームワークの抽象化が過度にプロンプトを変換し、意図しない動作を引き起こす
- デバッグ時にフレームワーク内部のLLM呼び出しを追跡できない
- フレームワークのバージョンアップにより動作が変化する

### 段階的複雑さの追加

「単純なプロンプトから始め、包括的な評価で最適化し、単純なソリューションでは不十分な場合にのみマルチステップのエージェントシステムを追加する」という原則が繰り返し強調されている。

### エラー管理

エージェントの各ターンでエラーが累積するリスクがあるため、以下の対策が推奨されている。

- 各ステップでの入力/出力の検証
- 人間介入ポイント（Human-in-the-Loop）の設計
- 失敗時のフォールバック戦略

## 学術研究との関連（Academic Connection）

本ガイドで紹介されているパターンは、以下の学術研究に基づいている。

- **ReAct** (arXiv:2210.03629): 推論と行動を交互に行うエージェントパターン。Anthropicの「エージェント」概念はReActの実践的な拡張と位置づけられる
- **Tree of Thoughts** (arXiv:2305.10601): 評価者-最適化者パターンの学術的基盤。複数の推論パスを評価・選択する手法
- **Toolformer** (arXiv:2302.04761): LLMのツール使用の学術的基盤。Anthropicの「ACI設計」はこの研究のプロダクション適用

## まとめと実践への示唆

Anthropicの「Building Effective Agents」は、LLMエージェント設計の実践的な指針を提供している。核心的なメッセージは「シンプルさが最も重要」であり、複雑なフレームワークに頼る前にAPI直接利用で十分なケースが多いということである。

Zenn記事のコードレビューエージェントの設計は、以下の点でAnthropicの推奨に合致している。

- **API直接利用**: Anthropic Python SDKを直接使用し、フレームワークの抽象化を避けている
- **構造化出力**: JSON形式でレビュー結果を返し、後続処理との統合を容易にしている
- **段階的な複雑さ**: まず1Mコンテキスト一括投入でレビューし、必要に応じてeffortレベルやコンテキスト量を調整する設計

一方、今後の改善として、ルーティングパターン（レビュー観点ごとの専門化）や評価者-最適化者パターン（レビュー品質の自己評価ループ）の導入が考えられる。

## 参考文献

- **Blog URL**: [https://www.anthropic.com/research/building-effective-agents](https://www.anthropic.com/research/building-effective-agents)
- **Context Engineering**: [https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **MCP**: [https://modelcontextprotocol.io/](https://modelcontextprotocol.io/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/a41a3cb117cc46](https://zenn.dev/0h_n0/articles/a41a3cb117cc46)
