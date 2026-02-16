---
layout: post
title: "OpenAI Agents SDK実践: Hub-and-Spokeパターンによるマルチエージェントポートフォリオ分析"
description: "OpenAI Agents SDKのAgent-as-Toolパターンを用いたマルチエージェント協調の実装を詳解。並列実行、MCP統合、トレーシングの実践的アーキテクチャを深掘りする"
categories: [blog, tech_blog]
tags: [OpenAI, agents-sdk, multi-agent, orchestration, claude, ai, agent, productivity]
date: 2026-02-17 09:00:00 +0900
source_type: tech_blog
source_domain: openai.com
source_url: https://cookbook.openai.com/examples/agents_sdk/multi-agent-portfolio-collaboration/multi_agent_portfolio_collaboration
zenn_article: c01f4e292ff1a7
zenn_url: https://zenn.dev/0h_n0/articles/c01f4e292ff1a7
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

OpenAI Cookbookで公開された本記事は、Agents SDKを用いたマルチエージェント協調の実践的な実装パターンを示している。投資分析を題材に、Portfolio Manager（PM）を中央オーケストレーターとし、Fundamental・Macro・Quantitativeの3つの専門エージェントをHub-and-Spoke型で統括するシステムを構築する。Agent-as-Toolパターンによる並列実行、MCP（Model Context Protocol）によるデータ連携、OpenAIトレーシングによる可観測性の確保が技術的な見どころである。

この記事は [Zenn記事: Claude Octopus: 複数AIを並列実行するオーケストレーションプラグイン](https://zenn.dev/0h_n0/articles/c01f4e292ff1a7) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（Cookbook）
- **URL**: https://cookbook.openai.com/examples/agents_sdk/multi-agent-portfolio-collaboration/multi_agent_portfolio_collaboration
- **組織**: OpenAI
- **発表日**: 2024-11-01

## 技術的背景（Technical Background）

マルチエージェントシステムの協調パターンは大きく2つに分類される。1つは**Handoff型**で、エージェントがタスクの途中で制御権を別のエージェントに移譲する方式である。会話的なワークフローに適するが、グローバルなタスク進捗の可視性が低下する。もう1つは**Agent-as-Tool型**で、中央エージェントがサブエージェントをツールとして呼び出す方式である。単一の制御スレッドを維持でき、並列実行との相性が良い。

本Cookbookが採用するのはAgent-as-Tool型であり、これはClaude Octopusのオーケストレーション設計思想と直接的に対応する。中央のPMエージェントが分析の全体像を把握しつつ、専門エージェントに個別の分析タスクを委譲し、結果を統合するパターンである。

学術的には、階層型マルチエージェントシステム（Hierarchical Multi-Agent Systems）のエージェントレベルが2段の特殊ケースとして位置付けられる。スーパーバイザーとワーカーの役割分担は、分散コンピューティングにおけるMaster-Workerパターンの応用でもある。

## 実装アーキテクチャ（Architecture）

### Hub-and-Spokeモデル

PMエージェント（Hub）が3つの専門エージェント（Spoke）を統括する構成を取る。

**Head Portfolio Manager**: 中央オーケストレーターとして、タスク分解・専門エージェント呼び出し・結果統合を担う。GPT-4.1をtemperature=0で使用し、再現性を確保する。

**Fundamental Agent**: 財務諸表、アナリストのセンチメント、企業指標を分析する。

**Macro Agent**: 経済指標、連邦準備制度（FRB）データ、セクターローテーションを分析する。

**Quantitative Agent**: 統計分析、テクニカル指標、相関分析を実施する。

### ツール統合の3層構造

Agents SDKは3種類のツール統合を提供する。

**MCP（Model Context Protocol）サーバー**: 外部データソースとの標準化された接続。本例ではYahoo Finance MCPサーバーを通じて市場データを取得する。

**OpenAI Managed Tools**: Code Interpreter（コード実行）やWebSearch（Web検索）等の組み込みツール。

**Custom Python Functions**: ドメイン固有のロジックをデコレータで関数ツールとして登録する。

```python
from openai.agents import Agent, function_tool
from typing import Any

def build_specialist_agents(
    model: str = "gpt-4.1",
) -> dict[str, Agent]:
    """3つの専門エージェントを構築する

    Args:
        model: 使用するモデルID

    Returns:
        エージェント名をキーとする辞書
    """
    fundamental = Agent(
        name="FundamentalAnalyst",
        model=model,
        instructions=(
            "あなたはファンダメンタル分析の専門家です。"
            "財務諸表、企業指標、アナリスト評価に基づく分析を行います。"
        ),
    )

    macro = Agent(
        name="MacroAnalyst",
        model=model,
        instructions=(
            "あなたはマクロ経済分析の専門家です。"
            "経済指標、金融政策、セクターローテーションを分析します。"
        ),
    )

    quant = Agent(
        name="QuantAnalyst",
        model=model,
        instructions=(
            "あなたは定量分析の専門家です。"
            "統計分析、テクニカル指標、相関分析を実施します。"
        ),
    )

    return {
        "fundamental": fundamental,
        "macro": macro,
        "quant": quant,
    }


def build_orchestrator(
    specialists: dict[str, Agent],
) -> Agent:
    """PMオーケストレーターを構築する

    各専門エージェントをfunction_toolでラップし、
    並列呼び出し可能なツールとしてPMに登録する。

    Args:
        specialists: 専門エージェントの辞書

    Returns:
        オーケストレーターエージェント
    """
    tools: list[Any] = []

    for name, agent in specialists.items():
        @function_tool(
            name_override=f"analyze_{name}",
            description_override=f"{name}分析を実行する",
        )
        async def agent_tool(
            query: str,
            _agent: Agent = agent,
        ) -> str:
            """専門エージェントに分析を委譲する

            Args:
                query: 分析クエリ

            Returns:
                分析結果の文字列
            """
            result = await _agent.run(task=query)
            return str(result)

        tools.append(agent_tool)

    pm = Agent(
        name="PortfolioManager",
        model="gpt-4.1",
        instructions=(
            "あなたはポートフォリオマネージャーです。"
            "3つの専門エージェントを並列に呼び出し、"
            "分析結果を統合して投資レポートを作成します。"
        ),
        tools=tools,
        parallel_tool_calls=True,  # 並列実行を有効化
    )

    return pm
```

### 並列実行メカニズム

`parallel_tool_calls=True`の設定により、PMエージェントは1回のLLM呼び出しで複数のツール（専門エージェント）を同時に起動できる。これにより、3つの専門エージェントの実行レイテンシは：

$$
L_{\text{total}} = L_{\text{PM}} + \max(L_{\text{fund}}, L_{\text{macro}}, L_{\text{quant}}) + L_{\text{synthesis}}
$$

ここで、
- $L_{\text{total}}$: 全体のレイテンシ
- $L_{\text{PM}}$: PMのタスク分解に要する時間
- $L_{\text{fund}}, L_{\text{macro}}, L_{\text{quant}}$: 各専門エージェントの実行時間
- $L_{\text{synthesis}}$: PMの結果統合に要する時間

逐次実行の場合は$L_{\text{fund}} + L_{\text{macro}} + L_{\text{quant}}$となるため、各エージェントの実行時間が同程度であれば約3倍の高速化が見込める。

## パフォーマンス最適化（Performance）

### モデル選択の最適化

PMエージェントにはGPT-4.1（temperature=0）を使用する。temperature=0とすることで、同一入力に対して同一の分析結果を生成でき、A/Bテストや回帰テストが容易になる。

各専門エージェントのモデルはタスクの複雑さに応じて選択可能であり、Fundamental分析のように構造化データの解釈が中心のタスクにはGPT-4.1-miniクラスの軽量モデルを、Macro分析のように複雑な推論が必要なタスクにはGPT-4.1を割り当てるなどの最適化が考えられる。

### トレーシングと可観測性

OpenAIのトレーシング機能により、各エージェントの判断プロセス、ツール呼び出し、推論過程をダッシュボードで可視化できる。これはプロダクション環境でのデバッグと品質監視に不可欠な機能である。

| 監視項目 | 取得方法 | 用途 |
|---------|---------|------|
| エージェント間メッセージ | OpenAI Tracing | デバッグ |
| ツール呼び出し履歴 | Function call logs | コスト分析 |
| 各エージェントのレイテンシ | タイムスタンプ差分 | ボトルネック特定 |
| トークン消費量 | Usage API | コスト最適化 |

## 運用での学び（Production Lessons）

### Agent-as-Toolパターンの利点と制約

**利点**: 中央制御により全体のタスク進捗を一元管理できる。サブエージェントの追加・削除がオーケストレーターの修正なしに可能である（Open/Closedの原則に準拠）。障害時のフォールバックが中央で統制できる。

**制約**: オーケストレーターがSPOF（Single Point of Failure）となる。サブエージェント間の直接通信が不可であり、すべての情報がオーケストレーター経由となるため、複雑な協調パターンには不向きである。

**対策**: オーケストレーターのリトライとタイムアウトを適切に設定し、サブエージェントの障害時にはデフォルト応答にフォールバックする。クリティカルな分析にはサーキットブレーカーパターンを導入する。

### プロンプト設計の重要性

各エージェントのシステムプロンプトには、以下の要素を明確に含める必要がある。

- **役割の明確な定義**: 何を分析し、何を分析しないか
- **出力フォーマットの指定**: 構造化されたレポート形式
- **ツール使用ルール**: どのツールをいつ使うか
- **品質基準**: 根拠のない主張の禁止、データソースの明記

## 学術研究との関連（Academic Connection）

Agent-as-Toolパターンは、階層型マルチエージェントシステム（HMAS）の2層特殊ケースとして形式化できる。AutoGen（Wu et al., 2023）のグループチャットがフラットな参加者構造を採るのに対し、本パターンは明確な階層を持つ。

また、タスク分解と並列実行の組み合わせは、MapReduceプログラミングモデルとの類似性がある。PMがMapperとReducerの両方の役割を担い、専門エージェントがMap処理を並列実行するモデルとして捉えられる。

## まとめと実践への示唆

OpenAI Agents SDKのAgent-as-Toolパターンは、Claude Octopusのようなオーケストレーションプラグインの設計に直接応用可能な実践的アーキテクチャを提供する。並列実行による3倍のレイテンシ削減、MCPによる標準化されたデータ接続、トレーシングによる完全な可観測性は、プロダクション環境でのマルチエージェントシステム構築に不可欠な要素である。

特に`parallel_tool_calls=True`による同時並行実行は、Claude Octopusの並列実行コンセプトと直接対応しており、OpenAI Agents SDKでの実装例として参考になる。今後はMCPの標準化が進むことで、異なるプロバイダーのエージェント間の相互運用性が向上し、より柔軟なマルチエージェントオーケストレーションが実現されるだろう。

## 参考文献

- **Blog URL**: https://cookbook.openai.com/examples/agents_sdk/multi-agent-portfolio-collaboration/multi_agent_portfolio_collaboration
- **Related Papers**: https://arxiv.org/abs/2308.08155 (AutoGen)
- **Related Zenn article**: https://zenn.dev/0h_n0/articles/c01f4e292ff1a7
