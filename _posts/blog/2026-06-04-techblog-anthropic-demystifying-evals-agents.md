---
layout: post
title: "Anthropic Engineering解説: Demystifying Evals for AI Agents"
description: "Anthropicが公開したAIエージェント評価の包括的ガイド。3種類のグレーダー設計、pass@k/pass^k指標、8ステップロードマップ、エージェント種別ごとの評価戦略を詳細解説"
categories: [blog, tech_blog]
tags: [evaluation, AI-agent, Anthropic, LLM-as-Judge, CI-CD, langsmith, langchain, pytest]
date: 2026-06-04 10:00:00 +0900
source_type: tech_blog
source_domain: anthropic.com
source_url: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
zenn_article: 6d33daf25f3dc7
zenn_url: https://zenn.dev/0h_n0/articles/6d33daf25f3dc7
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)（Anthropic Engineering Blog, 2026年1月9日公開）の解説記事です。

## ブログ概要（Summary）

Anthropicのエンジニアリングチームが公開した、AIエージェント評価（Evals）の包括的ガイドである。エージェント評価における基本概念の定義から、3種類のグレーダー（Code-Based / Model-Based / Human）の設計指針、非決定性を扱うpass@k・pass^k指標、エージェント種別（コーディング・会話・リサーチ・コンピュータ操作）ごとの具体的評価戦略、そして8ステップの導入ロードマップまでを体系的にカバーしている。

この記事は [Zenn記事: LangSmith Datasets×Experimentsでエージェント品質を自動テストする](https://zenn.dev/0h_n0/articles/6d33daf25f3dc7) の深掘りです。Zenn記事で紹介されている3層テスト戦略（Single-step / Full-turn / Multi-turn）やCI/CDパイプライン構築の設計思想は、本ブログの内容と直接対応しています。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- **組織**: Anthropic Engineering
- **著者**: Mikaela Grace, Jeremy Hadfield, Rodrigo Olivares, Jiri De Jonghe
- **発表日**: 2026年1月9日

## 技術的背景（Technical Background）

LLMエージェントの品質保証は、従来のソフトウェアテストとは根本的に異なる課題を持つ。エージェントはツール呼び出し、マルチターン対話、環境操作を含む複雑なワークフローを実行し、その出力は非決定的である。ブログではこの課題に対し、「Evaluations（Evals）とは、AIシステムに対するテストであり、出力に採点ロジックを適用して成功を測定する」と明確に定義している。

従来のユニットテストは決定論的な入出力を検証するが、エージェント評価ではモデルの非決定性、ツール使用の副作用、マルチステップの軌跡全体を考慮する必要がある。Zenn記事で紹介されているLangSmithのpytest統合は、この課題に対するLangChain側のアプローチであり、本ブログはAnthropicの視点から同様の問題にアプローチしている。

## 評価システムの基本概念

ブログでは以下の用語を定義している。

| 概念 | 定義 |
|------|------|
| **Task/Problem** | 定義された入力と成功基準を持つ単一テスト |
| **Trial** | 1つのタスクに対する1回の試行。モデルの非決定性を考慮し複数回実行する |
| **Grader** | エージェントの出力を採点するロジック。1タスクに複数のグレーダーを設定可能 |
| **Transcript/Trace** | 試行の完全な記録（出力、ツール呼び出し、推論過程、中間結果） |
| **Evaluation Suite** | 特定の能力を測定するためのタスク集合 |

## 3種類のグレーダー設計

### Code-Based Grader（コードベース）

文字列マッチング、正規表現、ファジーマッチング、静的解析、ツール呼び出し検証などの決定論的手法でエージェント出力を採点する。

**長所**: 高速、低コスト、客観的、再現可能、デバッグ容易
**短所**: 正当な出力バリエーションに対して脆弱。微妙なニュアンスを捉えられない

ブログでは、コーディングエージェントの評価においてユニットテストスイートの実行が最も信頼性の高いCode-Based Graderであると述べている。SWE-bench Verifiedではこのアプローチが標準採用されている。

### Model-Based Grader（モデルベース）

ルーブリックベースのスコアリング、自然言語アサーション、ペアワイズ比較、参照ベース評価、マルチジャッジコンセンサスなどの手法で、LLMを評価器として活用する。

**長所**: 柔軟、スケーラブル、ニュアンスを捉える、オープンエンドなタスクに対応
**短所**: 非決定的、高コスト、キャリブレーションが必要

これはZenn記事の`llm_judge_relevance`関数や`eval_no_hallucination`関数に直接対応する手法である。ブログでは、Model-Based Graderを使う際に「Unknown」を返すフォールバック指示を含めることを推奨している。

### Human Grader（人手評価）

SME（Subject Matter Expert）レビュー、クラウドソーシング、スポットチェック、A/Bテスト、評価者間一致度の測定を含む。

**長所**: 品質の金字標準、専門家の判断に一致
**短所**: 高コスト、低速、専門家へのアクセスが課題

ブログではHuman Graderを定期的なキャリブレーションに用い、Model-Based Graderの精度を人手評価で検証する運用を推奨している。

## 非決定性を扱う評価指標

エージェントの出力は非決定的であるため、1回の試行結果だけでは信頼性が低い。ブログでは2つの指標を紹介している。

### pass@k

$k$回の試行で少なくとも1回成功する確率。$k$を増やすほど成功確率が上がる。能力の上限（最良ケース）を測定する指標である。

$$
\text{pass@}k = 1 - (1 - p)^k
$$

ここで、$p$は1試行あたりの成功確率である。

### pass^k

$k$回の試行で全て成功する確率。$k$を増やすほど厳しくなる。一貫性（最悪ケース）を測定する指標である。

$$
\text{pass}^k = p^k
$$

ブログでは具体例として、1試行あたりの成功率75%の場合、3試行のpass^3は約42%になると示している。これは本番環境での信頼性を評価する際に重要な指標である。

```mermaid
flowchart LR
    A[エージェント実行] --> B{Trial 1}
    A --> C{Trial 2}
    A --> D{Trial 3}
    B --> E[pass@3: 少なくとも1回成功?]
    C --> E
    D --> E
    B --> F[pass^3: 全3回成功?]
    C --> F
    D --> F
```

### Capability Evals vs Regression Evals

ブログでは評価を2種類に分類している。

| 評価タイプ | 目標パスレート | 用途 |
|-----------|-------------|------|
| **Capability Evals** | 低い（難しいタスクをターゲット） | エージェントの能力上限を測定 |
| **Regression Evals** | ほぼ100% | 既存機能の維持を確認 |

Zenn記事のCI/CDパイプラインでの回帰テストはRegression Evalsに該当し、ナイトリービルドでの包括的評価はCapability Evalsに対応する。

## エージェント種別ごとの評価戦略

### コーディングエージェント

決定論的グレーダー（ユニットテストスイート）が最も効果的であるとブログは述べている。ベンチマークとしてSWE-bench VerifiedとTerminal-Benchが紹介されている。評価はpass/failのテスト結果に加え、コード品質（静的解析、セキュリティチェック、LLMルーブリック）も組み合わせる設計が推奨されている。

ブログではYAML形式のタスク設定例を示しており、認証バイパス修正タスクでは以下のグレーダーを組み合わせている:
- `deterministic_tests`: 必須テストの実行
- `llm_rubric`: コード品質のルーブリック評価
- `static_analysis`: ruff、mypy、banditによる静的解析
- `state_check`: セキュリティログの確認
- `tool_calls`: 必要なツール呼び出しの検証

### 会話エージェント

マルチ次元の成功基準（状態検証、ターン数制限、トーン適切性）を設定する。ユーザーシミュレーション（第2のLLMがユーザー役を演じる）が必要となる。ベンチマークとしてτ-Benchおよびτ2-Benchが紹介されている。

ブログのサポート対応タスク例では、共感表現の確認、解決策の明確な説明、ツール結果に基づいた回答の根拠確認を`llm_rubric`で評価し、チケットステータスと返金処理の状態を`state_check`で検証する設計となっている。

### リサーチエージェント

根拠確認（groundedness check）、カバレッジ検証、ソース品質評価を組み合わせる。主観的な品質判断と変動する正解（shifting ground truth）が課題であり、専門家判断との頻繁なキャリブレーションが必要であるとブログは述べている。

### コンピュータ操作エージェント

GUIインタラクション（スクリーンショット、クリック、キーボード入力）をテストする。ベンチマークとしてWebArena（ブラウザタスク）とOSWorld（OS操作）が紹介されている。DOM抽出とスクリーンショットベースのアプローチではトークン効率に差があり、コストとのトレードオフを考慮する必要があるとしている。

## 8ステップ導入ロードマップ

ブログでは以下の8ステップを推奨している。

**Step 0-1: 早期に小さく始める** — 実際の失敗事例から20-50の単純なタスクを作成する。包括的なカバレッジを待たず、既存の手動チェックを自動化することから始める。

**Step 2: タスク品質の確保** — 「2人のドメイン専門家が独立して同じpass/fail判定に達する」レベルの明確さが必要。曖昧なタスクはノイズを増やす。

**Step 3: バランスの取れた問題セット** — 行動すべき場合とすべきでない場合の両方をテストする。クラス不均衡を避ける。

**Step 4: 堅牢な評価ハーネス** — 「各試行はクリーンな環境から開始されるべき」。共有状態による相関失敗を防止する。Zenn記事でもDockerコンテナや一時ディレクトリの利用が推奨されている。

**Step 5: 思慮深いグレーダー設計** — 決定論的グレーダーを優先し、必要な場合のみLLMグレーダーを使用する。出力を評価し、経路を評価しない（硬直的なステップ検証を避ける）。

**Step 6: トランスクリプトレビュー** — 定期的に評価トランスクリプトを読み、グレーダーが正しく機能しているか確認する。

**Step 7: 飽和の監視** — パスレートが100%に近づいたら新しい困難なタスクを開発する。

**Step 8: 長期メンテナンス** — 評価インフラを所有する専任チームを設置し、ドメイン専門家がタスクを提供する体制を構築する。「eval-driven development」として、エージェントが問題を解決する前にテストを構築する。

## 実践的な評価フレームワーク

ブログでは評価手法を以下のように整理している。

| 手法 | タイミング | 用途 |
|------|-----------|------|
| **自動Evals** | プリローンチ、CI/CD | 高速イテレーション、品質の第一防衛線 |
| **本番モニタリング** | ポストローンチ | 分布ドリフト検出、実世界の障害発見 |
| **A/Bテスト** | ポストローンチ | 大きな変更のリアルトラフィック検証 |
| **ユーザーフィードバック** | 継続的 | 予期しない問題の発見 |
| **トランスクリプトレビュー** | 継続的 | 直感の構築、微妙な問題の発見 |
| **人手評価** | 定期的 | LLMグレーダーのキャリブレーション |

この分類は、Zenn記事で紹介されているOffline評価（CI/CD統合）とOnline評価（本番モニタリング）の使い分けと一致している。

## ベンチマーク実績

ブログでは以下の実績データが報告されている。

- **SWE-bench Verified**: フロンティアモデルが年初30%から80%超に到達し、飽和に近づいている
- **Opus 4.5 on CORE-Bench**: 当初42%のスコアが、グレーディングバグの修正後に95%に向上した。これは評価自体の品質がスコアに大きく影響することを示す事例である
- **Claude Code on SWE-bench**: 1年間で40%から80%超に進歩

ブログはフロンティアモデルが多くの試行で0%のスコアを示す場合、「モデルの能力不足ではなくタスク定義の問題を疑うべき」と指摘している。

## 学術研究との関連（Academic Connection）

本ブログの内容は、複数の学術研究と密接に関連している。

- **Judging LLM-as-a-Judge**（Zheng et al., 2023）: Model-Based Graderの理論的基盤。ブログのルーブリックベース評価やペアワイズ比較はこの論文の手法に基づく
- **SWE-bench**（Jimenez et al., 2024）: コーディングエージェント評価のベンチマーク。ブログではVerified版の使用を推奨
- **τ-Bench**（Yao et al., 2024）: 会話エージェント評価のベンチマーク。マルチターン対話の評価フレームワーク

## まとめと実践への示唆

Anthropicの本ブログは、エージェント評価を「神秘的なもの」ではなく「体系的なエンジニアリングプラクティス」として位置づけている。核心的なメッセージは、(1) 早期に小さく始めること、(2) 決定論的グレーダーを優先すること、(3) 評価自体の品質を継続的に改善すること、の3点に集約される。Zenn記事のLangSmith pytest統合と組み合わせることで、PRごとの自動品質検証からナイトリーの能力評価まで、多層的な品質保証体制を構築できる。

## 参考文献

- **Blog URL**: [https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- **Related Papers**: [https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)（Judging LLM-as-a-Judge）
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/6d33daf25f3dc7](https://zenn.dev/0h_n0/articles/6d33daf25f3dc7)

---

:::message
この記事はAI（Claude Code）により自動生成されました。内容の正確性については原ブログと照合していますが、最新の情報は公式ドキュメントもご確認ください。
:::
