---
layout: post
title: "Martin Fowler解説: Harness Engineering for Coding Agents — Guide/Sensorパターンの実践的設計"
description: "コーディングエージェントのためのハーネス設計をGuide（フィードフォワード制御）とSensor（フィードバック制御）の2軸で整理した実践的フレームワーク"
categories: [blog, tech_blog]
tags: [LLM, agent, harness-engineering, coding-agent, software-engineering, architecture]
date: 2026-04-26 11:00:00 +0900
source_type: tech_blog
source_domain: martinfowler.com
source_url: https://martinfowler.com/articles/harness-engineering.html
zenn_article: 73bdc5dd332f59
zenn_url: https://zenn.dev/0h_n0/articles/73bdc5dd332f59
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Harness engineering for coding agent users (martinfowler.com)](https://martinfowler.com/articles/harness-engineering.html) の解説記事です。

## ブログ概要（Summary）

Martin Fowlerは、AIコーディングエージェントにおける**ハーネスエンジニアリング（Harness Engineering）**の実践的設計パターンを体系化した記事を公開した。著者はハーネスを「AIエージェントからモデルを除いたすべて」（Agent = Model + Harness）と定義し、品質制御メカニズムを**Guide（フィードフォワード制御）**と**Sensor（フィードバック制御）**の2パターンに分類している。さらに計算的制御と推論的制御の使い分け、制御を配置するタイミング（Shift Left原則）、ハーネスの3つの規制カテゴリ（保守性・アーキテクチャ適合性・振る舞い）を提示し、実務的な設計指針を提供している。

この記事は [Zenn記事: LLMエージェントの外部化設計：Memory・Skills・Protocols・Harnessの統一的理解](https://zenn.dev/0h_n0/articles/73bdc5dd332f59) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（ThoughtWorks / Martin Fowler）
- **URL**: [https://martinfowler.com/articles/harness-engineering.html](https://martinfowler.com/articles/harness-engineering.html)
- **組織**: ThoughtWorks
- **発表日**: 2026年

## 技術的背景（Technical Background）

AIコーディングエージェント（Claude Code、GitHub Copilot Workspace、Cursor等）の普及に伴い、エンジニアの役割が「コードを直接書く」から「エージェントの出力を制御・検証する」に移行しつつある。この変化の中で、エージェントの出力品質をどのように体系的に担保するかが実務上の課題となっている。

Fowlerは、この課題に対してソフトウェアエンジニアリングの既存概念（CI/CDパイプライン、リンター、テスト駆動開発）をハーネスとして再構成し、エージェントの出力品質を制御するフレームワークを提案している。このフレームワークは、Zhou et al.（2026）の外部化フレームワークにおけるHarness次元の実践的実装として位置付けられる。

## 実装アーキテクチャ（Architecture）

### ハーネスの定義と構造

Fowlerはハーネスを以下の等式で定義している。

$$
\text{Agent} = \text{Model} + \text{Harness}
$$

ハーネスは、内部コンポーネント（システムプロンプト、オーケストレーション）と外部コンポーネント（ユーザー向け制御）の両方を含む。

### Guide（フィードフォワード制御）とSensor（フィードバック制御）

ハーネスの制御メカニズムは2つの基本パターンに分類される。

```mermaid
graph LR
    G[Guide<br/>フィードフォワード制御] -->|事前に注入| A[エージェント実行]
    A -->|出力を検査| S[Sensor<br/>フィードバック制御]
    S -->|修正を指示| A
```

**Guide（フィードフォワード制御）** — コード生成の**前**にエージェントの行動を方向付ける仕組み。

| Guideの種類 | 具体例 | 効果 |
|---|---|---|
| ドキュメント | AGENTS.md、設計ドキュメント | アーキテクチャ意図の伝達 |
| ブートストラップ | セットアップスクリプト、テンプレート | 初期構造の確立 |
| コーディング規約 | カスタムルール、アーキテクチャ制約 | 品質基準の事前注入 |
| LSP統合 | Language Server Protocol | リアルタイム型情報のフィードバック |

**Sensor（フィードバック制御）** — コード生成の**後**に出力を検査し、自己修正ループを駆動する仕組み。

| Sensorの種類 | 具体例 | 検出対象 |
|---|---|---|
| 静的解析 | ESLint、Semgrep、Ruff | スタイル違反、セキュリティ脆弱性 |
| テスト実行 | pytest、vitest、カバレッジ計測 | 機能的な不整合 |
| アーキテクチャ検証 | ArchUnit | 構造的な逸脱 |
| コードレビュー | LLMベースレビューエージェント | 意味的な品質問題 |
| ミューテーションテスト | テスト品質の評価 | テスト自体の品質 |

### 計算的制御と推論的制御の区別

Fowlerは制御メカニズムの実行特性を2つのカテゴリに分類している。

| 属性 | 計算的（Computational） | 推論的（Inferential） |
|---|---|---|
| **速度** | ミリ秒〜秒 | 秒〜分 |
| **コスト** | 低 | 高（LLM API呼び出し） |
| **信頼性** | 決定的（同じ入力→同じ出力） | 確率的（非決定的） |
| **適用例** | 型チェッカー、リンター、テスト実行 | LLMコードレビュー、意味分析 |

計算的制御は決定的に成功確率を向上させる一方、推論的制御はより豊かな意味的判断を可能にするが、コストとレイテンシのトレードオフが生じる。

### Shift Left原則 — 制御の配置タイミング

Fowlerは制御メカニズムの配置をソフトウェア開発ライフサイクルに沿って3段階に整理し、安価で高速な制御を前段に配置する「Shift Left」を推奨している。

**第1段階: Pre-Integration（高速トラック）**
- LSPフィードバック
- 基本的なリンティングとスタイルチェック
- 高速テストスイート
- 初期コードレビューエージェント

**第2段階: Post-Integration（パイプライン）**
- 高コストなセンサー（ミューテーションテスト、網羅的コードレビュー）
- 全高速制御の再実行
- 詳細なアーキテクチャレビュー

**第3段階: Continuous Monitoring（変更ライフサイクル外）**
- デッドコード検出
- テストカバレッジ品質分析
- 依存関係スキャン
- ランタイムSLO監視
- ログ異常検知

この3段階配置により、低コストで高速なフィードバックを最初に提供し、高コストな検証は後段で実施するパイプラインが構成される。

### ハーネスの3つの規制カテゴリ

Fowlerはハーネスが規制すべき品質属性を3つのカテゴリに分類している。

**1. Maintainability Harness（保守性ハーネス）** — 内部コード品質を規制。重複コード検出、循環複雑度、カバレッジギャップ、スタイル違反を計算的センサーが検出する。推論的センサーは意味的重複やテスト冗長性を検出するが、誤診やスコープ外機能の追加は検出が困難とされる。

**2. Architecture Fitness Harness（アーキテクチャ適合性ハーネス）** — アーキテクチャ特性をフィットネス関数で強制。パフォーマンス要件をパフォーマンステストで、ログ標準をオブザーバビリティチェックで検証する。

**3. Behaviour Harness（振る舞いハーネス）** — 機能的正確性を検証。Fowlerはこのカテゴリが**最も未成熟**であると指摘しており、現状ではAI生成テストスイートの品質に依存するか、手動テストとの組み合わせが必要とされる。「Approved Fixtures（承認済みフィクスチャ）」パターンが新興のアプローチとして言及されている。

### Harnessability（ハーネス適合性）

Fowlerは、すべてのコードベースが均等にハーネスを構築しやすいわけではないとして「Harnessability」概念を提唱している。

**ハーネス構築を促進する要因（Ambient Affordances）**:
- 強く型付けされた言語（型チェックセンサーが機能）
- 明確なモジュール境界（アーキテクチャ制約ルールが定義可能）
- フレームワーク抽象化（例: Spring、Next.js）
- 事前定義されたサービストポロジー

レガシーシステムや技術的負債の蓄積したコードベースでは、ハーネス構築自体が困難であるにもかかわらず、最もハーネスを必要とするケースであるという矛盾が指摘されている。

### Harness Templates（ハーネステンプレート）

Fowlerは、共通のサービストポロジーに整合するGuideとSensorをバンドルした「Harness Templates」の概念を提案している。

- **データダッシュボード（Node.js）**: フロントエンド中心のテスト・Lighthouseセンサー
- **CRUDビジネスサービス（JVM）**: ArchUnit + 統合テスト + DB migration検証
- **イベントプロセッサ（Go）**: イベントスキーマ検証 + 冪等性テスト

各テンプレートは構造定義、技術スタック、適切な制御システムを含む。ただし、チームがテンプレートをカスタマイズするにつれてバージョニングとドリフト（乖離）が課題になるとされている。

## パフォーマンス最適化（Performance）

Fowlerの記事は具体的なベンチマーク数値を提示していないが、計算的制御と推論的制御の配置順序がコストとフィードバック速度に直接影響することを強調している。

**最適化の原則**:
- 計算的センサー（リンター、型チェッカー）はミリ秒〜秒単位で結果を返すため、最初の防御線として配置
- 推論的センサー（LLMレビュー）は秒〜分単位のレイテンシを持つため、計算的センサーを通過したコードにのみ適用
- この順序により、不要なLLM API呼び出しが削減されコスト効率が向上

## 運用での学び（Production Lessons）

Fowlerは業界事例として以下の実践を紹介している。

**OpenAI**: カスタムリンター、構造テスト、定期的なドリフトスキャンエージェントを組み合わせた階層型アーキテクチャ。コード規約の強制をリンター（計算的制御）で行い、設計意図の検証をLLMエージェント（推論的制御）で補完。

**Stripe**: Pre-pushフックによるヒューリスティックなリンター選択と、「Blueprints」と呼ばれるワークフローへのセンサー統合を実践。

**ThoughtWorksチーム**: 計算的センサーと推論的センサーを組み合わせたアーキテクチャドリフト緩和策。品質改善のための「Janitor Army（清掃チーム）」パターンの適用。

**Steering Loop（操舵ループ）**: エンジニアの役割は、再発する問題を監視し、将来の発生を防ぐためにGuideとSensorを改善するイテレーションにあるとされている。Fowlerは「良いハーネスは人間の入力を排除するのではなく、最も重要な箇所に人間の注意を向ける」と述べている。

## 学術研究との関連（Academic Connection）

Fowlerのハーネスエンジニアリングの分類は、Zhou et al.（2026）の外部化フレームワーク（arXiv 2604.08224）におけるHarness次元の6設計次元のうち、特にConfiguration & Policy（ガバナンス）とObservability（可視化）に直接対応している。

- **Guide = Harness内のConfiguration & Policy**: アーキテクチャルール、コーディング規約がポリシーとして事前定義される
- **Sensor = Harness内のObservability + Human Oversight**: 実行結果の可視化と品質ゲートの組み合わせ

また、OpenAIのHarness Engineering実践（2026年）はFowlerのフレームワークに基づいて構築されたと推察される。3名のエンジニアで100万行を生成した事例の背後には、GuideとSensorの体系的な設計があったことがFowlerの記事で確認できる。

## まとめと実践への示唆

Fowlerのハーネスエンジニアリングのフレームワークは、AIコーディングエージェントの出力品質制御を体系的に設計するための実践的指針を提供している。Guide/Sensorの2パターン分類、計算的/推論的制御の使い分け、Shift Left原則による配置最適化は、エージェント開発に取り組むチームが即座に適用できる設計ツールである。

未解決の課題として、異なるGuide/Sensor間の矛盾する信号への対処、ハーネスカバレッジの測定方法（コードカバレッジに類似するメトリクスの未整備）、Behaviour Harness（振る舞い検証）の成熟度向上が挙げられている。

## 参考文献

- **Blog URL**: [https://martinfowler.com/articles/harness-engineering.html](https://martinfowler.com/articles/harness-engineering.html)
- **Related Papers**: [arXiv:2604.08224](https://arxiv.org/abs/2604.08224)（Externalization in LLM Agents）
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/73bdc5dd332f59](https://zenn.dev/0h_n0/articles/73bdc5dd332f59)
- **OpenAI Harness Engineering**: [https://openai.com/index/harness-engineering/](https://openai.com/index/harness-engineering/)
