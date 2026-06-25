---
layout: post
title: "テックブログ解説: Gemini 3.5 Flash — 高速推論とThinking Modeの設計思想"
description: "Google DeepMindが2026年5月にリリースしたGemini 3.5 Flashの技術的特徴を解説。Thinking Mode、4倍高速出力、ベンチマーク結果を分析する。"
categories: [blog, techblog, google]
tags: [gemini, LLM, function-calling, thinking-mode, google-deepmind]
date: 2026-06-26 12:00:00 +0900
source_type: tech_blog
source_url: https://deepmind.google/models/gemini-flash/
zenn_article: 60ad7eec7ce63c
zenn_url: https://zenn.dev/0h_n0/articles/60ad7eec7ce63c
target_audience: "修士学生レベル"
---

本記事は [Google DeepMind — Gemini Flash](https://deepmind.google/models/gemini-flash/) および [Google AI Blog: Gemini 3.5 Flash](https://blog.google/technology/google-deepmind/gemini-3-5-flash/) の解説記事です。

## ブログ概要（Overview）

Google DeepMindは2026年5月19日にGemini 3.5 Flashをリリースした。Googleはこのモデルを「our fastest, most efficient model」と位置付け、前世代の2.0 Flashに対して出力速度4倍、ベンチマーク性能の大幅な向上を実現したと発表している。Gemini APIおよびGoogle AI Studioで利用可能であり、`thinking_level`パラメータによる推論深度の動的制御（Thinking Mode）を特徴とする。

この記事は [Zenn記事: Gemini 3.5 Flash×階層型エピソード記憶でCSエージェントの応答精度を高める](https://zenn.dev/0h_n0/articles/60ad7eec7ce63c) の深掘りである。Zenn記事ではGemini 3.5 FlashのFunction Callingを階層型エピソード記憶のツール呼び出しに活用しているが、本記事ではモデル自体の技術的特徴を掘り下げる。

## 情報源

- **公式ページ**: [https://deepmind.google/models/gemini-flash/](https://deepmind.google/models/gemini-flash/)
- **ブログ記事**: [https://blog.google/technology/google-deepmind/gemini-3-5-flash/](https://blog.google/technology/google-deepmind/gemini-3-5-flash/)
- **API ドキュメント**: [https://ai.google.dev/gemini-api/docs/models#gemini-3.5-flash](https://ai.google.dev/gemini-api/docs/models#gemini-3.5-flash)
- **発表日**: 2026年5月19日（Google I/O 2026）
- **開発元**: Google DeepMind

## 技術的背景（Technical Background）

### モデルポジショニング

Gemini 3.5 Flashは、Geminiモデルファミリーにおいて「コスト効率と速度を重視」するFlashラインの最新世代である。Googleは以下の3つのモデルラインを展開している：

| モデル | 特徴 | ユースケース |
|--------|------|-------------|
| Gemini 3.5 Pro | 高精度・大規模推論 | 複雑な分析、研究 |
| **Gemini 3.5 Flash** | **高速・低コスト** | **エージェント、リアルタイム応答** |
| Gemini 3.5 Nano | オンデバイス | モバイル、エッジ |

CSエージェントのように、ユーザーとのリアルタイム対話でFunction Callingを多用するシナリオでは、レイテンシとコストの両方が重要であり、Flashラインが適切な選択となる。

### Thinking Mode

Gemini 3.5 Flashの最も注目すべき技術的特徴は`thinking_level`パラメータによる推論深度の動的制御である。Googleの公式ドキュメントによると、以下の4段階が提供されている：

| thinking_level | 挙動 | 想定ユースケース |
|---------------|------|----------------|
| `minimal` | 最小限の内部推論 | 単純な分類、ルーティング |
| `low` | 簡潔な推論 | FAQ応答、パターンマッチ |
| `medium` | 標準的な推論 | 一般的な質問応答 |
| `high` | 深い推論チェーン | 複雑な問題解決、数学 |

Zenn記事の階層型エピソード記憶パターンでは、記憶の検索・分類には`minimal`〜`low`、顧客への最終回答生成には`medium`〜`high`と使い分けることで、コストとレイテンシの最適化が可能である。

```python
from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="この顧客の過去のやり取りから、返品傾向を分析してください。",
    config=genai.types.GenerateContentConfig(
        thinking_config=genai.types.ThinkingConfig(
            thinking_level="HIGH"
        )
    )
)
```

### 速度とコスト

Googleが公表している性能指標は以下の通りである。

**速度**: 前世代（Gemini 2.0 Flash）比で出力速度4倍。Googleは「4x faster output」と公式に述べている。

**料金体系**（Gemini API、Google公式より）：

| 項目 | 料金 |
|------|------|
| 入力トークン | $1.50 / 100万トークン |
| 出力トークン | $9.00 / 100万トークン |
| キャッシュ入力 | $0.15 / 100万トークン |
| コンテキストウィンドウ | 1,048,576トークン |

CSエージェントにおけるコスト試算：1回のFunction Calling往復を平均1,000トークン（入力800 + 出力200）と仮定すると、1,000回のツール呼び出しで約$0.003（入力$0.0012 + 出力$0.0018）となる。Zenn記事の階層型記憶パターンでは1顧客対応あたり3〜5回のFunction Callingが発生するため、1,000顧客対応で約$0.009〜$0.015となる。

キャッシュ入力料金が通常入力の1/10であることは、エピソード記憶のシステムプロンプトやツール定義をキャッシュする設計において大きなコスト削減効果を持つ。

## ベンチマーク結果（Benchmark Results）

Googleが発表しているベンチマーク結果から、エージェント関連の指標を中心に整理する。

### エージェント系ベンチマーク

| ベンチマーク | スコア | 内容 |
|------------|--------|------|
| Terminal-Bench 2.1 | 76.2% | ターミナル操作タスク |
| MCP Atlas | 83.6% | MCP（Model Context Protocol）統合 |
| GDPval-AA | 1656 Elo | エージェント能力のEloレーティング |

Googleのブログによると、Terminal-Bench 2.1はCLI操作の自律的実行を評価するベンチマークであり、76.2%はFlashクラスのモデルとしては高い水準とされている。MCP Atlas 83.6%は、外部ツールとの統合能力を示す指標であり、Zenn記事のようなFunction Callingを多用するエージェントアーキテクチャとの親和性を裏付ける。

### Function Callingの特徴

Zenn記事の階層型エピソード記憶パターンでは、以下のFunction Callingツールが定義されている：
- `search_episodic_memory`: エピソード記憶の検索
- `store_episodic_memory`: エピソード記憶の保存
- `consolidate_to_semantic`: セマンティック記憶への統合
- `search_semantic_memory`: セマンティック記憶の検索

Gemini 3.5 FlashのFunction Callingは、ツール定義をJSON Schemaで受け取り、モデルがツール呼び出しの必要性を判断して構造化された呼び出しを生成する。Googleのドキュメントによると、`function_calling_config`で`ANY`（必ずツールを呼ぶ）、`AUTO`（モデルが判断）、`NONE`（ツールを呼ばない）の3モードが選択可能である。

CSエージェントでは`AUTO`モードが適切であり、ユーザーの質問内容に応じてモデルが記憶検索の必要性を自律的に判断できる。

## パフォーマンス最適化（Performance Optimization）

### Context Caching

Gemini 3.5 Flashはコンテキストキャッシュ機能を提供しており、繰り返し使用されるプロンプト部分のコスト削減が可能である。Googleのドキュメントによると、キャッシュの最小トークン数は32,768トークンである。

階層型エピソード記憶パターンへの適用：
1. **システムプロンプト + ツール定義**: 固定部分をキャッシュ化（$0.15/100万トークン）
2. **セマンティック記憶サマリ**: 顧客ごとの長期記憶をキャッシュ化
3. **エピソード記憶**: セッションごとに変化するため、キャッシュ対象外

### バッチ処理

Gemini APIはバッチリクエストをサポートしており、非リアルタイムの記憶統合処理（エピソード→セマンティック変換）をバッチで実行することで、さらなるコスト削減が可能である。Zenn記事のセマンティック記憶への統合処理は、顧客対応終了後に非同期で実行されるため、バッチ処理との相性が良い。

## 運用での学び（Operational Insights）

### Thinking Modeの使い分け戦略

CSエージェントの各処理フェーズに対するThinking Levelの推奨設定を以下に整理する。

| 処理フェーズ | thinking_level | 理由 |
|-------------|---------------|------|
| 記憶検索クエリ生成 | `minimal` | 単純なキーワード抽出 |
| エピソード記憶の関連度判定 | `low` | パターンマッチ中心 |
| 顧客意図の分類 | `low` | 定型的な分類タスク |
| 回答生成 | `medium` | 文脈を踏まえた自然な応答 |
| クレーム対応・エスカレーション判断 | `high` | 複雑な状況判断が必要 |

この使い分けにより、全処理を`high`で実行する場合と比較して、推論トークンの消費を削減できる。ただし、Thinking Modeの推論トークンは出力トークンとしてカウントされるため（$9.00/100万トークン）、`high`の多用はコストに直接影響する点に注意が必要である。

### レート制限と可用性

Gemini APIのレート制限はティアによって異なる。Googleのドキュメントによると、無料ティアでは15 RPM（Requests Per Minute）、有料ティアではより高い制限が適用される。CSエージェントのように複数の顧客を同時に処理する場合、レート制限を考慮した非同期処理設計が必須である。

### 安全性フィルタ

Gemini 3.5 Flashには組み込みの安全性フィルタがあり、`safety_settings`で閾値を調整可能である。CSエージェントでは、顧客からの攻撃的な入力に対してモデルが応答を拒否するケースが発生しうるため、安全性設定の適切なチューニングが運用上の課題となる。

## 学術研究との関連（Academic Context）

Gemini 3.5 Flashの設計思想は、以下の研究潮流と関連する。

**Mixture of Experts (MoE)**: Geminiファミリーは（公式には詳細非公開であるが）MoEアーキテクチャを採用していると広く推測されている。MoEは入力に応じて活性化するエキスパートを切り替えることで、パラメータ数に対する計算コストを削減する。Flashラインの高速性はこの設計に起因する可能性がある。

**Adaptive Computation**: Thinking Modeの`thinking_level`パラメータは、入力の複雑さに応じて計算量を動的に調整するAdaptive Computationの実用的実装と解釈できる。学術的には、Universal Transformers（Dehghani et al., 2019）やPonderNet（Banino et al., 2021）が提案した「考える時間を動的に割り当てる」概念に対応する。

**Tool Use / Function Calling**: Geminiを含む最近のLLMにおけるFunction Calling能力は、Toolformer（Schick et al., 2023）やGorilla（Patil et al., 2023）の研究成果を実用化したものである。Zenn記事の記憶ツール呼び出しパターンは、この研究方向の直接的な応用例である。

## まとめと今後の展望

Gemini 3.5 Flashは、エージェント用途に最適化された高速・低コストモデルとして、Zenn記事の階層型エピソード記憶パターンの実行基盤に適した特徴を備えている。Thinking Modeによる推論深度の動的制御は、記憶検索（軽量推論）と回答生成（深い推論）の使い分けを可能にし、コスト最適化の手段を提供する。

MCP Atlas 83.6%のスコアが示すツール統合能力と、コンテキストキャッシュによるコスト削減は、記憶層をFunction Callingで操作するアーキテクチャにおいて実用的な価値を持つ。ただし、Googleが公開しているベンチマーク結果は自社評価であり、独立した第三者評価との比較が重要である点を付記する。

## 参考文献

- **Google DeepMind — Gemini Flash**: [https://deepmind.google/models/gemini-flash/](https://deepmind.google/models/gemini-flash/)
- **Google AI Blog**: [https://blog.google/technology/google-deepmind/gemini-3-5-flash/](https://blog.google/technology/google-deepmind/gemini-3-5-flash/)
- **Gemini API Docs**: [https://ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/60ad7eec7ce63c](https://zenn.dev/0h_n0/articles/60ad7eec7ce63c)
