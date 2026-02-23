---
layout: post
title: "Anthropic公式解説: Claude Prompt Cachingの技術仕様と最適化戦略"
description: "Anthropic公式ドキュメントに基づくClaude Prompt Cachingの完全技術解説。自動キャッシュ・明示的ブレークポイント・価格構造・キャッシュ無効化条件を詳述"
categories: [blog, tech_blog]
tags: [anthropic, claude, prompt-caching, KV-cache, LLM, cost-optimization]
date: 2026-02-23 14:00:00 +0900
source_type: tech_blog
source_domain: anthropic.com
source_url: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
zenn_article: 555a4e799660de
zenn_url: https://zenn.dev/0h_n0/articles/555a4e799660de
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Anthropic公式ドキュメント: Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) および [Anthropic公式ブログ: Prompt caching with Claude](https://claude.com/blog/prompt-caching) の解説記事です。

## ブログ概要（Summary）

Anthropicが提供するPrompt Cachingは、Claude APIのリクエスト間で共通するプロンプトプレフィックスのKV（Key-Value）テンソルをサーバー側で保存・再利用する機能である。公式ドキュメントによると、キャッシュ読取はベース入力トークン価格の10%で処理され、**APIコスト最大90%削減・レイテンシ最大85%改善**が報告されている。2026年2月時点で、Claude Opus 4.6、Sonnet 4.6、Haiku 4.5を含む全現行モデルで利用可能である。

この記事は [Zenn記事: LangGraph×Claude Sonnet 4.6のプロンプトキャッシュ最適化でAgentic RAGコスト90%削減](https://zenn.dev/0h_n0/articles/555a4e799660de) の深掘りです。

## 情報源

- **種別**: 企業テックブログ / 公式ドキュメント
- **URL**: [https://platform.claude.com/docs/en/build-with-claude/prompt-caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- **組織**: Anthropic
- **更新日**: 2026年2月（workspace-level isolation変更の告知含む）

## 技術的背景（Technical Background）

LLMの推論では、入力トークンに対してTransformerの各レイヤーでKey/Valueテンソルを計算する。マルチターン会話やエージェント型アプリケーションでは、システムプロンプト・ツール定義・過去の会話履歴が各リクエストで繰り返し送信され、そのたびにKV計算が発生する。

Prompt Cachingは、プロンプトの「プレフィックス」（先頭からキャッシュブレークポイントまでの部分）のKVテンソルを暗号学的ハッシュでインデックスし、同一プレフィックスが再利用された場合にKV計算をスキップする。これにより、入力トークンの処理コスト（計算時間と課金）を大幅に削減する。

**Anthropicの公式ドキュメントによる補足:**
- キャッシュは暗号学的ハッシュとKV表現を保存するが、**生のプロンプトテキストは保存しない**
- キャッシュはOrganization単位（2026年2月5日以降はWorkspace単位）で分離される
- 100%完全一致するプロンプトセグメントのみがキャッシュヒットする

## 2つのキャッシュ方式

### 自動キャッシュ（Automatic Caching）

リクエストレベルで`cache_control`を1つ追加するだけでキャッシュが有効になる方式。

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6-20250929",
    max_tokens=1024,
    cache_control={"type": "ephemeral"},  # これだけで自動キャッシュ有効
    system="あなたは社内文書検索アシスタントです。...",
    messages=[
        {"role": "user", "content": "売上レポートを教えてください"}
    ],
)
```

**動作原理**: 最後のキャッシュ可能ブロックに自動的にブレークポイントが配置される。マルチターン会話では、会話が成長するにつれてブレークポイントが自動的に前方へ移動する。

| リクエスト | キャッシュ動作 |
|-----------|-------------|
| リクエスト1: System + User:A + Asst:B + **User:C** | 全コンテンツをキャッシュ書込 |
| リクエスト2: 上記 + Asst:D + **User:E** | User:Cまでキャッシュ読取、D+Eを書込 |
| リクエスト3: 上記 + Asst:F + **User:G** | User:Eまでキャッシュ読取、F+Gを書込 |

### 明示的ブレークポイント（Explicit Cache Breakpoints）

個々のコンテンツブロックに`cache_control`を配置する方式。最大4つのブレークポイントを設定可能。

```python
response = client.messages.create(
    model="claude-sonnet-4-6-20250929",
    max_tokens=4096,
    tools=[
        {"name": "search", "description": "...", "input_schema": {...},
         "cache_control": {"type": "ephemeral"}},  # BP1: ツール定義
    ],
    system=[
        {"type": "text", "text": "システムプロンプト...",
         "cache_control": {"type": "ephemeral"}},  # BP2: システムプロンプト
        {"type": "text", "text": "RAGコンテキスト...",
         "cache_control": {"type": "ephemeral"}},  # BP3: RAGコンテキスト
    ],
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "質問...",
             "cache_control": {"type": "ephemeral"}},  # BP4: 会話
        ]},
    ],
)
```

**ブレークポイントのコスト**: ブレークポイント自体は無料。課金されるのはキャッシュ書込（ベースの1.25倍）とキャッシュ読取（ベースの0.1倍）のみ。

## 価格構造の詳細

Anthropic公式ドキュメントに記載されている価格表（2026年2月時点）：

| モデル | ベース入力 | 5分キャッシュ書込 | 1時間キャッシュ書込 | キャッシュ読取 | 出力 |
|--------|----------|-----------------|-------------------|-------------|------|
| Claude Opus 4.6 | $5/MTok | $6.25/MTok | $10/MTok | $0.50/MTok | $25/MTok |
| Claude Sonnet 4.6 | $3/MTok | $3.75/MTok | $6/MTok | $0.30/MTok | $15/MTok |
| Claude Haiku 4.5 | $1/MTok | $1.25/MTok | $2/MTok | $0.10/MTok | $5/MTok |

**価格乗数:**
- 5分キャッシュ書込: ベース × 1.25
- 1時間キャッシュ書込: ベース × 2.0
- キャッシュ読取: ベース × 0.1

**損益分岐点の計算:**

5分キャッシュの場合、キャッシュ書込のコストが通常入力より25%高いため、**少なくとも2回以上の再利用**がなければコスト削減にならない：

$$
\text{損益分岐} = \frac{C_{\text{write}}}{C_{\text{base}} - C_{\text{read}}} = \frac{1.25 C_{\text{base}}}{C_{\text{base}} - 0.1 C_{\text{base}}} = \frac{1.25}{0.9} \approx 1.39
$$

つまり、1回の書込後に1.39回以上の読取でコスト効率が改善する。実用上は**5分以内に2回以上**同一プレフィックスが使用されるユースケースで効果的。

## キャッシュ無効化の条件

公式ドキュメントは、キャッシュが無効化される条件を以下の表で整理している：

| 変更内容 | ツールキャッシュ | システムキャッシュ | メッセージキャッシュ |
|---------|---------------|-----------------|-----------------|
| ツール定義の変更 | ✘ | ✘ | ✘ |
| Web検索の切替 | ✓ | ✘ | ✘ |
| Citations切替 | ✓ | ✘ | ✘ |
| Speed設定変更 | ✓ | ✘ | ✘ |
| tool_choice変更 | ✓ | ✓ | ✘ |
| 画像の追加/削除 | ✓ | ✓ | ✘ |
| Thinking設定変更 | ✓ | ✓ | ✘ |

**重要な注意点**: キャッシュの階層は `tools` → `system` → `messages` の順序で構成されている。上位層の変更は下位層すべてのキャッシュを無効化する。ツール定義の変更は最も影響が大きく、全キャッシュが無効化される。

## 最小キャッシュトークン閾値

モデルによって最小キャッシュ可能トークン数が異なる（公式ドキュメントより）：

| モデル | 最小トークン数 |
|--------|-------------|
| Claude Opus 4.6 / 4.5 | 4096 |
| Claude Sonnet 4.6 / 4.5 / 4 | 1024 |
| Claude Haiku 4.5 | 4096 |

この閾値未満のプロンプトではキャッシュが適用されない。短いシステムプロンプトの場合、ツール定義と合わせて閾値を超えるよう設計する必要がある。

## 20ブロックルックバックウィンドウ

明示的ブレークポイント使用時、システムはブレークポイントから最大20ブロック前までしかキャッシュヒットを探索しない。

**具体例（公式ドキュメントより）:**
- 30ブロックの会話でブロック30にのみ`cache_control`を設定した場合
- ブロック25を変更 → ブロック24でキャッシュヒット（25-30を再処理）
- ブロック5を変更 → ブロック10まで探索して停止。キャッシュヒットなし

**対策**: 20ブロックを超える長い会話では、途中に追加のブレークポイントを配置する。

## 1時間キャッシュの使いどころ

公式ドキュメントが推奨する1時間TTLの使用シナリオ：

1. **エージェントのサブタスク実行**: サイドエージェントの処理が5分を超える場合
2. **低頻度ユーザー対話**: ユーザーの応答間隔が5分〜1時間の場合
3. **バッチAPIとの組み合わせ**: 共通プレフィックスを1時間キャッシュし、後続バッチリクエストで再利用

**5分と1時間の混在:**

同一リクエスト内で1時間と5分のTTLを混在可能だが、制約がある：
- 長いTTL（1時間）が短いTTL（5分）より前に配置されなければならない
- 逆順にするとAPI 400エラーが返る

## パフォーマンスベンチマーク

公式ブログに記載されたベンチマーク結果：

| シナリオ | キャッシュなし | キャッシュあり | レイテンシ改善 | コスト改善 |
|---------|-------------|-------------|-------------|----------|
| 100Kトークン書籍との対話 | 11.5秒 | 2.4秒 | -79% | -90% |
| 10K Many-shotプロンプト | 1.6秒 | 1.1秒 | -31% | -86% |
| 10ターン会話 | 約10秒 | 約2.5秒 | -75% | -53% |

**注意**: これらはAnthropic公式が報告した値であり、実際の効果はプロンプト構造や利用パターンにより変動する。

## 運用での学び（Production Lessons）

Anthropicの公式情報およびClaude Codeチームの知見：

1. **キャッシュミスは本番インシデント**: Anthropicは公式にキャッシュヒット率の低下をページャーアラート対象としていると言及している。キャッシュ効率がインフラコストに直結するため

2. **全ツールを毎回含める**: Claude Codeチームの設計方針として、リクエストごとにツールを動的に選択するのではなく、全ツールを常に含めてキャッシュプレフィックスを安定させることが推奨されている

3. **キャッシュメトリクスの監視**: レスポンスの`usage`フィールドで以下を追跡する
   - `cache_read_input_tokens`: キャッシュから読取られたトークン数
   - `cache_creation_input_tokens`: キャッシュに書込まれたトークン数
   - `input_tokens`: キャッシュ対象外のトークン数（最終ブレークポイント以降）

4. **Workspace分離への対応**: 2026年2月5日以降、キャッシュがOrganization単位からWorkspace単位に変更された。複数Workspaceを使用する場合、キャッシュ戦略の見直しが必要

## 学術研究との関連（Academic Connection）

Anthropicのプロンプトキャッシュは、以下の学術研究の実用化実装と位置づけられる：

- **Don't Break the Cache (arXiv:2601.06007)**: プロンプトキャッシュの設計原則を体系化した論文。AnthropicのStatic-before-Dynamic設計はこの論文の知見と一致
- **PagedAttention (vLLM)**: GPUメモリのページング管理によるKVキャッシュ効率化。Anthropicの内部実装詳細は非公開だが、類似のメモリ管理が行われていると推測される
- **Prefix Sharing**: マルチテナント環境でのプレフィックス共有（ChunkAttention等）。AnthropicのOrganization/Workspace単位のキャッシュ分離はこの方向性の実装

## まとめと実践への示唆

Anthropicのプロンプトキャッシュは、LLMアプリケーションの推論コスト削減において最も実装コストの低い最適化手法である。自動キャッシュは1パラメータの追加で有効化でき、明示的ブレークポイントはより細かい制御を提供する。

実践上の要点は以下の3点に集約される：
1. **静的コンテンツを先頭に配置**し、キャッシュヒット率を最大化する
2. **キャッシュメトリクスを監視**し、ヒット率80%以上を維持する
3. **5分TTLと1時間TTLを使い分け**、利用パターンに応じたコスト最適化を行う

## 参考文献

- **Anthropic Docs**: [https://platform.claude.com/docs/en/build-with-claude/prompt-caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- **Anthropic Blog**: [https://claude.com/blog/prompt-caching](https://claude.com/blog/prompt-caching)
- **Anthropic Pricing**: [https://platform.claude.com/docs/en/about-claude/pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/555a4e799660de](https://zenn.dev/0h_n0/articles/555a4e799660de)
