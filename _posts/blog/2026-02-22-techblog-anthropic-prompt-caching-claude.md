---
layout: post
title: "Anthropic解説: Claude APIプロンプトキャッシュ — コスト90%削減・レイテンシ85%短縮の実装と料金設計"
description: "AnthropicのClaude APIプロンプトキャッシュ機能の技術的仕組み、料金体系、ユースケース別パフォーマンスを詳細解説する"
categories: [blog, tech_blog]
tags: [anthropic, claude, prompt-caching, llm, cost-optimization, api, aws, bedrock, rag]
date: 2026-02-22 18:30:00 +0900
source_type: tech_blog
source_domain: anthropic.com
source_url: https://www.anthropic.com/news/prompt-caching
zenn_article: d027acf4081b9d
zenn_url: https://zenn.dev/0h_n0/articles/d027acf4081b9d
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Anthropic公式ブログ: Prompt caching with Claude](https://www.anthropic.com/news/prompt-caching) および [Claude APIドキュメント](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) の解説記事です。

## ブログ概要（Summary）

Anthropicは2024年8月にClaude APIのプロンプトキャッシュ機能をGA（一般提供）としてリリースした。この機能は、APIコール間で頻繁に使われるコンテキスト（システムプロンプト、ドキュメント、ツール定義等）をキャッシュし、後続リクエストでの再処理を省略する。公式ブログによれば、長いプロンプトに対してコストを最大90%、レイテンシを最大85%削減できると報告されている。Amazon BedrockおよびGoogle Cloud Vertex AIでもプレビュー提供されている。

この記事は [Zenn記事: Bedrock AgentCore×1時間キャッシュで社内RAGコスト90%削減](https://zenn.dev/0h_n0/articles/d027acf4081b9d) の深掘りです。

## 情報源

- **種別**: 企業テックブログ / 公式ドキュメント
- **URL**: [https://www.anthropic.com/news/prompt-caching](https://www.anthropic.com/news/prompt-caching)
- **組織**: Anthropic
- **発表日**: 2024年8月（GA）、以降継続的に機能追加

## 技術的背景（Technical Background）

### なぜプロンプトキャッシュが必要か

現代のLLMアプリケーションでは、プロンプトの大部分がリクエスト間で共通するケースが多い。例えば：

- **チャットボット**: 数千トークンのシステムプロンプトが全リクエストで同一
- **RAGアプリ**: 検索されたドキュメントコンテキスト（数万トークン）が一定期間変化しない
- **コーディングアシスタント**: コードベースの要約（数千〜数万トークン）が編集ごとに大部分は不変
- **エージェント**: ツール定義（数百〜数千トークン）が全ターンで同一

これらの共通部分に対して毎回KVキャッシュ（Attention States）を再計算するのは計算資源の浪費である。Prompt Cacheの学術研究（arXiv:2311.04934、本ブログの関連記事として別途解説）がこの問題に対する理論的基盤を提供し、Anthropicはこれを商用API機能として実装した。

### KVキャッシュ再利用の仕組み

Claude APIのプロンプトキャッシュは、以下の手順で動作する：

1. **キャッシュ書き込み（初回）**: プロンプト内の`cache_control`マーカーが付与された部分までのKVキャッシュを計算し、Anthropicのインフラストラクチャに保存する
2. **キャッシュ読み取り（2回目以降）**: 同一のプレフィックスを持つ後続リクエストで、保存済みKVキャッシュを直接読み込み、プロンプト処理をスキップする
3. **差分計算**: キャッシュされていない部分（ユーザーメッセージ等の動的部分）のみAttention計算を実行する

重要な制約として、キャッシュは**プレフィックス一致**で動作する。すなわち、キャッシュされたプレフィックスの後に異なるテキストが続く場合でもキャッシュヒットとなるが、プレフィックス自体が変更されるとキャッシュミスとなる。

## 実装アーキテクチャ（Architecture）

### Claude API（Messages API）での実装

Claude APIでは、`cache_control`フィールドをコンテンツブロックに追加することでキャッシュポイントを指定する：

```python
import anthropic

client = anthropic.Anthropic()

# 1回目: キャッシュ書き込み
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "あなたは社内ドキュメントQAアシスタントです。...",
        },
        {
            "type": "text",
            "text": "【社内ドキュメント全文】\n...(数万トークン)...",
            "cache_control": {"type": "ephemeral"},  # キャッシュポイント
        },
    ],
    messages=[
        {"role": "user", "content": "休暇申請の手順は？"}
    ],
)

# usage.cache_creation_input_tokens: キャッシュ書き込みトークン数
# usage.cache_read_input_tokens: キャッシュ読み取りトークン数（初回は0）
```

### Amazon Bedrock Converse APIでの実装

Bedrock Converse APIでは、`cachePoint`ブロックとTTL指定を使用する：

```python
import boto3

client = boto3.client("bedrock-runtime", region_name="us-west-2")

response = client.converse(
    modelId="us.anthropic.claude-sonnet-4-20250514-v1:0",
    system=[
        {"text": "システムプロンプト..."},
        {"text": "ドキュメントコンテキスト...(大量テキスト)"},
        {
            "cachePoint": {
                "type": "default",
                "ttl": "1h",  # 1時間TTL（2026年1月GA）
            }
        },
    ],
    messages=[
        {"role": "user", "content": [{"text": "質問テキスト"}]}
    ],
)
```

Zenn記事で詳述されている通り、Bedrock版では5分と1時間の2種類のTTLが選択可能であり、1時間TTLは社内RAGのようなユーザー応答間隔が不定のユースケースに適している。

### キャッシュ制約と要件

Claude APIドキュメントより、以下の制約が明記されている：

| 制約項目 | 仕様 |
|---------|------|
| 最低トークン要件 | Claude Sonnet: 1,024トークン、Claude Haiku: 2,048トークン |
| 最大キャッシュブレークポイント | 4個/リクエスト |
| TTL（Claude API直接） | 5分（デフォルト）。アクセスごとにリフレッシュ |
| TTL（Bedrock版） | 5分または1時間を選択可能 |
| キャッシュ無効化条件 | プレフィックスの変更、ツール定義の変更、画像の追加・削除 |

## 料金体系と損益分岐分析

### 料金テーブル（2026年2月時点）

Claude APIドキュメント記載の料金体系を以下にまとめる：

| モデル | 通常入力 | 5分キャッシュ書込 | 1時間キャッシュ書込 | キャッシュ読取 | 出力 |
|--------|---------|-----------------|-------------------|-------------|------|
| Claude Sonnet 4 | $3.00/MTok | $3.75/MTok (1.25x) | $6.00/MTok (2.0x) | $0.30/MTok (0.1x) | $15.00/MTok |
| Claude Haiku 3.5 | $0.25/MTok | $0.30/MTok (1.2x) | — | $0.03/MTok (0.12x) | $1.25/MTok |
| Claude Opus 4 | $15.00/MTok | $18.75/MTok (1.25x) | — | $1.50/MTok (0.1x) | $75.00/MTok |

**料金設計の原則**:
- キャッシュ書き込みは基本入力料金の1.25倍（5分）または2.0倍（1時間）
- キャッシュ読み取りは基本入力料金の約10%
- 読み取り回数が増えるほど、書き込みコストの差（1.25x vs 2.0x）は希釈される

### 損益分岐分析

キャッシュなしの場合と比較した損益分岐点を計算する。50,000トークンのドキュメントコンテキストを$N$回再利用する場合のコスト関数は以下の通り：

**キャッシュなし（毎回全トークン処理）**:

$$
C_{\text{none}}(N) = p_{\text{input}} \times T \times N
$$

**1時間キャッシュ**:

$$
C_{\text{1h}}(N) = p_{\text{write\_1h}} \times T + p_{\text{read}} \times T \times (N - 1)
$$

ここで、
- $T = 0.05$ MTok（50,000トークン）
- $p_{\text{input}} = 3.00$ $/MTok
- $p_{\text{write\_1h}} = 6.00$ $/MTok
- $p_{\text{read}} = 0.30$ $/MTok

損益分岐点は $C_{\text{none}}(N) = C_{\text{1h}}(N)$ を解くと：

$$
N = \frac{p_{\text{write\_1h}}}{p_{\text{input}} - p_{\text{read}}} = \frac{6.00}{3.00 - 0.30} \approx 2.2
$$

すなわち、1時間以内に**3回以上**同一コンテキストが参照されれば1時間キャッシュが有利となる。Zenn記事の「損益分岐点は約2.2回」という分析と一致する。

### ユースケース別パフォーマンス（公式ブログより）

Anthropic公式ブログで報告されているベンチマーク結果は以下の通りである：

| ユースケース | キャッシュなしTTFT | キャッシュありTTFT | TTFT削減率 | コスト削減率 |
|------------|-----------------|-----------------|----------|----------|
| 書籍全文QA（100Kトークン） | 11.5秒 | 2.4秒 | 79% | 90% |
| Many-shot prompting（10Kトークン） | 1.6秒 | 1.1秒 | 31% | 86% |
| マルチターン会話（10ターン） | ~10秒 | ~2.5秒 | 75% | 53% |

## パフォーマンス最適化（Performance）

### キャッシュヒット率を最大化する設計パターン

1. **静的コンテンツを先頭に配置**: システムプロンプト → ツール定義 → ドキュメントコンテキスト → 会話履歴 → ユーザーメッセージの順に配置し、変化頻度が低いものを先頭に置く

2. **キャッシュブレークポイントの戦略的配置**: 最大4つのブレークポイントのうち、以下の配置が推奨される：
   - BP1: ツール定義の後（ほぼ変化しない）
   - BP2: ドキュメントコンテキストの後（1時間TTL推奨）
   - BP3: 会話履歴の累積部分の後（5分TTL）
   - BP4: 予備（自動キャッシュ用）

3. **ドキュメント更新頻度の制御**: Knowledge Basesの同期を日次に固定し、ドキュメント変更によるキャッシュ無効化を最小限に抑える

### キャッシュが無効化される条件

Claude APIドキュメントに基づき、以下の操作でキャッシュが無効化される：

- プレフィックステキストの変更（1文字でも変わればミス）
- ツール定義の追加・削除・変更
- 画像コンテンツの追加・削除
- `tool_choice`パラメータの変更
- **キャッシュが維持される**: `max_tokens`、`temperature`の変更、キャッシュブレークポイント以降のメッセージ変更

## 運用での学び（Production Lessons）

### Notionの導入事例

Anthropic公式ブログによれば、NotionはNotion AI機能にプロンプトキャッシュを統合している。Notion共同創業者のSimon Last氏は、社内オペレーションの最適化に活用していると述べている。具体的な数値は公開されていないが、Notionのようなナレッジマネジメントツールでは「同一ワークスペースの文書に対する複数ユーザーからの質問」パターンが頻出するため、プロンプトキャッシュの恩恵が大きいと推測される。

### レート制限への効果

Anthropicおよび Zenn記事で言及されている通り、1時間キャッシュのヒットはレート制限にカウントされない。これは、社内RAGで朝のラッシュ時に多数のユーザーが同時にクエリする場面で、スループットのボトルネック回避に寄与する。

### 監視すべきメトリクス

本番運用では以下のメトリクスを追跡することが推奨される：

| メトリクス | 取得方法 | 目標値 |
|-----------|---------|--------|
| キャッシュヒット率 | `cache_read_input_tokens / (cache_read + cache_write + input_tokens)` | 80%以上 |
| キャッシュ書き込み頻度 | `cache_creation_input_tokens > 0` の割合 | 20%以下（ヒット率高ければ低下） |
| TTFT改善率 | キャッシュヒット時 vs ミス時のTTFT比較 | 50%以上の短縮 |
| コスト削減率 | `(cache_read_cost) / (non_cache_cost)` | 70%以上の削減 |

## 学術研究との関連（Academic Connection）

AnthropicのPrompt Caching機能は、Yale大学のGimらによるPrompt Cache論文（arXiv:2311.04934）の概念を商用実装したものと位置づけられる。主な差異は以下の通り：

| 観点 | 論文Prompt Cache | Claude API Prompt Caching |
|------|-----------------|--------------------------|
| 位置制約 | 任意位置（PMLで明示指定） | プレフィックス一致のみ |
| ユーザーインターフェース | PML（XMLライクスキーマ） | `cache_control`フィールド |
| TTL管理 | LRUエビクション | 5分 / 1時間の明示的TTL |
| インフラ | 単一GPU上のキャッシュ | クラウドインフラ上の分散キャッシュ |

Prompt Cache論文がプロンプト内の**任意位置**でのキャッシュ再利用を可能にしていたのに対し、Claude APIはプレフィックス一致に限定することで実装の単純化と信頼性を確保している。

## まとめと実践への示唆

AnthropicのClaude APIプロンプトキャッシュは、LLMアプリケーションのコスト最適化における実用的なソリューションである。損益分岐点が約2.2回と低く、社内RAGのようなユースケースでは条件が容易に満たされる。Zenn記事で解説されているBedrock Converse APIでの1時間TTLと組み合わせることで、クエリ間隔が不定なシナリオでもキャッシュの恩恵を享受できる。

実装にあたっては、静的コンテンツの先頭配置、キャッシュブレークポイントの戦略的配置、ドキュメント更新頻度の制御が重要である。

## 参考文献

- **Blog URL**: [https://www.anthropic.com/news/prompt-caching](https://www.anthropic.com/news/prompt-caching)
- **API Documentation**: [https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- **Related Paper (Prompt Cache)**: [https://arxiv.org/abs/2311.04934](https://arxiv.org/abs/2311.04934)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/d027acf4081b9d](https://zenn.dev/0h_n0/articles/d027acf4081b9d)
