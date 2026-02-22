---
layout: post
title: "Anthropic解説: Code Execution with MCP — プログラマティックツール呼び出しでトークン消費98.7%削減"
description: "Anthropicが提案するMCPサーバのコード実行統合パターンとTool Search機能によるトークン・レイテンシ最適化手法の解説"
categories: [blog, tech_blog]
tags: [MCP, Anthropic, tool-search, code-execution, latency, LLM]
date: 2026-02-22 12:00:00 +0900
source_type: tech_blog
source_domain: anthropic.com
source_url: https://www.anthropic.com/engineering/code-execution-with-mcp
zenn_article: 2929e45a5bf12b
zenn_url: https://zenn.dev/0h_n0/articles/2929e45a5bf12b
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Anthropic Engineering Blog: Code execution with MCP: building more efficient AI agents](https://www.anthropic.com/engineering/code-execution-with-mcp) の解説記事です。

## ブログ概要（Summary）

Anthropic のエンジニアリングチームは、MCP サーバと連携する LLM エージェントの効率化のために、**コード実行によるプログラマティックツール呼び出し**パターンを提案している。従来の「1 ツール呼び出し = 1 API ラウンドトリップ」モデルでは、20 以上のツール呼び出しが必要なタスクでは各呼び出しにモデル推論（数百ミリ秒〜数秒）が発生する。コード実行モードでは、LLM が 1 回のレスポンスで TypeScript コードを生成し、そのコード内で複数のツールをプログラマティックに呼び出すことで、推論パスの回数を大幅に削減する。ブログでは Google Drive から Salesforce へのトランスクリプト転送の例で、トークン消費を 150,000 から 2,000 に削減（98.7% 削減）した事例が紹介されている。

この記事は [Zenn記事: LangGraph×MCPツール呼び出しレイテンシ最適化：社内検索エージェントの応答を5倍速くする](https://zenn.dev/0h_n0/articles/2929e45a5bf12b) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://www.anthropic.com/engineering/code-execution-with-mcp](https://www.anthropic.com/engineering/code-execution-with-mcp)
- **組織**: Anthropic Engineering
- **発表日**: 2025

## 技術的背景（Technical Background）

### 従来のツール呼び出しモデルの制約

従来の LLM ツール呼び出しは、以下のループで動作する。

```
1. LLM が tool_call を生成
2. ランタイムがツールを実行
3. ツール結果を LLM に返却
4. LLM が次の tool_call を生成（または最終回答を出力）
```

このモデルでは、$n$ 回のツール呼び出しに $n$ 回の LLM 推論パスが必要となる。各推論パスには最低でも数百ミリ秒のレイテンシが発生するため、ツール呼び出しが多いタスクではレイテンシが線形に増加する。

$$
L_{\text{total}} = \sum_{i=1}^{n} (L_{\text{LLM},i} + L_{\text{tool},i})
$$

$n = 20$ で $L_{\text{LLM}} = 800\text{ms}$、$L_{\text{tool}} = 200\text{ms}$ の場合、$L_{\text{total}} = 20{,}000\text{ms} = 20$ 秒となる。

### コード実行モードの提案

Anthropic が提案するコード実行モードでは、LLM は 1 回の推論で TypeScript コードを生成し、そのコード内で複数のツールを呼び出す。

$$
L_{\text{code}} = L_{\text{LLM,generate}} + L_{\text{code\_exec}} + \max_{j=1}^{k} L_{\text{tool},j}
$$

ここで $L_{\text{code\_exec}}$ はコード実行のオーバーヘッド（通常 10-50ms）であり、ツール呼び出しは**コード内でプログラマティックに並列化**できる。$n = 20$ のツール呼び出しを 1 回のコード生成で処理すると、推論パスが $n$ 回から 1 回に削減される。

### MCPとの統合

Anthropic のブログでは、MCP サーバをコード実行環境から直接呼び出すパターンを提示している。

```typescript
// 従来: 各ツール呼び出しが個別のLLM推論パスを必要とする
// tool_call 1: gdrive.getDocument(...)  → LLM推論
// tool_call 2: salesforce.createNote(...) → LLM推論
// tool_call 3: slack.postMessage(...)     → LLM推論

// コード実行モード: 1回のLLM推論で全ツールを呼び出し
const transcript = (await gdrive.getDocument({documentId})).content;
const note = await salesforce.createNote({
    subject: "Meeting Summary",
    body: summarize(transcript),
});
await slack.postMessage({
    channel: "#team",
    text: `Notes created: ${note.url}`,
});
```

この例では 3 回の LLM 推論パスが 1 回に削減される。

## 実装アーキテクチャ（Architecture）

### Tool Search による動的ツール発見

Anthropic は Claude API に **Tool Search** 機能を導入している。これは MCP-Zero (arXiv:2503.23278) と類似のアプローチで、全ツール定義を LLM のコンテキストに含めるのではなく、タスクに応じて必要なツールのみを動的に取得する。

Tool Search の動作フロー:

1. エージェント起動時に `search_tools` ツールのみをバインド
2. タスク実行中、LLM が `search_tools` を呼び出してツールを発見
3. 発見されたツールの定義が動的にコンテキストに追加される
4. LLM がツールを呼び出す

**詳細レベル選択**:

| レベル | 返却情報 | トークンコスト |
|---|---|---|
| names_only | ツール名一覧 | 低い |
| basic | ツール名 + 説明 | 中程度 |
| full_schema | 完全な JSON Schema | 高い |

エージェントはまず `names_only` で関連ツールを特定し、必要なツールについてのみ `full_schema` を取得することで、段階的にコンテキストコストを最小化できる。

### ファイルシステムベースのスキル永続化

Anthropic のブログでは、コード実行環境に**ファイルシステムアクセス**を付与し、再利用可能な関数（スキル）を `./skills/` ディレクトリに保存するパターンが紹介されている。

```
./skills/
├── search_confluence.ts    # Confluence検索ラッパー
├── summarize_thread.ts     # Slackスレッド要約
└── format_report.ts        # レポート整形
```

一度生成されたスキルは後続のタスクで再利用され、LLM の推論負荷を削減する。これは Zenn 記事の「動的ツールローディング」を発展させた概念であり、ツール定義の取得だけでなく**使用パターンのキャッシング**まで含む。

### セキュリティ考慮事項

コード実行環境に MCP サーバへのアクセスを付与する場合、以下のセキュリティ対策がブログで言及されている。

- **PII 自動トークン化**: 個人情報を含むデータがコード実行環境内で自動的にトークン化され、LLM のコンテキストに平文で露出しない
- **サンドボックス実行**: コードはサンドボックス内で実行され、ファイルシステムアクセスは `./skills/` ディレクトリに制限される
- **中間結果の保持**: ツール実行結果はコード実行環境内に保持され、LLM のコンテキストには要約のみが返却される

## パフォーマンス最適化（Performance）

### トークン消費の劇的削減

Anthropic のブログで紹介された Google Drive → Salesforce トランスクリプト転送の事例:

| 方式 | トークン消費 | 推論パス数 |
|---|---|---|
| 従来のツール呼び出し | 150,000 tokens | 20+ 回 |
| コード実行モード | 2,000 tokens | 1 回 |
| **削減率** | **98.7%** | **95%** |

Anthropic のブログによると、この削減の主因は以下の 2 点である。

1. **推論パス削減**: 20 回の LLM 推論が 1 回に削減され、各推論のコンテキスト反復コストが排除される
2. **中間結果のコード内処理**: 大きなツール出力（例: ドキュメント全文）がコード変数に保持され、LLM のコンテキストに注入されない

### LLMコスト試算

月間 10 万リクエスト、リクエストあたり平均 10 ツール呼び出しの場合:

| 方式 | リクエストあたりトークン | 月間トークン | 月額コスト (Claude Sonnet) |
|---|---|---|---|
| 従来 | ~50,000 | 5B | ~$15,000 |
| コード実行 | ~5,000 | 500M | ~$1,500 |

**月額約 $13,500 のコスト削減**が見込まれる。ただし、この試算はツール出力のサイズとタスクの複雑さに大きく依存する。

### レイテンシ削減効果

推論パス数の削減はレイテンシにも直結する。

$$
L_{\text{conventional}} = n \times (L_{\text{prefill}} + L_{\text{decode}} + L_{\text{tool}}) \approx n \times 1{,}200\text{ms}
$$

$$
L_{\text{code\_exec}} = 1 \times (L_{\text{prefill}} + L_{\text{code\_gen}} + L_{\text{exec}}) \approx 2{,}500\text{ms}
$$

$n = 10$ の場合、$12{,}000\text{ms} \rightarrow 2{,}500\text{ms}$ で **79% のレイテンシ削減**となる。

## 運用での学び（Production Lessons）

### 制約事項

Anthropic のブログでは以下の制約が指摘されている。

**コード生成の正確性**: LLM が生成するコードにバグが含まれる可能性がある。特に複雑な制御フロー（ネストしたループ、例外処理）では生成品質が低下しうる。

**デバッグの困難さ**: コード実行モードでは中間ステップが不透明になり、従来の「1 ツール = 1 ステップ」のトレーサビリティが失われる。OpenTelemetry による計装が推奨される。

**ツールのAPI互換性**: コード実行環境からMCPサーバを呼び出す場合、TypeScript/JavaScript のランタイムとMCPクライアントSDKの互換性を確認する必要がある。

### 適用判断のガイドライン

| 条件 | 推奨方式 |
|---|---|
| ツール呼び出し 1-3 回 | 従来の tool_call |
| ツール呼び出し 4+ 回、独立 | コード実行（並列） |
| ツール呼び出し間に依存 | コード実行（直列） |
| 大量のデータ変換 | コード実行（変数内処理） |

## 学術研究との関連（Academic Connection）

### MCP-Zero との関係

Anthropic の Tool Search 機能は、MCP-Zero (arXiv:2503.23278) の Active Tool Discovery と概念的に同一のアプローチである。いずれも全ツール定義のコンテキスト注入を避け、必要なツールのみを動的に取得する。

| 側面 | Tool Search (Anthropic) | MCP-Zero (論文) |
|---|---|---|
| 取得方式 | API ネイティブ | embedding 検索 |
| 対象環境 | Claude API | 汎用 LLM |
| 取得粒度 | 段階的（name→schema） | 固定（top-k） |

### OctoTools との関係

コード実行モードの並列ツール呼び出しは、OctoTools (arXiv:2502.18145) の DAG 並列実行と同様の効果を達成するが、実現手段が異なる。OctoTools はフレームワーク側で DAG を管理するのに対し、Anthropic のアプローチは LLM 自身がコードで並列性を表現する。

### Prefix Sharing との関連

推論パス数の削減は、KV キャッシュの観点でも有利である。Prefix Sharing (arXiv:2408.01812) では、反復される system prompt + tool definitions の KV キャッシュを再利用するが、コード実行モードでは推論パス自体が 1 回なのでキャッシュの必要性が低下する。

## まとめと実践への示唆

Anthropic のコード実行 + MCP 統合パターンは、Zenn 記事で紹介した 5 層の最適化戦略に対して**6番目のレイヤー**として位置づけられる。特に、ツール呼び出しが 4 回以上のタスクでは、コード実行モードによる推論パス削減がレイテンシとコストの両面で大きな効果をもたらす。

ただし、コード生成の正確性やデバッグの困難さという制約があるため、本番環境への導入にはサンドボックス実行、エラーハンドリング、および OpenTelemetry による計装が不可欠である。

## 参考文献

- **Blog URL**: [https://www.anthropic.com/engineering/code-execution-with-mcp](https://www.anthropic.com/engineering/code-execution-with-mcp)
- **Advanced Tool Use**: [https://www.anthropic.com/engineering/advanced-tool-use](https://www.anthropic.com/engineering/advanced-tool-use)
- **MCP Specification**: [https://modelcontextprotocol.io/specification/2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/2929e45a5bf12b](https://zenn.dev/0h_n0/articles/2929e45a5bf12b)
