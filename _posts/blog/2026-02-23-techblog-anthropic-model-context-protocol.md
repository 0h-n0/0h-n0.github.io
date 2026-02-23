---
layout: post
title: "Anthropic解説: Model Context Protocol (MCP) — AIアシスタントと外部データを接続する標準プロトコル"
description: "AnthropicがオープンソースとしてリリースしたMCPの技術アーキテクチャと設計思想を詳細解説"
categories: [blog, tech_blog]
tags: [MCP, Anthropic, LLM, agent, protocol, malleable-software]
date: 2026-02-23 10:00:00 +0900
source_type: tech_blog
source_domain: anthropic.com
source_url: https://www.anthropic.com/news/model-context-protocol
zenn_article: c0712fa2cd13b2
zenn_url: https://zenn.dev/0h_n0/articles/c0712fa2cd13b2
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

2024年11月25日、Anthropicは**Model Context Protocol（MCP）**を発表した。MCPは、AIアシスタントと外部データソース・ツールを接続するためのオープンスタンダードであり、コンテンツリポジトリ、ビジネスツール、開発環境など、データが存在するシステムとAIを統合するための標準的な方法を提供する。従来の断片的なインテグレーションを単一のプロトコルに置き換えることで、AIシステムが必要なデータにより信頼性高くアクセスできるようになることを目的としている。

本記事は [Zenn記事: Malleable Softwareとは何か——LLM時代の動的ソフトウェアとその実現要件](https://zenn.dev/0h_n0/articles/c0712fa2cd13b2) の深掘り記事です。元のブログ記事は [Anthropic News: Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) で公開されています。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://www.anthropic.com/news/model-context-protocol](https://www.anthropic.com/news/model-context-protocol)
- **組織**: Anthropic
- **発表日**: 2024年11月25日

## 技術的背景（Technical Background）

AIアシスタントの能力が向上しても、その有用性は利用可能なコンテキスト（情報）に制約される。高度なLLMでさえ、必要なデータが組織内の異なるシステムに分散している場合、その力を十分に発揮できない。Anthropicはこの問題を「情報の孤立（information isolation）」と表現している。

従来のアプローチでは、各データソースごとに個別のインテグレーションを構築する必要があった。例えば、AIアシスタントがGitHubのコード、Slackのメッセージ、PostgreSQLのデータベースにアクセスする場合、それぞれ別々のコネクタを開発・保守する必要があった。$N$個のAIアプリケーションと$M$個のデータソースがある場合、最大$N \times M$個のカスタムインテグレーションが必要となる。

$$
\text{従来のインテグレーション数} = O(N \times M)
$$

MCPはこの問題を、USBの比喩で説明している。USBが多様なデバイスを標準的なインターフェースで接続したように、MCPは多様なAIアプリケーションとデータソースを標準プロトコルで接続する。

$$
\text{MCPによるインテグレーション数} = O(N + M)
$$

この計算量の削減は、Malleable Softwareの実現において重要な意味を持つ。ユーザーがソフトウェアを自由にカスタマイズするには、異なるツール間のデータ相互運用性が前提条件となるためである。

## 実装アーキテクチャ（Architecture）

### MCPの基本構成

MCPは**クライアント・サーバーモデル**を採用している。Anthropicのブログ記事によると、開発者は以下の2つの方法でMCPを利用できる。

1. **MCPサーバー**: 自身のデータをMCPプロトコルで公開する
2. **MCPクライアント**: MCPサーバーに接続するAIアプリケーションを構築する

```
┌──────────────────────────────────────────────────────┐
│                   MCP Architecture                    │
│                                                      │
│  ┌──────────┐                      ┌──────────────┐  │
│  │  MCP     │  JSON-RPC 2.0 over  │  MCP Server  │  │
│  │  Client  │◄────────────────────►│  (GitHub)    │  │
│  │          │   stdio / SSE        │              │  │
│  └──────────┘                      └──────────────┘  │
│       │                                              │
│       │  JSON-RPC 2.0             ┌──────────────┐   │
│       └──────────────────────────►│  MCP Server  │   │
│                                   │  (Slack)     │   │
│  ┌──────────┐                     └──────────────┘   │
│  │  MCP     │                                        │
│  │  Host    │  JSON-RPC 2.0       ┌──────────────┐   │
│  │ (Claude  │────────────────────►│  MCP Server  │   │
│  │ Desktop) │                     │  (Postgres)  │   │
│  └──────────┘                     └──────────────┘   │
└──────────────────────────────────────────────────────┘
```

### 3つのコアコンポーネント

MCPの仕様は3つの主要な抽象化を提供している。

**1. Resources（リソース）**: MCPサーバーが公開するデータ。ファイルの内容、データベースのスキーマ、APIレスポンスなどが該当する。クライアントは`resources/list`と`resources/read`の2つのメソッドでリソースにアクセスする。

**2. Tools（ツール）**: MCPサーバーが公開する操作。ファイルの作成、データベースのクエリ実行、APIの呼び出しなどが該当する。各ツールはJSON Schemaで入力パラメータを定義する。

**3. Prompts（プロンプト）**: MCPサーバーが提供する再利用可能なプロンプトテンプレート。特定のドメイン知識をカプセル化し、クライアント間で共有可能にする。

```python
from mcp.server import Server
from mcp.types import Resource, Tool

server = Server("example-server")

@server.list_resources()
async def list_resources() -> list[Resource]:
    """利用可能なリソースを返す"""
    return [
        Resource(
            uri="file:///project/config.yaml",
            name="Project Configuration",
            mimeType="application/yaml",
        )
    ]

@server.list_tools()
async def list_tools() -> list[Tool]:
    """利用可能なツールを返す"""
    return [
        Tool(
            name="query_database",
            description="SQLクエリを実行する",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query"}
                },
                "required": ["query"],
            },
        )
    ]
```

### 通信プロトコル

MCPはJSON-RPC 2.0をワイヤプロトコルとして使用する。トランスポート層は2つのモードをサポートしている。

- **stdio**: ローカル環境でのサーバー実行（標準入出力経由）
- **SSE（Server-Sent Events）**: リモートサーバーへのHTTPベースの接続

### 事前構築済みサーバー

Anthropicによると、初期リリース時点で以下のサービスに対するMCPサーバーが提供されている。

| サービス | 機能 | 用途 |
|---------|------|------|
| GitHub | リポジトリ操作、PR管理 | コードレビュー・開発支援 |
| Google Drive | ファイル検索・読み取り | ドキュメント分析 |
| Slack | メッセージ検索・送信 | チームコラボレーション |
| Postgres | SQLクエリ実行 | データ分析 |
| Puppeteer | Webスクレイピング | 情報収集 |
| Git | リポジトリ操作 | バージョン管理 |

## Malleable Softwareとの関連（Connection to Malleable Software）

MCPは、Zenn記事で紹介されているMalleable Softwareの設計原則と密接に関連している。以下にその対応関係を整理する。

### 「Tools over Applications」原則との対応

Ink & Switchの「Tools over Applications」パターンは、モノリシックなアプリではなく組み合わせ可能なツールを提唱している。MCPは、この原則を技術的に実現するための標準インターフェースを提供する。各MCPサーバーが独立したツールとして機能し、クライアント側で自由に組み合わせることが可能になる。

### 「Composability（合成可能性）」の実現

MCPの$O(N + M)$モデルは、Geoffrey Littが提唱する「合成可能性」の技術的基盤と位置づけられる。異なるツールが共通のデータ層を通じて相互運用できるため、ユーザーは既存のツールを組み合わせて独自のワークフローを構築できる。

```python
# MCPによるツール合成の概念例
class MalleableWorkspace:
    """MCPサーバーを組み合わせて個人ワークスペースを構築"""

    def __init__(self):
        self.servers: list[MCPClient] = []

    async def add_tool(self, server_uri: str) -> None:
        """MCPサーバーを動的に追加"""
        client = MCPClient(server_uri)
        await client.connect()
        self.servers.append(client)

    async def list_all_capabilities(self) -> dict:
        """全サーバーの機能を統合リスト"""
        capabilities = {}
        for server in self.servers:
            tools = await server.list_tools()
            resources = await server.list_resources()
            capabilities[server.name] = {
                "tools": tools,
                "resources": resources,
            }
        return capabilities
```

### エコシステムの成長

2025年から2026年にかけて、MCPのエコシステムは急速に拡大した。MCP仕様サイトによると、月間SDK ダウンロード数は9,700万以上に達し、Anthropicだけでなく、OpenAI、Google、Microsoftを含む主要なAI企業がサポートを表明している。2025年12月には、AnthropicがMCPをLinux Foundation傘下のAgentic AI Foundation（AAIF）に寄贈した。

開発ツール企業のZed、Replit、Codeium、Sourcegraphも早期にMCPを統合しており、AIエージェントがコンテキスト情報を取得してより機能的なコードを生成できるようにしている。

## パフォーマンス最適化（Performance）

### レイテンシに関する考慮事項

MCPの設計における重要な考慮事項の一つはレイテンシである。ローカルのstdioトランスポートでは通常1-10msのオーバーヘッドで済むが、リモートSSEトランスポートではネットワークレイテンシが加算される。

MCPサーバーの実装においては、以下の最適化が推奨されている。

- **リソースのキャッシュ**: 頻繁にアクセスされるリソースはサーバー側でキャッシュする
- **バッチリクエスト**: JSON-RPC 2.0のバッチ機能を活用して複数リクエストをまとめる
- **非同期処理**: ツール実行は非同期で行い、長時間の処理をブロックしない

### セキュリティモデル

MCPは、ユーザーの承認なしにツールを実行しない設計となっている。ツール呼び出しは「提案」として扱われ、MCPホスト（Claude Desktop等）がユーザーに確認を求める。

## 運用での学び（Production Lessons）

### 早期採用企業の事例

Anthropicのブログによると、BlockとApolloがMCPの早期採用企業として紹介されている。Blockは決済プラットフォームのAPIをMCPサーバーとして公開し、AIアシスタントが決済データにアクセスできるようにしている。

### 課題と制約

MCPの現時点での課題として以下が挙げられる。

1. **認証・認可の標準化**: 初期リリースではOAuth2統合が限定的。企業環境でのデプロイには追加のセキュリティ層が必要
2. **スキーマのバージョニング**: MCPサーバーのスキーマが変更された場合のクライアント側の互換性管理
3. **ディスカバリ**: 利用可能なMCPサーバーを動的に発見する仕組みの標準化

## 学術研究との関連（Academic Connection）

MCPは、学術的にはいくつかの研究潮流と関連している。

- **Agent-Tool Interaction**: LLMエージェントがツールを使用する際のインターフェース設計に関する研究（ToolLLM, Gorilla等）との関連。MCPはこれらの研究成果を産業レベルの標準として実装したものと位置づけられる
- **Semantic Web / Linked Data**: W3Cが推進してきたデータの相互運用性に関する取り組みとの類似性。MCPはAIアプリケーション特化のセマンティックレイヤーと見なすことができる
- **End-User Programming**: MCPはZenn記事で紹介されているMalleable Softwareの「Communal Creation」パターンの技術的基盤を提供する可能性がある

## まとめと実践への示唆

MCPは、AIアシスタントと外部データの接続を標準化するプロトコルとして、Malleable Softwareの実現に向けた重要なインフラ層を提供している。主要な示唆は以下の通りである。

1. **$O(N \times M)$から$O(N + M)$へ**: インテグレーションの計算量削減により、多様なツールの組み合わせが実用的になる
2. **クライアント・サーバーモデル**: データの提供者（サーバー）と消費者（クライアント）を分離することで、ツールの合成可能性を実現する
3. **エコシステムの急速な拡大**: 主要AI企業の支持により、事実上の標準としての地位を確立しつつある

Malleable Softwareを実現するためのツール設計においては、MCPへの対応を前提とした設計が推奨される。

## 参考文献

- **Blog URL**: [https://www.anthropic.com/news/model-context-protocol](https://www.anthropic.com/news/model-context-protocol)
- **MCP仕様**: [https://modelcontextprotocol.io](https://modelcontextprotocol.io)
- **GitHub**: [https://github.com/modelcontextprotocol](https://github.com/modelcontextprotocol)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/c0712fa2cd13b2](https://zenn.dev/0h_n0/articles/c0712fa2cd13b2)
