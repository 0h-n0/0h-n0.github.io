---
layout: post
title: "LangChain公式解説: 知識グラフ構築×ハイブリッド検索でRAG精度を向上させる実装パターン"
description: "LangChainブログのKG強化RAG実装を詳細解説。LLMGraphTransformerによるKG自動構築、Neo4jベクトル＋キーワード＋グラフの3系統ハイブリッド検索の実装手法を分析する"
categories: [blog, tech_blog]
tags: [LangChain, knowledge-graph, Neo4j, RAG, hybrid-search, LLMGraphTransformer]
date: 2026-02-22 20:40:00 +0900
source_type: tech_blog
source_domain: blog.langchain.com
source_url: https://blog.langchain.com/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/
zenn_article: f894fb3fa04a59
zenn_url: https://zenn.dev/0h_n0/articles/f894fb3fa04a59
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

LangChainの公式ブログでは、非構造化テキストからLLMを用いて知識グラフ（KG）を自動構築し、ベクトル検索＋キーワード検索＋グラフトラバーサルの3系統ハイブリッド検索でRAG精度を向上させる実装パターンが紹介されている。Neo4jをバックエンドとし、`LLMGraphTransformer`によるエンティティ・関係抽出から`Neo4jVector`によるベクトルインデックス、構造化出力によるエンティティ認識まで、一貫した実装が解説されている。

本記事は [Zenn記事: LangGraph×GraphRAGハイブリッド検索で社内文書の複合質問精度を向上させる](https://zenn.dev/0h_n0/articles/f894fb3fa04a59) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（LangChain公式）
- **URL**: [https://blog.langchain.com/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/](https://blog.langchain.com/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/)
- **組織**: LangChain / Neo4j（共同執筆）
- **発表日**: 2024年

## 技術的背景（Technical Background）

LangChainのブログでは、RAGの検索精度を向上させるために知識グラフを活用する動機を以下のように説明している。

1. **構造化された関係性の活用**: テキストチャンクのベクトル類似度だけでなく、エンティティ間の構造化された関係（「所属」「担当」「使用」等）をたどることで、関係性クエリ（「AがBに所属し、Bの部署長は誰か」）に正確に回答できる。
2. **文脈の保持**: チャンク分割により失われるドキュメント間の文脈情報を、KGのエッジとして保持することで、cross-documentの質問にも対応できる。
3. **Neo4jの統合優位性**: ベクトルインデックスとグラフDB機能を同一のNeo4jインスタンスで提供するため、外部ベクトルDBとの同期問題が不要になる。

このブログは、Zenn記事で紹介されている`LLMGraphTransformer`と`Neo4jVector`の使用方法の**原典**であり、Zenn記事のGraphRAG実装の基盤となっている技術情報源である。

## 実装アーキテクチャ（Architecture）

### Step 1: LLMGraphTransformerによるKG自動構築

ブログでは、`LLMGraphTransformer`を用いてテキストからエンティティと関係を自動抽出する実装が紹介されている。

```python
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph

# LLM設定（温度0で決定論的な抽出）
llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview")

# GraphTransformer初期化
llm_transformer = LLMGraphTransformer(llm=llm)

# テキスト → グラフドキュメント変換
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Neo4jへの格納
graph = Neo4jGraph()
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,     # 全ノードに __Entity__ ラベル追加
    include_source=True,      # ソースドキュメントへのリンク保持
)
```

**`baseEntityLabel=True`の重要性**: ブログでは、このパラメータの設定により全ノードに`__Entity__`ラベルが追加され、ノードタイプに依存しない横断的なインデックス作成が可能になると説明されている。これにより、ベクトルインデックスを特定のノードラベルに限定せず、全エンティティに対して統一的に構築できる。

**`include_source=True`の意義**: エンティティからソースドキュメントへのリンクを保持することで、回答生成時にソースの引用・トレーサビリティを確保できる。

### Step 2: Neo4jベクトルインデックスの構築

ブログでは、`Neo4jVector.from_existing_graph`を使用して、既存のグラフノードにベクトルインデックスを追加する方法が紹介されている。

```python
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings

# 既存グラフノードにベクトルインデックスを作成
vector_index = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    search_type="hybrid",        # ベクトル + キーワードのハイブリッド
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)
```

**`search_type="hybrid"`**: ブログでは、このパラメータによりベクトル類似度検索とBM25キーワード検索の両方が有効化されると説明されている。Neo4j 5.xのネイティブ全文インデックスとベクトルインデックスが同時に利用される。

### Step 3: エンティティベースのグラフリトリーバー

ブログの核心部分は、クエリからエンティティを認識し、そのエンティティの近傍グラフをリトリーバルに活用する実装である。

**構造化出力によるエンティティ認識**:

```python
from pydantic import BaseModel, Field
from typing import List

class Entities(BaseModel):
    """クエリ中のエンティティ"""
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities "
                    "appearing in the text",
    )

# LLMの構造化出力でエンティティを抽出
entity_chain = prompt | llm.with_structured_output(Entities)
```

**ファジーマッチングによるグラフ検索**:

ブログでは、抽出されたエンティティ名でNeo4j内のノードを検索する際、**ファジーマッチング**（Levenshtein距離2文字以内の許容）を使用して表記ゆれに対応する実装が紹介されている。

```python
# Cypherクエリ: ファジーマッチングで近傍取得
cypher_query = """
CALL db.index.fulltext.queryNodes('entity', $query, {limit: 2})
YIELD node, score
CALL {
  WITH node
  MATCH (node)-[r]->(neighbor)
  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
  UNION ALL
  WITH node
  MATCH (node)<-[r]-(neighbor)
  RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
}
RETURN output LIMIT 50
"""
```

このCypherクエリは、エンティティの全文インデックスでファジーマッチを行い、マッチしたノードの1ホップ近傍（入出両方向）を取得する。これにより、「田中」で検索しても「田中太郎」にマッチし、その近傍の関係性情報（所属部署、担当プロジェクト等）が取得される。

### Step 4: RAGチェーンの統合

ブログでは、`RunnableParallel`を使用してベクトル検索とグラフ検索を並列実行し、結果を統合するRAGチェーンが紹介されている。

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ベクトル検索 + グラフ検索の並列実行
retriever = RunnableParallel(
    {
        "vector_context": vector_index.as_retriever(),
        "graph_context": graph_retriever,  # エンティティベースの近傍取得
    }
)

# コンテキスト統合 → LLM生成
chain = (
    RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    | prompt
    | llm
    | StrOutputParser()
)
```

**会話履歴の処理**: ブログでは、フォローアップ質問（「彼はどこに住んでいますか？」のような代名詞を含む質問）に対処するため、会話履歴を用いてクエリを書き換えるリライトチェーンも紹介されている。これはZenn記事の`rewrite_query`関数と同等の機能である。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $70-180 | Lambda + Bedrock + Neptune Serverless |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $400-1,000 | ECS Fargate + Neptune + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,500-6,000 | EKS + Neptune + OpenSearch |

LangChainのブログで紹介されているアーキテクチャは、ベクトル検索とグラフ検索を同一のNeo4j（AWS上はNeptune）で実行するため、外部ベクトルDBのコストが不要である点がコスト効率の源泉となる。

**コスト試算の注意事項**: 上記は2026年2月時点のAWS ap-northeast-1料金に基づく概算値です。最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください。

### Terraformインフラコード

```hcl
# --- Neptune Serverless (KG + ベクトルインデックス統合) ---
resource "aws_neptune_cluster" "langchain_graphrag" {
  cluster_identifier  = "langchain-kg-rag"
  engine              = "neptune"
  serverless_v2_scaling_configuration {
    min_capacity = 1.0
    max_capacity = 4.0
  }
  storage_encrypted = true
}

# --- Lambda (ハイブリッド検索) ---
resource "aws_lambda_function" "hybrid_search" {
  function_name = "langchain-hybrid-search"
  role          = aws_iam_role.lambda_hybrid.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 30
  memory_size   = 1536

  environment {
    variables = {
      NEPTUNE_ENDPOINT = aws_neptune_cluster.langchain_graphrag.endpoint
      BEDROCK_MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"
      SEARCH_TYPE      = "hybrid"
      FUZZY_DISTANCE   = "2"
    }
  }
}

# --- IAMロール（最小権限） ---
resource "aws_iam_role_policy" "lambda_hybrid_policy" {
  role = aws_iam_role.lambda_hybrid.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["neptune-db:ReadDataViaQuery", "neptune-db:WriteDataViaQuery"]
        Resource = aws_neptune_cluster.langchain_graphrag.arn
      },
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = "arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.claude-3-5-haiku*"
      }
    ]
  })
}
```

### コスト最適化チェックリスト

- [ ] Neptune統合: ベクトルDB不要（コスト30%削減）
- [ ] Neptune Serverless: 最小NCU設定
- [ ] Bedrock: Haiku利用でエンティティ抽出コスト最小化
- [ ] Prompt Caching: KGスキーマプロンプトのキャッシュ
- [ ] ファジーマッチング: 全文インデックス利用（追加コスト不要）
- [ ] 差分更新: LLMGraphTransformerのインクリメンタル実行
- [ ] AWS Budgets: 月額予算設定
- [ ] CloudWatch: Neptune NCU使用量の監視

## パフォーマンス最適化（Performance）

ブログでは定量的なベンチマーク結果は提供されていないが、以下の定性的な観察が述べられている。

- **ベクトル検索のみ vs ハイブリッド**: エンティティ間の関係性クエリ（「AはBの部署に所属し、Bの部署長は誰か」）において、ベクトル検索のみでは正確な回答が得られないが、グラフ検索の追加により正答率が向上する。
- **ファジーマッチングの効果**: 表記ゆれ（「田中太郎」「田中」「Tanaka」）に対して、Levenshtein距離2文字以内のファジーマッチングにより、厳密一致では取りこぼすエンティティを補完できる。
- **並列実行によるレイテンシ**: `RunnableParallel`によりベクトル検索とグラフ検索を並列実行することで、直列実行と比較してレイテンシを約50%削減できる（ブログの定性的記述に基づく）。

## 運用での学び（Production Lessons）

ブログおよびNeo4jの関連ドキュメントから得られる実践的な知見は以下の通りである。

1. **`baseEntityLabel=True`の必須性**: このパラメータを省略すると、ノードタイプごとに個別のインデックスが必要になり、横断的な検索が困難になる。ブログでは「best practice」として常にTrueに設定することを推奨している。
2. **全文インデックスの事前作成**: ファジーマッチングに必要な全文インデックスは、`LLMGraphTransformer`によるグラフ構築後に明示的に作成する必要がある。自動作成されないため、初期セットアップ時の見落としに注意が必要である。
3. **チャンクサイズとKG品質のトレードオフ**: チャンクサイズが大きすぎると、LLMがエンティティ・関係を見落とす確率が上がる。ブログでは2000トークン程度のチャンクサイズが推奨されている。

## 学術研究との関連（Academic Connection）

- **LLMGraphTransformer**: Neo4jが開発し、LangChainの`langchain_experimental`パッケージに統合されたツール。テキストからのKG自動構築をLLMで行うアプローチは、IE（Information Extraction）分野の長年の課題に対するLLMベースの解法として位置付けられる。
- **GraphCypherQAChain**: LangChainの`langchain_community`パッケージに含まれる、自然言語→Cypher変換によるグラフQAチェーン。ブログではこのチェーンの代わりに、ファジーマッチング＋近傍取得のカスタム実装を使用しており、text2Cypherの精度問題を回避している。

## まとめと実践への示唆

LangChainの公式ブログは、知識グラフ強化RAGの**リファレンス実装**として位置付けられる。`LLMGraphTransformer`によるKG自動構築、`Neo4jVector`によるハイブリッドインデックス、構造化出力によるエンティティ認識、ファジーマッチングによるグラフ検索という一連の実装パターンは、Zenn記事のGraphRAGハイブリッド検索の直接的な基盤である。

特に、ベクトル検索とグラフ検索の並列実行を`RunnableParallel`で実現するパターンは、LangGraphの`StateGraph`への移行時に`search_vector`と`search_graph`ノードの並列実行に直接対応する。LangGraphによるワークフロー制御（ルーティング、自己修正ループ）の追加は、このブログの実装を拡張する自然な次のステップである。

## 参考文献

- **Blog URL**: [https://blog.langchain.com/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/](https://blog.langchain.com/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/)
- **Neo4j GraphRAG Documentation**: [https://neo4j.com/docs/neo4j-graphrag-python/current/](https://neo4j.com/docs/neo4j-graphrag-python/current/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/f894fb3fa04a59](https://zenn.dev/0h_n0/articles/f894fb3fa04a59)
