---
layout: post
title: "Anthropic解説: Contextual Retrieval — チャンク文脈付与で検索失敗率67%削減"
description: "Anthropicが提案するContextual Embeddings + Contextual BM25によるRAG検索精度改善手法を詳細解説し、マルチソースRAGへの適用方法を分析"
categories: [blog, tech_blog]
tags: [RAG, chunking, embeddings, BM25, Anthropic, Claude, langgraph, python, retrieval]
date: 2026-02-21 13:00:00 +0900
source_type: tech_blog
source_domain: anthropic.com
source_url: https://www.anthropic.com/news/contextual-retrieval
zenn_article: e4a4b18478c692
zenn_url: https://zenn.dev/0h_n0/articles/e4a4b18478c692
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

Anthropicが2024年9月に公開した「Introducing Contextual Retrieval」は、RAGにおけるチャンキングの根本的な課題—**チャンク分割時にコンテキスト情報が失われる問題**—を解決する手法を提案している。Contextual Embeddings（文脈付き埋め込み）とContextual BM25（文脈付きキーワード検索）の2つのサブ技術により、検索失敗率を最大67%削減した。さらにリランキングを組み合わせることで、従来RAGのリコールを大幅に改善した。

この記事は [Zenn記事: LangGraphマルチソースRAGの本番構築：権限制御×HITLで社内検索を安全運用](https://zenn.dev/0h_n0/articles/e4a4b18478c692) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://www.anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval)
- **組織**: Anthropic
- **著者**: Daniel Ford（et al.）
- **発表日**: 2024年9月19日

## 技術的背景（Technical Background）

### チャンキングの根本的問題

RAGパイプラインでは、長い文書を数百トークン単位のチャンクに分割してベクトルDBに格納する。しかし、この分割プロセスで**チャンクが元の文書内でどのような文脈に位置していたか**の情報が失われる。

例: 財務報告書のチャンク「Q2の売上は前年比15%増の$5Mでした。」

このチャンクだけでは、以下の情報が欠落している。
- **どの会社の**財務報告か
- **何年の**Q2か
- **どの事業部門の**売上か

従来のRAGでは、このような文脈欠落チャンクがベクトル化され、検索時に「ACME社の2024年Q2売上」というクエリに対して適切にマッチしない。これが**検索失敗（retrieval failure）**の主要因である。

### Zenn記事のチャンキング設計との関連

Zenn記事では、ソースタイプ別のチャンキング戦略を以下のように設計している。

| ソース | チャンク戦略 | サイズ |
|--------|-----------|--------|
| Confluence | 見出し構造保持型 | 512トークン（見出し単位） |
| Slack | スレッド単位 | 最大1024トークン |
| Google Drive | 段落単位 | 512トークン |

Contextual Retrievalは、これらのチャンキング戦略に**文脈アノテーション**を追加するレイヤーとして機能する。チャンキング戦略自体を変更するのではなく、既存のチャンクに文脈情報を付加する。

## 実装アーキテクチャ（Architecture）

### Contextual Retrievalの2つのサブ技術

#### 1. Contextual Embeddings（文脈付き埋め込み）

各チャンクに対して、LLM（Claude 3 Haiku）を使って**チャンクが元の文書内でどのような位置づけか**を説明する短いテキスト（50-100トークン）を生成し、チャンクの前に付加してから埋め込みベクトルを計算する。

```python
from anthropic import Anthropic

client = Anthropic()

CONTEXT_PROMPT = """<document>
{whole_document}
</document>

<chunk>
{chunk_content}
</chunk>

上記のチャンクについて、文書全体の文脈を踏まえた短い説明文を生成してください。
このチャンクが文書内でどのような位置づけにあるか、50-100トークンで説明してください。"""


def generate_context(
    whole_document: str,
    chunk_content: str,
) -> str:
    """チャンクの文脈説明を生成

    Args:
        whole_document: 元の文書全体
        chunk_content: チャンクの内容

    Returns:
        チャンクの文脈説明テキスト（50-100トークン）
    """
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": CONTEXT_PROMPT.format(
                whole_document=whole_document,
                chunk_content=chunk_content,
            ),
        }],
    )
    return response.content[0].text


def create_contextual_chunk(
    whole_document: str,
    chunk_content: str,
    chunk_metadata: dict,
) -> dict:
    """文脈付きチャンクを作成

    Args:
        whole_document: 元の文書全体
        chunk_content: チャンクの内容
        chunk_metadata: チャンクのメタデータ

    Returns:
        文脈情報が付加されたチャンク
    """
    context = generate_context(whole_document, chunk_content)

    # 文脈をチャンクの前に付加
    contextual_content = f"{context}\n\n{chunk_content}"

    return {
        "content": contextual_content,
        "original_content": chunk_content,
        "context": context,
        "metadata": chunk_metadata,
    }
```

**出力例**:

```
[文脈]: このチャンクはACME社の2024年度第2四半期財務報告書から抽出されたもので、
北米事業部門の売上実績について記述しています。

[元のチャンク]: Q2の売上は前年比15%増の$5Mでした。新規顧客獲得が主因です。
```

#### 2. Contextual BM25（文脈付きキーワード検索）

同じ文脈情報をBM25インデックスにも適用する。これにより、キーワードベースの検索でも文脈情報を活用でき、「ACME社」「2024年」「北米」などのキーワードでの検索ヒット率が向上する。

```python
from rank_bm25 import BM25Okapi


class ContextualBM25Index:
    """文脈付きBM25インデックス

    通常のBM25に加え、文脈情報を含めたインデックスを構築する。
    """

    def __init__(self, contextual_chunks: list[dict]):
        """インデックスを構築

        Args:
            contextual_chunks: create_contextual_chunkの出力リスト
        """
        self.chunks = contextual_chunks
        # 文脈付きコンテンツでBM25インデックスを構築
        tokenized = [
            chunk["content"].lower().split()
            for chunk in contextual_chunks
        ]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """文脈付きBM25検索

        Args:
            query: 検索クエリ
            top_k: 返却する文書数

        Returns:
            スコア付きチャンクリスト
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            {**self.chunks[i], "bm25_score": scores[i]}
            for i in top_indices
        ]
```

### ハイブリッド検索 + リランキング

Anthropicの推奨構成は、Contextual Embeddings + Contextual BM25 + リランキングの3段階パイプラインである。

```python
from langchain_voyageai import VoyageAIEmbeddings
from langchain_postgres import PGVectorStore


class ContextualHybridRetriever:
    """Contextual Retrievalによるハイブリッド検索

    Anthropicの推奨構成:
    1. Contextual Embeddings（セマンティック検索）
    2. Contextual BM25（キーワード検索）
    3. Reciprocal Rank Fusionによるスコア統合
    4. リランキング（オプション）
    """

    def __init__(
        self,
        vector_store: PGVectorStore,
        bm25_index: ContextualBM25Index,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def search(
        self,
        query: str,
        top_k: int = 20,
        rerank: bool = True,
    ) -> list[dict]:
        """ハイブリッド検索を実行

        Args:
            query: 検索クエリ
            top_k: 返却する文書数
            rerank: リランキングを実行するか

        Returns:
            検索結果リスト
        """
        # 1. Contextual Embeddings検索
        embedding_results = self.vector_store.similarity_search(
            query, k=top_k * 3  # 初期検索は多めに
        )

        # 2. Contextual BM25検索
        bm25_results = self.bm25_index.search(query, top_k=top_k * 3)

        # 3. RRFによるスコア統合
        fused = self._reciprocal_rank_fusion(
            embedding_results, bm25_results, k=60
        )

        if rerank:
            # 4. リランキング（上位150件→上位20件）
            return self._rerank(query, fused[:150])[:top_k]
        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self, emb_results, bm25_results, k: int = 60
    ) -> list[dict]:
        """RRFによるスコア統合"""
        scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for rank, doc in enumerate(emb_results):
            doc_id = doc.metadata.get("id", str(rank))
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc

        for rank, doc in enumerate(bm25_results):
            doc_id = doc.get("metadata", {}).get("id", f"bm25_{rank}")
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        return [doc_map[did] for did in sorted_ids if did in doc_map]

    def _rerank(self, query: str, docs: list) -> list:
        """Cross-encoderリランキング"""
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        pairs = [
            (query, getattr(doc, 'page_content', doc.get('content', '')))
            for doc in docs
        ]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked]
```

## パフォーマンス最適化（Performance）

### Anthropicの公式ベンチマーク結果

| 構成 | 検索失敗率 | 改善率 |
|------|----------|--------|
| 標準RAG（ベースライン） | 5.7% | — |
| Contextual Embeddings | 3.7% | **-35%** |
| Contextual Embeddings + BM25 | 2.9% | **-49%** |
| **Contextual Embeddings + BM25 + Reranking** | **1.9%** | **-67%** |

**評価指標**: 1 - Recall@20（上位20件に正解チャンクが含まれない率）

### ドメイン別の効果

Anthropicは以下の4ドメインでテストを実施している。

| ドメイン | ベースライン | Contextual Retrieval | 改善率 |
|---------|------------|---------------------|--------|
| コードベース | 8.2% | 2.1% | -74% |
| フィクション | 4.3% | 1.8% | -58% |
| arXiv論文 | 5.1% | 2.0% | -61% |
| 科学論文 | 5.2% | 1.7% | -67% |

コードベースでの改善率が最も高い（-74%）。これはコードチャンクが関数名・変数名のみで、クラス名やモジュール名の文脈が失われやすいためである。

### コスト効率

Prompt Cachingを使用した場合の文脈生成コスト:

$$
\text{Cost}_{\text{contextual}} = \frac{\$1.02}{\text{1M document tokens}}
$$

つまり、100万トークン分の文書（約2,500ページ）に文脈を付加するコストは**わずか$1.02**である。これは一回限りの前処理コストであり、検索精度の大幅な改善に対して極めて安価である。

### 最適パラメータ

Anthropicの実験から判明した最適パラメータ:

| パラメータ | 推奨値 | 根拠 |
|-----------|--------|------|
| チャンク取得数（top-K） | 20 | パフォーマンスとノイズのバランス |
| 初期検索数（リランク前） | 150 | 十分な候補を確保 |
| 文脈トークン数 | 50-100 | 簡潔さと情報量のバランス |
| 埋め込みモデル | Voyage, Gemini | Anthropicの実験で最高精度 |

## 運用での学び（Production Lessons）

### マルチソースRAGへの適用

Contextual Retrievalをマルチソース環境に適用する場合、ソースタイプごとの文脈生成プロンプトをカスタマイズすることが重要である。

```python
# ソースタイプ別の文脈生成プロンプト
SOURCE_CONTEXT_PROMPTS = {
    "confluence": """このチャンクはConfluence文書「{title}」のセクション「{section}」から
抽出されました。スペース: {space_key}、最終更新: {updated_at}。
チャンクの文脈を50-100トークンで説明してください。""",

    "slack": """このチャンクはSlackチャンネル #{channel} のスレッド
（開始日: {thread_date}）から抽出されました。参加者: {participants}。
チャンクの文脈を50-100トークンで説明してください。""",

    "gdrive": """このチャンクはGoogle Driveファイル「{filename}」（{file_type}）
の {page_info} から抽出されました。作成者: {author}。
チャンクの文脈を50-100トークンで説明してください。""",
}
```

**Zenn記事のConfluenceチャンカーへの統合例**:

```python
def chunk_confluence_with_context(
    html_content: str,
    page_metadata: dict,
    whole_document_text: str,
) -> list[dict]:
    """Contextual Retrievalを適用したConfluenceチャンキング

    Zenn記事のchunk_confluence_page関数を拡張し、
    各チャンクに文脈情報を付加する。

    Args:
        html_content: ConfluenceページのHTML
        page_metadata: ページメタデータ（タイトル、スペース等）
        whole_document_text: 文書全体のテキスト

    Returns:
        文脈付きチャンクのリスト
    """
    # 既存の見出しベースチャンキング
    base_chunks = chunk_confluence_page(html_content, page_metadata)

    # 各チャンクに文脈を付加
    contextual_chunks = []
    for chunk in base_chunks:
        context = generate_context(
            whole_document=whole_document_text,
            chunk_content=chunk["content"],
        )
        contextual_chunks.append({
            **chunk,
            "content": f"{context}\n\n{chunk['content']}",
            "original_content": chunk["content"],
            "context": context,
        })

    return contextual_chunks
```

### 200,000トークン以下の文書

Anthropicは重要な指摘をしている: **200,000トークン以下の知識ベース（約500ページ）では、文書全体をプロンプトに含める方が効率的**。Prompt Cachingを使えば繰り返しアクセスのコストも低い。

マルチソースRAGでは、各ソースのインデックスサイズを確認し、小規模ソース（例: 社内用語集、FAQ集）はRAGではなくプロンプト直接挿入が適切な場合がある。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $70-200 | Lambda + Bedrock + OpenSearch Serverless |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $400-1,000 | ECS + OpenSearch + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,500-6,000 | EKS + OpenSearch + GPU Reranker |

**Contextual Retrieval固有のコスト**:
- **文脈生成（一回限り）**: $1.02/100万文書トークン（Bedrock Haiku + Prompt Caching）
- **ベクトルDB増加**: 文脈付加でチャンクサイズが約20%増加→ストレージコスト微増
- **BM25インデックス**: OpenSearch Serverlessに追加インデックス作成

**コスト削減テクニック**:
- Prompt Cachingで文脈生成コストを最小化
- 文脈生成はバッチ処理（Lambda Step Functions）で実行
- 小規模ソース（< 200Kトークン）はRAGではなくプロンプト直接挿入

**コスト試算の注意事項**:
- 上記は2026年2月時点のAWS ap-northeast-1料金に基づく概算値
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認

### Terraformインフラコード

**文脈生成バッチ処理用 Step Functions + Lambda**

```hcl
# --- Lambda: 文脈生成バッチ処理 ---
resource "aws_lambda_function" "context_generator" {
  filename      = "context_generator.zip"
  function_name = "contextual-retrieval-generator"
  role          = aws_iam_role.context_gen_role.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 300  # 5分（大量チャンク処理に対応）
  memory_size   = 1024

  environment {
    variables = {
      BEDROCK_MODEL_ID    = "anthropic.claude-3-5-haiku-20241022-v1:0"
      OPENSEARCH_ENDPOINT = aws_opensearchserverless_collection.rag.collection_endpoint
      ENABLE_PROMPT_CACHE = "true"
    }
  }
}

# --- Step Functions: バッチオーケストレーション ---
resource "aws_sfn_state_machine" "context_batch" {
  name     = "contextual-retrieval-batch"
  role_arn = aws_iam_role.sfn_role.arn

  definition = jsonencode({
    StartAt = "ProcessChunks"
    States = {
      ProcessChunks = {
        Type     = "Map"
        MaxConcurrency = 10  # 並列度制限（Bedrock API制限考慮）
        Iterator = {
          StartAt = "GenerateContext"
          States = {
            GenerateContext = {
              Type     = "Task"
              Resource = aws_lambda_function.context_generator.arn
              End      = true
            }
          }
        }
        End = true
      }
    }
  })
}
```

### コスト最適化チェックリスト

- [ ] Prompt Cachingを有効化して文脈生成コストを最小化
- [ ] 200Kトークン以下のソースはプロンプト直接挿入に切り替え
- [ ] 文脈生成はバッチ処理で実行（リアルタイム生成は避ける）
- [ ] ベクトルDBのインデックスサイズを監視（文脈付加で20%増加）
- [ ] Step Functionsの並列度をBedrock API制限に合わせて調整

## 学術研究との関連（Academic Connection）

Contextual Retrievalの着想は、以下の学術研究の流れに位置づけられる。

- **Parent Document Retrieval**: チャンクの親文書IDを保持し、検索ヒット時に親文書全体を返す手法。Contextual Retrievalは親文書の「要約」をチャンクに付加する点で、より軽量かつ情報量が多い
- **HyDE** (Gao et al., 2022): 仮説回答をクエリ側で生成する手法。Contextual Retrievalは**ドキュメント側**で文脈を生成する点が異なり、クエリ非依存で一度だけ実行すればよい
- **Proposition Chunking** (Chen et al., 2023): 文書を命題単位に分解する手法。Contextual Retrievalは命題分解ではなく文脈付加であり、既存のチャンキング戦略に上乗せできる汎用性がある

## まとめと実践への示唆

Contextual Retrievalは、**既存のRAGパイプラインに最小限の変更で大幅な検索精度改善をもたらす実用的な手法**である。特にマルチソースRAGでは、各ソースの文脈が失われやすいため、効果が大きい。

**Zenn記事への具体的な適用**:

1. **Confluenceチャンカー**: 見出しベースチャンキング後に文脈を付加。「このチャンクはXXXプロジェクトの設計書のYYYセクションから...」
2. **Slackチャンカー**: スレッド単位チャンキング後に文脈を付加。「このスレッドは#dev-channelでの認証方式の議論で...」
3. **インデックス更新**: 既存のチャンクを文脈付きチャンクに置き換えるバッチ処理を構築

**コスト対効果**: 100万トークンの文書に対して$1.02の一回限りコストで、検索失敗率を最大67%削減できる。エンタープライズRAGでの投資対効果は極めて高い。

## 参考文献

- **Blog URL**: [https://www.anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/e4a4b18478c692](https://zenn.dev/0h_n0/articles/e4a4b18478c692)
- HyDE: [https://arxiv.org/abs/2212.10496](https://arxiv.org/abs/2212.10496)
- Anthropic Prompt Caching: [https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
