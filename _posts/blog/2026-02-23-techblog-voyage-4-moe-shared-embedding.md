---
layout: post
title: "Voyage AI解説: Voyage 4 — MoEアーキテクチャと共有埋め込み空間で非対称検索を実現"
description: "Embeddingモデル初のMoEアーキテクチャを採用しサービングコスト40%削減、4モデル間の共有埋め込み空間で非対称検索を可能にしたVoyage 4の技術詳細を解説"
categories: [blog, tech_blog]
tags: [embedding, Voyage, MoE, retrieval, RAG, asymmetric-retrieval]
date: 2026-02-23 12:00:00 +0900
source_type: tech_blog
source_domain: blog.voyageai.com
source_url: https://blog.voyageai.com/2026/01/15/voyage-4/
zenn_article: 6388d71c6bcb23
zenn_url: https://zenn.dev/0h_n0/articles/6388d71c6bcb23
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Voyage AI Blog: The Voyage 4 model family: shared embedding space with MoE architecture](https://blog.voyageai.com/2026/01/15/voyage-4/) の解説記事です。

## ブログ概要（Summary）

Voyage AI社は2026年1月、Embeddingモデルとして業界初のMixture of Experts（MoE）アーキテクチャを採用したVoyage 4ファミリーを発表した。voyage-4-large、voyage-4、voyage-4-lite、voyage-4-nanoの4モデルで構成され、全モデルが互換性のある**共有埋め込み空間（Shared Embedding Space）**を持つ。これにより、ドキュメント側はvoyage-4-largeで高精度にエンコードし、クエリ側はvoyage-4-liteで低コストに処理する「非対称検索」が可能になった。RTEB 29データセットでOpenAI text-embedding-3-largeを14.05%上回るスコアを記録したと報告されている。

この記事は [Zenn記事: MTEB×JMTEBで選ぶEmbeddingモデル：精度評価の実践ガイド](https://zenn.dev/0h_n0/articles/6388d71c6bcb23) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://blog.voyageai.com/2026/01/15/voyage-4/](https://blog.voyageai.com/2026/01/15/voyage-4/)
- **組織**: Voyage AI（MongoDB傘下）
- **発表日**: 2026年1月15日

## 技術的背景（Technical Background）

従来のEmbeddingモデルはDense（密）アーキテクチャを採用しており、推論時に全パラメータを活性化していた。モデルサイズを大きくして精度を向上させると、サービングコストも比例して増加するというトレードオフがあった。

一方、LLMの分野ではMistral（Mixtral 8x7B）やSwitch Transformerに代表されるMoEアーキテクチャが、パラメータ効率と精度のバランスに優れることが実証されていた。しかし、Embeddingモデルへの応用は2026年1月のVoyage 4まで実現されていなかった。

Voyage AI社のブログによると、MoEアーキテクチャにより「denseモデルと比較してサービングコストを40%削減しながらSOTA精度を達成した」と報告されている。

また、従来の課題として「モデル切り替え時のインデックス再構築コスト」があった。異なるモデルが異なる埋め込み空間を持つため、モデルを更新する際にはコーパス全体の再エンコードが必要だった。共有埋め込み空間はこの課題を解決する。

## 実装アーキテクチャ（Architecture）

### MoEアーキテクチャの概要

Voyage 4のMoEアーキテクチャは、各Transformerレイヤーのフィードフォワード部分に複数のエキスパートネットワークを配置し、入力トークンに応じて一部のエキスパートのみを活性化する仕組みである。

MoEの基本原理を数式で示す。入力 $\mathbf{x}$ に対して、ゲーティングネットワーク $G$ が各エキスパート $E_i$ の重みを決定する。

$$
\text{MoE}(\mathbf{x}) = \sum_{i=1}^{N} G(\mathbf{x})_i \cdot E_i(\mathbf{x})
$$

ここで、
- $N$: エキスパートの総数
- $G(\mathbf{x})_i$: エキスパート$i$のゲーティング重み（softmaxで正規化、Top-Kのみ非ゼロ）
- $E_i$: $i$番目のエキスパートネットワーク

各推論ステップで活性化されるのはTop-K個のエキスパートのみであるため、パラメータ総数に対して計算コストを大幅に削減できる。Voyage AI社のブログでは具体的なエキスパート数は公開されていないが、サービングコスト40%削減という数値からMoEの効率性が確認できる。

### 共有埋め込み空間（Shared Embedding Space）

Voyage 4ファミリーの4モデルが生成するベクトルは互換性がある。Voyage AI社のブログでは「query embeddings generated using voyage-4-lite can be used to search for document embeddings generated using voyage-4-large」と明記されている。

```python
# shared_embedding_space_example.py
# Voyage 4の非対称検索の概念実装
# voyage-4-largeでドキュメントをエンコード（高精度・高コスト）
doc_embeddings = voyage_client.embed(
    texts=documents,
    model="voyage-4-large",
    input_type="document"
)

# voyage-4-liteでクエリをエンコード（低コスト・低レイテンシ）
query_embedding = voyage_client.embed(
    texts=[query],
    model="voyage-4-lite",
    input_type="query"
)

# 異なるモデルのベクトルでも直接コサイン類似度検索が可能
# 共有埋め込み空間により互換性が保証される
similarities = cosine_similarity(query_embedding, doc_embeddings)
```

この設計により、以下の運用パターンが可能になる。

| ユースケース | ドキュメント側 | クエリ側 | 利点 |
|------------|-------------|---------|------|
| コスト最適化 | voyage-4-large | voyage-4-lite | ドキュメントは1回のみエンコード、クエリは毎回低コストで処理 |
| 段階的移行 | voyage-4 | voyage-4 → voyage-4-large | インデックス再構築なしでモデルアップグレード |
| ローカル開発 | voyage-4-nano（OSS） | voyage-4-nano | Apache 2.0でローカル開発・テスト |

### モデルラインナップと性能

| モデル | 位置づけ | 次元数 | 最大トークン | ライセンス |
|--------|---------|--------|-----------|-----------|
| **voyage-4-large** | フラッグシップ | 2048/1024/512/256 | 32K | 商用API |
| **voyage-4** | バランス型 | 2048/1024/512/256 | 32K | 商用API |
| **voyage-4-lite** | 軽量版 | 2048/1024/512/256 | 32K | 商用API |
| **voyage-4-nano** | 開発用 | - | - | Apache 2.0（HuggingFace公開） |

次元数の選択はMatryoshka Representation Learning（Kusupati et al., 2022）に基づく。2048次元がフル精度であり、1024/512/256次元は精度を少し犠牲にしてストレージ・計算コストを削減する。

### ベンチマーク結果

Voyage AI社のブログによると、RTEB 29データセット（top-10文書検索、NDCG@10で評価）での比較結果は以下のとおりである。

| モデル | voyage-4-largeとの差 |
|--------|---------------------|
| voyage-4 | -1.87% |
| voyage-4-lite | -4.80% |
| Gemini Embedding 001 | -8.20% |
| Cohere Embed v4 | -3.87% |
| OpenAI text-embedding-3-large | -14.05% |

voyage-4-largeがOpenAI text-embedding-3-largeを14.05%上回ったとVoyage AI社のブログで報告されている。ただし、この数値はVoyage AI社自身による評価結果であり、第三者による独立した検証結果ではない点に留意が必要である。

## パフォーマンス最適化（Performance）

### コスト最適化戦略

Voyage 4の非対称検索を活用したコスト最適化の戦略を示す。

**バッチインデックス構築**:
- ドキュメントコーパスはvoyage-4-largeで一括エンコード（1回のみ）
- 新規ドキュメント追加時もvoyage-4-largeでインクリメンタルにエンコード
- ドキュメント側のエンコードコストはN×1回のみ発生

**リアルタイムクエリ処理**:
- ユーザークエリはvoyage-4-liteで低レイテンシ・低コストに処理
- voyage-4-liteはvoyage-4-largeの-4.80%の精度低下で、大幅なコスト削減を実現

**次元削減の併用**:
- Matryoshka対応により、用途に応じて256/512/1024/2048次元を選択
- ストレージコスト: 256次元は2048次元の1/8
- 検索レイテンシ: 次元削減に比例して高速化

### 量子化サポート

Voyage 4は以下の量子化精度をサポートしている。
- 32ビット浮動小数点（float32）: 最高精度
- 8ビット符号付き整数（int8）: メモリ1/4、精度微減
- 8ビット符号なし整数（uint8）: int8と同等
- バイナリ精度: メモリ1/32、ハミング距離検索に適用

## 運用での学び（Production Lessons）

### モデル切り替えの運用パターン

共有埋め込み空間の最大の実用的利点は、インデックス再構築なしのモデルアップグレードである。従来のEmbeddingモデルでは、新しいモデルに切り替える際にコーパス全体の再エンコードが必要だった。数百万〜数十億ドキュメントの大規模コーパスでは、再エンコードに数日〜数週間かかるケースがある。

Voyage 4では、voyage-4からvoyage-4-largeへのアップグレード時にドキュメント側の再エンコードが不要である。クエリ側のモデルを切り替えるだけで精度向上が得られる。

### API利用時の注意点

Voyage AI社は「最初の200Mトークンは無料」としているが、大規模コーパスの初期インデックス構築では200Mトークンを超えるケースが多い。コスト見積もりの際は、ドキュメントの平均トークン数 × ドキュメント数で総トークン数を事前に計算することが推奨される。

## 学術研究との関連（Academic Connection）

### MoEアーキテクチャの学術的背景

Voyage 4のMoEアーキテクチャは、以下の学術研究の成果に基づいている。

- **Switch Transformer** (Fedus et al., 2021): MoEを大規模言語モデルに適用し、パラメータ効率を実証
- **Mixtral 8x7B** (Jiang et al., 2024): LLMにおけるMoEの実用化を推進

Voyage 4はこれらのLLM向けMoE手法をEmbeddingモデルに初めて適用した事例であると、Voyage AI社のブログで位置づけられている。

### Matryoshka Representation Learningとの関連

次元の柔軟な選択（2048/1024/512/256）は、Matryoshka Representation Learning（Kusupati et al., 2022, arXiv:2212.09741）の手法に基づいている。1つのモデルで複数の次元の埋め込みを同時に最適化することで、用途に応じた精度・コストのトレードオフを実現している。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

Voyage 4の非対称検索パイプラインをAWS上で構築する場合の構成を示す。

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $50-150 | Lambda + Voyage API + OpenSearch Serverless |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $300-800 | ECS Fargate + Voyage API + OpenSearch |
| **Large** | 300,000+ (10,000/日) | Container | $2,000-5,000 | EKS + voyage-4-nano(自ホスト) + OpenSearch |

**Small構成の詳細**（月額$50-150）:
- **Lambda**: クエリ前処理・Voyage API呼び出し（$20/月）
- **Voyage API**: voyage-4-liteでクエリ、voyage-4-largeでドキュメント（$80/月、200Mトークン無料枠活用）
- **OpenSearch Serverless**: ベクトル検索インデックス（$30/月）
- **CloudWatch**: レイテンシ・コスト監視（$5/月）

**コスト試算の注意事項**:
- 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です
- Voyage APIの料金は別途発生します（200Mトークン無料、以降は従量課金）
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) および [Voyage AI Pricing](https://docs.voyageai.com/docs/pricing) で確認してください

### Terraformインフラコード

**Small構成（Serverless）: Lambda + OpenSearch Serverless**

```hcl
# --- OpenSearch Serverless Collection（ベクトル検索） ---
resource "aws_opensearchserverless_collection" "embedding_search" {
  name = "voyage4-embedding-search"
  type = "VECTORSEARCH"
}

resource "aws_opensearchserverless_security_policy" "encryption" {
  name = "voyage4-encryption"
  type = "encryption"
  policy = jsonencode({
    Rules = [{
      ResourceType = "collection"
      Resource     = ["collection/voyage4-embedding-search"]
    }]
    AWSOwnedKey = true
  })
}

# --- IAMロール（最小権限） ---
resource "aws_iam_role" "lambda_voyage" {
  name = "lambda-voyage4-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

# --- Lambda関数（Voyage API呼び出し） ---
resource "aws_lambda_function" "voyage_query" {
  filename      = "lambda.zip"
  function_name = "voyage4-query-handler"
  role          = aws_iam_role.lambda_voyage.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 30
  memory_size   = 512

  environment {
    variables = {
      VOYAGE_MODEL_QUERY = "voyage-4-lite"
      VOYAGE_MODEL_DOC   = "voyage-4-large"
      OPENSEARCH_ENDPOINT = aws_opensearchserverless_collection.embedding_search.collection_endpoint
    }
  }
}

# --- Secrets Manager（Voyage APIキー） ---
resource "aws_secretsmanager_secret" "voyage_api_key" {
  name = "voyage4-api-key"
}
```

### コスト最適化チェックリスト

- [ ] 非対称検索: ドキュメント=voyage-4-large、クエリ=voyage-4-lite
- [ ] Matryoshka次元削減: 256次元でストレージ1/8（精度許容範囲の場合）
- [ ] int8量子化: メモリ使用量1/4
- [ ] 200Mトークン無料枠: 初期インデックス構築に活用
- [ ] バッチエンコード: ドキュメントは一括処理でAPI呼び出し回数削減
- [ ] OpenSearch Serverless: アイドル時のコスト最小化
- [ ] voyage-4-nano: 開発・テスト環境はOSSモデルでAPI費用ゼロ
- [ ] AWS Budgets: 月額予算設定（80%で警告）
- [ ] CloudWatch: API呼び出し数・レイテンシの監視
- [ ] コスト比較: 自ホスト(voyage-4-nano) vs API(voyage-4-lite)のブレークイーブン分析

## まとめと実践への示唆

Voyage 4は、MoEアーキテクチャと共有埋め込み空間という2つの技術的革新を組み合わせたEmbeddingモデルファミリーである。Voyage AI社のブログによると、MoEによりサービングコスト40%削減を実現し、共有埋め込み空間により非対称検索とインデックス再構築なしのモデルアップグレードを可能にしたとされている。

実務的な示唆として、大規模コーパスの検索システムでは非対称検索（ドキュメント=高精度モデル、クエリ=軽量モデル）がコスト最適化に有効である。ただし、Voyage 4は商用APIモデルであり、APIサービスへの依存が生まれる点に注意が必要である。オンプレミスや自社GPU環境が必須の場合は、Apache 2.0のvoyage-4-nanoまたはbge-m3が代替候補となる。

## 参考文献

- **Blog URL**: [https://blog.voyageai.com/2026/01/15/voyage-4/](https://blog.voyageai.com/2026/01/15/voyage-4/)
- **Voyage AI Docs**: [https://docs.voyageai.com/docs/embeddings](https://docs.voyageai.com/docs/embeddings)
- **Related Papers**: Matryoshka Representation Learning ([arXiv:2212.09741](https://arxiv.org/abs/2212.09741))
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/6388d71c6bcb23](https://zenn.dev/0h_n0/articles/6388d71c6bcb23)
