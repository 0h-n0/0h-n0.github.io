---
layout: post
title: "AWS公式解説: Letta（旧MemGPT）がAurora PostgreSQL + pgvectorでプロダクション向けAIエージェントを構築する方法"
description: "LLMエージェントの長期記憶バックエンドとしてAurora PostgreSQL + pgvectorを活用するLettaの本番アーキテクチャ詳解"
categories: [blog, tech_blog]
tags: [AWS, pgvector, PostgreSQL, LLM, agent, memory, langgraph]
date: 2026-02-23 12:00:00 +0900
source_type: tech_blog
source_domain: aws.amazon.com
source_url: https://aws.amazon.com/blogs/database/how-letta-builds-production-ready-ai-agents-with-amazon-aurora-postgresql/
zenn_article: 3901eb498f526c
zenn_url: https://zenn.dev/0h_n0/articles/3901eb498f526c
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [AWS Database Blog: How Letta builds production-ready AI agents with Amazon Aurora PostgreSQL](https://aws.amazon.com/blogs/database/how-letta-builds-production-ready-ai-agents-with-amazon-aurora-postgresql/) の解説記事です。

## ブログ概要（Summary）

AWS Database Blogで公開されたこの記事は、MemGPTの後継プロジェクトであるLetta（letta-ai/letta）が、Amazon Aurora PostgreSQL-Compatible Edition + pgvector拡張をバックエンドとして、プロダクション向けAIエージェントの長期記憶を構築する方法を解説している。Lettaは3層メモリアーキテクチャ（Core Memory, Archival Memory, Recall Memory）をPostgreSQLの単一データベース内で統一的に管理し、pgvectorのHNSWインデックスによるセマンティック検索で高速なメモリ検索を実現する。

この記事は [Zenn記事: LangGraph Store APIで実装するマルチエージェントRAGの共有メモリと長期記憶](https://zenn.dev/0h_n0/articles/3901eb498f526c) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（AWS Database Blog）
- **URL**: [https://aws.amazon.com/blogs/database/how-letta-builds-production-ready-ai-agents-with-amazon-aurora-postgresql/](https://aws.amazon.com/blogs/database/how-letta-builds-production-ready-ai-agents-with-amazon-aurora-postgresql/)
- **組織**: Amazon Web Services / Letta (letta-ai)
- **発表日**: 2024年（pgvector 0.7.0以降の機能を前提）

## 技術的背景（Technical Background）

LLMエージェントが長期記憶を保持するためには、コンテキストウィンドウ外の情報を永続化し、必要に応じて検索・取得する外部ストレージが不可欠である。Letta（旧MemGPT）は、OSの仮想メモリ概念をLLMに適用したアーキテクチャであり、その永続化バックエンドとしてPostgreSQL + pgvectorを採用している。

**なぜPostgreSQLが選ばれたか**（ブログの記述に基づく）:

1. **統一ストレージ**: 構造化データ（メタデータ、会話履歴）とベクトルデータ（embedding）を単一DBで管理。マイクロサービス間のデータ同期が不要
2. **トランザクション保証**: メモリの更新と会話履歴の保存をACIDトランザクションで一貫性を保証
3. **pgvectorの成熟度**: HNSW / IVFFlatインデックスによるANN検索、スカラー量子化、バイナリ量子化をサポート
4. **Aurora互換性**: マネージドサービスとして運用負荷を削減、Multi-AZでの高可用性

## 実装アーキテクチャ（Architecture）

### Lettaの3層メモリとPostgreSQLテーブル設計

Lettaは3つの論理的メモリ層を、PostgreSQLの各テーブルに以下のようにマッピングしている。

```sql
-- 1. Core Memory テーブル
-- 常時コンテキスト内に存在する重要情報（ユーザープロファイル等）
CREATE TABLE core_memory (
    agent_id UUID NOT NULL,
    section_name TEXT NOT NULL,  -- 'human', 'persona' 等
    content TEXT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (agent_id, section_name)
);

-- 2. Archival Memory テーブル（pgvector対応）
-- セマンティック検索可能な長期記憶
CREATE TABLE archival_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),  -- pgvector型
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSWインデックス（コサイン類似度）
CREATE INDEX idx_archival_hnsw ON archival_memory
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- エージェント別フィルタ用のBTreeインデックス
CREATE INDEX idx_archival_agent ON archival_memory (agent_id);

-- 3. Recall Memory テーブル
-- 過去の会話メッセージの時系列ストア
CREATE TABLE recall_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL,
    role TEXT NOT NULL,       -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    embedding vector(1024),  -- セマンティック検索用
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_recall_agent_time ON recall_memory (agent_id, created_at DESC);
CREATE INDEX idx_recall_hnsw ON recall_memory
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### LangGraph Store APIとの設計対応

LettaのPostgreSQLベースのメモリ管理は、LangGraph PostgresStoreの内部実装と高い類似性を持つ。

| Letta | LangGraph PostgresStore | 備考 |
|-------|------------------------|------|
| `archival_memory` テーブル | `store` テーブル | 名前空間 + ベクトル + JSONB |
| `core_memory` テーブル | Checkpointerの状態 | セッション内の短期メモリ |
| `recall_memory` テーブル | Checkpointerのメッセージ履歴 | 会話ログの永続化 |
| `agent_id` でフィルタ | 名前空間タプルでフィルタ | メモリの分離制御 |
| HNSWインデックス | PostgresStoreのindex設定 | セマンティック検索の高速化 |

```python
# LangGraph PostgresStoreの設定（Lettaと同等のpgvector構成）
from langgraph.store.postgres import AsyncPostgresStore

store = AsyncPostgresStore.from_conn_string(
    "postgresql://user:pass@aurora-cluster:5432/rag_memory",
    index={
        "dims": 1024,
        "embed": voyage_embed_fn,
        "fields": ["content", "metadata.summary"],
    },
)
```

### pgvectorインデックスの選択と最適化

ブログでは、pgvectorが提供する3種類のANNインデックスの使い分けを以下のように説明している。

| インデックス | 推奨メモリ件数 | Recall@10 | 構築時間 | メモリ使用量 |
|------------|-------------|-----------|---------|------------|
| **HNSW** | ~100万件 | 99%+ | 長い | 多い |
| **IVFFlat** | 100万件以上 | 95%程度 | 短い | 少ない |
| **Flat**（完全スキャン） | ~1万件 | 100% | なし | 最小 |

**HNSWパラメータのチューニング指針**（ブログの推奨値に基づく）:

```sql
-- 構築パラメータ
-- m: グラフのビーム幅（大きいほど精度↑、メモリ↑）
-- ef_construction: 構築時の探索幅（大きいほど精度↑、構築速度↓）
CREATE INDEX ON archival_memory
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 検索パラメータ
-- ef_search: 検索時の探索幅（大きいほど精度↑、検索速度↓）
SET hnsw.ef_search = 40;  -- デフォルト40、高精度が必要なら100
```

**pgvector 0.7.0以降の新機能**:
- **パラレルインデックス構築**: `maintenance_work_mem`を増やすことで構築速度が向上
- **スカラー量子化**: `halfvec`型で精度を維持しつつメモリ使用量を50%削減
- **バイナリ量子化**: `bit`型でさらに削減（精度は低下する可能性あり）

```sql
-- スカラー量子化（halfvec）によるメモリ削減
ALTER TABLE archival_memory
ADD COLUMN embedding_half halfvec(1024)
GENERATED ALWAYS AS (embedding::halfvec) STORED;

CREATE INDEX idx_archival_halfvec ON archival_memory
USING hnsw (embedding_half halfvec_cosine_ops);
```

## パフォーマンス最適化（Performance）

### Aurora PostgreSQLでの実測パフォーマンス

ブログおよび関連AWSドキュメントで報告されている性能指標:

| 指標 | 値 | 条件 |
|------|-----|------|
| ベクトル検索レイテンシ | 5-15ms | HNSW, 100万ベクトル, Top-10 |
| メモリ追加（INSERT） | 2-5ms | 単一レコード、embedding含む |
| インデックス構築 | ~30分 | 100万ベクトル, 1024次元, m=16 |
| Recall@10 | 99%+ | HNSW, ef_search=40 |

**pgvector 0.8.0（2025年11月リリース）の改善点**:
AWSブログの別記事によると、pgvector 0.8.0はクエリ処理速度が最大9倍、検索精度が100倍向上したと報告されている。具体的には、ストリーミングディスクANN（SD-ANN）の導入により、ディスクベースのインデックスでもインメモリに近い性能を実現している。

### 接続プーリングとセッション管理

Lettaの本番環境では、Aurora PostgreSQLへの接続管理にRDS Proxyを使用することが推奨されている。

```python
# 本番環境の接続設定
import asyncpg

async def create_pool():
    """Aurora PostgreSQL用の接続プール作成"""
    return await asyncpg.create_pool(
        host="aurora-cluster.proxy-xxx.ap-northeast-1.rds.amazonaws.com",
        port=5432,
        database="letta_memory",
        user="letta_app",
        password="${SECRETS_MANAGER}",
        min_size=5,
        max_size=20,
        command_timeout=30,
        # pgvector拡張の初期化
        init=lambda conn: conn.execute("SET hnsw.ef_search = 40"),
    )
```

## 運用での学び（Production Lessons）

### メモリの肥大化とガベージコレクション

長期運用では、Archival Memoryが蓄積し続けてインデックスサイズが肥大化する問題がある。Lettaの運用では以下の対策が取られている。

1. **TTLベースの自動削除**: 一定期間アクセスされないメモリを自動削除
2. **類似度ベースの重複排除**: 新規メモリ追加時に類似度0.95以上の既存メモリが存在する場合はスキップ
3. **定期的なVACUUM**: pgvectorインデックスの断片化を防ぐため、`VACUUM FULL`を週次で実行

```sql
-- TTLベースのクリーンアップ（90日以上前のメモリを削除）
DELETE FROM archival_memory
WHERE created_at < NOW() - INTERVAL '90 days'
  AND agent_id = $1;

-- 重複検知クエリ
SELECT id FROM archival_memory
WHERE agent_id = $1
  AND 1 - (embedding <=> $2::vector) > 0.95
LIMIT 1;
```

### マルチテナント運用での注意点

複数ユーザー/エージェントが同一データベースを共有する場合、`agent_id`によるフィルタリングが検索性能に影響する。

```sql
-- 非効率なクエリ（全レコードスキャン後にフィルタ）
SELECT * FROM archival_memory
ORDER BY embedding <=> $1::vector
LIMIT 10;

-- 効率的なクエリ（パーティショニング + HNSW）
-- agent_idでパーティション分割し、各パーティションにHNSWインデックス
SELECT * FROM archival_memory
WHERE agent_id = $2
ORDER BY embedding <=> $1::vector
LIMIT 10;
```

AWSブログでは、エージェント数が100を超える場合はテーブルパーティショニングの導入を推奨している。

## 学術研究との関連（Academic Connection）

LettaはMemGPT論文（arXiv:2310.03744）の直接的な後継実装であり、学術研究からプロダクション実装への橋渡しとして位置づけられる。

- **MemGPT論文からの実装変更点**: FAISSベースのArchival StorageをPostgreSQL + pgvectorに移行、REST APIの追加、マルチエージェント対応
- **LangGraph Store APIとの関係**: LangGraphのPostgresStoreもpgvector拡張を使用しており、テーブル設計の基本思想（JSONB + vector列 + HNSW）がLettaと一致している

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 | Serverless | $100-200 | Aurora Serverless v2 + Lambda |
| **Medium** | ~30,000 | Provisioned | $400-900 | Aurora Provisioned + ECS Fargate |
| **Large** | 300,000+ | Multi-AZ | $1,500-4,000 | Aurora Multi-AZ + EKS |

**コスト試算の注意事項**: 上記は2026年2月時点のAWS ap-northeast-1料金に基づく概算値です。最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください。

### Terraformインフラコード

```hcl
# --- Aurora PostgreSQL + pgvector (本番構成) ---
module "aurora_letta" {
  source  = "terraform-aws-modules/rds-aurora/aws"
  version = "~> 9.0"

  name           = "letta-memory-store"
  engine         = "aurora-postgresql"
  engine_version = "15.4"

  # Provisioned構成（Medium規模）
  instance_class = "db.r6g.large"
  instances      = {
    writer = {}
    reader = {}  # 読み取りレプリカ（検索負荷分散）
  }

  vpc_id  = module.vpc.vpc_id
  subnets = module.vpc.private_subnets

  storage_encrypted            = true
  performance_insights_enabled = true
  monitoring_interval          = 60

  # pgvector初期化スクリプト
  # CREATE EXTENSION IF NOT EXISTS vector;
}

# --- RDS Proxy (接続プーリング) ---
resource "aws_db_proxy" "letta_proxy" {
  name                   = "letta-db-proxy"
  debug_logging          = false
  engine_family          = "POSTGRESQL"
  idle_client_timeout    = 1800
  require_tls            = true
  vpc_security_group_ids = [aws_security_group.proxy_sg.id]
  vpc_subnet_ids         = module.vpc.private_subnets

  auth {
    auth_scheme = "SECRETS"
    iam_auth    = "REQUIRED"
    secret_arn  = aws_secretsmanager_secret.letta_db.arn
  }
}
```

### 運用・監視設定

```sql
-- pgvectorインデックスの健全性チェック
SELECT indexname, indexdef,
       pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size
FROM pg_indexes
WHERE tablename = 'archival_memory';

-- メモリ検索レイテンシの監視
SELECT agent_id,
       count(*) as query_count,
       avg(duration_ms) as avg_latency,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95
FROM memory_search_logs
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY agent_id;
```

### コスト最適化チェックリスト

- [ ] Aurora Serverless v2: 低トラフィック時0.5 ACUで$43/月
- [ ] 読み取りレプリカ: 検索クエリをレプリカに分散
- [ ] RDS Proxy: Lambda接続プーリングで接続数削減
- [ ] スカラー量子化（halfvec）: メモリ使用量50%削減
- [ ] パーティショニング: 100エージェント以上ではagent_id別パーティション
- [ ] VACUUM定期実行: インデックス断片化防止

## まとめと実践への示唆

LettaのAurora PostgreSQL + pgvector統合は、LLMエージェントの長期記憶を単一のマネージドデータベースで実現する実用的なアーキテクチャパターンを示している。構造化データ（会話履歴、メタデータ）とベクトルデータ（semantic embedding）を同一データベースで管理することで、トランザクション一貫性の確保と運用の簡素化を両立している。LangGraph PostgresStoreの本番環境構築において、Lettaのテーブル設計とインデックス最適化の知見は直接参考になる。

## 参考文献

- **Blog URL**: [https://aws.amazon.com/blogs/database/how-letta-builds-production-ready-ai-agents-with-amazon-aurora-postgresql/](https://aws.amazon.com/blogs/database/how-letta-builds-production-ready-ai-agents-with-amazon-aurora-postgresql/)
- **pgvector Optimization**: [https://aws.amazon.com/blogs/database/optimize-generative-ai-applications-with-pgvector-indexing-a-deep-dive-into-ivfflat-and-hnsw-techniques/](https://aws.amazon.com/blogs/database/optimize-generative-ai-applications-with-pgvector-indexing-a-deep-dive-into-ivfflat-and-hnsw-techniques/)
- **Letta GitHub**: [https://github.com/letta-ai/letta](https://github.com/letta-ai/letta)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/3901eb498f526c](https://zenn.dev/0h_n0/articles/3901eb498f526c)
