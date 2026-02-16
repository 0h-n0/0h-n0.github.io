---
layout: post
title: "ElasticsearchとLangGraphで構築するマルチエージェントシステム：Reflectionパターンの実践"
description: "Elastic Search Labs公式ブログ。ハイブリッド検索とELSER embeddings、Reflectionパターンで自己修正するマルチエージェントシステムの実装詳解"
categories: [blog, tech_blog]
tags: [Elasticsearch, LangGraph, multi-agent, reflection, RAG, ELSER, hybrid-search]
date: 2026-02-16 10:00:00 +0900
source_type: tech_blog
source_domain: elastic.co
source_url: https://www.elastic.co/search-labs/blog/multi-agent-system-llm-agents-elasticsearch-langgraph
zenn_article: 8487a08b378cf1
zenn_url: https://zenn.dev/0h_n0/articles/8487a08b378cf1
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

Elasticsearch Search LabsのAlex Salgado氏による2025年11月の公式ブログ記事。LangGraphのReflectionパターンを用いて、複数のLLMエージェントがElasticsearchの**ハイブリッド検索（BM25 + ELSER）** と**長期記憶（LTM）** を活用しながら、インシデントログの根本原因分析を自己修正するシステムの実装を解説します。従来のDAG（有向非巡回グラフ）では不可能だった「批評→改善→再評価」の循環フローを実現し、品質スコア0.8以上の分析レポートを2イテレーション以内で生成します。

この記事は [Zenn記事: LangGraphで作るマルチエージェント：30分で構築する実践ガイド](https://zenn.dev/0h_n0/articles/8487a08b378cf1) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（Elasticsearch Labs）
- **URL**: https://www.elastic.co/search-labs/blog/multi-agent-system-llm-agents-elasticsearch-langgraph
- **組織**: Elastic（Elasticsearch開発元）
- **著者**: Alex Salgado（Senior Developer Advocate）
- **発表日**: 2025年11月17日

## 技術的背景（Technical Background）

### なぜReflectionパターンが必要か

単一のLLMエージェントは以下の問題を抱えます。

1. **コンテキストウィンドウ限界**: 大量のログを一度に処理できない
2. **幻覚（Hallucination）**: 存在しないエラー原因を生成
3. **品質のばらつき**: 出力の正確性が安定しない

Reflectionパターンは、**専門エージェント間の相互批評**により、これらを解決します。

### Reflectionパターンの定義

以下の3つのエージェントが循環的に動作します。

```mermaid
graph LR
    A[SearchAgent] --> B[AnalyserAgent]
    B --> C[ReflectionAgent]
    C -->|Score < 0.8| B
    C -->|Score ≥ 0.8| D[END]
```

- **SearchAgent**: Elasticsearchでハイブリッド検索を実行
- **AnalyserAgent**: 検索結果から根本原因分析レポートを生成
- **ReflectionAgent**: レポートを評価し、品質スコア（0.0〜1.0）とフィードバックを返す

### 学術研究との関連

- **ReAct論文（arXiv 2210.03629）**: 推論（Thought）と行動（Action）の交互実行パラダイム
- **Constitutional AI（Anthropic, 2022）**: 自己批評による出力改善手法
- **Tree of Thoughts（Yao et al., 2023）**: 複数の推論パスを探索・評価

このブログ記事は、ReActのAction部分にElasticsearch検索を組み込み、Reflectionループで品質を担保する実装例です。

## 実装アーキテクチャ（Architecture）

### システム構成

| コンポーネント | 役割 | 技術スタック |
|--------------|------|-------------|
| **LangGraph** | エージェント間のワークフローオーケストレーション | Python SDK 1.0+ |
| **Elasticsearch** | ハイブリッド検索 + 長期記憶ストレージ | 8.15+ with ELSER v2 |
| **Ollama** | ローカルLLMエンジン | llama3.2:latest |
| **State Management** | エージェント間の状態共有 | `IncidentState` TypedDict |

### データフロー

```python
from typing import TypedDict

class IncidentState(TypedDict):
    """エージェント間で共有される状態"""
    query: str                   # ユーザーの問い合わせ
    search_results: list[dict]   # Elasticsearch検索結果
    analysis: str                # AnalyserAgentの出力
    quality_score: float         # ReflectionAgentの評価（0.0〜1.0）
    feedback: str                # 改善点のフィードバック
    iteration: int               # 現在のイテレーション回数
```

### Elasticsearch インデックス設計

**incident-logs インデックス（RAGデータソース）**:

```json
{
  "mappings": {
    "properties": {
      "semantic_text": {
        "type": "semantic_text",
        "inference_id": "elser_embeddings"
      },
      "content": {"type": "text"},
      "timestamp": {"type": "date"},
      "severity": {"type": "keyword"}
    }
  }
}
```

**agent-memory インデックス（長期記憶）**:

```json
{
  "mappings": {
    "properties": {
      "query": {"type": "text"},
      "analysis": {"type": "text"},
      "quality_score": {"type": "float"},
      "timestamp": {"type": "date"},
      "success": {"type": "boolean"}
    }
  }
}
```

### ハイブリッド検索の実装

Elasticsearchの**BM25（キーワードベース）** と**ELSER（セマンティック）** を組み合わせます。

```python
def hybrid_search(es_client, query: str, top_k: int = 15) -> list[dict]:
    """ハイブリッド検索（BM25 + ELSER）

    Args:
        es_client: Elasticsearch client
        query: 検索クエリ
        top_k: 取得件数

    Returns:
        検索結果のリスト
    """
    response = es_client.search(
        index="incident-logs",
        query={
            "bool": {
                "should": [
                    # BM25キーワード検索
                    {
                        "match": {
                            "content": {
                                "query": query,
                                "boost": 1.0
                            }
                        }
                    },
                    # ELSER セマンティック検索
                    {
                        "semantic": {
                            "field": "semantic_text",
                            "query": query,
                            "boost": 2.0  # セマンティック検索を重視
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        },
        size=top_k,
        sort=[
            {"_score": {"order": "desc"}},
            {"timestamp": {"order": "desc"}}
        ]
    )

    return [
        {
            "content": hit["_source"]["content"],
            "score": hit["_score"],
            "timestamp": hit["_source"]["timestamp"]
        }
        for hit in response["hits"]["hits"]
    ]
```

**なぜハイブリッドか**:
- BM25: 正確なキーワードマッチ（例: "ConnectionTimeout"）
- ELSER: 意味的類似性（例: "接続失敗" → "ネットワークエラー"）
- 組み合わせにより再現率（Recall）と適合率（Precision）を両立

## パフォーマンス最適化（Performance）

### 実測値（記事での実行例）

| メトリクス | 値 |
|----------|---|
| 検索レイテンシ | 250ms（15件取得） |
| Iteration 1 品質スコア | 0.65 |
| Iteration 2 品質スコア | 0.85（終了条件達成） |
| 総処理時間 | 約8秒（LLM推論含む） |
| 長期記憶ヒット時 | 1イテレーションで完了（50%時間削減） |

### チューニング手法

**1. ELSER Boostの調整**

```python
# デフォルト設定（均等）
"boost": 1.0  # BM25
"boost": 1.0  # ELSER

# 最適化後（セマンティック重視）
"boost": 1.0  # BM25
"boost": 2.0  # ELSER
```

実験により、ELSER boostを2.0にすることで、類義語を含むログの再現率が15%向上。

**2. 品質スコア閾値の設定**

| 閾値 | イテレーション平均 | 品質スコア平均 |
|------|------------------|--------------|
| 0.7 | 1.2回 | 0.74 |
| 0.8 | 1.8回 | 0.83 |
| 0.9 | 3.1回 | 0.91 |

0.8が精度と速度のトレードオフ最適点（記事推奨値）。

**3. 長期記憶（LTM）キャッシング**

```python
def query_long_term_memory(es_client, query: str) -> dict | None:
    """過去の成功事例を検索"""
    response = es_client.search(
        index="agent-memory",
        query={
            "bool": {
                "must": [
                    {"match": {"query": query}},
                    {"range": {"quality_score": {"gte": 0.8}}},
                    {"term": {"success": True}}
                ]
            }
        },
        size=1,
        sort=[{"quality_score": {"order": "desc"}}]
    )

    if response["hits"]["total"]["value"] > 0:
        return response["hits"]["hits"][0]["_source"]
    return None
```

**効果**: 過去に解決済みのインシデントは、LTMから即座に回答取得。処理時間50%削減。

## 運用での学び（Production Lessons）

### 障害事例1: 無限ループ

**症状**: ReflectionAgentが常に `quality_score < 0.8` を返し、3回超過してもループ継続。

**原因**: `iteration` カウンターのインクリメント漏れ。

**対策**:

```python
def should_continue(state: IncidentState) -> Literal["continue", "end"]:
    """終了判定"""
    if state["quality_score"] >= 0.8:
        return "end"
    if state["iteration"] >= 3:  # 最大イテレーション数
        logger.warning(f"Max iterations reached. Final score: {state['quality_score']}")
        return "end"
    return "continue"
```

### 障害事例2: Elasticsearch タイムアウト

**症状**: 大量のログに対して検索が5秒超過し、タイムアウト。

**原因**: インデックスの肥大化（1000万件超）。

**対策**:
- **ILM（Index Lifecycle Management）** でログをホット/ウォーム/コールドに階層化
- 検索対象を過去30日のホットログに限定

```python
"query": {
    "bool": {
        "must": [...],
        "filter": [
            {"range": {"timestamp": {"gte": "now-30d"}}}
        ]
    }
}
```

結果、検索レイテンシが250ms → 80msに短縮。

### モニタリング戦略

**Kibana Dashboard**:
- エージェント別の処理時間分布（ヒストグラム）
- 品質スコアの時系列推移
- イテレーション回数の分布（1回/2回/3回の割合）
- 長期記憶ヒット率

**アラート条件**:
- 品質スコア平均が0.75を下回る
- イテレーション3回の割合が20%超過
- Elasticsearch検索レイテンシが500ms超

## 学術研究との関連（Academic Connection）

### 関連論文との比較

| 論文/手法 | アプローチ | 本記事との違い |
|----------|----------|--------------|
| **ReAct (ICLR 2023)** | Thought + Action + Observation | ActionにElasticsearch検索を組み込み |
| **Reflexion (NeurIPS 2023)** | 自己反省によるポリシー改善 | 強化学習ではなくLLM評価ループ |
| **Self-Refine (arXiv 2303.17651)** | 単一モデルの自己批評 | 専門エージェント分離で役割明確化 |

### プロダクションならではの工夫

- **ハイブリッド検索**: 論文では単一の検索手法。本記事はBM25+ELSERで再現率向上
- **長期記憶**: 学術研究ではエピソード記憶が未実装。Elasticsearchで永続化
- **品質スコア定量化**: 論文では主観評価。本記事は4軸（完全性・証拠・実行可能性・論理性）で客観評価

## まとめと実践への示唆

### 主要な成果

1. **Reflectionパターンの実装**: LangGraphで循環フローを実現し、2イテレーション以内に品質スコア0.85達成
2. **ハイブリッド検索の効果**: BM25+ELSERで再現率15%向上
3. **長期記憶による効率化**: 過去事例活用で処理時間50%削減
4. **プロダクション実績**: Elasticの実環境でインシデント分析に導入

### Zenn記事との連携

Zenn記事で紹介した以下のパターンの実装詳細を補完します。

- **Supervisorパターン**: 本記事のReflectionAgentがSupervisor的役割
- **条件付きルーティング**: `should_continue` 関数がZenn記事の例と同じ設計
- **Elasticsearch統合**: Zenn記事の「長期記憶」セクションの具体的実装

### 次のステップ

1. **Human-in-the-loop追加**: 品質スコア0.8未満の場合、人間の承認フローを挟む
2. **マルチモーダル対応**: ログだけでなくメトリクス（CPU/メモリグラフ）も解析
3. **A/Bテスト**: LLMモデル（GPT-4 vs Claude）による品質スコア比較

## 参考文献

- **Blog URL**: https://www.elastic.co/search-labs/blog/multi-agent-system-llm-agents-elasticsearch-langgraph
- **Code Repository**: https://github.com/elastic/elasticsearch-labs/tree/main/notebooks/langgraph
- **Related Papers**:
  - ReAct (arXiv 2210.03629)
  - Reflexion (arXiv 2303.11366)
- **Related Zenn article**: https://zenn.dev/0h_n0/articles/8487a08b378cf1
