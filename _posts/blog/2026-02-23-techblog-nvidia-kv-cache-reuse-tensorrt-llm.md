---
layout: post
title: "NVIDIA解説: TensorRT-LLMのKVキャッシュ再利用最適化 — 優先度ベースEvictionとイベントAPI"
description: "TensorRT-LLMに導入された優先度ベースKVキャッシュEvictionとKV Cache Event APIにより、キャッシュヒット率約20%改善とKV認識ルーティングを実現する技術の解説"
categories: [blog, tech_blog]
tags: [nvidia, tensorrt-llm, kv-cache, inference-optimization, llm-serving, gpu]
date: 2026-02-23 12:00:00 +0900
source_type: tech_blog
source_domain: developer.nvidia.com
source_url: https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/
zenn_article: 555a4e799660de
zenn_url: https://zenn.dev/0h_n0/articles/555a4e799660de
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Introducing New KV Cache Reuse Optimizations in NVIDIA TensorRT-LLM](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/) の解説記事です。

## ブログ概要（Summary）

NVIDIAは2025年1月、TensorRT-LLMに2つの新機能を導入した。**優先度ベースKVキャッシュEviction**は、トークン範囲ごとに優先度と保持期間を設定し、システムプロンプトなどの頻繁に再利用されるKVキャッシュの保持を優先する。**KV Cache Event API**は、キャッシュの状態変化をイベントとして外部システムに通知し、キャッシュ認識型リクエストルーティングを可能にする。NVIDIAの報告によると、これらの機能によりキャッシュヒット率が約20%改善されるとしている。

この記事は [Zenn記事: LangGraph×Claude Sonnet 4.6のプロンプトキャッシュ最適化でAgentic RAGコスト90%削減](https://zenn.dev/0h_n0/articles/555a4e799660de) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/)
- **組織**: NVIDIA（John Thomson, Anjali Shah, Laikh Tewari）
- **発表日**: 2025年1月16日

## 技術的背景（Technical Background）

LLMの推論サービングでは、KVキャッシュのメモリ管理がスループットとレイテンシの両方に影響を与える。TensorRT-LLMはNVIDIAのLLM推論エンジンであり、PagedKVキャッシュ、量子化KVキャッシュ、Circular Buffer KVキャッシュなどの基本機能を提供してきた。

従来のPagedKVキャッシュでは、メモリが逼迫した際にLRU（Least Recently Used）ポリシーでブロックを解放していた。しかし、LRUは**ワークロードの特性を考慮しない**ため、頻繁に再利用されるシステムプロンプトのKVキャッシュが不用意に解放されるケースが発生していた。特に、長い出力生成（数千トークン）を処理するリクエストがKVキャッシュメモリを大量に消費する場合、短いが頻出するシステムプロンプトのキャッシュが追い出される問題があった。

Zenn記事で解説されているAnthropicのプロンプトキャッシュ（`cache_control`によるブレークポイント設定）は**APIプロバイダ側**の最適化であるのに対し、TensorRT-LLMの最適化は**自社推論サーバーを運用する場合**のサーバーサイド最適化である。両者は異なるレイヤーで動作するが、「頻繁に再利用されるプレフィックスのKVキャッシュを優先的に保持する」という設計思想は共通している。

## 実装アーキテクチャ（Architecture）

### 優先度ベースKVキャッシュEviction

この機能は、KVキャッシュのブロックにトークン範囲ごとの優先度と保持期間を設定する仕組みである。

**設定の構造**:

```python
# TensorRT-LLM の KV Cache Retention Config（ブログの記載に基づく）
from tensorrt_llm.runtime import KvCacheRetentionConfig, TokenRangeRetentionConfig

config = KvCacheRetentionConfig(
    token_range_retention_configs=[
        # システムプロンプト（0-1000トークン）: 最高優先度、30秒保持
        TokenRangeRetentionConfig(
            start=0,
            end=1000,
            priority=100,  # 0-100のスケール
            duration_seconds=30,
        ),
        # RAGコンテキスト（1000-3000トークン）: 中優先度、10秒保持
        TokenRangeRetentionConfig(
            start=1000,
            end=3000,
            priority=50,
            duration_seconds=10,
        ),
    ],
    # デコードフェーズの生成済みトークンの優先度
    decode_priority=30,
)
```

**動作メカニズム**:

1. リクエスト到着時に、プロンプトのトークン範囲に対して優先度を割り当てる
2. メモリが逼迫した際、優先度が低いブロックから解放される
3. 保持期間（duration）が経過したブロックは優先度に関わらず解放候補になる
4. 同一優先度のブロックはLRU順で解放される

NVIDIAの報告では、この機能により**キャッシュヒット率が約20%改善**されるとしている。改善幅はワークロードの特性（システムプロンプトの再利用パターン、リクエストの到着率等）に依存する。

### KV Cache Event API

この機能は、KVキャッシュの状態変化をリアルタイムにイベントとして通知する仕組みである。

**イベントの種類**（ブログの記載に基づく）:

| イベント | 説明 | ユースケース |
|---------|------|-----------|
| `BLOCK_CREATED` | 新規KVキャッシュブロックが作成された | キャッシュウォームアップの監視 |
| `BLOCK_STORED` | KVキャッシュブロックが永続化された | ティアードキャッシュ管理 |
| `BLOCK_REMOVED` | KVキャッシュブロックがevictされた | キャッシュ効率のモニタリング |
| `BLOCK_UPDATED` | 既存ブロックが更新された | 増分デコードの追跡 |

**KV認識型ルーティングの実装パターン**:

```python
from typing import Any


class KVCacheAwareRouter:
    """KVキャッシュ状態に基づくリクエストルーティング

    NVIDIAブログで紹介されたKV Cache Event APIを活用し、
    各エグゼキューター（GPU/サーバー）のキャッシュ状態に基づいて
    リクエストを最適なエグゼキューターにルーティングする。
    """

    def __init__(self, executors: list[str]):
        self.executors = executors
        # 各エグゼキューターのキャッシュ状態を追跡
        self.cache_state: dict[str, set[str]] = {
            ex: set() for ex in executors
        }

    def on_cache_event(self, executor: str, event: dict[str, Any]) -> None:
        """KVキャッシュイベントのハンドラ（最終整合性モデル）"""
        if event["type"] == "BLOCK_STORED":
            self.cache_state[executor].add(event["prefix_hash"])
        elif event["type"] == "BLOCK_REMOVED":
            self.cache_state[executor].discard(event["prefix_hash"])

    def route_request(self, request_prefix_hash: str) -> str:
        """リクエストを最適なエグゼキューターにルーティング

        キャッシュヒットが期待できるエグゼキューターを優先する。
        キャッシュが存在しない場合は負荷が最も低いエグゼキューターを選択。
        """
        # キャッシュヒットが期待できるエグゼキューターを探す
        for executor in self.executors:
            if request_prefix_hash in self.cache_state[executor]:
                return executor  # キャッシュヒットが期待できる

        # キャッシュがない場合はラウンドロビン（簡略化）
        return self.executors[0]
```

NVIDIAのブログによると、Event APIは**最終整合性（eventually consistent）** モデルで動作する。イベントは非同期に配信されるため、ルーティング決定時のキャッシュ状態は厳密にはリアルタイムではない。しかし、著者らはほとんどのワークロードでこの遅延が実用上問題にならないと述べている。

## パフォーマンス最適化（Performance）

### 優先度設定の指針

NVIDIAの報告に基づき、推奨される優先度設定パターンを整理する。

| プロンプト構成要素 | 推奨優先度 | 保持期間 | 理由 |
|----------------|----------|---------|------|
| システムプロンプト | 90-100 | 30-60秒 | 全リクエストで共通、最も再利用される |
| ツール定義 | 80-90 | 30秒 | システムプロンプトに次いで安定 |
| RAGコンテキスト | 40-60 | 10-15秒 | ドキュメントに依存、中程度の再利用 |
| 会話履歴 | 20-40 | 5-10秒 | セッション固有、再利用機会は限定的 |
| デコード出力 | 10-30 | 5秒 | 生成中のみ必要 |

### Zenn記事の4層キャッシュブレークポイントとの対応

Zenn記事で解説されているAnthropicの4層キャッシュブレークポイント設計は、TensorRT-LLMの優先度設定に以下のように対応する。

| Anthropic ブレークポイント | TensorRT-LLM 優先度 |
|--------------------------|-------------------|
| BP1: ツール定義（最上位） | priority: 90-100 |
| BP2: システムプロンプト | priority: 80-90 |
| BP3: RAGコンテキスト | priority: 40-60 |
| BP4: 会話履歴 | priority: 20-40 |

両者のアプローチは異なる（Anthropicは「キャッシュする/しない」の二値、TensorRT-LLMは連続的な優先度スケール）が、**変更頻度が低いコンテンツほど高い優先度を設定する**という原則は共通している。

## 運用での学び（Production Lessons）

### マルチGPU環境でのキャッシュ管理

NVIDIAのブログでは、マルチエグゼキューター（マルチGPU/マルチサーバー）環境でのKVキャッシュ管理の課題を取り上げている。

**課題**: 各エグゼキューターが独立にKVキャッシュを管理するため、同一プレフィックスのリクエストが異なるエグゼキューターに分散されるとキャッシュヒット率が低下する。

**解決策**: KV Cache Event APIを活用したKV認識型ルーティングにより、同一プレフィックスのリクエストを同一エグゼキューターに集約する。これにより、CPU/メモリの負荷分散だけでなく、キャッシュ再利用の機会も最大化される。

### 監視メトリクス

NVIDIAのブログで推奨されている監視項目:

- **キャッシュヒット率**: KV Cache Event APIのSTOREDイベントとCREATEDイベントの比率
- **Eviction率**: REMOVEDイベントの頻度（高すぎる場合はメモリ不足の兆候）
- **優先度分布**: 各優先度レベルでのキャッシュ使用量（偏りがないか確認）

## 学術研究との関連（Academic Connection）

TensorRT-LLMのKVキャッシュ再利用最適化は、以下の学術研究の知見を実装レベルで統合したものと位置づけられる。

- **PagedAttention** (Kwon et al., SOSP 2023): KVキャッシュのブロック管理の基盤。TensorRT-LLMのPagedKVキャッシュはこの論文の手法を実装している
- **Prompt Cache** (Gim et al., 2023): プロンプトモジュール単位でのKVキャッシュ再利用。TensorRT-LLMのトークン範囲指定はこの概念の一般化
- **RAGCache** (Jiang et al., 2024): ドキュメントレベルのKVキャッシュ管理。TensorRT-LLMの優先度設定はRAGCacheの適応的ポリシーを簡略化した形で提供

## まとめと実践への示唆

TensorRT-LLMの2つの新機能は、自社でLLM推論サーバーを運用する場合のKVキャッシュ最適化を強化するものである。優先度ベースEvictionはキャッシュヒット率を約20%改善し、KV Cache Event APIはマルチエグゼキューター環境でのキャッシュ認識型ルーティングを可能にする。

APIプロバイダ（Anthropic、OpenAI等）を利用する場合は`cache_control`やプレフィックスキャッシュがサーバーサイドで自動管理されるが、自社推論サーバーを運用する場合はTensorRT-LLMの優先度設定が同等の効果を提供する。

## 参考文献

- **Blog URL**: [https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/)
- **TensorRT-LLM GitHub**: [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/555a4e799660de](https://zenn.dev/0h_n0/articles/555a4e799660de)
