---
layout: post
title: "NVIDIA TensorRT-LLM解説: KV Cache Early Reuseによるシステムプロンプト共有でTTFT 5倍高速化"
description: "TensorRT-LLMのKV Cache Early Reuse、Flexible Block Sizing、Intelligent Evictionの3手法によるLLMサービング最適化の技術解説"
categories: [blog, tech_blog]
tags: [kv-cache, tensorrt-llm, nvidia, llm-serving, inference-optimization, performance]
date: 2026-04-29 09:00:00 +0900
source_type: tech_blog
source_domain: developer.nvidia.com
source_url: https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/
zenn_article: 80b83bf28e8353
zenn_url: https://zenn.dev/0h_n0/articles/80b83bf28e8353
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [NVIDIA Technical Blog: 5x Faster Time to First Token with NVIDIA TensorRT-LLM KV Cache Early Reuse](https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/) の解説記事です。

## ブログ概要（Summary）

NVIDIAは TensorRT-LLM における KV Cache の再利用効率を大幅に向上させる3つの最適化技術を発表した。第一に **KV Cache Early Reuse** は、計算途中のKVキャッシュをリアルタイムで他のリクエストと共有し、システムプロンプトを含むシナリオで Time to First Token（TTFT）を最大5倍高速化する。第二に **Flexible Block Sizing** は、キャッシュブロック粒度を従来の64トークンから最小2トークンまで縮小可能にし、LLAMA-70Bで最大7%のTTFT改善を実現する。第三に **Intelligent Eviction Protocols** は、依存関係ツリーを追跡してカスケード無効化を防止する。これら3つの技術の組み合わせにより、大規模LLMサービングにおけるKVキャッシュの再利用率とレイテンシが大幅に改善されることが報告されている。

この記事は [Zenn記事: プロンプトキャッシュの本番運用設計 -- ヒット率7%→84%改善の実装パターン](https://zenn.dev/0h_n0/articles/80b83bf28e8353) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（NVIDIA Technical Blog）
- **URL**: [https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/](https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/)
- **組織**: NVIDIA
- **関連ドキュメント**: [TensorRT-LLM KV Cache Reuse Documentation](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html)

## 技術的背景（Technical Background）

### KV Cacheの基礎

Transformer ベースの大規模言語モデル（LLM）では、自己注意機構（Self-Attention）の計算において、各トークンに対する Key（K）と Value（V）のテンソルを生成する。これらを再計算せずにキャッシュとして保持する手法が KV Cache である。

自己注意の計算は以下の式で表される。

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

ここで $Q$ はクエリ、$K$ はキー、$V$ はバリュー、$d_k$ はキー次元数である。KV Cacheは計算済みの $K$ と $V$ を保持し、新トークン生成時の過去トークンに対する再計���を不要にする。

### TTFTのボトルネック

TTFT（Time to First Token）は、リクエスト送信から最初のトークン生成までの時間である。ボトルネックの主因は、(1) prefill phaseの $O(n^2)$ 計算コスト、(2) 全ユーザー共通のシステムプロンプトの重複計算、(3) バースト時のGPUリソース圧迫の3点である。

### 既存手法の課題

| 手法 | 概要 | 制約 |
|:---|:---|:---|
| vLLM PagedAttention | OS仮想メモリのようにページ単位でKV Cache管理。断片化を4%未満に削減 | ブロック粒度が固定、プレフィックス共有が限定的 |
| SGLang RadixAttention | Radix TreeでKV Cacheを管理し、プレフィックスを自動共有 | 計算途中のKV Cacheリアルタイム共有は不可 |
| Anthropic Prompt Caching | APIレベルでプレフィックスKV Cacheを保持、85%レイテンシ削減 | クラウドAPI限定、セルフホスト型には適用不可 |

これらに共通する制約は、**計算完了を待たなければキャッシュを再利用できない**点である。NVIDIAのKV Cache Early Reuseは、この制約を解消するアプローチである。

## 実装アーキテクチャ（Architecture）

NVIDIAが発表した3つの最適化技術について、それぞれの仕組みを詳細に解説する。

### 1. KV Cache Early Reuse

従来のKVキャッシュ再利用は、リクエスト全体のprefill計算が完了してからキャッシュをストレージに格納し、後続リクエストで再利用するフローであった。Early Reuseはこの「完了待ち」を撤廃し、**計算途中のKVキャッシュをリアルタイムで他のリクエストと共有する**。

NVIDIAのブログによれば、エンタープライズチャットボットではシステムプロンプトが「全ユーザーに対してモデルの応答を指示するための事前定義された命令」として機能する。Early Reuseにより、バースト発生時にシステムプロンプトのKV Cacheを「リアルタイムで生成しながら全ユーザー間で共有」できるため、各ユーザーごとの再計算が不要になると報告されている。

主なメリットは、(1) 最初のリクエストの計算途中でも後続リクエストが逐次的にKV Cacheを利用可能、(2) 同時接続数が増加してもシステムプロンプト部分の計算は1回で済むバースト耐性、(3) NVIDIAが報告するTTFT最大5倍高速化の3点である。

ただし、Early Reuseの効果は共通プレフィックスを持つリクエストが同時に大量到達するシナリオで最大化され、リクエストごとにプロンプトが大きく異なる場合には効果が限定的である。

### 2. Flexible Block Sizing

TensorRT-LLM のデフォルトでは、KV Cache のブロックサイズは128トークンである（ブログでは64トークンの例で説明されている）。Flexible Block Sizing は、このブロック粒度を最小2トークンまで縮小可能にする機能である。

ブロックサイズが大きい場合の問題を具体例で示す。

$$
\text{KV Cache 長} = 80 \text{ tokens}, \quad \text{Block Size} = 64 \text{ tokens}
$$

この場合、64トークン分のブロック1個のみが再利用対象となり、残りの16トークンは再計算が必要になる。再利用率は以下の通りとなる。

$$
\text{Reuse Rate} = \frac{\lfloor 80 / 64 \rfloor \times 64}{80} = \frac{64}{80} = 80\%
$$

一方、ブロックサイズを16トークンに縮小した場合は次のようになる。

$$
\text{Reuse Rate} = \frac{\lfloor 80 / 16 \rfloor \times 16}{80} = \frac{80}{80} = 100\%
$$

80トークン全体を5ブロックとして保存・再利用でき、再計算が不要になる。

NVIDIAは、LLAMA-70BをH100 Tensor Core GPU上で実行した際、ブロックサイズを64トークンから8トークンに縮小することでTTFTが最大7%改善したと報告している。

**ブロックサイズ選択のトレードオフ**:

| ブロックサイズ | メリット | デメリット |
|:---:|:---|:---|
| 大（64-128トークン） | メモリ管理オーバーヘッド小、長シーケンスで効率的 | 部分再利用率が低い、短プロンプトでの無駄 |
| 小（2-16トークン） | 部分再利用率が高い、短プロンプトに有利 | ブロック管理オーバーヘッド増、依存関係の複雑化 |

TensorRT-LLM でのブロックサイズ設定は、モデルビルド時に `tokens_per_block` パラメータで指定する。

```bash
trtllm-build --use_paged_context_fmha enable \
             --tokens_per_block 16 \
             # ... 他のビルドオプション
```

なお、`tokens_per_block` は2の累乗（2, 4, 8, 16, 32, 64, 128）でなければならないという制約がある。

### 3. Intelligent Eviction Protocols

ブロックサイズを小さくすると、ブロック間の依存関係がツリー構造として複雑化する。NVIDIAのブログでは、この問題をLRU（Least Recently Used）単純適用の限界として指摘している。

例として、System Prompt（Block 0-3、最終使用: T-100）を根として、User Query A（Block 4-5、T-1）、B（Block 6-7、T-50）、C（Block 8-9、T-30）が分岐するツリーを考える。単純なLRUでは最も古いBlock 0-3が最初にevictされるが、これはBlock 4-9全ての依存元であるため、**カスケード無効化**が発生する。

NVIDIAが導入した Intelligent Eviction は、「依存ノードからソースノードへの依存関係ツリーをトレースし、依存ノード（leaf）をソースノード（root）より先にevictする」アルゴリズムである。たとえ依存ノードの方が最近使用されていたとしても、ツリーの末端から優先的に解放することで、共有ブロックの不要な無効化を防ぐ。

さらに、NVIDIAは別のブログ記事（[Introducing New KV Cache Reuse Optimizations](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/)）において、**Priority-based Eviction** を導入している。0-100のスケールでブロックに優先度を付与し、優先度の低いブロックから先にevictする仕組みである。

```python
# TensorRT-LLM の KvCacheRetentionConfig 設定例（公式ドキュメントより）
from tensorrt_llm.executor import (
    KvCacheRetentionConfig,
    TokenRangeRetentionConfig,
)

retention_config = KvCacheRetentionConfig(
    token_ranges=[
        # システムプロンプト: 最高優先度で保持
        TokenRangeRetentionConfig(
            start=0,
            end=1000,
            priority=100,
        ),
        # ユーザーコンテキスト: 中程度の優先度、30秒間保持
        TokenRangeRetentionConfig(
            start=1000,
            end=None,  # シーケンス末尾まで
            priority=50,
            duration=30,  # 秒
        ),
    ],
    decode_priority=20,  # デコードフェーズのトークンは低優先度
    decode_duration=10,  # 10秒間保持
)
```

NVIDIAによれば、このPriority-based Evictionにより**キャッシュヒット率が約20%向上した**と報告されている。

## Production Deployment Guide

TensorRT-LLM のKV Cache最適化を本番環境で活用するための実装ガイドを示す。

### AWS実装パターン（コスト最適化重視）

TensorRT-LLM によるLLMサービングは GPU インスタンスが必須であるため、AWS上では以下の構成が考えられる。コスト試算は2026年4月時点の東京リージョン（ap-northeast-1）概算であり、実際のコストはトラフィックパターンにより変動する。

**Small構成（~100 req/日）**: 単一GPU

- **インスタンス**: g5.xlarge（NVIDIA A10G、24GB VRAM） x 1
- **推論エンジン**: TensorRT-LLM + Triton Inference Server
- **モデル**: 7B-13Bパラメータクラス（量子化済み）
- **月額概算**: $700-1,000（オンデマンド）、$280-400（1年RI）
- **KV Cache設定**: `tokens_per_block=32`、`free_gpu_memory_fraction=0.85`

**Medium構成（~1,000 req/日）**: マルチGPU + ALB

- **インスタンス**: g5.2xlarge x 2（ALB経由でロードバランシング）
- **推論エンジン**: TensorRT-LLM + Triton + KV-aware routing
- **月額概算**: $2,000-3,000（オンデマンド）、$800-1,200（1年RI）
- **KV Cache設定**: `tokens_per_block=16`、Priority-based Eviction有効化、Host Offloading有効（`host_cache_size=45GB`）

**Large構成（10,000+ req/日）**: EKS + Tensor Parallelism

- **インスタンス**: p4d.24xlarge（A100 x 8）または p5.48xlarge（H100 x 8）
- **オーケストレーション**: EKS + Karpenter（GPU Spot対応）
- **推論エンジン**: TensorRT-LLM + Triton + KV Cache Event API による分散ルーティング
- **月額概算**: $15,000-30,000（オンデマンド）、$6,000-12,000（1年RI + Spot混合）
- **KV Cache設定**: `tokens_per_block=8`、Early Reuse有効、分散KVキャッシュ管理

**コスト削減テクニック**: Spot Instances（最大70%削減、要フォールバック設計）、Reserved Instances 1年（最大40%削減）、KV Cache Early Reuse自体による prefill計算削減（同一GPUでの処理能力向上）。

### Terraformインフラコード

**Small構成（単一GPU）** の主要リソース:

```hcl
resource "aws_instance" "tensorrt_llm_server" {
  ami           = data.aws_ami.deep_learning.id
  instance_type = "g5.xlarge"
  subnet_id     = aws_subnet.private.id

  root_block_device {
    volume_size = 200  # モデルウェイト + KV Cache host offloading用
    volume_type = "gp3"
    throughput  = 500
  }

  user_data = base64encode(templatefile("${path.module}/scripts/setup_tensorrt.sh", {
    model_s3_path       = var.model_s3_path
    tokens_per_block    = 32
    gpu_memory_fraction = 0.85
    host_cache_size_gb  = 30
  }))
}
```

**Large構成（EKS + Karpenter）** ではGPU Spot Instancesを優先するNodePoolを定義する:

```hcl
# Karpenter NodePool: GPU Spot優先
resource "kubectl_manifest" "gpu_nodepool" {
  yaml_body = yamlencode({
    apiVersion = "karpenter.sh/v1"
    kind       = "NodePool"
    metadata   = { name = "gpu-inference" }
    spec = {
      template.spec.requirements = [
        { key = "node.kubernetes.io/instance-type", operator = "In",
          values = ["g5.2xlarge", "g5.4xlarge", "p4d.24xlarge"] },
        { key = "karpenter.sh/capacity-type", operator = "In",
          values = ["spot", "on-demand"] },
      ]
      limits     = { "nvidia.com/gpu" = 16 }
      disruption = { consolidationPolicy = "WhenEmptyOrUnderutilized" }
    }
  })
}
```

### 運用・監視設定

**CloudWatch Logs Insights クエリ（主要メトリクス）**:

```
# KV Cache ヒット率の推移
fields @timestamp, @message
| parse @message "cache_hit_rate=*," as hit_rate
| stats avg(hit_rate) as avg_hit, p95(hit_rate) by bin(5m)

# TTFT レイテンシ分布
fields @timestamp, @message
| parse @message "ttft_ms=*," as ttft
| stats p50(ttft), p95(ttft), p99(ttft) by bin(5m)
```

**主要アラーム**: GPU利用率85%超過（スケールアウト検討）、KV Cacheヒット率50%未満（`tokens_per_block`・eviction policy見直し）をCloudWatch Alarmで監視する。Triton Inference Serverは `/metrics` エンドポイントでPrometheus形式のメトリクスを公開しており、KV Cache Event APIと組み合わせてキャッシュ状態を可視化できる。

### コスト最適化チェックリスト

- [ ] トラフィック量に応じた構成選択（A10G / A100 / H100）
- [ ] `tokens_per_block` をワークロードに合わせて調整
- [ ] Priority-based Eviction でシステムプロンプトに高優先度設定
- [ ] Host Offloading有効化（`secondary_offload_min_priority` デフォルト35）
- [ ] Reserved Instances / Spot Instances の併用
- [ ] KV Cache ヒット率・TTFT p95/p99の継続的モニタリング
- [ ] AWS Budgets + Cost Anomaly Detection 設定

## パフォーマンス最適化（Performance）

### NVIDIAが報告しているベンチマーク結果

| 最適化技術 | モデル | GPU | 改善指標 | 改善幅 |
|:---|:---|:---|:---|:---|
| KV Cache Early Reuse | 非公開 | H100 | TTFT | 最大5倍高速化 |
| Flexible Block Sizing（64→8トークン） | LLAMA-70B | H100 | TTFT | 最大7%改善 |
| KV Cache Offloading | 非公開 | H100（x86） | TTFT | 14倍高速化 |
| KV Cache Offloading | 非公開 | GH200 | TTFT | 28倍高速化 |
| Priority-based Eviction | 非公開 | 非公開 | Cache Hit Rate | 約20%向上 |

これらの数値はNVIDIAが自社ブログで報告したものであり、ワークロードの特性（システムプロンプト長、バッチサイズ、同時接続数等）によって実際の改善幅は変動する点に注意が必要である。

### チューニングパラメータの指針

TensorRT-LLM の KV Cache に関する主要な設定パラメータを以下にまとめる。

```python
from tensorrt_llm.executor import KvCacheConfig

kv_cache_config = KvCacheConfig(
    # GPU メモリの90%をKV Cacheに割り当て（デフォルト）
    free_gpu_memory_fraction=0.9,
    # ブロック再利用の有効化（デフォルトで有効）
    enable_block_reuse=True,
    # Host Offloading: 45GBのホストメモリを二次キャッシュに使用
    host_cache_size=45_000_000_000,
    # 優先度35未満のブロックはhost offloadingをスキップ
    secondary_offload_min_priority=35,
    # 部分的なブロック再利用を有効化（デフォルトで有効）
    enable_partial_reuse=True,
    # KV Cache Event API: イベントバッファサイズ（0で無効）
    event_buffer_max_size=16384,
)
```

**`tokens_per_block` の選択基準**: 短いプロンプト中心（<512トークン）なら8-16、中程度（512-2048）なら16-32、長い（>2048）なら32-64を推奨。NVIDIAのブログでも「短い入力シーケンスには小さいブロックが効果的で、長いシーケンスには大きいブロックが有利」と述べられている。70B以上のモデルでは `free_gpu_memory_fraction` を0.8程度に下げてOOMを回避する。

### KV Cache Event API による分散ルーティング

大規模環境では、各TensorRT-LLMインスタンスのKVキャッシュ状態に基づいてリクエストをルーティングすることで、キャッシュヒット率を向上できる。Event API（`KvCacheConfig(event_buffer_max_size=16384)` で有効化）は `CreatedData`、`StoredData`、`RemovedData`、`UpdatedData` の4種類のイベントを発行し、各ブロックの `blockHash`、`tokens`、`priority`、`cacheLevel`（0: GPU, 1: Host）を追跡する。

NVIDIAのドキュメントによれば、このEvent APIは「複数のexecutor間でのキャッシュ状態の集約を可能にし、キャッシュされたコンテンツの可用性に基づいてインテリジェントなリクエストルーティング決定を行える」と説明されている。ただし、**結果整合性（eventually consistent）**のビューである点に留意が必要である。

## 運用での学び（Production Lessons）

### KV Cache管理のモニタリング観点

NVIDIAのブログおよびドキュメントから読み取れる運用上の注意点をまとめる。

**バッチサイズとキャッシュ再利用の関係**: NVIDIAのドキュメントでは、「高いバッチサイズでは、多くのリクエストが先行リクエストの完了前に起動する可能性があるため、再利用が妨げられる場合がある」と指摘されている。Early Reuseはこの問題を部分的に解決するが、バッチスケジューリング戦略との組み合わせが重要となる。

**Host Offloading の有効活用**: KV Cacheをホストメモリにオフロードし、GPUメモリを推論に集中させつつ、オフロードされたブロックもプレフィックスマッチング対象として維持できる。NVIDIAはGrace-Hopper（GH200）で特に効果が高いと報告しており、NVLink-C2Cの高帯域接続によるものと考えられる。

**キャッシュセキュリティ**: マルチテナント環境では `cache_salt` パラメータでテナントごとにキャッシュを分離し、ユーザー間のキャッシュ流出を防止する。

**キャッシュの温め戦略**: 高トラフィック時間帯の前にシステムプロンプトでダミーリクエストを送信しキャッシュをウォームアップしておく。Priority-based Evictionで高優先度を付与すれば不用意なevictも防げる。

## 学術研究との関連（Academic Connection）

**PagedAttention（Kwon et al., 2023）**: vLLMで導入されたOS仮想メモリページングの概念をKV Cacheに適用した先駆的研究。TensorRT-LLMのブロックベース管理もこの考え方を基盤とし、その上にFlexible Block SizingやIntelligent Evictionを追加した形である。vLLMではブロックサイズが実装依存で固定される傾向があるのに対し、TensorRT-LLMはビルド時に柔軟に設定可能。

**SGLang / RadixAttention（Zheng et al., 2024）**: Radix Treeによるプレフィックス自動共有で最大5倍のスループット向上を報告。TensorRT-LLMもRadix Tree検索構造を採用しているが、Early Reuseによる「計算途中のキャッシュ共有」はSGLangにはない独自機能である。一方、SGLangのHiCacheによる多段キャッシングではSGLangが先行している。

**Anthropic Prompt Caching（2024）**: APIレベルでプレフィックスKV Cacheを保持し、100Kトークンでレイテンシ11.5秒→2.4秒の削減事例がある。セルフホスト型のTensorRT-LLMとは対象レイヤーが異なるが、プレフィックス再利用という基本アプローチは共通している。

### フレームワーク間の性能比較

SqueezeBitsの比較記事によれば、APC有効時にTensorRT-LLMはスループット約34.7%向上・TPOT約20.9%改善を示したのに対し、vLLMはスループット約13.3%向上・TPOT約9.8%改善にとどまり、高並行性環境でvLLMのAPC有効時に性能低下が発生するケースも指摘されている。

## まとめと実践への示唆

NVIDIAのTensorRT-LLMにおけるKV Cache最適化は、(1) Early ReuseによるTTFT最大5倍高速化、(2) Flexible Block Sizingによる部分再利用率向上（最大7%改善）、(3) Intelligent EvictionとPriority-based Evictionによるキャッシュヒット率約20%向上の3技術で構成されている。

本番環境ではワークロード特性に基づく `tokens_per_block` や優先度設定の最適化が重要であり、システムプロンプトを共有するエンタープライズチャットボットでEarly Reuseの効果が顕著に現れる。なお、これらはTensorRT-LLM固有の機能であり、フレームワーク選択にはモデルサポートやデプロイ容易性も含めた総合判断が必要である。

## 参考文献

- **Blog URL**: [5x Faster Time to First Token with NVIDIA TensorRT-LLM KV Cache Early Reuse](https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/)
- **Related Blog**: [Introducing New KV Cache Reuse Optimizations in NVIDIA TensorRT-LLM](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/)
- **TensorRT-LLM KV Cache Docs**: [KV Cache Reuse Documentation](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html)
- **TensorRT-LLM KV Cache System**: [KV Cache System Documentation](https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html)
- **vLLM / PagedAttention**: Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
- **SGLang / RadixAttention**: Zheng, L., et al. "SGLang: Efficient Execution of Structured Language Model Programs." 2024. [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- **Anthropic Prompt Caching**: [Prompt Caching - Claude API Docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- **vLLM vs TensorRT-LLM APC比較**: [SqueezeBits: Automatic Prefix Caching Comparison](https://blog.squeezebits.com/vllm-vs-tensorrtllm-12-automatic-prefix-caching-38189)
- **Related Zenn article**: [プロンプトキャッシュの本番運用設計](https://zenn.dev/0h_n0/articles/80b83bf28e8353)
