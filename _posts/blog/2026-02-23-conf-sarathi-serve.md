---
layout: post
title: "OSDI 2024論文解説: Sarathi-Serve — Chunked-Prefillsで実現するLLM推論のスループット・レイテンシ最適化"
description: "Prefillのチャンク分割とStall-Freeスケジューリングでvllm比2.6-5.6倍の容量向上を達成"
categories: [blog, paper, conference]
tags: [LLM-inference, scheduling, chunked-prefill, latency, throughput, OSDI]
date: 2026-02-23 13:00:00 +0900
source_type: conference
conference: "OSDI 2024"
source_url: https://arxiv.org/abs/2403.02310
zenn_article: a5be5c172a5a99
zenn_url: https://zenn.dev/0h_n0/articles/a5be5c172a5a99
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve (arXiv:2403.02310, OSDI 2024)](https://arxiv.org/abs/2403.02310) の解説記事です。

## 論文概要（Abstract）

Sarathi-Serveは、LLM推論サービングにおけるスループットとレイテンシのトレードオフを解消するスケジューラである。著者ら（Microsoft ResearchのAmey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee）は、Prefillフェーズのチャンク分割（Chunked-Prefills）とDecode-Maximalバッチングの組み合わせにより、進行中のDecodeリクエストを停止させることなく新規リクエストをバッチに追加するStall-Freeスケジューリングを実現した。著者らの実験では、vLLMと比較してMistral-7Bで2.6倍、Yi-34Bで3.7倍、Falcon-180Bで5.6倍のサービング容量向上が報告されている。

この記事は [Zenn記事: LangChain LCEL実践ガイド：LLMチェーンのレイテンシを50%削減する最適化手法](https://zenn.dev/0h_n0/articles/a5be5c172a5a99) の深掘りです。Zenn記事ではLCELのストリーミング機能やバッチ処理による応答性能の改善を解説していますが、本記事ではLLMサーバー側のスケジューリング最適化という基盤技術を深掘りします。

## 情報源

- **会議名**: OSDI 2024（USENIX Symposium on Operating Systems Design and Implementation）
- **年**: 2024（7月開催）
- **URL**: [https://arxiv.org/abs/2403.02310](https://arxiv.org/abs/2403.02310)
- **著者**: Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee
- **所属**: Microsoft Research, Georgia Tech
- **コード**: [https://github.com/microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve)

## カンファレンス情報

**OSDIについて**:
- OSDIはオペレーティングシステム・分散システム分野のトップカンファレンスの一つ
- USENIXが主催し、採択率は通常15-20%程度
- Sarathi-ServeはLLMサービングの効率化という実用的な貢献で採択された

## 技術的詳細（Technical Details）

### LLM推論の2フェーズ問題

LLM推論は本質的に2つの異なるフェーズで構成される：

1. **Prefillフェーズ**: 入力プロンプト全体を並列処理し、最初のトークンを生成する。GPU利用率が高い（Compute-bound）
2. **Decodeフェーズ**: 1トークンずつ逐次生成する。GPU利用率が低い（Memory-bandwidth-bound）

この2フェーズの計算特性の違いが、スループットとレイテンシのトレードオフの根本原因である。

$$
T_{\text{iteration}} = \max(T_{\text{prefill}}, T_{\text{decode}})
$$

従来のシステム（vLLM等）では、PrefillとDecodeを同一イテレーションで混在実行するため、長いPrefill処理が進行中のDecodeリクエストのTBT（Time-Between-Tokens）を悪化させる。

### Chunked-Prefills

Sarathi-Serveの中核技術であるChunked-Prefillsは、長いPrefillリクエストをチャンクに分割して複数のイテレーションに分散実行する。

$$
\text{Prefill}(P) = \text{Chunk}_1(P) \circ \text{Chunk}_2(P) \circ \ldots \circ \text{Chunk}_K(P)
$$

ここで、
- $P$: Prefillリクエスト（プロンプトトークン列）
- $K = \lceil |P| / C \rceil$: チャンク数
- $C$: チャンクサイズ（ハイパーパラメータ）
- $\circ$: 逐次実行

チャンクサイズ $C$ はTTFTとTBTのトレードオフを制御するパラメータである：
- **$C$ が大きい**: TTFTは短くなるが、Decodeリクエストのストール（停止）時間が長くなる
- **$C$ が小さい**: Decodeリクエストへの影響は小さくなるが、Prefillの完了（= TTFT）が遅くなる

```python
from dataclasses import dataclass

@dataclass
class ChunkedPrefillConfig:
    """Chunked-Prefillsの設定

    Args:
        chunk_size: Prefillチャンクのトークン数
        max_batch_tokens: バッチあたりの最大トークン数
    """
    chunk_size: int = 512
    max_batch_tokens: int = 4096

def split_prefill_into_chunks(
    prompt_tokens: list[int],
    chunk_size: int,
) -> list[list[int]]:
    """Prefillリクエストをチャンクに分割

    Args:
        prompt_tokens: プロンプトのトークンID列
        chunk_size: チャンクサイズ（トークン数）

    Returns:
        チャンクのリスト
    """
    chunks = []
    for i in range(0, len(prompt_tokens), chunk_size):
        chunks.append(prompt_tokens[i:i + chunk_size])
    return chunks
```

### Decode-Maximal Batching

Chunked-Prefillsと組み合わせるDecode-Maximal Batchingは、各イテレーションで進行中のDecodeリクエストを最優先でバッチに含め、残りの容量をPrefillチャンクで埋める戦略である。

```python
def decode_maximal_batch(
    running_decodes: list["Request"],
    pending_prefills: list["PrefillChunk"],
    max_batch_tokens: int,
) -> tuple[list["Request"], list["PrefillChunk"]]:
    """Decode-Maximalバッチの構築

    Args:
        running_decodes: 進行中のDecodeリクエスト（各1トークン生成）
        pending_prefills: 待機中のPrefillチャンク
        max_batch_tokens: バッチの最大トークン数

    Returns:
        (バッチに含めるDecodeリクエスト, バッチに含めるPrefillチャンク)
    """
    batch_decodes = list(running_decodes)  # Decodeを全て含める
    used_tokens = len(batch_decodes)  # Decodeは各1トークン

    batch_prefills = []
    remaining_tokens = max_batch_tokens - used_tokens

    for chunk in pending_prefills:
        if chunk.token_count <= remaining_tokens:
            batch_prefills.append(chunk)
            remaining_tokens -= chunk.token_count
        else:
            break

    return batch_decodes, batch_prefills
```

### Stall-Free Scheduling

上記の2つの手法を組み合わせることで、Stall-Freeスケジューリングが実現される。新しいPrefillリクエストが到着しても、進行中のDecodeリクエストは停止せず、次のイテレーションでPrefillチャンクとDecodeリクエストが共存する。

$$
\text{TBT}_{\text{Sarathi}} \leq \frac{C + D}{T_{\text{GPU}}}
$$

ここで、
- $C$: Prefillチャンクサイズ
- $D$: Decodeバッチサイズ（進行中のリクエスト数）
- $T_{\text{GPU}}$: GPU処理速度（tokens/s）

一方、vLLMのTBTは：

$$
\text{TBT}_{\text{vLLM}} \leq \frac{|P| + D}{T_{\text{GPU}}}
$$

$C \ll |P|$ であるため、Sarathi-ServeのTBTはvLLMと比較して大幅に小さくなる。

### Uniform Batch Sizes

著者らの論文によると、Chunked-Prefillsにはもう一つの利点がある。各イテレーションのバッチサイズ（トークン数）が均一化されることで、パイプライン並列化を使用する際のパイプラインバブル（アイドル時間）が削減される。

## 実装のポイント（Implementation）

### vLLMとの統合

Sarathi-Serveの手法はvLLMにも取り込まれており（chunked prefillオプション）、LCELから呼び出すOpenAI互換サーバーの性能に直接影響する。

```python
# vLLMサーバーの起動（chunked prefill有効化）
# vllm serve --model meta-llama/Llama-2-7b-chat-hf \
#            --enable-chunked-prefill \
#            --max-num-batched-tokens 4096
```

### LCELのストリーミングとの関連

Zenn記事で解説されているLCELの`.stream()`メソッドは、アプリケーション層でTTFTを最小化する手法である。Sarathi-Serveはサーバー層でTBTを最小化する手法であり、両者は相補的に機能する。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Sarathi-Serve（vLLM）をバックエンドとするLCELチェーン
model = ChatOpenAI(
    model="meta-llama/Llama-2-7b-chat-hf",
    openai_api_base="http://localhost:8000/v1",  # vLLMサーバー
    streaming=True,  # ストリーミング有効化
)

chain = (
    ChatPromptTemplate.from_template("{query}")
    | model
    | StrOutputParser()
)

# ストリーミング実行: Sarathi-ServeのTBT最適化 + LCELのTTFT最小化
for chunk in chain.stream({"query": "RAGの仕組みを説明して"}):
    print(chunk, end="", flush=True)
```

**階層的な最適化の整理**:

| レイヤー | 最適化手法 | 対象メトリクス |
|---------|-----------|-------------|
| **アプリケーション層** | LCEL `.stream()` | TTFT（ユーザー体感） |
| **アプリケーション層** | LCEL `RunnableParallel` | 全体レイテンシ |
| **アプリケーション層** | LCEL `.batch()` | スループット |
| **サーバー層** | Sarathi-Serve Chunked-Prefills | TBT |
| **サーバー層** | Decode-Maximal Batching | スループット + TBT |
| **システム層** | RAGO スケジューリング | QPS + TTFT |

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $50-150 | Lambda + Bedrock |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $300-800 | ECS Fargate + Bedrock |
| **Large** | 300,000+ (10,000/日) | Container | $2,000-5,000 | EKS + vLLM + GPU Spot |

**Large構成でのSarathi-Serve活用**:

vLLMのchunked prefillオプションを有効化することで、Sarathi-Serveの最適化を直接活用できる。

```yaml
# EKS上のvLLMデプロイメント（Kubernetes manifest）
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-sarathi
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - "--model"
          - "meta-llama/Llama-2-7b-chat-hf"
          - "--enable-chunked-prefill"
          - "--max-num-batched-tokens"
          - "4096"
        resources:
          limits:
            nvidia.com/gpu: 1
```

**コスト試算の注意事項**:
- 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です
- GPU Spot Instancesは最大90%割引だが、回収リスクがあるため冗長構成が必要
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

```hcl
# vLLM + Sarathi-Serve のEKSデプロイ
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "vllm-sarathi-cluster"
  cluster_version = "1.31"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = true
  enable_cluster_creator_admin_permissions = true
}

# Karpenter: GPU Spot Instances自動管理
resource "kubectl_manifest" "karpenter_gpu" {
  yaml_body = <<-YAML
    apiVersion: karpenter.sh/v1
    kind: NodePool
    metadata:
      name: gpu-spot
    spec:
      template:
        spec:
          requirements:
            - key: karpenter.sh/capacity-type
              operator: In
              values: ["spot"]
            - key: node.kubernetes.io/instance-type
              operator: In
              values: ["g5.xlarge", "g5.2xlarge"]
          nodeClassRef:
            group: karpenter.k8s.aws
            kind: EC2NodeClass
            name: default
      limits:
        cpu: "32"
        memory: "128Gi"
        nvidia.com/gpu: "4"
  YAML
}
```

### コスト最適化チェックリスト

- [ ] vLLMの`--enable-chunked-prefill`を有効化（TBT改善）
- [ ] GPU Spot Instances + Karpenterで最大90%コスト削減
- [ ] `max-num-batched-tokens`の最適値をベンチマークで決定
- [ ] EKSノードプールの自動スケールダウン（夜間0台）
- [ ] Bedrock使用時はPrompt Caching有効化で30-90%削減
- [ ] AWS Budgets: GPU費用の月額予算設定
- [ ] CloudWatch: TBT・TTFTメトリクス監視

## 実験結果（Results）

著者らの論文で報告された主要な実験結果：

| モデル | GPU構成 | ベースライン（vLLM） | Sarathi-Serve | 容量向上 |
|--------|---------|-------------------|--------------|---------|
| Mistral-7B | A100 ×1 | 1.0x | 2.6x | **2.6倍** |
| Yi-34B | A100 ×2 | 1.0x | 3.7x | **3.7倍** |
| Falcon-180B | パイプライン並列 | 1.0x | 5.6x | **5.6倍** |

（論文の実験結果より）

**分析ポイント**:
- モデルサイズが大きくなるほど改善幅が拡大する傾向がある。これはPrefillフェーズの計算量がモデルサイズに比例して増大し、Chunked-Prefillsによる分割効果が大きくなるためと著者らは分析している
- Falcon-180Bでの5.6倍の改善は、パイプライン並列化との組み合わせによるUniform Batch Sizeの効果が加わったものである
- TBTの改善は特にリアルタイムチャット応用で重要であり、ユーザーのトークン間待ち時間を大幅に短縮する

## 実運用への応用（Practical Applications）

Sarathi-Serveの手法は、以下の実運用シナリオに直接適用可能である：

1. **LLMサービングの効率化**: vLLMのchunked prefillオプションとして既に実装されており、設定変更のみで利用可能
2. **リアルタイムチャットボット**: TBTの改善により、ユーザー体感のトークン生成速度が向上。LCELのストリーミングと組み合わせることで、アプリケーション層とサーバー層の両方で最適化
3. **マルチモデルサービング**: Chunked-Prefillsの原理は異なるモデルサイズに対して一般的に適用可能

**制約と限界**: Chunked-Prefillsはchunk size $C$ のチューニングが必要であり、ワークロード特性（プロンプト長の分布、リクエストレート）に応じて最適値が変わる。著者らの論文ではこの調整の自動化は今後の課題として言及されている。

## 関連研究（Related Work）

- **vLLM** (Kwon et al., 2023): PagedAttentionによるKVキャッシュ管理の効率化。Sarathi-ServeはvLLMをベースラインとし、スケジューリング層で改善を追加
- **Orca** (Yu et al., 2022): Continuous batchingによるLLM推論効率化の先駆的研究。Sarathi-ServeはOrcaのcontinuous batchingをChunked-Prefillsで改良
- **PipeRAG** (Jiang et al., 2024): RAGパイプライン内での検索・生成の並列化。Sarathi-ServeのUniform Batch Size効果はパイプライン並列化の効率向上に寄与

## まとめと今後の展望

Sarathi-Serveは、LLM推論サービングにおけるスループットとレイテンシのトレードオフをChunked-PrefillsとDecode-Maximal Batchingで解消した研究である。vLLMと比較して最大5.6倍のサービング容量向上は、LLMサービングのコスト効率を大幅に改善する。LCELのアプリケーション層最適化（ストリーミング、バッチ処理、並列化）と組み合わせることで、エンドユーザーの体感性能を多層的に向上させることが可能である。

## 参考文献

- **arXiv**: [https://arxiv.org/abs/2403.02310](https://arxiv.org/abs/2403.02310)
- **OSDI 2024**: [https://www.usenix.org/conference/osdi24/presentation/agrawal](https://www.usenix.org/conference/osdi24/presentation/agrawal)
- **Code**: [https://github.com/microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/a5be5c172a5a99](https://zenn.dev/0h_n0/articles/a5be5c172a5a99)
