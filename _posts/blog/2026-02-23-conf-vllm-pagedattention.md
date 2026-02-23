---
layout: post
title: "SOSP 2023論文解説: vLLM — PagedAttentionによるKVキャッシュメモリ管理の革新"
description: "OSの仮想メモリ・ページング手法をKVキャッシュに応用し、メモリ断片化をほぼゼロにしてスループット最大24倍向上を達成したvLLMの技術詳細解説"
categories: [blog, paper, conference]
tags: [vllm, pagedattention, kv-cache, llm-serving, memory-management, inference-optimization]
date: 2026-02-23 13:00:00 +0900
source_type: conference
conference: "SOSP 2023 (Symposium on Operating Systems Principles)"
source_url: https://arxiv.org/abs/2309.06180
zenn_article: 555a4e799660de
zenn_url: https://zenn.dev/0h_n0/articles/555a4e799660de
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) の解説記事です。

**注記**: 本論文はarXiv ID 2309.06180として公開されているが、正式にはSOSP 2023（Symposium on Operating Systems Principles）で採択されたvLLMシステムの論文 (Kwon et al., 2023) である。vLLMのPagedAttention手法はarXiv 2310.07240で詳述されている。

## 論文概要（Abstract）

著者らは、LLM推論サービングにおけるKVキャッシュのメモリ管理の非効率性を解決するPagedAttentionを提案している。OSの仮想メモリ管理（ページング）の手法をKVキャッシュに応用し、KVキャッシュを固定サイズのブロック（ページ）に分割して不連続なメモリ領域に格納する。著者らの報告によると、メモリ断片化による無駄を4%以下に削減し（従来手法では60〜80%の無駄）、スループットを既存システム比で最大24倍向上させている。

この記事は [Zenn記事: LangGraph×Claude Sonnet 4.6のプロンプトキャッシュ最適化でAgentic RAGコスト90%削減](https://zenn.dev/0h_n0/articles/555a4e799660de) の深掘りです。

## 情報源

- **会議名**: SOSP 2023（ACM Symposium on Operating Systems Principles）
- **年**: 2023
- **URL**: [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- **著者**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al.（UC Berkeley）
- **採択率**: SOSPは例年15〜20%程度の採択率を持つトップ会議

## カンファレンス情報

**SOSPについて**:
SOSP（ACM Symposium on Operating Systems Principles）はオペレーティングシステム・分散システム分野の最高峰会議の1つであり、2年に1度開催される。採択率は通常15〜20%程度であり、厳格な査読プロセスを経る。本論文がSOSPに採択されたことは、KVキャッシュのメモリ管理がシステム研究として高い評価を受けたことを示している。

## 技術的詳細（Technical Details）

### LLM推論のメモリボトルネック

LLMの自己回帰生成（autoregressive generation）では、各トークンの生成時に過去の全トークンのKey-Valueテンソルが必要となる。生成が進むにつれてKVキャッシュは線形に成長し、GPUメモリの大部分を占有する。

具体的には、LLaMA-13Bモデルの場合:

$$
\text{KV cache per token} = 2 \times L \times H \times d_k \times \text{precision}
$$

ここで、
- $L = 40$: レイヤー数
- $H = 40$: アテンションヘッド数
- $d_k = 128$: ヘッド次元数
- $\text{precision} = 2$: FP16のバイト数

$$
\text{KV cache per token} = 2 \times 40 \times 40 \times 128 \times 2 = 819,200 \text{ bytes} \approx 800 \text{ KB}
$$

1リクエストで2048トークン生成する場合、KVキャッシュだけで約1.6GBを消費する。A100 80GBでも、モデルパラメータ（26GB）を除くと複数リクエストの同時処理には厳しい制約がある。

### 従来手法の問題: メモリ断片化

著者らは、従来のLLM推論システム（FasterTransformer、Orca等）のKVキャッシュ管理に3種類のメモリ断片化があることを指摘している。

**1. 内部断片化（Internal Fragmentation）**: 最大シーケンス長でメモリを事前確保するため、実際の生成長が短い場合にメモリが無駄になる。

**2. 外部断片化（External Fragmentation）**: 異なるサイズのリクエストが終了・開始を繰り返すことで、連続したメモリ領域が確保できなくなる。

**3. 予約過多（Reservation Waste）**: バッチ処理のために全リクエストの最大可能長分のメモリを予約するが、多くのリクエストはその長さまで生成しない。

著者らの分析によると、従来手法ではKVキャッシュに割り当てたメモリの**60〜80%が断片化により無駄になっている**。

### PagedAttentionの仕組み

著者らは、OSの仮想メモリ管理のアナロジーでKVキャッシュを管理するPagedAttentionを提案している。

**OSの仮想メモリ → KVキャッシュ管理の対応**:

| OS概念 | PagedAttention対応 |
|-------|-------------------|
| プロセスのアドレス空間 | リクエストのKVキャッシュ |
| 仮想ページ | KVキャッシュの論理ブロック |
| 物理フレーム | GPUメモリ上の物理ブロック |
| ページテーブル | ブロックテーブル |
| ページフォールト | ブロック割り当て（on-demand） |

**ブロック構造**:

各ブロックは固定数のトークン（ブロックサイズ $B$）のKVテンソルを格納する。

$$
\text{Block size (bytes)} = 2 \times L \times H \times d_k \times B \times \text{precision}
$$

著者らのデフォルト設定は $B = 16$（16トークン/ブロック）である。

**Attention計算の修正**:

従来のAttention計算では、KVテンソルは連続メモリ上に格納されている前提だった。PagedAttentionでは、ブロックテーブルを介して不連続なブロックにアクセスする。

$$
\text{Attention}(q, K, V) = \text{softmax}\left(\frac{q \cdot [K_1 \| K_2 \| \cdots \| K_n]^T}{\sqrt{d_k}}\right) \cdot [V_1 \| V_2 \| \cdots \| V_n]
$$

ここで $K_i, V_i$ は$i$番目のブロックに格納されたKVテンソル。実装上は、各ブロックのアテンションスコアを個別に計算し、softmaxの数値安定性を保つためにonline softmax（Milakov & Gimelshein, 2018）を使用する。

### Prefix Sharing（共有プレフィックスキャッシュ）

PagedAttentionのブロック管理は、プレフィックス共有を自然にサポートする。複数のリクエストが共通のプレフィックス（システムプロンプト等）を持つ場合、そのプレフィックスに対応するブロックを共有できる。

```
リクエストA: [System Prompt][Query A]
リクエストB: [System Prompt][Query B]

ブロックテーブル:
A: [Block0, Block1, Block2, Block_A3, Block_A4]
B: [Block0, Block1, Block2, Block_B3, Block_B4]
     ^^^^^^^^^^^^^^^^^^^^^^
     共有ブロック（Copy-on-Writeで管理）
```

この機構は、OSのCopy-on-Write（CoW）ページ共有と同一の原理で動作する。共有ブロックは参照カウントで管理され、いずれかのリクエストがブロックを変更する必要が生じた場合のみコピーが作成される。

### アルゴリズムの擬似コード

```python
from dataclasses import dataclass, field
import torch


@dataclass
class KVBlock:
    """KVキャッシュの物理ブロック"""
    block_id: int
    k_cache: torch.Tensor  # shape: (block_size, num_heads, head_dim)
    v_cache: torch.Tensor  # shape: (block_size, num_heads, head_dim)
    num_filled: int = 0     # 使用済みスロット数
    ref_count: int = 1      # 参照カウント（CoW用）


@dataclass
class BlockTable:
    """リクエストのブロックテーブル（ページテーブルに相当）"""
    logical_blocks: list[int] = field(default_factory=list)
    # logical_blocks[i] = physical_block_id


class PagedKVCacheManager:
    """PagedAttention KVキャッシュマネージャ（著者らのアルゴリズムを再現）

    OSの仮想メモリマネージャに相当する。
    """

    def __init__(self, num_blocks: int, block_size: int,
                 num_layers: int, num_heads: int, head_dim: int):
        self.block_size = block_size
        self.free_blocks: list[int] = list(range(num_blocks))
        self.blocks: dict[int, KVBlock] = {}

        # 物理ブロックの事前割り当て
        for bid in range(num_blocks):
            self.blocks[bid] = KVBlock(
                block_id=bid,
                k_cache=torch.zeros(block_size, num_heads, head_dim),
                v_cache=torch.zeros(block_size, num_heads, head_dim),
            )

    def allocate_block(self) -> int:
        """空きブロックを割り当て（ページフォールトに相当）"""
        if not self.free_blocks:
            raise MemoryError("No free KV cache blocks available")
        block_id = self.free_blocks.pop()
        self.blocks[block_id].ref_count = 1
        self.blocks[block_id].num_filled = 0
        return block_id

    def free_block(self, block_id: int) -> None:
        """ブロックを解放（参照カウントが0になった場合）"""
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block_id)

    def copy_on_write(self, block_id: int) -> int:
        """Copy-on-Write: 共有ブロックの変更時にコピーを作成"""
        src_block = self.blocks[block_id]
        if src_block.ref_count == 1:
            return block_id  # 共有されていないのでコピー不要

        new_block_id = self.allocate_block()
        new_block = self.blocks[new_block_id]
        new_block.k_cache.copy_(src_block.k_cache)
        new_block.v_cache.copy_(src_block.v_cache)
        new_block.num_filled = src_block.num_filled

        src_block.ref_count -= 1
        return new_block_id

    def append_token(
        self,
        block_table: BlockTable,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """トークンのKVテンソルをブロックテーブルに追加"""
        if not block_table.logical_blocks:
            # 最初のブロックを割り当て
            bid = self.allocate_block()
            block_table.logical_blocks.append(bid)

        last_bid = block_table.logical_blocks[-1]
        block = self.blocks[last_bid]

        if block.num_filled >= self.block_size:
            # 現在のブロックが満杯、新しいブロックを割り当て
            new_bid = self.allocate_block()
            block_table.logical_blocks.append(new_bid)
            block = self.blocks[new_bid]

        # CoWチェック
        if block.ref_count > 1:
            new_bid = self.copy_on_write(last_bid)
            block_table.logical_blocks[-1] = new_bid
            block = self.blocks[new_bid]

        # KVテンソルを書き込み
        idx = block.num_filled
        block.k_cache[idx] = k
        block.v_cache[idx] = v
        block.num_filled += 1
```

## 実験結果（Results）

著者らはShareGPTおよびAlpacaデータセットで評価を行っている（論文Section 6より）。

| 構成 | 比較対象 | スループット向上倍率 |
|------|---------|-------------------|
| A100 80GB, OPT-13B | FasterTransformer | 14-24x |
| A100 80GB, OPT-13B | Orca | 2.2-4.3x |
| A100 80GB, OPT-175B (TP=8) | FasterTransformer | 3.5-8.0x |

メモリ効率（論文Table 2より）:

| 手法 | メモリ断片化率 | 同時処理可能リクエスト数 |
|------|-------------|---------------------|
| FasterTransformer | 60-80% | ベースライン |
| Orca | 30-50% | 1.5-2.0x |
| vLLM (PagedAttention) | <4% | 2.2-4.3x（Orca比） |

著者らの報告によると、PagedAttentionのメモリ管理により、A100 80GBで同時処理可能なリクエスト数が大幅に増加し、これがスループットの向上に直結している。

### Prefix Sharingの効果

共通のシステムプロンプト（2048トークン）を持つリクエストでPrefix Sharingを有効化した場合:

- メモリ削減: システムプロンプトのKVキャッシュを共有することで、リクエストあたりのメモリ使用量が約40%削減
- スループット向上: 共有により空いたメモリで追加リクエストを処理可能

## 実装のポイント（Implementation）

### vLLMの利用方法

vLLMはPythonパッケージとして提供されており、OpenAI互換APIとして起動できる。

```bash
# インストール
pip install vllm

# OpenAI互換APIサーバーとして起動
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enable-prefix-caching  # プレフィックスキャッシュの有効化
```

### ブロックサイズの選択

著者らはブロックサイズ16をデフォルトとしている。ブロックサイズの選択はトレードオフがある:

- **小さいブロック**: 内部断片化が減少するが、ブロックテーブルが大きくなりメタデータのオーバーヘッドが増加
- **大きいブロック**: メタデータのオーバーヘッドが減少するが、内部断片化が増加

### Flash Attentionとの統合

vLLMはFlash Attention 2 / xformersと統合可能である。PagedAttentionのブロックレベルアクセスパターンはFlash Attentionのタイル処理と互換性があり、両者の恩恵を同時に受けることができる。

## 実運用への応用（Practical Applications）

### Zenn記事との関連

Zenn記事で解説されているAnthropicのプロンプトキャッシュは、APIプロバイダのサーバーサイドで動作するキャッシュ機構である。vLLMのPagedAttentionは、自社でLLM推論サーバーを構築する場合の基盤技術として位置づけられる。

特にPrefix Sharing機能は、Anthropicの`cache_control`ブレークポイントと同様に、共通のシステムプロンプトやツール定義のKVキャッシュを複数リクエスト間で共有する。自社サーバー環境でAgentic RAGパイプラインを運用する場合、vLLMのPrefix Cachingが有効な選択肢となる。

### スケーリング

vLLMはTensor Parallelism（TP）によるマルチGPUスケーリングをサポートしている。各GPUが独立にPagedAttentionを実行し、KVキャッシュを分散管理する。

## 関連研究（Related Work）

- **Orca** (Yu et al., OSDI 2022): 反復レベルスケジューリングによるLLMサービング効率化。vLLMはOrcaのスケジューリングにPagedAttentionのメモリ管理を追加
- **Prompt Cache** (Gim et al., 2023): vLLMのPrefix Cachingはこの概念をシステムレベルで実装したもの
- **FlexGen** (Sheng et al., ICML 2023): CPUオフロードベースの高スループット推論。vLLMとは異なるアプローチだが、メモリ効率化の目標は共通

## まとめと今後の展望

vLLMのPagedAttentionは、OSのページング手法をKVキャッシュに応用することで、メモリ断片化をほぼゼロにし、LLM推論のスループットを最大24倍向上させた。この手法はLLM推論サービングの事実上の標準となっており、2026年2月時点でGitHubスター数は40,000以上に達している。

Prefix Sharing（共有プレフィックスキャッシュ）機能は、Anthropic/OpenAIのプロンプトキャッシュの自社サーバー版として、Agentic RAGパイプラインのコスト最適化に直接活用できる。

## 参考文献

- **Conference URL**: [https://dl.acm.org/doi/10.1145/3600006.3613165](https://dl.acm.org/doi/10.1145/3600006.3613165)
- **arXiv**: [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- **Code**: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/555a4e799660de](https://zenn.dev/0h_n0/articles/555a4e799660de)
