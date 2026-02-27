---
layout: post
title: "NeurIPS 2024論文解説: Toward Efficient Inference for Mixture of Experts — MoE推論のスループットを最大11.55倍改善する3手法"
description: "NeurIPS 2024で発表されたMoE推論最適化論文を解説。Dynamic Gating、Expert Buffering、Load Balancingの3手法でスループットを最大11.55倍改善しメモリ使用量を1.47倍削減する技術詳細。"
categories: [blog, paper, conference]
tags: [MoE, inference-optimization, NeurIPS, GPU, llm, qwen, gpu]
date: 2026-02-27 13:00:00 +0900
source_type: conference
conference: NeurIPS 2024
source_url: https://neurips.cc/virtual/2024/poster/93368
zenn_article: b1c2ee45a42db3
zenn_url: https://zenn.dev/0h_n0/articles/b1c2ee45a42db3
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Toward Efficient Inference for Mixture of Experts](https://neurips.cc/virtual/2024/poster/93368) (NeurIPS 2024) の解説記事です。

## 論文概要（Abstract）

本論文は、MoE（Mixture-of-Experts）モデルの推論効率化に取り組んだ研究である。著者らは、言語モデリングと機械翻訳のワークロードを体系的に特性分析し、非効率性の原因を特定した上で、3つの最適化手法を提案している。(1) **Dynamic Gating**はスループットを6.21〜11.55倍改善し、(2) **Expert Buffering**はGPUメモリの静的割り当てを1.47倍削減し、(3) **Load Balancing**はワークロードのロバスト性を向上させる。これらの手法はZenn記事で扱うQwen3.5-MoEモデルの推論効率を理解する上で直接的に関連する。

この記事は [Zenn記事: Qwen3.5×RTX 3090でバイブコーディング環境を構築する実践ガイド](https://zenn.dev/0h_n0/articles/b1c2ee45a42db3) の深掘りです。Zenn記事で扱うQwen3.5-35B-A3B（MoEモデル）の推論時に発生するエキスパート管理の課題と最適化手法を解説します。

## 情報源

- **会議名**: NeurIPS 2024（38th Conference on Neural Information Processing Systems）
- **年**: 2024
- **URL**: [https://neurips.cc/virtual/2024/poster/93368](https://neurips.cc/virtual/2024/poster/93368)
- **著者**: Haiyang Huang, Newsha Ardalani, Anna Sun, Liu Ke, Shruti Bhosale, Hsien-Hsin Lee, Carole-Jean Wu, Benjamin Lee
- **コード**: [https://github.com/hyhuang00/moe_inference](https://github.com/hyhuang00/moe_inference)

## カンファレンス情報

**NeurIPS**（Neural Information Processing Systems）は機械学習・計算神経科学分野の最高峰会議の1つであり、2024年のNeurIPSは38回目の開催となる。MoEモデルの推論効率化は、大規模LLMのデプロイメントコスト削減に直結するため、実用的な重要性が高い研究テーマである。

## 技術的詳細（Technical Details）

### MoE推論の3つの非効率性

著者らは、標準的なMoE推論における非効率性を以下の3カテゴリに分類している：

**1. 静的ゲーティングの非効率性**

標準的なTop-K MoEでは、各トークンに対して固定数（$K$個）のエキスパートを選択する：

$$
g_{i,t} = \begin{cases}
s_{i,t} & \text{if } s_{i,t} \in \text{TopK}(\{s_{j,t}\})_{j=1}^{N} \\
0 & \text{otherwise}
\end{cases}
$$

ここで $s_{i,t} = \text{Softmax}(\mathbf{h}_t^\top \mathbf{e}_i)$ はトークン $t$ とエキスパート $i$ の親和度スコアである。

問題点は、全トークンに対して常に $K$ 個のエキスパートを活性化する点にある。実際には、一部のトークンは少数のエキスパートで十分に処理でき、残りのエキスパートのゲーティングスコアは非常に小さい。これにより不必要な計算が発生する。

**2. エキスパートメモリの静的割り当て**

標準的な実装では、全エキスパートの重みをGPU VRAMに常時保持する。エキスパート数が多いモデル（Qwen3.5-35B-A3Bでは128個のルーティングエキスパート）では、大量のVRAMが「使われていないエキスパートの重み」に消費される。

**3. ロードインバランス**

Top-Kルーティングでは、一部のエキスパートに負荷が集中し、他のエキスパートがアイドル状態になる。GPUの計算リソースの無駄であり、スループットが低下する。

### 手法1: Dynamic Gating

著者らは、固定のTop-Kルーティングを動的なゲーティングに置き換えることを提案している。具体的には、ゲーティングスコアが閾値 $\tau$ を超えるエキスパートのみを活性化する：

$$
g_{i,t}^{\text{dynamic}} = \begin{cases}
s_{i,t} & \text{if } s_{i,t} > \tau \\
0 & \text{otherwise}
\end{cases}
$$

この動的ゲーティングにより、トークンごとに活性化するエキスパート数が可変になる。「簡単な」トークン（一般的な語彙等）は1〜2個のエキスパートで処理され、「難しい」トークン（専門用語等）は多くのエキスパートが活性化する。

著者らの実験結果によれば、この手法だけで言語モデリングのスループットが6.21〜11.55倍改善されている。

### 手法2: Expert Buffering

全エキスパートをGPU VRAMに常時保持する代わりに、アクティブなエキスパートのみをGPUに保持し、非アクティブなエキスパートをCPUメモリに退避させるバッファリング戦略を導入する。

```python
class ExpertBufferManager:
    """エキスパートバッファリングマネージャ

    アクティブなエキスパートのみGPU VRAMに保持し、
    非アクティブなエキスパートはCPUメモリに退避する。

    Args:
        total_experts: 全エキスパート数
        buffer_size: GPUバッファに保持するエキスパート数
    """
    def __init__(
        self,
        total_experts: int,
        buffer_size: int,
    ):
        self.total_experts = total_experts
        self.buffer_size = buffer_size
        self.gpu_buffer: dict[int, torch.Tensor] = {}
        self.cpu_storage: dict[int, torch.Tensor] = {}
        self.access_counts: dict[int, int] = {
            i: 0 for i in range(total_experts)
        }

    def get_expert(self, expert_id: int) -> torch.Tensor:
        """エキスパートの重みを取得

        GPUバッファにあればそのまま返し、
        なければCPUからロードしてLRUエビクションを実行。
        """
        self.access_counts[expert_id] += 1

        if expert_id in self.gpu_buffer:
            return self.gpu_buffer[expert_id]

        # GPUバッファが満杯ならLRU方式でエビクション
        if len(self.gpu_buffer) >= self.buffer_size:
            lru_id = min(
                self.gpu_buffer,
                key=lambda x: self.access_counts[x],
            )
            self.cpu_storage[lru_id] = (
                self.gpu_buffer[lru_id].cpu()
            )
            del self.gpu_buffer[lru_id]

        # CPUからGPUにロード
        self.gpu_buffer[expert_id] = (
            self.cpu_storage[expert_id].cuda()
        )
        return self.gpu_buffer[expert_id]
```

著者らの報告によれば、Expert Bufferingにより静的メモリ割り当てを1.47倍削減できる。これはRTX 3090（24GB VRAM）のような制約の厳しい環境で特に有効である。

### 手法3: Load Balancing

エキスパート間の負荷不均衡を動的に検出・修正する手法を導入している。ルーティング確率の分散を監視し、不均衡が検出された場合にゲーティング閾値を調整する。

これにより、特定のエキスパートへの負荷集中を防ぎ、GPU計算リソースの利用効率を向上させる。

## 実装のポイント（Implementation）

**Qwen3.5-35B-A3Bへの適用可能性**: Zenn記事で扱うQwen3.5-35B-A3Bは128個のルーティングエキスパートから約10個を選択するMoEアーキテクチャである。本論文のDynamic Gatingを適用すれば、トークンによっては10個未満のエキスパートで推論が可能になり、計算量とメモリアクセスを削減できる可能性がある。

**Expert BufferingとRTX 3090**: 128個のエキスパート全てを24GB VRAMに保持するのは、量子化を併用しても困難な場合がある。Expert Bufferingにより、頻繁に使用されるエキスパートのみをVRAMに保持し、残りをシステムRAMに退避させることで、VRAM制約を緩和できる。ただし、llama.cppの現行実装ではこの最適化はまだ統合されていない。

**llama.cppとの統合状況**: 2026年2月時点で、llama.cppにはDynamic GatingやExpert Bufferingの実装は含まれていない。これらの最適化は研究段階であり、実用化にはllama.cppやOllamaへの統合が必要である。

## 実験結果（Results）

著者らの実験結果をワークロード別に示す：

### スループット改善

| ワークロード | Dynamic Gatingの効果 | メモリ削減 |
|------------|---------------------|-----------|
| 言語モデリング | 6.21-11.55倍 | 1.36倍 |
| 機械翻訳（エンコーダ） | 5.75-10.98倍 | - |
| 機械翻訳（デコーダ） | 2.58-5.71倍 | - |

### Expert Bufferingの効果

著者らの報告によれば、Expert Bufferingにより静的メモリ割り当てを1.47倍削減できる。これはエキスパート数が多いモデルほど効果が大きい。

### ワークロード別の分析

言語モデリングとデコーダタスクでは、各トークンが順次処理されるため、エキスパートの活性化パターンが予測しやすく、バッファリングとDynamic Gatingの効果が高い。エンコーダタスクでは入力全体を並列処理するため、より多くのエキスパートが同時に必要になり、効果がやや控えめになる。

## 実運用への応用（Practical Applications）

**バイブコーディング環境への示唆**: Zenn記事で構築するバイブコーディング環境では、コード生成タスク（言語モデリングの一種）が主要なワークロードである。本論文の結果から、MoEモデルのコード生成ではDynamic Gatingにより6〜11倍のスループット改善が理論的に期待できるが、実際の実装（llama.cpp等）への統合が前提となる。

**将来的なllama.cppの最適化方向**: Expert Bufferingの概念は、llama.cppの `--n-gpu-layers` オプションとは異なるアプローチでVRAM効率を改善する。エキスパート単位での動的ロード/アンロードが実装されれば、RTX 3090でより大きなMoEモデルの実行が可能になる。

**Qwen3-Coder-Next（80B-A3B）への応用**: Zenn記事で言及されているQwen3-Coder-Next（80B、512エキスパート中10個選択）は、RTX 3090単体ではVRAM不足となるモデルである。Expert Bufferingが実装されれば、頻出エキスパートのみをVRAMに保持することで、このモデルの実用的な推論が可能になる可能性がある。

## 関連研究（Related Work）

- **DeepSeek-V2** (DeepSeek-AI 2024): MLA + DeepSeekMoEアーキテクチャ。共有エキスパートによるロード分散はLoad Balancingと類似のアプローチ。
- **Mixtral** (Jiang et al. 2024): 8x22B MoE。Top-2固定ルーティングを採用しており、Dynamic Gatingの適用余地がある。
- **FlexMoE** (Huang et al. 2024): GPU-CPUハイブリッドメモリでMoE推論を行う手法。Expert Bufferingと関連するが、FlexMoEは動的スケジューリングに焦点を当てている。
- **ScatterMoE** (Tan et al. 2024): スパース行列演算によるMoE実装の効率化。MegaBlocksの1.35倍の高速化とピークメモリ30%削減を達成。

## まとめと今後の展望

本論文は、MoE推論の3つの非効率性（静的ゲーティング、静的メモリ割り当て、ロードインバランス）を体系的に分析し、それぞれに対応する最適化手法（Dynamic Gating、Expert Buffering、Load Balancing）を提案した。著者らの実験によれば、言語モデリングで最大11.55倍のスループット改善が達成されている。

Zenn記事で扱うQwen3.5-MoEモデルの推論において、これらの手法が実装レベルで統合されれば、RTX 3090のような制約の厳しい環境でもMoE推論の効率を大幅に改善できる可能性がある。現時点ではllama.cppへの統合は未実施であるが、MoEモデルの普及に伴い、今後の実装が期待される。

## 参考文献

- **Conference URL**: [https://neurips.cc/virtual/2024/poster/93368](https://neurips.cc/virtual/2024/poster/93368)
- **Code**: [https://github.com/hyhuang00/moe_inference](https://github.com/hyhuang00/moe_inference)
- **OpenReview**: [https://openreview.net/forum?id=stXtBqyTWX](https://openreview.net/forum?id=stXtBqyTWX)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/b1c2ee45a42db3](https://zenn.dev/0h_n0/articles/b1c2ee45a42db3)

---

:::message
本記事はNeurIPS 2024論文 [Toward Efficient Inference for Mixture of Experts](https://neurips.cc/virtual/2024/poster/93368) の解説記事です。論文の主張・実験結果を正確に伝えることを目的としており、筆者自身が実験を行ったものではありません。内容の正確性については原論文をご確認ください。
:::
