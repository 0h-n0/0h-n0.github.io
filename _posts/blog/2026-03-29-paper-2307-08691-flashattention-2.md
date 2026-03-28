---
layout: post
title: "論文解説: FlashAttention-2 — GPU並列性とワーク分割の改善によるAttention高速化"
description: "FlashAttention-2の技術詳細を解説。ループ順序変更・シーケンス方向並列化・warpレベル最適化により、A100上でGPU理論性能の50-73%を達成した手法とrvLLMへの影響"
categories: [blog, paper, arxiv]
tags: [FlashAttention, GPU, CUDA, attention, LLM, inference, optimization, A100, rust, vllm]
date: 2026-03-29 12:00:00 +0900
source_type: arxiv
arxiv_id: "2307.08691"
source_url: https://arxiv.org/abs/2307.08691
zenn_article: 48d89cb18bf0e1
zenn_url: https://zenn.dev/0h_n0/articles/48d89cb18bf0e1
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [arXiv:2307.08691 "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"](https://arxiv.org/abs/2307.08691) の解説記事です。

## 論文概要（Abstract）

Tri Dao（Princeton/Together AI）は、FlashAttention v1のGPU利用率を25-40%から**50-73%**に改善するFlashAttention-2を提案した。3つのアルゴリズム改善——非matmul FLOPsの削減、シーケンス方向の並列化、warpレベルのワーク分割最適化——により、A100 GPU上でFlashAttention v1比**約2倍**の高速化を達成したと報告されている。

この記事は [Zenn記事: rvLLM：Rust製vLLM代替で学ぶGPU推論エンジンの実装最適化](https://zenn.dev/0h_n0/articles/48d89cb18bf0e1) の深掘りです。rvLLMが15個のCUDAカーネルにFlashAttention-2を実装しており、そのアルゴリズムの原理を理解する上で必読の論文です。

## 情報源

- **arXiv ID**: 2307.08691
- **URL**: [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
- **著者**: Tri Dao
- **発表年**: 2023
- **分野**: cs.LG

## 背景と動機（Background & Motivation）

TransformerのAttention機構は $O(N^2)$ の時間・空間計算量を持ち、長シーケンスではメモリと速度の両面でボトルネックとなる。FlashAttention v1（Dao et al., 2022）は、GPUのメモリ階層（HBM↔SRAM）のI/O特性を活用したタイリング手法でメモリを $O(N)$ に削減し、2-4倍の実測高速化を達成した。

しかし、FlashAttention v1はGPUの理論性能（FLOP/s）の25-40%しか達成できていなかった。著者は2つの原因を特定している：

1. **warp間の非最適なワーク分割**: 共有メモリへの不要な読み書きが発生
2. **シーケンス方向の並列化不足**: バッチ×ヘッド数のみで並列化し、単一ヘッド内でのシーケンス方向並列化がない

## 主要な貢献（Key Contributions）

- **非matmul FLOPsの削減**: backward passのリスケーリング操作を効率化し、約15% FLOPs削減
- **ループ順序の変更**: 外側ループをQ方向、内側ループをK/V方向に変更し、スレッドブロック間の独立性を確保
- **シーケンス方向並列化**: (batch, head, seq_block) の3次元で並列化し、長シーケンスでのGPU occupancyを向上
- **warpレベルワーク分割の改善**: warpがK/V列方向を分担し、Qはレジスタに保持することで共有メモリアクセスを削減

## 技術的詳細（Technical Details）

### GPUメモリ階層

A100 GPUのメモリ構成：
- **SRAM（共有メモリ）**: 約19MB、帯域幅約19TB/s
- **HBM（高帯域幅メモリ）**: 40-80GB、帯域幅約2TB/s

SRAMはHBMの約10倍高速であり、FlashAttentionの核心はHBMアクセスを最小化するタイリング戦略にある。

### 標準AttentionのI/O問題

クエリ $Q$、キー $K$、バリュー $V \in \mathbb{R}^{N \times d}$ に対する標準Attention：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

標準実装では、$S = QK^\top$ をHBMに書き出し（$O(N^2)$ メモリ）、softmaxを計算し、再びHBMから読み出して $PV$ を計算する。HBMアクセスは $O(N^2)$ となる。

### FlashAttention v1のタイリング戦略

FlashAttention v1はQ, K, VをSRAMに収まるブロックに分割し、online softmax（タイル間で正規化定数を累積更新）でブロック単位にAttentionを計算する。

**online softmaxの更新式**（論文Section 3.1）：

$$
m_i^{\text{new}} = \max(m_i, \text{rowmax}(S_{ij}))
$$

$$
\ell_i^{\text{new}} = e^{m_i - m_i^{\text{new}}} \ell_i + \text{rowsum}(e^{S_{ij} - m_i^{\text{new}}})
$$

$$
O_i^{\text{new}} = \text{diag}(e^{m_i - m_i^{\text{new}}})^{-1} O_i + e^{S_{ij} - m_i^{\text{new}}} V_j
$$

ここで、$m_i$ は行方向最大値（数値安定化用）、$\ell_i$ はsoftmax正規化定数の累積和、$O_i$ は出力の累積和である。

**HBMアクセス量**: $O(N^2 d / M)$（$M$: SRAM容量）。$N^2 \gg M$ の場合、標準の $O(N^2)$ より大幅に削減。

### FlashAttention-2の3つの改善

#### 改善1: ループ順序の変更

**FlashAttention v1**: 外側K/Vブロック、内側Qブロック

**FlashAttention-2**: 外側Qブロック、内側K/Vブロック

```python
# FlashAttention-2 forward pass（概念的な擬似コード）
def flash_attention_2_forward(
    Q: torch.Tensor,  # (N, d)
    K: torch.Tensor,  # (N, d)
    V: torch.Tensor,  # (N, d)
    block_size_q: int,
    block_size_kv: int,
) -> torch.Tensor:
    """FlashAttention-2のフォワードパス

    Args:
        Q, K, V: クエリ・キー・バリュー行列
        block_size_q: Qブロックサイズ
        block_size_kv: K/Vブロックサイズ

    Returns:
        Attention出力
    """
    N, d = Q.shape
    O = torch.zeros_like(Q)

    # 外側ループ: Qブロック（並列化可能）
    for i in range(0, N, block_size_q):
        Q_i = Q[i:i+block_size_q]
        m_i = torch.full((block_size_q,), float('-inf'))
        l_i = torch.zeros(block_size_q)
        O_i = torch.zeros(block_size_q, d)

        # 内側ループ: K/Vブロック（逐次処理）
        for j in range(0, N, block_size_kv):
            K_j = K[j:j+block_size_kv]
            V_j = V[j:j+block_size_kv]

            S_ij = Q_i @ K_j.T / (d ** 0.5)

            # Online softmax更新
            m_i_new = torch.max(m_i, S_ij.max(dim=-1).values)
            P_ij = torch.exp(S_ij - m_i_new.unsqueeze(-1))
            l_i_new = torch.exp(m_i - m_i_new) * l_i + P_ij.sum(dim=-1)

            O_i = torch.diag(torch.exp(m_i - m_i_new)) @ O_i + P_ij @ V_j
            m_i, l_i = m_i_new, l_i_new

        O[i:i+block_size_q] = O_i / l_i.unsqueeze(-1)

    return O
```

**利点**: 各Qブロックが独立に計算可能なため、異なるスレッドブロック間でQブロックを並列処理できる。FlashAttention v1では同一Qブロックが複数のK/Vイテレーションで更新されるため、スレッドブロック間の同期が必要だった。

#### 改善2: シーケンス方向並列化

**FlashAttention v1**: (batch, head) の2次元で並列化

**FlashAttention-2**: (batch, head, **seq_block**) の3次元で並列化

長シーケンス・小バッチの推論では、batch × head数だけではGPUのSM（Streaming Multiprocessor）を埋め切れない。シーケンス方向にも並列化することで、GPU occupancyが向上する。

#### 改善3: warpレベルワーク分割

**FlashAttention v1**: 各warpがO（出力）の異なる行を担当。K, VはすべてのwarpがSRAMから読み取るため、共有メモリ帯域幅がボトルネック。

**FlashAttention-2**: warpがK/Vの列方向を分担。Qはレジスタに保持し、warp shuffle命令で交換。共有メモリへの書き込みを大幅に削減。

### I/O計算量の理論分析

FlashAttentionのHBMアクセス量（論文Theorem 1より）：

$$
\Theta\left(\frac{N^2 d}{M}\right)
$$

これは情報理論的な下界と一致し、**I/O最適**であることが証明されている。FlashAttention-2は漸近的な計算量は同一だが、定数係数を改善している。

### MQA/GQAサポート

FlashAttention-2はMulti-Query Attention（MQA）とGrouped-Query Attention（GQA）をネイティブサポートする：

- **MQA**: 複数のQヘッドが単一のK/Vヘッドを共有。K/Vを1回ロードし、複数Qヘッドで再利用
- **GQA**: $h$ 個のQヘッドを $g$ グループに分割し、各グループが1つのK/Vヘッドを共有

KVキャッシュサイズが $h/g$ 倍に削減されるため、推論時のメモリ効率が向上する。

## 実装のポイント（Implementation）

**CUDA vs Triton**: FlashAttention-2はCUDA C++（高性能）とTriton（可読性・ポータビリティ）の両実装を提供。warp shuffle命令やレジスタ割り当ての直接制御が必要なため、CUDA版がTriton版を上回る。

**rvLLMとの関連**: rvLLMは15個のCUDAカーネルをPTXで手書きしており、そのうちFlashAttention-2カーネルが最も複雑な実装である。共有メモリ並列リダクション、warpレベル削減、ベクトル化float4ロードを組み合わせ、GQA対応のPagedAttention V2カーネルとして統合している。

**精度要件**: FP16またはBF16が必要。FP32は非対応。rvLLMがPhase 2でFP32→FP16に切り替えた際にFlashAttention-2の恩恵を受けた理由はここにある。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト |
|------|--------------|---------|-----------|
| **Small** | ~3,000 | Lambda + Bedrock | $50-150 |
| **Medium** | ~30,000 | ECS Fargate + Bedrock | $300-800 |
| **Large** | 300,000+ | EKS + Karpenter + Spot | $2,000-5,000 |

**コスト試算の注意事項**:
上記は2026年3月時点のAWS ap-northeast-1料金に基づく概算値です。最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください。

### Terraformインフラコード

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "flashattn-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["ap-northeast-1a", "ap-northeast-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]

  enable_nat_gateway   = false
  enable_dns_hostnames = true
}

resource "aws_lambda_function" "inference" {
  filename      = "lambda.zip"
  function_name = "flashattn-inference"
  role          = aws_iam_role.lambda_bedrock.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 60
  memory_size   = 1024
}

resource "aws_budgets_budget" "monthly" {
  name         = "flashattn-monthly"
  budget_type  = "COST"
  limit_amount = "5000"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["ops@example.com"]
  }
}
```

### コスト最適化チェックリスト

- [ ] Spot Instances優先（最大90%削減）
- [ ] Reserved Instances: 1年コミットで72%削減
- [ ] Bedrock Batch API使用で50%割引
- [ ] Prompt Caching有効化で30-90%削減
- [ ] AWS Budgets設定（80%で警告）
- [ ] CloudWatch: GPU利用率・トークン使用量モニタリング
- [ ] Cost Anomaly Detection有効化
- [ ] タグ戦略: 環境別コスト可視化
- [ ] S3ライフサイクル: キャッシュ自動削除
- [ ] 開発環境: 夜間GPU停止

## 実験結果（Results）

### フォワードパス速度

A100 80GB上でのAttention計算速度（論文Table 1, causal attention）：

| 手法 | TFLOPs/s | 理論性能比 |
|---|---|---|
| PyTorch naive | ~3 | ~3% |
| xFormers memory-efficient | ~10-12 | ~15% |
| FlashAttention v1 | 20-30 | 25-40% |
| **FlashAttention-2** | **40-60** | **50-73%** |

### シーケンス長別の高速化

FlashAttention v1に対するFlashAttention-2の高速化倍率（論文Figure 3, A100）：

| シーケンス長 | FA v1 (TFLOPs/s) | FA-2 (TFLOPs/s) | 高速化 |
|---|---|---|---|
| 512 | 25 | 40 | 1.6× |
| 1024 | 30 | 50 | 1.7× |
| 4096 | 30 | 58 | 1.9× |
| 8192 | 28 | 60 | 2.1× |
| 16384 | 25 | 58 | **2.3×** |

シーケンス長が長くなるほど高速化倍率が増大する。これはシーケンス方向並列化の効果であり、長コンテキスト推論でFlashAttention-2の恩恵が特に大きいことを示している。

### エンドツーエンドの訓練高速化

GPT-3相当モデルの4×A100訓練（論文Table 2）：

| モデル | PyTorch比 FA v1 | PyTorch比 FA-2 |
|---|---|---|
| GPT3-1.3B | 1.8× | 2.0× |
| GPT3-2.7B | 2.0× | 2.2× |
| GPT3-6.7B | 2.2× | 2.4× |

## 実運用への応用（Practical Applications）

FlashAttention-2は事実上すべての現代的LLM推論・訓練で採用されている標準技術である。PyTorch 2.0以降の`F.scaled_dot_product_attention`はFlashAttention-2を内部的に呼び出す。

**rvLLMへの影響**: rvLLMがPhase 2でFP16に切り替えた際、cuBLASのhgemmとFlashAttention-2の組み合わせにより、Phase 1（FP32）の8,339 tok/sから2.6倍の高速化を達成している。FlashAttention-2の恩恵は、Tensor Coreの活用とメモリ帯域幅の削減の両面で現れる。

## 関連研究（Related Work）

- **FlashAttention v1 (Dao et al., 2022)**: I/O-awareなタイリング手法の原論文。FlashAttention-2はこの上に構築
- **FlashAttention-3 (Dao & Gu, 2024)**: H100のWGMMAとTMAを活用し、最大740 TFLOPs/sを達成。FlashInferがMLSys 2025最優秀論文賞を受賞
- **PagedAttention (Kwon et al., SOSP 2023)**: FlashAttention-2のタイリングはPagedAttentionの非連続KVレイアウトと互換性がある

## まとめと今後の展望

FlashAttention-2は、GPU利用率を25-40%→50-73%に改善した実用的に重要な最適化である。3つの改善（ループ順序変更、シーケンス並列化、warp分割改善）はいずれもアルゴリズムの正確性を保ちながらハードウェア利用率のみを改善している点が特徴的である。rvLLMのようなRust製推論エンジンがPTXカーネルでFlashAttention-2を再実装していることは、この手法がハードウェア効率の観点で不可欠であることを示している。

## 参考文献

- **arXiv**: [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
- **Code**: [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) (BSD-3)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/48d89cb18bf0e1](https://zenn.dev/0h_n0/articles/48d89cb18bf0e1)
