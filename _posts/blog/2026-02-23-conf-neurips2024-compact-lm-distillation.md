---
layout: post
title: "NeurIPS 2024論文解説: Compact Language Models via Pruning and Knowledge Distillation (Minitron)"
description: "NVIDIAのMinitronファミリー。大規模LLMのプルーニングと知識蒸留を組み合わせ、学習コスト40分の1で同等性能の圧縮モデルを生成"
categories: [blog, paper, conference]
tags: [knowledge-distillation, pruning, LLM, compression, NeurIPS, NVIDIA, bedrock]
date: 2026-02-23 12:00:00 +0900
source_type: conference
conference: "NeurIPS 2024"
source_url: https://arxiv.org/abs/2407.14679
zenn_article: f5fa165860f5e8
zenn_url: https://zenn.dev/0h_n0/articles/f5fa165860f5e8
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Compact Language Models via Pruning and Knowledge Distillation (arXiv:2407.14679)](https://arxiv.org/abs/2407.14679)（NeurIPS 2024採択）の解説記事です。

## 論文概要（Abstract）

著者らは、大規模LLMをプルーニング（枝刈り）した後に知識蒸留で再学習することで、スクラッチ学習の3%未満のデータ量で同等性能のコンパクトモデルを生成する手法を提案している。Nemotron-4 15Bモデルから8Bおよび4Bのモデルを導出し、スクラッチ学習と比較して最大40倍少ないトレーニングトークンで、MMLUスコアが最大16ポイント向上したと報告されている。生成されたMinitronファミリーは、Mistral 7B、Gemma 7B、Llama-3 8Bと競合的な性能を示す。

この記事は [Zenn記事: Bedrock Intelligent Prompt Routingで社内RAGコスト最大60%削減](https://zenn.dev/0h_n0/articles/f5fa165860f5e8) の深掘りです。

## 情報源

- **会議名**: NeurIPS 2024（Thirty-eighth Conference on Neural Information Processing Systems）
- **年**: 2024
- **URL**: [https://arxiv.org/abs/2407.14679](https://arxiv.org/abs/2407.14679)
- **著者**: Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, Pavlo Molchanov（NVIDIA）
- **採択形式**: Conference Paper

## カンファレンス情報

**NeurIPS**について: NeurIPS（Neural Information Processing Systems）は機械学習・深層学習分野の最高峰国際会議の1つであり、採択率は通常25〜30%程度である。Minitron論文がNeurIPS 2024に採択されたことは、LLM圧縮手法の実用的重要性が学術コミュニティに広く認められていることを示している。

## 技術的詳細（Technical Details）

### Bedrock Model Distillationとの関連

この研究はAmazon Bedrock Model Distillation（Zenn記事のLayer 3）と密接に関連する。Bedrock Model Distillationは教師モデル（Sonnet）の知識を生徒モデル（Haiku）に蒸留する機能であり、Minitron論文はその理論的・実験的基盤を提供している。

### 圧縮パイプラインの概要

Minitronの圧縮パイプラインは2つのフェーズで構成される。

```
┌─────────────────────┐    ┌──────────────────────┐    ┌───────────────────┐
│   Phase 1: Pruning   │    │  Phase 2: Distillation│    │   Final Model     │
│                      │    │                       │    │                   │
│ Nemotron-4 15B       │───▶│ Pruned 8B/4B          │───▶│ Minitron 8B/4B    │
│ (教師モデル)          │    │ + 蒸留による再学習      │    │ (圧縮モデル)       │
│                      │    │ (<3% データ量)          │    │                   │
└─────────────────────┘    └──────────────────────┘    └───────────────────┘
```

### Phase 1: マルチ軸プルーニング

著者らは4つの軸に沿ったプルーニング戦略を体系的に探索している。

**1. Depth Pruning（層方向）**

Transformerの層を選択的に除去する。重要度スコアは以下で計算される：

$$
\text{Importance}(\ell) = \frac{1}{|D_{\text{cal}}|} \sum_{x \in D_{\text{cal}}} \| h^{(\ell+1)}(x) - h^{(\ell)}(x) \|_2
$$

ここで、
- $h^{(\ell)}(x)$: 入力 $x$ の第 $\ell$ 層における隠れ状態
- $D_{\text{cal}}$: 校正データセット（1024サンプル程度）
- 重要度が低い層（= 入出力の差が小さい層）を除去

**2. Width Pruning（幅方向）**

各層のニューロン数（hidden dimension）を削減する。Attention headとFFN intermediate dimensionを同時に削減。

$$
\text{Score}(n_j) = \frac{1}{|D_{\text{cal}}|} \sum_{x \in D_{\text{cal}}} |a_j(x)| \cdot \| w_j \|_2
$$

ここで、
- $a_j(x)$: ニューロン $j$ の活性化値
- $w_j$: ニューロン $j$ に接続する重みベクトル
- 活性化値と重みの積が小さいニューロンを除去

**3. Attention Pruning**

Attention headを選択的に除去。Multi-Head Attention (MHA) からGrouped-Query Attention (GQA)への変換も含む。

**4. MLP Pruning**

FFN（Feed-Forward Network）のintermediate dimensionを削減。

### Phase 2: 知識蒸留

プルーニング後のモデルを教師モデル（元の15B）の知識で再学習する。蒸留損失は以下の通りである：

$$
\mathcal{L}_{\text{distill}} = \alpha \cdot \mathcal{L}_{\text{CE}}(y, \hat{y}_S) + (1 - \alpha) \cdot T^2 \cdot \text{KL}(p_T \| p_S)
$$

ここで、
- $\mathcal{L}_{\text{CE}}$: 正解ラベルとの交差エントロピー損失
- $\text{KL}(p_T \| p_S)$: 教師モデルと生徒モデルのKLダイバージェンス
- $T$: 温度パラメータ（ソフトラベルの平滑化度）
- $\alpha$: 損失の重み（著者らは $\alpha = 0.5$, $T = 2.0$ を使用と報告）
- $p_T$: 教師モデルの出力確率分布
- $p_S$: 生徒モデルの出力確率分布

**重要な発見**: 著者らは蒸留において、元の学習データのわずか3%未満で十分な性能回復が可能であることを実験的に確認している。これは計算コストの面で実用上非常に大きな利点である。

### プルーニング戦略の比較

著者らは様々なプルーニング戦略を比較し、以下のベストプラクティスを報告している（論文Table 1, 2より）：

| 圧縮比 | 推奨戦略 | MMLU改善（vs スクラッチ） |
|--------|---------|----------------------|
| 15B → 8B (約2x) | Width + Attention | +16ポイント |
| 15B → 4B (約4x) | Depth + Width + MLP | +9ポイント |

**幅方向プルーニングの優位性**: 2x圧縮では幅方向プルーニングが深さ方向プルーニングより一貫して優れているが、4x圧縮では深さ方向の除去も必要になると報告されている。

## 実装のポイント（Implementation）

### 最小限のコードで再現する蒸留パイプライン

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def distillation_step(
    teacher: AutoModelForCausalLM,
    student: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> torch.Tensor:
    """1ステップの蒸留損失を計算する

    Args:
        teacher: 教師モデル（フリーズ済み）
        student: 生徒モデル（学習対象）
        input_ids: 入力トークンID
        alpha: CE損失の重み
        temperature: ソフトラベルの温度

    Returns:
        蒸留損失
    """
    with torch.no_grad():
        teacher_logits = teacher(input_ids).logits

    student_logits = student(input_ids).logits
    labels = input_ids[:, 1:]

    # 交差エントロピー損失（ハードラベル）
    ce_loss = F.cross_entropy(
        student_logits[:, :-1].reshape(-1, student_logits.size(-1)),
        labels.reshape(-1),
    )

    # KLダイバージェンス損失（ソフトラベル）
    kl_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)

    return alpha * ce_loss + (1 - alpha) * kl_loss
```

### Bedrock Model Distillationへの対応

Amazon Bedrockでは、上記の蒸留プロセスがマネージドサービスとして提供される：

```python
import boto3

bedrock = boto3.client("bedrock", region_name="us-east-1")

# Bedrock Model Distillation ジョブの作成
response = bedrock.create_model_customization_job(
    jobName="rag-domain-distillation",
    customModelName="rag-distilled-haiku",
    roleArn="arn:aws:iam::123456789012:role/BedrockDistillationRole",
    baseModelIdentifier="anthropic.claude-3-5-haiku-20241022-v1:0",
    trainingDataConfig={
        "s3Uri": "s3://training-data/rag-prompts.jsonl"
    },
    customizationType="DISTILLATION",
    customizationConfig={
        "distillationConfig": {
            "teacherModelConfig": {
                "teacherModelIdentifier": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "maxResponseLengthForInference": 2048,
            },
        },
    },
)
```

AWSの公式ドキュメントによると、蒸留モデルは元のモデルと比較して最大500%高速で、コストは最大75%削減されると報告されている。

## 実験結果（Results）

### ベンチマーク比較

著者らが報告するMinitronファミリーの性能（論文Table 3より）：

| モデル | パラメータ数 | MMLU | HellaSwag | ARC-C |
|--------|------------|------|-----------|-------|
| Nemotron-4 15B（教師） | 15B | 78.7 | 82.4 | 72.1 |
| **Minitron 8B** | 8B | 72.3 | 79.8 | 68.5 |
| **Minitron 4B** | 4B | 63.1 | 73.2 | 60.8 |
| Llama-3 8B | 8B | 66.2 | 79.1 | 59.4 |
| Mistral 7B | 7B | 62.5 | 81.0 | 60.0 |
| Gemma 7B | 7B | 63.4 | 78.0 | 57.5 |

**注目すべき点**: Minitron 8Bは同サイズのLlama-3 8BやMistral 7Bを上回る性能を達成している。著者らはこれを「プルーニング+蒸留」アプローチの有効性の証拠として位置づけている。

### 学習効率

| 手法 | 必要トレーニングトークン数 | スクラッチ比 |
|------|------------------------|------------|
| スクラッチ学習（8B） | 8T tokens | 1x |
| **Minitron 8B** | 200B tokens | **40x削減** |
| スクラッチ学習（4B） | 4T tokens | 1x |
| **Minitron 4B** | 120B tokens | **33x削減** |

## 実運用への応用（Practical Applications）

### Bedrock 3層最適化戦略との統合

Minitron論文の知見は、Zenn記事で紹介した3層コスト最適化戦略のLayer 3（Model Distillation）に直接適用できる：

1. **Layer 1（IPR）**: クエリの複雑度に応じたSonnet/Haiku振り分け
2. **Layer 2（Cross-Region Inference）**: スロットリング回避
3. **Layer 3（Distillation）**: Minitron手法を参考に、ドメイン特化の蒸留モデルを作成。RAGの過去ログ（Sonnetが処理した高品質回答5,000件以上）を蒸留データとして使用

### 蒸留モデル導入の判断基準

- **5,000件以上の高品質回答ログ**がある場合に検討開始を推奨
- **ドメインが限定的**な場合（例: 社内IT FAQのみ）に効果が大きい
- **レイテンシ要件が厳しい**場合（蒸留モデルは推論速度最大500%向上）
- **コスト削減が不十分**な場合（IPR + Cachingで50-70%削減後、さらに75%削減を目指す）

## まとめ

Minitron論文は、大規模LLMの圧縮においてプルーニングと知識蒸留の組み合わせが有効であることを大規模実験で実証した。スクラッチ学習の3%未満のデータで同等以上の性能を達成できるという結果は、Amazon Bedrock Model Distillationの理論的基盤を提供している。社内RAGシステムにおいて、IPRとCachingによるコスト削減が不十分な場合の次のステップとして、蒸留モデルの導入を検討する価値がある。

## 参考文献

- **Conference URL**: [https://arxiv.org/abs/2407.14679](https://arxiv.org/abs/2407.14679)
- **NVIDIA Blog**: [Pruning and Distilling LLMs Using NVIDIA TensorRT Model Optimizer](https://developer.nvidia.com/blog/pruning-and-distilling-llms-using-nvidia-tensorrt-model-optimizer/)
- **AWS Documentation**: [Amazon Bedrock Model Distillation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-distillation.html)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/f5fa165860f5e8](https://zenn.dev/0h_n0/articles/f5fa165860f5e8)
