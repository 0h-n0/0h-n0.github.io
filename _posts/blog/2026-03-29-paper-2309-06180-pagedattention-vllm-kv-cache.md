---
layout: post
title: "論文解説: PagedAttention — 仮想メモリ着想のKVキャッシュ管理でLLMサービングを高効率化"
description: "vLLMの中核技術PagedAttentionを詳細解説。OS仮想メモリのページング手法をKVキャッシュに適用し、メモリ断片化を4%未満に抑え最大24倍のスループット向上を実現した手法の技術的詳細"
categories: [blog, paper, arxiv]
tags: [LLM, vLLM, PagedAttention, KV-cache, GPU, inference, memory-management, rust, cuda]
date: 2026-03-29 09:00:00 +0900
source_type: arxiv
arxiv_id: "2309.06180"
source_url: https://arxiv.org/abs/2309.06180
zenn_article: 48d89cb18bf0e1
zenn_url: https://zenn.dev/0h_n0/articles/48d89cb18bf0e1
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [arXiv:2309.06180 "Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180) の解説記事です。

## 論文概要（Abstract）

Kwon et al. (UC Berkeley) は、LLMサービングにおけるKVキャッシュのメモリ断片化問題に取り組み、OSの仮想メモリ・ページング機構からの着想でPagedAttentionアルゴリズムを提案した。この手法を実装したvLLMは、KVキャッシュメモリの無駄を4%未満に抑え、HuggingFace Transformersに対して最大24倍のスループット向上を達成したと報告されている。

この記事は [Zenn記事: rvLLM：Rust製vLLM代替で学ぶGPU推論エンジンの実装最適化](https://zenn.dev/0h_n0/articles/48d89cb18bf0e1) の深掘りです。rvLLMがRustで再実装したPagedAttentionの原理を理解する上で、本論文は必読文献です。

## 情報源

- **arXiv ID**: 2309.06180
- **URL**: [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- **著者**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **発表年**: 2023（SOSP 2023に採択）
- **分野**: cs.CL, cs.LG

## 背景と動機（Background & Motivation）

LLMサービングでは、リクエストごとにKVキャッシュ（Attention計算のKey-Value中間状態）をGPUメモリ上に保持する必要がある。各リクエストのKVキャッシュサイズは動的に変化し、シーケンス長やモデルサイズに依存する。

著者らが問題として指摘したのは、従来のLLMサービングシステムにおけるメモリ管理の非効率性である。HuggingFace Transformersでは最大シーケンス長分のメモリを事前確保するため、約80%がメモリの無駄となっていた。HuggingFace TGIでも40-60%の無駄が生じていたと報告されている（論文Section 3.1）。

この問題は3つの要因に分解される：

1. **内部断片化**: 最大シーケンス長まで事前確保するが、実際の出力は短い場合が多い
2. **外部断片化**: 異なるサイズのリクエストが断片的にメモリを占有
3. **過剰予約**: 将来の生成に備えた予約分が未使用のまま保持される

## 主要な貢献（Key Contributions）

- **PagedAttentionアルゴリズム**: 非連続な物理メモリ上に格納されたKVキャッシュに対してAttention計算を実行する手法
- **ブロックテーブル方式のメモリ管理**: OSページテーブルの概念をGPUメモリに適用し、論理ブロック→物理ブロックの間接参照を実現
- **Copy-on-Write KVキャッシュ共有**: 同一プレフィックスを持つリクエスト間でブロックを安全に共有する機構
- **vLLMシステム**: PagedAttentionを実装したオープンソースLLMサービングエンジン

## 技術的詳細（Technical Details）

### KVキャッシュのメモリ要件

各トークンのKVキャッシュサイズは以下の式で計算される（論文Section 2.2）：

$$
\text{KVcache\_size} = 4 \times t \times L \times H \times D_h \text{ bytes}
$$

ここで、
- $t$: シーケンス中のトークン数
- $L$: Transformerレイヤー数
- $H$: Attentionヘッド数
- $D_h$: ヘッド次元
- $4 = 2(\text{KeyとValue}) \times 2(\text{FP16のバイト数})$

OPT-13Bの場合、1トークンあたり約0.8MBのKVキャッシュが必要となる。2048トークンのシーケンスでは約1.6GBに達し、40GB GPUでは同時に処理できるリクエスト数が著しく制限される。

### PagedAttentionアルゴリズム

PagedAttentionの中核は、KVキャッシュを固定サイズの**ブロック**（ページ）に分割し、非連続な物理メモリに格納する点にある。

**ブロック定義**:
- 各ブロックは $B$ トークン分のKVペアを格納（デフォルト $B=16$）
- シーケンス長 $T$ のリクエストは $\lceil T/B \rceil$ 個のブロックを使用
- ブロックはGPUメモリ上の任意の場所に配置可能

**ブロックテーブル**: OSのページテーブルと同様に、論理ブロック番号→物理ブロック番号のマッピングを管理する。

```
Request A (7トークン, B=4):
  論理ブロック0 → 物理ブロック7  [t1, t2, t3, t4]
  論理ブロック1 → 物理ブロック1  [t5, t6, t7, _]

Request B (5トークン, B=4):
  論理ブロック0 → 物理ブロック3  [t1, t2, t3, t4]
  論理ブロック1 → 物理ブロック5  [t5, _, _, _]
```

デコード時のAttention計算は、ブロック単位で実行される。現在のトークンのクエリ $q_t$ に対し：

$$
o_t = \sum_{j=0}^{S-1} \frac{\exp(q_t K_j^\top / \sqrt{d}) \cdot V_j}{\sum_{l=0}^{S-1} \sum_{k} \exp(q_t k_{l,k} / \sqrt{d})}
$$

ここで、$S$ は論理ブロック数、$K_j, V_j$ はブロックテーブル経由で物理メモリから取得される。online softmax（FlashAttention方式）で、ブロックをストリーミング処理しながら正規化定数を累積計算する。

### メモリ断片化の定量的分析

ブロックサイズ $B$ における無駄の分析（論文Section 4.4）：

| 断片化タイプ | 従来方式 | PagedAttention |
|---|---|---|
| 内部断片化 | 最大シーケンス長の50%以上 | 最大 $B-1$ トークン/リクエスト |
| 外部断片化 | リクエスト間の隙間 | **完全に排除** |
| 過剰予約 | 最大長まで事前確保 | **完全に排除**（オンデマンド確保） |

$B=16$ の場合、平均無駄は8トークン/リクエストであり、数百〜数千トークンのシーケンスに対して無視可能な水準となる。

### Copy-on-Write KVキャッシュ共有

同一プロンプトから複数サンプルを並列生成する場合（parallel sampling）、プロンプト部分のKVキャッシュブロックを共有できる。著者らはOSのCopy-on-Write（CoW）機構を適用し、以下のように実装している：

1. 共有ブロックに参照カウントを付与（参照カウント = サンプル数）
2. デコード中、全サンプルが同一プロンプトブロックを参照
3. 書き込みが発生した場合、参照カウント > 1 ならブロックをコピーしてから書き込み

この機構はbeam searchにも適用され、共通プレフィックスのブロックをビーム候補間で共有することで、冗長な再計算を回避する。

### アルゴリズムの擬似コード

```python
def paged_attention_decode(
    query: torch.Tensor,       # shape: (d,) — 現在のトークンのクエリ
    block_table: list[int],    # 論理ブロック→物理ブロックのマッピング
    kv_cache: torch.Tensor,    # shape: (num_physical_blocks, B, 2, d)
    block_size: int,
) -> torch.Tensor:
    """PagedAttentionのデコードステップ（単一トークン生成）

    Args:
        query: 現在トークンのクエリベクトル
        block_table: ブロックテーブル（論理→物理ブロックID）
        kv_cache: 全物理ブロックのKVキャッシュ
        block_size: 1ブロックあたりのトークン数

    Returns:
        Attention出力ベクトル
    """
    d = query.shape[-1]
    max_score = float('-inf')
    sum_exp = 0.0
    output = torch.zeros(d)

    for logical_idx, physical_idx in enumerate(block_table):
        K_block = kv_cache[physical_idx, :, 0, :]  # (B, d)
        V_block = kv_cache[physical_idx, :, 1, :]  # (B, d)

        # ブロック内Attentionスコア計算
        scores = (query @ K_block.T) / (d ** 0.5)  # (B,)

        # Online softmax更新
        block_max = scores.max()
        new_max = max(max_score, block_max)

        # 既存の累積値を再スケーリング
        scale_old = torch.exp(torch.tensor(max_score - new_max))
        output = output * scale_old
        sum_exp = sum_exp * scale_old.item()

        # 新ブロックの寄与を加算
        exp_scores = torch.exp(scores - new_max)
        output = output + exp_scores @ V_block
        sum_exp += exp_scores.sum().item()
        max_score = new_max.item()

    return output / sum_exp
```

## 実装のポイント（Implementation）

**ブロックサイズの選択**: デフォルト16トークン。大きすぎると内部断片化が増加し、小さすぎるとブロックテーブルのオーバーヘッドが増大する。ワークロードに応じたチューニングが必要。

**CUDAカーネル実装**: PagedAttentionのCUDAカーネルはgather操作で非連続メモリブロックを取得し、online softmaxでメモリ効率を確保する。prefill/decodeの両フェーズをサポート。

**スケジューラ**: FCFSスケジューリングにプリエンプション機構を追加。GPU メモリ逼迫時はスワッピング（KVキャッシュをCPUに退避）またはリコンピュテーション（KVキャッシュを破棄し再計算）を実行する。

**rvLLMとの関連**: Zenn記事で解説されているrvLLMは、このPagedAttention方式を**Rustの所有権モデル**で再実装している。単一の`cudaMalloc`スラブ上に自前ブロックアロケータを構築し、GCなしの決定論的メモリ解放を実現している点が、Python vLLMとの主要な実装差異である。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $50-150 | Lambda + Bedrock + DynamoDB |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $300-800 | Lambda + ECS Fargate + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,000-5,000 | EKS + Karpenter + EC2 Spot |

**Small構成の詳細** (月額$50-150):
- **Lambda**: 1GB RAM, 60秒タイムアウト ($20/月)
- **Bedrock**: Claude 3.5 Haiku, Prompt Caching有効 ($80/月)
- **DynamoDB**: On-Demand ($10/月)
- **CloudWatch**: 基本監視 ($5/月)

**Large構成の詳細** (月額$2,000-5,000):
- **EKS**: コントロールプレーン ($72/月)
- **EC2 Spot Instances**: g5.xlarge × 2-4台 (平均$800/月)
- **Karpenter**: 自動スケーリング（追加コストなし）
- **Bedrock Batch**: 50%割引活用 ($2,000/月)

**コスト試算の注意事項**:
上記は2026年3月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です。実際のコストはトラフィックパターンやリージョンにより変動します。最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください。

### Terraformインフラコード

**Small構成 (Serverless): Lambda + Bedrock + DynamoDB**

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "llm-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["ap-northeast-1a", "ap-northeast-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]

  enable_nat_gateway   = false
  enable_dns_hostnames = true
}

resource "aws_iam_role" "lambda_bedrock" {
  name = "lambda-bedrock-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "bedrock_invoke" {
  role = aws_iam_role.lambda_bedrock.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
      Resource = "arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.claude-3-5-haiku*"
    }]
  })
}

resource "aws_lambda_function" "llm_handler" {
  filename      = "lambda.zip"
  function_name = "llm-bedrock-handler"
  role          = aws_iam_role.lambda_bedrock.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 60
  memory_size   = 1024

  environment {
    variables = {
      BEDROCK_MODEL_ID    = "anthropic.claude-3-5-haiku-20241022-v1:0"
      DYNAMODB_TABLE      = aws_dynamodb_table.cache.name
      ENABLE_PROMPT_CACHE = "true"
    }
  }
}

resource "aws_dynamodb_table" "cache" {
  name         = "llm-prompt-cache"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "prompt_hash"

  attribute {
    name = "prompt_hash"
    type = "S"
  }

  ttl {
    attribute_name = "expire_at"
    enabled        = true
  }
}
```

**Large構成 (Container): EKS + Karpenter + Spot Instances**

```hcl
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "llm-inference-cluster"
  cluster_version = "1.31"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  cluster_endpoint_public_access = true
  enable_cluster_creator_admin_permissions = true
}

resource "kubectl_manifest" "karpenter_provisioner" {
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
          limits:
            cpu: "32"
            memory: "128Gi"
      disruption:
        consolidationPolicy: WhenEmptyOrUnderutilized
        consolidateAfter: 30s
  YAML
}

resource "aws_budgets_budget" "llm_monthly" {
  name         = "llm-monthly-budget"
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

### セキュリティベストプラクティス

- IAMロール: 最小権限の原則（Bedrock特定モデルのみ許可）
- ネットワーク: EKSは`cluster_endpoint_public_access = false`推奨（VPN経由）
- シークレット: AWS Secrets Manager使用、環境変数ハードコード禁止
- 暗号化: S3/DynamoDB/EBS全てKMS暗号化
- 監査: CloudTrail/Config/GuardDuty有効化

### コスト最適化チェックリスト

- [ ] ~100 req/日 → Lambda + Bedrock (Serverless) - $50-150/月
- [ ] ~1000 req/日 → ECS Fargate + Bedrock (Hybrid) - $300-800/月
- [ ] 10000+ req/日 → EKS + Spot Instances (Container) - $2,000-5,000/月
- [ ] Spot Instances優先（最大90%削減）
- [ ] Bedrock Batch API使用で50%割引
- [ ] Prompt Caching有効化で30-90%削減
- [ ] AWS Budgets: 月額予算設定（80%で警告）
- [ ] CloudWatch アラーム: トークン使用量スパイク検知
- [ ] Cost Anomaly Detection: 自動異常検知
- [ ] 未使用リソース削除: Lambda Insights活用
- [ ] タグ戦略: 環境別（dev/staging/prod）でコスト可視化

## 実験結果（Results）

### メインスループット結果

著者らのベンチマーク（論文Table 1, A100 40GB）によると、OPT-13BでShareGPTデータセットを使用した場合：

| システム | スループット (req/s) | 相対性能 |
|---|---|---|
| HuggingFace Transformers | 0.31 | 1× |
| HuggingFace TGI | 0.91 | 2.9× |
| vLLM | 7.10 | **22.9×** |

Alpacaデータセットでは、vLLMがTGI比1.7倍のスループットを達成している。モデルサイズが大きくなるほどvLLMの優位性は拡大し、OPT-175B（8×A100）ではTGI比約1.5倍と報告されている。

### メモリ効率

KVキャッシュメモリの利用効率（論文Figure 5）：

| システム | メモリ無駄 | 有効利用率 |
|---|---|---|
| HF | ~80%無駄 | ~20%利用 |
| TGI | ~60%無駄 | ~40%利用 |
| vLLM | **<4%無駄** | **>96%利用** |

### KVキャッシュ共有の効果

並列サンプリング（同一プロンプトからk個のサンプル生成）でのプレフィックス共有による高速化（論文Figure 7）：

- $k=2$: 1.1× 高速化
- $k=4$: 1.2× 高速化
- $k=8$: 1.5× 高速化

## 実運用への応用（Practical Applications）

PagedAttentionの実用的な意義は、同一GPU上で処理可能な同時リクエスト数を大幅に増やせる点にある。メモリ利用率が20%→96%に改善することで、バッチサイズを約5倍に拡大でき、GPU投資効率が劇的に改善する。

**rvLLMとの比較**: Zenn記事で解説されているように、rvLLMはこのPagedAttention方式をRustで再実装し、さらにGPU側argmaxやCUDAグラフなどの最適化を追加している。vLLMが73GBのGPUメモリを必要とする場面で、rvLLMは40GBで動作すると報告されており、Rustの所有権モデルによるメモリ管理の効率化が寄与していると考えられる。

## 関連研究（Related Work）

- **Orca (Yu et al., OSDI 2022)**: iteration-level schedulingを提案したが、メモリ管理は静的確保。PagedAttentionはOrcaのスケジューリングと直交する最適化
- **FlashAttention (Dao et al., 2022)**: HBM↔SRAM間のI/O最適化。PagedAttentionのブロック内Attention計算にFlashAttentionが適用可能
- **FasterTransformer (NVIDIA)**: 固定バッチサイズ・連続メモリ方式。PagedAttentionの柔軟性を持たない

## まとめと今後の展望

PagedAttentionは、OSの仮想メモリ概念をGPU KVキャッシュ管理に適用することで、LLMサービングのメモリ効率を根本的に改善した。rvLLMを含む後続の推論エンジンがこの設計を採用していることからも、本論文のインパクトは大きい。今後はvAttention（CUDA仮想メモリAPIの直接利用）など、さらにオーバーヘッドの少ない手法への発展が期待される。

## 参考文献

- **arXiv**: [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- **Code**: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache 2.0)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/48d89cb18bf0e1](https://zenn.dev/0h_n0/articles/48d89cb18bf0e1)
