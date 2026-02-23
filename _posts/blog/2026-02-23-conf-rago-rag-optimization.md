---
layout: post
title: "ISCA 2025論文解説: RAGO — RAGサービングのシステムレベル性能最適化"
description: "RAGSchema抽象化とスケジューリング最適化でQPS 2倍・TTFT 55%削減を達成したRAGOを解説"
categories: [blog, paper, conference]
tags: [RAG, system-optimization, scheduling, TTFT, latency, ISCA]
date: 2026-02-23 12:00:00 +0900
source_type: conference
conference: "ISCA 2025"
source_url: https://arxiv.org/abs/2503.14649
zenn_article: a5be5c172a5a99
zenn_url: https://zenn.dev/0h_n0/articles/a5be5c172a5a99
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving (arXiv:2503.14649, ISCA 2025)](https://arxiv.org/abs/2503.14649) の解説記事です。

## 論文概要（Abstract）

RAGO（Retrieval-Augmented Generation Optimizer）は、RAGサービングのシステムレベル性能最適化フレームワークである。著者ら（ETH Zurich/Google DeepMindのWenqi Jiang, Suvinay Subramanian, Cat Graves, Gustavo Alonso, Amir Yazdanbakhsh, Vidushi Dadu）は、RAGアルゴリズムの多様性を捉えるRAGSchema抽象化を提案し、タスク配置・リソース配分・バッチングポリシーの3軸でスケジューリングを最適化する。著者らの実験では、LLMシステム拡張ベースのRAGと比較して、チップあたりのQPS（Queries Per Second）を最大2倍に向上し、TTFT（Time-to-First-Token）を55%削減したと報告されている。

この記事は [Zenn記事: LangChain LCEL実践ガイド：LLMチェーンのレイテンシを50%削減する最適化手法](https://zenn.dev/0h_n0/articles/a5be5c172a5a99) の深掘りです。Zenn記事ではアプリケーション層（LCELの`RunnableParallel`、ストリーミング等）でのレイテンシ最適化を解説していますが、本記事ではシステム層（MLアクセラレータのスケジューリング）でのRAG性能最適化を深掘りします。

## 情報源

- **会議名**: ISCA 2025（International Symposium on Computer Architecture）
- **年**: 2025（6月21-25日、東京）
- **URL**: [https://arxiv.org/abs/2503.14649](https://arxiv.org/abs/2503.14649)
- **著者**: Wenqi Jiang, Suvinay Subramanian, Cat Graves, Gustavo Alonso, Amir Yazdanbakhsh, Vidushi Dadu
- **所属**: ETH Zurich, Google DeepMind

## カンファレンス情報

**ISCAについて**:
- ISCAはコンピュータアーキテクチャ分野の最高峰カンファレンスの一つ
- 採択率は通常20%程度（高い競争率）
- RAGOはRAGシステムのアーキテクチャレベル最適化として採択された

## 技術的詳細（Technical Details）

### RAGSchema: RAGアルゴリズムの構造化抽象

RAGOの最初の貢献は、多様なRAGアルゴリズムを統一的に表現するRAGSchema抽象化である。著者らの論文によると、RAGアルゴリズムは以下のコンポーネントの組み合わせで表現される：

1. **Retriever**: クエリに基づいて外部データベースから関連情報を検索するコンポーネント
2. **Reader/Generator**: 検索結果とクエリを入力としてLLMが回答を生成するコンポーネント
3. **Reranker**: 検索結果の関連性を再評価して並べ替えるコンポーネント（オプション）
4. **Iterative Steps**: 検索→生成→再検索のループ（Self-RAG等）

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGSchema:
    """RAGアルゴリズムの構造化表現

    多様なRAGアルゴリズムを統一的に表現するための抽象化。
    """
    retriever_type: str  # "dense", "sparse", "hybrid"
    generator_model: str  # LLMモデル名
    reranker: Optional[str] = None  # リランカーモデル名
    num_iterations: int = 1  # 反復検索の回数
    retrieval_count: int = 10  # 検索件数
    chunk_size: int = 512  # チャンクサイズ（トークン数）

    @property
    def has_reranker(self) -> bool:
        return self.reranker is not None

    @property
    def is_iterative(self) -> bool:
        return self.num_iterations > 1
```

著者らの論文によると、異なるRAGSchemaを持つワークロードでは性能特性が大きく異なることが分析で明らかになった。例えば、リランカーを含むRAGパイプラインはリランカーなしのパイプラインと比較して計算リソースの配分が大きく異なり、単一のスケジューリングポリシーでは最適化できない。

### スケジューリングポリシー空間

RAGOは以下の3軸でスケジューリングを最適化する：

#### 1. タスク配置（Task Placement）

推論コンポーネント（LLM生成、リランキング）をMLアクセラレータ上にどう配置するかを決定する。

$$
\text{Placement}: \{C_1, C_2, \ldots, C_n\} \rightarrow \{A_1, A_2, \ldots, A_m\}
$$

ここで、
- $C_i$: RAGパイプラインのコンポーネント（Retriever, Generator, Reranker等）
- $A_j$: 利用可能なMLアクセラレータ（GPU/TPU）
- $n$: コンポーネント数
- $m$: アクセラレータ数

**Colocation vs. Disaggregation**: すべてのコンポーネントを同一アクセラレータに配置（Colocation）するか、異なるアクセラレータに分散（Disaggregation）するかの選択がある。著者らの論文によると、Disaggregationは各コンポーネントの独立スケーリングを可能にし、高スループットワークロードで有効であると報告されている。

#### 2. リソース配分（Resource Allocation）

各コンポーネントに割り当てるリソース量（GPU数、メモリ等）を決定する。

$$
\text{Allocation}: C_i \rightarrow (r_{\text{compute}}, r_{\text{memory}}, r_{\text{bandwidth}})
$$

- $r_{\text{compute}}$: 計算リソース（FLOPS）
- $r_{\text{memory}}$: メモリ容量（GB）
- $r_{\text{bandwidth}}$: メモリ帯域幅（GB/s）

#### 3. バッチングポリシー（Batching Policies）

検索リクエストと推論リクエストのバッチサイズを調整する。

$$
\text{Throughput} \propto \frac{B_{\text{inference}} \cdot B_{\text{retrieval}}}{T_{\text{total}}}
$$

- $B_{\text{inference}}$: 推論バッチサイズ
- $B_{\text{retrieval}}$: 検索バッチサイズ
- $T_{\text{total}}$: 全体のレイテンシ

バッチサイズが大きいほどスループットは向上するが、個々のリクエストのレイテンシは増加するというトレードオフがある。

### TTFT削減のメカニズム

著者らの論文によると、RAGOがTTFTを55%削減できた主要因は以下の2つである：

1. **検索と推論の分離スケジューリング**: 検索リクエストのバッチ完了を待たずに、早期に完了した検索結果から順次推論を開始する
2. **コンポーネント間のパイプライン化**: PipeRAG（Jiang et al., 2024）と同様に、検索と推論の時間的オーバーラップを実現するが、RAGOはサービングレベル（複数リクエストの同時処理）で最適化を行う点が異なる

## 実装のポイント（Implementation）

### LCELとの関連

RAGOの知見をLCELパイプラインに応用する際のポイント：

1. **バッチサイズの調整**: LCELの`batch()`メソッドの`max_concurrency`パラメータは、RAGOのバッチングポリシーに相当する。ワークロードに応じて最適値を調整することでスループットとレイテンシのバランスを取る
2. **コンポーネントの分離デプロイ**: LCELチェーン内の検索処理とLLM呼び出しを別サービスとしてデプロイすることで、独立したスケーリングが可能になる
3. **ストリーミングとTTFT**: LCELの`.stream()`メソッドによるストリーミングは、アプリケーション層でTTFTを最小化する手法であり、RAGOのシステム層最適化と組み合わせることでさらなる改善が期待できる

```python
from langchain_core.runnables import RunnableParallel

# RAGOの知見を活かしたLCELパイプライン設計
# 検索とLLM呼び出しを分離し、独立スケーリング可能に
retrieval_service = RunnableParallel({
    "dense_results": dense_retriever,
    "sparse_results": sparse_retriever,
})

reranker_service = reranker_chain
generator_service = llm_chain

# パイプライン全体
rag_pipeline = (
    retrieval_service  # 検索サービス（CPUスケーリング）
    | merge_and_rerank  # リランキング（GPUスケーリング）
    | generator_service  # 生成（GPUスケーリング）
)
```

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $50-150 | Lambda + Bedrock + OpenSearch Serverless |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $300-800 | ECS Fargate + Bedrock + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,000-5,000 | EKS + Karpenter + EC2 Spot |

**RAGOの知見を活かした構成設計**:

RAGOの研究から、検索コンポーネントと推論コンポーネントのDisaggregation（分離配置）が高スループットワークロードで有効であることが示されている。AWS上では以下のように実現する：

- **Small**: Lambda（検索+推論一体型）— 低トラフィックでは分離のオーバーヘッドが利点を上回る
- **Medium**: ECS Fargate（検索用タスク）+ Bedrock（推論）— 検索と推論を分離、独立スケーリング
- **Large**: EKS（検索Pod + 推論Pod）— Karpenterで各コンポーネントを独立オートスケーリング

**コスト試算の注意事項**:
- 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

```hcl
# RAGOの分離アーキテクチャをAWSで実現
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "rago-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["ap-northeast-1a", "ap-northeast-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  enable_dns_hostnames = true
}

# 検索サービス（OpenSearch Serverless）
resource "aws_opensearchserverless_collection" "vectors" {
  name = "rago-vectors"
  type = "VECTORSEARCH"
}

# 推論サービス（Lambda + Bedrock）
resource "aws_lambda_function" "inference" {
  filename      = "inference.zip"
  function_name = "rago-inference"
  role          = aws_iam_role.lambda_role.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 60
  memory_size   = 1024
}

# バッチサイズ監視（CloudWatch）
resource "aws_cloudwatch_metric_alarm" "batch_latency" {
  alarm_name          = "rago-batch-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "p95"
  threshold           = 30000
  alarm_description   = "RAGパイプラインP95レイテンシ30秒超過"
}
```

### コスト最適化チェックリスト

- [ ] 検索と推論の分離デプロイ（独立スケーリング）
- [ ] バッチサイズの最適化（`max_concurrency`調整）
- [ ] OpenSearch Serverless: アイドル時自動スケールダウン
- [ ] Bedrock Batch API: 非リアルタイム処理で50%割引
- [ ] Spot Instances: EKSワーカーノードで最大90%削減
- [ ] AWS Budgets: 月額予算設定
- [ ] CloudWatch: TTFT・QPS監視ダッシュボード

## 実験結果（Results）

著者らの論文で報告された主要な実験結果：

| 指標 | ベースライン（LLM拡張型RAG） | RAGO | 改善率 |
|------|---------------------------|------|--------|
| QPS/chip | 1.0x（基準） | 2.0x | **2倍** |
| TTFT | 1.0x（基準） | 0.45x | **55%削減** |

（論文の実験結果より）

**分析ポイント**:
- QPS向上はタスク配置とバッチングポリシーの最適化に起因
- TTFT削減は検索・推論の分離スケジューリングとパイプライン化に起因
- RAGSchemaが異なるワークロードでは最適なスケジューリングポリシーが大きく異なることが実証された

## 実運用への応用（Practical Applications）

RAGOの知見は以下の実運用シナリオに適用可能である：

1. **マルチテナントRAGサービス**: 異なるRAGパイプラインを持つ複数テナントを同一インフラでサービングする際の最適化
2. **コスト効率の高いRAGデプロイ**: チップあたりのQPS向上により、同一ハードウェアでのサービング容量が2倍に
3. **TTFT重視のチャットボット**: TTFT 55%削減により、ユーザー体感の応答速度が大幅に向上

**制約と限界**: RAGOはGoogle DeepMindのTPUベースのインフラで検証されており、GPUベースの環境（NVIDIA A100/H100等）での性能特性は異なる可能性がある。

## 関連研究（Related Work）

- **PipeRAG** (Jiang et al., 2024): 同じ著者（Wenqi Jiang）による、モデルレベルでのパイプライン並列化。RAGOはサービングレベルでの最適化であり相補的
- **vLLM** (Kwon et al., 2023): PagedAttentionによるLLM推論の効率化。RAGOはvLLMをベースラインとして使用している
- **Sarathi-Serve** (Agrawal et al., 2024): Chunked-prefillsによるLLM推論のスループット・レイテンシ最適化。RAGOのGenerator部分の最適化に応用可能

## まとめと今後の展望

RAGOは、RAGサービングのシステムレベル最適化をRAGSchema抽象化と3軸スケジューリングで体系化した研究である。チップあたりQPS 2倍、TTFT 55%削減という成果は、RAGアプリケーションのスケーラビリティとユーザー体験の両方に大きなインパクトを持つ。LCELのアプリケーション層最適化（`RunnableParallel`、ストリーミング等）とRAGOのシステム層最適化を組み合わせることで、エンドツーエンドの性能向上が期待される。

## 参考文献

- **arXiv**: [https://arxiv.org/abs/2503.14649](https://arxiv.org/abs/2503.14649)
- **ISCA 2025**: [https://dl.acm.org/doi/10.1145/3695053.3731093](https://dl.acm.org/doi/10.1145/3695053.3731093)
- **Code**: [https://github.com/google/rago](https://github.com/google/rago)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/a5be5c172a5a99](https://zenn.dev/0h_n0/articles/a5be5c172a5a99)
