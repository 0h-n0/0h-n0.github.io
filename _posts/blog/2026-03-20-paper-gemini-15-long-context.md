---
layout: post
title: "論文解説: Gemini 1.5 — 100万トークン長文脈理解を実現するMoEアーキテクチャ"
description: "Gemini 1.5の技術報告を解説。MoEアーキテクチャにより100万トークンの長文脈処理とマルチモーダル理解を同時に実現した設計思想と実験結果を詳述する"
categories: [blog, paper, arxiv]
tags: [Gemini, MoE, long-context, multimodal, LLM, gemini, vertexai, rag, gcp]
date: 2026-03-20 09:00:00 +0900
source_type: arxiv
arxiv_id: "2403.05530"
source_url: https://arxiv.org/abs/2403.05530
zenn_article: 81e707a2ab8751
zenn_url: https://zenn.dev/0h_n0/articles/81e707a2ab8751
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context](https://arxiv.org/abs/2403.05530) の解説記事です。

## 論文概要（Abstract）

Gemini 1.5は、Googleが2024年3月に公開したマルチモーダル大規模言語モデルの技術報告である。Mixture-of-Experts（MoE）アーキテクチャを採用し、最大100万トークン（研究用途では1000万トークン）の超長文脈処理を実現した。テキスト・音声・画像・動画を統一モデルで処理しつつ、長文脈中の情報検索タスク（Needle-in-a-Haystack）で99%以上の精度を達成したと著者らは報告している。

この記事は [Zenn記事: Gemini 2.0 Flash×コンテキストキャッシュで社内検索のコストとレイテンシを削減する実装手法](https://zenn.dev/0h_n0/articles/81e707a2ab8751) の深掘りです。

## 情報源

- **arXiv ID**: 2403.05530
- **URL**: [https://arxiv.org/abs/2403.05530](https://arxiv.org/abs/2403.05530)
- **著者**: Gemini Team, Google（Machel Reid, Nikolay Savinov, Denis Teplyashin et al.）
- **発表年**: 2024
- **分野**: cs.CL, cs.AI, cs.CV

## 背景と動機（Background & Motivation）

大規模言語モデルの文脈長は、長らくアーキテクチャ上の制約だった。GPT-4 Turboの128Kトークン、Claude 2.1の200Kトークンなど、2023年時点では10万トークン台が上限であり、書籍一冊分（約10万トークン）や長時間の動画（1時間の動画で約100万トークン相当）を一度に処理することは困難であった。

従来のTransformerアーキテクチャでは、Self-Attentionの計算量が系列長 $n$ に対して $O(n^2)$ でスケールするため、長い系列の処理にはメモリと計算コストの両面で壁があった。また、Dense（全パラメータ活性型）モデルでは、パラメータ数の増大に伴う計算コストの増加が顕著であり、推論時の効率性が課題となっていた。

Gemini 1.5は、これらの課題に対してMoEアーキテクチャを採用することで「推論時の活性パラメータ数を抑えつつモデル容量を拡大する」というアプローチをとり、長文脈処理と計算効率の両立を目指した。

## 主要な貢献（Key Contributions）

著者らが主張する主要な貢献は以下の通りである：

- **100万トークン長文脈**: プロダクション環境で100万トークン、研究環境で最大1000万トークンの文脈長をサポートし、従来モデルの約8倍の文脈長を実現
- **MoEによる計算効率**: Mixture-of-Experts構造により、推論時にはモデル全体のパラメータの一部のみを活性化することで、Dense型モデルと同等の精度を少ない計算量で達成
- **マルチモーダル統合**: テキスト・画像・音声・動画を単一モデルで処理し、クロスモーダルな推論を実現
- **長文脈 Needle-in-a-Haystack**: 100万トークンの文脈中から特定の情報を99%以上の精度で検索可能であると報告

## 技術的詳細（Technical Details）

### Mixture-of-Experts（MoE）アーキテクチャ

Gemini 1.5 ProはMoE（Mixture-of-Experts）アーキテクチャを採用している。MoEの基本的な考え方は、Transformerの各層のFeed-Forward Network（FFN）を複数の「エキスパート」に分割し、入力に応じて一部のエキスパートのみを活性化するものである。

標準的なMoE層の出力は以下の式で表される：

$$
y = \sum_{i=1}^{E} g_i(x) \cdot \text{FFN}_i(x)
$$

ここで、
- $E$: エキスパートの総数
- $x$: 入力トークンの表現ベクトル
- $g_i(x)$: ゲーティング関数（ルーター）の出力。入力 $x$ に対するエキスパート $i$ の重み
- $\text{FFN}_i(x)$: $i$ 番目のエキスパートネットワークの出力

ゲーティング関数 $g(x)$ は通常、Top-K選択を行う。上位 $K$ 個のエキスパートのみを活性化し、残りのエキスパートの重みをゼロにする：

$$
g_i(x) = \begin{cases}
\frac{\exp(w_i^\top x)}{\sum_{j \in \text{Top-K}} \exp(w_j^\top x)} & \text{if } i \in \text{Top-K}(w^\top x) \\
0 & \text{otherwise}
\end{cases}
$$

ここで、$w_i$ はルーターの学習可能パラメータ、Top-Kは上位K個のスコアを持つエキスパートの集合である。

この機構により、**全パラメータ数は大きいが推論時の活性パラメータ数はDenseモデルより少ない**という特性が生まれる。例えば、エキスパート数 $E=16$、Top-K $K=2$ の場合、推論時には各トークンに対して約12.5%のFFNパラメータのみが活性化される。

### 長文脈処理の実現

Gemini 1.5が100万トークンの長文脈を処理できる背景には、以下の技術的要素が寄与していると考えられる（著者らはアーキテクチャの詳細を完全には公開していない）：

1. **効率的なAttention機構**: 標準的なSelf-Attentionの $O(n^2)$ 計算量を削減する手法（Ring Attention、Flash Attentionなどの可能性）
2. **MoEによるメモリ効率**: 活性パラメータの削減により、長い系列のKVキャッシュに多くのメモリを割り当てられる
3. **段階的な学習**: 短い文脈長から徐々に長い文脈長へと学習を拡張するカリキュラム学習

```python
# MoE層の概念的な実装（PyTorch風の擬似コード）
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """Mixture-of-Experts層

    Args:
        d_model: モデルの隠れ次元数
        num_experts: エキスパートの総数
        top_k: 各トークンで活性化するエキスパート数
        d_ff: FFNの内部次元数
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 16,
        top_k: int = 2,
        d_ff: int = 4096,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # ルーター（ゲーティングネットワーク）
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # 各エキスパートのFFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MoE層の順伝播

        Args:
            x: 入力テンソル (batch_size, seq_len, d_model)

        Returns:
            出力テンソル (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # ルーターでエキスパート選択
        router_logits = self.router(x)  # (B, S, E)
        top_k_values, top_k_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        top_k_weights = F.softmax(top_k_values, dim=-1)  # (B, S, K)

        # 選択されたエキスパートの出力を加重平均
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]  # (B, S)
            weight = top_k_weights[:, :, k].unsqueeze(-1)  # (B, S, 1)

            for e in range(self.num_experts):
                mask = (expert_idx == e)  # (B, S)
                if mask.any():
                    expert_input = x[mask]  # (N, d_model)
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weight[mask] * expert_output

        return output
```

### Needle-in-a-Haystack（NIAH）テスト

著者らは、100万トークンのコンテキスト中に挿入された特定の情報を検索するNeedle-in-a-Haystackテストで、Gemini 1.5 Proが**99%以上の精度**を達成したと報告している（論文Figure 3より）。

このテストでは、長い無関係なテキスト（haystack）の中にターゲット情報（needle）を様々な位置に挿入し、モデルがその情報を正確に取り出せるかを評価する。Gemini 1.5 Proは文脈長10万～100万トークンの範囲で安定した検索精度を維持し、テキストだけでなく動画・音声のマルチモーダルNIAHでも高精度を示したと報告されている。

## 実験結果（Results）

著者らが報告する主要ベンチマーク結果（論文Table 1, 2より）：

| ベンチマーク | Gemini 1.5 Pro | GPT-4 Turbo | Claude 2.1 | Gemini 1.0 Ultra |
|-------------|---------------|-------------|------------|-----------------|
| MMLU | 90.0% | 86.5% | 78.2% | 90.0% |
| HumanEval | 71.9% | 67.0% | 70.1% | 74.4% |
| MATH | 58.5% | 52.2% | — | 53.2% |
| GSM8K | 90.2% | 92.0% | 88.0% | 94.4% |

**長文脈性能**（論文Figure 3, 4より）:

| 文脈長 | NIAH精度 | 備考 |
|--------|---------|------|
| 128K | >99% | GPT-4 Turboの上限と同等の文脈長 |
| 500K | >99% | 書籍2-3冊分に相当 |
| 1M | >99% | プロダクション上限 |
| 10M | ~99% | 研究用途のみ |

**注意**: 上記の数値は著者らが論文内で報告したものであり、独立した第三者による再現実験の結果ではない。

### Many-Shot In-Context Learning

著者らは、長文脈を活用した**Many-Shot In-Context Learning**の有効性も報告している。従来のFew-Shot（数例）ではなく、数百～数千のデモンストレーション例をコンテキストに含めることで、Fine-tuningなしにタスク性能を向上させることが可能であると主張している。翻訳タスクにおいて、500例以上のデモンストレーションで性能が飽和する傾向が観察されたと報告されている（論文Section 5.3より）。

## 実装のポイント（Implementation）

Gemini 1.5はプロプライエタリモデルであり、API経由でのみ利用可能である。実装時の考慮点を以下に整理する。

### 長文脈利用時の設計指針

```python
# Vertex AI SDK を使ったGemini 1.5 Proの長文脈利用例
from google import genai
from google.genai.types import GenerateContentConfig


def query_with_long_context(
    project_id: str,
    location: str,
    documents: list[str],
    query: str,
    max_output_tokens: int = 8192,
) -> str:
    """長文脈を活用した文書検索・回答生成

    Args:
        project_id: GCPプロジェクトID
        location: リージョン
        documents: 参照文書のリスト
        query: ユーザークエリ
        max_output_tokens: 最大出力トークン数

    Returns:
        生成された回答テキスト
    """
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )

    # 文書をコンテキストとして結合
    # 重要: 重要な情報は先頭と末尾に配置する
    # "Lost in the Middle" 問題の軽減策
    context = "\n\n---\n\n".join(documents)

    prompt = f"""以下の文書群に基づいて質問に回答してください。

## 参照文書
{context}

## 質問
{query}

## 回答指示
- 文書に記載のある情報のみを使用してください
- 参照した文書を明示してください
- 情報が見つからない場合はその旨を回答してください
"""

    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=prompt,
        config=GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=0.1,
        ),
    )

    return response.text
```

### Lost in the Middle問題

長文脈モデルにおける既知の課題として、**Lost in the Middle問題**がある。コンテキストの先頭と末尾に配置された情報は検索されやすいが、中間部分の情報は取り出しにくい傾向がある。著者らもこの問題を認識しており、Gemini 1.5 ProはNIAHテストで高精度を示しているものの、実運用では以下の対策が推奨される：

1. **重要情報の配置最適化**: システムプロンプトと最重要文書を先頭に配置
2. **適度なチャンキング**: 全文書をコンテキストに詰め込むのではなく、関連文書のみを選択的に含める
3. **コンテキストキャッシュとの併用**: 頻繁に参照する文書はキャッシュに格納し、クエリ依存の文書のみを動的に追加

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

Gemini 1.5の長文脈機能をAWS上のシステムから利用する場合の推奨構成を示す。GCPのVertex AI APIをバックエンドとしつつ、フロントエンドとオーケストレーションをAWS上に構築するハイブリッドパターンである。

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト目安 | 主要サービス |
|------|--------------|---------|--------------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $200-500 | Lambda + Vertex AI API + DynamoDB |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $1,500-3,000 | ECS Fargate + ElastiCache + Vertex AI API |
| **Large** | 300,000+ (10,000/日) | Container | $5,000-15,000 | EKS + Redis Cluster + Vertex AI API |

**Small構成の詳細**（月額$200-500）:
- **Lambda**: 1GB RAM, 120秒タイムアウト（$30/月）。長文脈クエリはVertex AI API呼び出しの待ち時間が長いためタイムアウトを長めに設定
- **Vertex AI API**: Gemini 1.5 Pro、平均入力50Kトークン/リクエスト（$300/月）
- **DynamoDB**: On-Demand、クエリログ・レスポンスキャッシュ（$20/月）
- **API Gateway**: REST API（$10/月）
- **CloudWatch**: 基本監視（$5/月）

**コスト削減テクニック**:
- Vertex AI コンテキストキャッシュ活用でキャッシュ済みトークンのコストを75%削減（Gemini 2.0）/ 90%削減（Gemini 2.5以降）
- DynamoDBでレスポンスキャッシュを実装し、同一クエリの再処理を回避
- Lambda Provisioned Concurrencyは使用せず、コールドスタートを許容（長文脈処理自体が数秒かかるため）

**コスト試算の注意事項**:
- 上記は2026年3月時点のAWS ap-northeast-1（東京）リージョンおよびVertex AI料金に基づく概算値です
- Vertex AI APIの料金は入力トークン数に大きく依存するため、実際のコストは文書量により変動します
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) および [Vertex AI料金ページ](https://cloud.google.com/vertex-ai/generative-ai/pricing) で確認してください

### Terraformインフラコード

**Small構成（Serverless）: Lambda + Vertex AI API + DynamoDB**

```hcl
# --- IAMロール（最小権限） ---
resource "aws_iam_role" "lambda_vertexai" {
  name = "lambda-vertexai-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  role = aws_iam_role.lambda_vertexai.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:Query"
        ]
        Resource = aws_dynamodb_table.response_cache.arn
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = aws_secretsmanager_secret.gcp_credentials.arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# --- Lambda関数 ---
resource "aws_lambda_function" "long_context_handler" {
  filename      = "lambda.zip"
  function_name = "gemini-long-context-handler"
  role          = aws_iam_role.lambda_vertexai.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 120  # 長文脈処理のため長めに設定
  memory_size   = 1024

  environment {
    variables = {
      GCP_SECRET_ARN   = aws_secretsmanager_secret.gcp_credentials.arn
      DYNAMODB_TABLE   = aws_dynamodb_table.response_cache.name
      VERTEX_AI_REGION = "us-central1"
    }
  }
}

# --- DynamoDB（レスポンスキャッシュ） ---
resource "aws_dynamodb_table" "response_cache" {
  name         = "gemini-response-cache"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "query_hash"

  attribute {
    name = "query_hash"
    type = "S"
  }

  ttl {
    attribute_name = "expire_at"
    enabled        = true
  }
}

# --- GCP認証情報（Secrets Manager） ---
resource "aws_secretsmanager_secret" "gcp_credentials" {
  name = "gemini-gcp-service-account"
}

# --- CloudWatchアラーム ---
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "gemini-lambda-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "Lambda エラー率異常（Vertex AI APIタイムアウトの可能性）"

  dimensions = {
    FunctionName = aws_lambda_function.long_context_handler.function_name
  }
}
```

### セキュリティベストプラクティス

- **GCP認証情報**: AWS Secrets Managerに格納し、環境変数へのハードコード禁止
- **IAMロール**: 最小権限の原則（DynamoDB・Secrets Manager・CloudWatch Logsのみ）
- **ネットワーク**: Lambda VPC配置時はNAT Gateway経由でVertex AI APIにアクセス
- **暗号化**: DynamoDB・Secrets Manager・CloudWatch LogsすべてKMS暗号化

### 運用・監視設定

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Vertex AI API レイテンシ監視
cloudwatch.put_metric_alarm(
    AlarmName='vertex-ai-latency-spike',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=2,
    MetricName='VertexAILatency',
    Namespace='Custom/GeminiProxy',
    Period=300,
    Statistic='p95',
    Threshold=30000,  # 30秒超過でアラート
    AlarmDescription='Vertex AI API P95レイテンシ異常'
)
```

### コスト最適化チェックリスト

- [ ] ~100 req/日 → Lambda + Vertex AI API（Serverless）$200-500/月
- [ ] ~1000 req/日 → ECS Fargate + ElastiCache（Hybrid）$1,500-3,000/月
- [ ] 10000+ req/日 → EKS + Redis Cluster（Container）$5,000-15,000/月
- [ ] Vertex AI コンテキストキャッシュ有効化（75-90%削減）
- [ ] DynamoDBレスポンスキャッシュ（同一クエリの再処理回避）
- [ ] Lambda タイムアウト最適化（不要な待機時間削減）
- [ ] CloudWatch アラームでVertex AI APIエラー率監視
- [ ] Secrets Manager でGCP認証情報の自動ローテーション設定

## 実運用への応用（Practical Applications）

Gemini 1.5の長文脈処理は、Zenn記事で紹介されている社内検索システムの基盤技術である。具体的な応用シナリオとして：

1. **社内文書の一括処理**: 100万トークン（約700ページ相当）の社内マニュアルを1リクエストで処理可能。チャンキング不要で文脈断片化を回避
2. **コードベース分析**: 大規模リポジトリ全体をコンテキストに含めた質問応答
3. **動画コンテンツ理解**: 社内研修動画（1時間分）を直接入力し、要約・検索を実施

ただし、100万トークンすべてを使用するとAPIコストが増大するため、Zenn記事で紹介されているコンテキストキャッシュとの併用が実用上は重要である。

## 関連研究（Related Work）

- **GPT-4 Turbo（OpenAI, 2023）**: 128Kトークンの文脈長をサポートするDenseモデル。Gemini 1.5 Proと比較して文脈長は約1/8だが、短い文脈でのベンチマーク性能は同等
- **Claude 2.1（Anthropic, 2023）**: 200Kトークンの文脈長をサポート。NIAHテストでは文脈長が伸びるにつれ精度低下が報告されている
- **Ring Attention（Liu et al., 2023）**: 複数デバイス間でAttention計算を分散し、メモリ制約を超えた長文脈処理を可能にする手法。Gemini 1.5の内部でも類似手法が使われている可能性がある

## まとめと今後の展望

Gemini 1.5は、MoEアーキテクチャにより100万トークンの長文脈処理を実現し、マルチモーダル理解と計算効率を両立させたモデルである。NIAHテストでの99%以上の精度は、長文脈LLMの実用性を示す重要な成果として著者らは位置づけている。

後継モデルであるGemini 2.0 Flash（Zenn記事で使用）はこの長文脈技術を継承しつつ推論速度を向上させており、さらにGemini 2.5、Gemini 3へと進化が続いている。MoEの設計思想は今後のLLMアーキテクチャにおける標準的なアプローチとなる可能性がある。

## 参考文献

- **arXiv**: [https://arxiv.org/abs/2403.05530](https://arxiv.org/abs/2403.05530)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/81e707a2ab8751](https://zenn.dev/0h_n0/articles/81e707a2ab8751)
- **Gemini 1.0 Technical Report**: [https://arxiv.org/abs/2312.11805](https://arxiv.org/abs/2312.11805)
- **Mixture-of-Experts Survey**: [https://arxiv.org/abs/2209.01667](https://arxiv.org/abs/2209.01667)

---

:::message
この記事はAI（Claude Code）により自動生成されました。論文の内容を正確に伝えることを心がけていますが、解釈の誤りがある可能性があります。正確な情報は原論文をご確認ください。
:::
