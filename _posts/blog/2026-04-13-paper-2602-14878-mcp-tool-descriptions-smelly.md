---
layout: post
title: "論文解説: MCP Tool Descriptions Are Smelly! — MCPツール記述の品質問題と改善手法"
description: "856ツールの分析で97.1%に品質問題を発見。6コンポーネントフレームワークによる記述改善でタスク成功率が中央値5.85ポイント向上した研究を解説"
categories: [blog, paper, arxiv]
tags: [MCP, tool-design, AI-agent, LLM, function-calling, ai, agent, python, mcp]
date: 2026-04-13 09:00:00 +0900
source_type: arxiv
arxiv_id: "2602.14878"
source_url: https://arxiv.org/abs/2602.14878
zenn_article: c1f033224797db
zenn_url: https://zenn.dev/0h_n0/articles/c1f033224797db
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [arXiv:2602.14878 "Model Context Protocol (MCP) Tool Descriptions Are Smelly!"](https://arxiv.org/abs/2602.14878) の解説記事です。

## 論文概要（Abstract）

Model Context Protocol（MCP）エコシステムにおけるツール記述の品質問題を体系的に分析した研究である。著者らは103のMCPサーバから856のツールを収集・分析し、97.1%のツール記述に少なくとも1つの品質問題（「スメル」）が存在することを報告している。6つの記述コンポーネントからなる評価フレームワークを提案し、記述の拡張によりタスク成功率が中央値で5.85ポイント向上する一方、実行ステップ数が67.46%増加するという精度-コストのトレードオフを明らかにした。

この記事は [Zenn記事: AIエージェントツール設計の7原則：Anthropic・OpenAI公式ガイドに学ぶ実装パターン](https://zenn.dev/0h_n0/articles/c1f033224797db) の深掘りです。

## 情報源

- **arXiv ID**: 2602.14878
- **URL**: [https://arxiv.org/abs/2602.14878](https://arxiv.org/abs/2602.14878)
- **著者**: Mohammed Mehedi Hasan, Hao Li, Gopi Krishnan Rajbahadur, Bram Adams, Ahmed E. Hassan
- **発表年**: 2026
- **分野**: cs.SE, cs.AI

## 背景と動機（Background & Motivation）

MCPはAnthropicが2024年にリリースしたオープンプロトコルであり、LLMアプリケーションと外部ツール・データソースの統合を標準化する。2026年時点でMCP Registryには10,000以上のパブリックサーバが登録されており、ChatGPT、Cursor、Geminiなど主要プラットフォームが採用している。

しかし、MCPにおけるツール記述はLLMエージェントがツールを選択・パラメータ化する際の唯一の意味的インターフェースであるにもかかわらず、その品質に関する体系的な研究は行われていなかった。著者らは、ソフトウェア工学における「コードスメル」の概念をツール記述に適用し、品質問題を体系的に分類・定量化することで、エージェントの信頼性向上に寄与する知見を提供することを目指している。

## 主要な貢献（Key Contributions）

- **貢献1**: 6つの記述コンポーネントからなる品質評価フレームワークの提案と、対応する「ツール記述スメル」の定義
- **貢献2**: 103のMCPサーバ・856ツールの大規模分析により、97.1%のツールに品質問題が存在することの実証
- **貢献3**: 記述拡張による性能改善と精度-コストのトレードオフの定量的評価
- **貢献4**: スメルスキャナ・拡張ツール・評価フレームワークのオープンソース公開

## 技術的詳細（Technical Details）

### 6つの記述コンポーネントフレームワーク

著者らはAnthropicの公式MCPドキュメント、15のコミュニティソース、および先行研究の分析から、高品質なツール記述に必要な6つのコンポーネントを特定した（評価者間一致度: Jaccard類似度0.92）。

| コンポーネント | 定義 | 対応するスメル |
|---|---|---|
| **Purpose（目的）** | ツールの機能と役割の明示 | Unclear Purpose（56%） |
| **Guidelines（ガイドライン）** | 使用条件と操作手順 | Missing Usage Guidelines |
| **Limitations（制約）** | 制約事項と失敗シナリオ | Unstated Limitations |
| **Parameter Explanation（パラメータ説明）** | 入力パラメータの詳細説明 | Opaque Parameters |
| **Length and Completeness（詳細度）** | ツール複雑性に対する記述の十分性 | Underspecified or Incomplete |
| **Examples（使用例）** | 正しい呼び出しパターンの例示 | Exemplar Issues |

### スメル検出手法: LLM-as-Jury

各コンポーネントの品質を5段階で評価するため、著者らは**LLM-as-Jury**手法を採用した。GPT-4.1-mini、Claude-Haiku-3.5、Qwen3-30Bの3モデルが独立にルーブリックベースのプロンプトで評価を行い、スコアが3未満のコンポーネントを「スメルあり」と判定する。評価者間信頼性はICC（級内相関係数）で0.62-0.90の範囲であったと報告されている。

$$
\text{SmellScore}(t, c) = \begin{cases} 1 & \text{if } \text{median}(s_1, s_2, s_3) < 3 \\ 0 & \text{otherwise} \end{cases}
$$

ここで、$t$はツール、$c$はコンポーネント、$s_i$は各LLM評価者のスコア（1-5）である。

### Tool Description Router

記述拡張のトレードオフに対処するため、著者らは**Tool Description Router**を提案した。これは実行時にドメインとモデルの組み合わせに応じて、最適な記述コンポーネントの組み合わせを動的に選択するコンポーネントである。全コンポーネントを拡張するのではなく、コンテキスト固有の最適なバランスポイントを選択する。

### MCP-Universeベンチマーク

評価には6ドメイン・231タスクからなるMCP-Universeベンチマークが使用された。各タスクはMCPツールの呼び出しを必要とし、タスク成功率・実行ステップ数・評価者スコアの3指標で測定される。

## 実装のポイント（Implementation）

著者らが公開したスメルスキャナの実装において、以下の点が重要である。

```python
from dataclasses import dataclass


@dataclass
class ToolDescriptionAudit:
    """MCPツール記述の品質監査結果"""

    tool_name: str
    purpose_score: float
    guidelines_score: float
    limitations_score: float
    parameter_score: float
    completeness_score: float
    examples_score: float

    @property
    def has_smell(self) -> bool:
        """いずれかのコンポーネントがスコア3未満であればスメルあり"""
        scores = [
            self.purpose_score,
            self.guidelines_score,
            self.limitations_score,
            self.parameter_score,
            self.completeness_score,
            self.examples_score,
        ]
        return any(s < 3.0 for s in scores)

    @property
    def smell_components(self) -> list[str]:
        """スメルが検出されたコンポーネント一覧"""
        mapping = {
            "purpose": self.purpose_score,
            "guidelines": self.guidelines_score,
            "limitations": self.limitations_score,
            "parameters": self.parameter_score,
            "completeness": self.completeness_score,
            "examples": self.examples_score,
        }
        return [name for name, score in mapping.items() if score < 3.0]
```

実装上の注意点として、著者らはExamplesコンポーネントの除去がドメイン-モデル間で統計的に有意な性能低下を引き起こさなかったことを報告している。これは、開発者がPurposeとGuidelinesの品質に集中し、使用例の作成コストを削減できることを意味する。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

MCPツール記述の品質監査パイプラインをAWS上に構築する場合の推奨構成を示す。

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $50-150 | Lambda + Bedrock + DynamoDB |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $300-800 | Lambda + ECS Fargate + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,000-5,000 | EKS + Karpenter + EC2 Spot |

**Small構成の詳細**（月額$50-150）:
- **Lambda**: 1GB RAM、60秒タイムアウト（$20/月）
- **Bedrock**: Claude 3.5 Haiku（LLM-as-Jury評価用）、Prompt Caching有効（$80/月）
- **DynamoDB**: On-Demand、スメル監査結果キャッシュ（$10/月）
- **S3**: ツール記述スナップショット保存（$5/月）

**コスト削減テクニック**:
- Bedrock Batch APIで非リアルタイム監査を50%割引実行
- Prompt Cachingで3モデル評価のシステムプロンプトを30-90%削減
- DynamoDB TTLで古い監査結果を自動削除

**コスト試算の注意事項**: 上記は2026年4月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です。実際のコストはトラフィックパターンにより変動します。最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください。

### Terraformインフラコード

**Small構成（Serverless）: Lambda + Bedrock + DynamoDB**

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "mcp-audit-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["ap-northeast-1a", "ap-northeast-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]

  enable_nat_gateway   = false
  enable_dns_hostnames = true
}

resource "aws_iam_role" "lambda_auditor" {
  name = "mcp-audit-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "bedrock_invoke" {
  role = aws_iam_role.lambda_auditor.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
      Resource = "arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.claude-3-5-haiku*"
    }]
  })
}

resource "aws_lambda_function" "mcp_auditor" {
  filename      = "lambda.zip"
  function_name = "mcp-tool-description-auditor"
  role          = aws_iam_role.lambda_auditor.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 120
  memory_size   = 1024

  environment {
    variables = {
      BEDROCK_MODEL_ID    = "anthropic.claude-3-5-haiku-20241022-v1:0"
      DYNAMODB_TABLE      = aws_dynamodb_table.audit_cache.name
      ENABLE_PROMPT_CACHE = "true"
    }
  }
}

resource "aws_dynamodb_table" "audit_cache" {
  name         = "mcp-audit-cache"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "tool_id"

  attribute {
    name = "tool_id"
    type = "S"
  }

  ttl {
    attribute_name = "expire_at"
    enabled        = true
  }
}
```

### 運用・監視設定

**CloudWatch Logs Insights クエリ**:

```sql
fields @timestamp, tool_name, smell_count, total_score
| stats avg(total_score) as avg_score, sum(smell_count) as total_smells by bin(1h)
| filter total_smells > 50
```

**CloudWatch アラーム**:

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_alarm(
    AlarmName='mcp-audit-bedrock-token-spike',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='TokenUsage',
    Namespace='Custom/MCPAudit',
    Period=3600,
    Statistic='Sum',
    Threshold=300000,
    AlarmDescription='MCP監査のBedrockトークン使用量異常'
)
```

### コスト最適化チェックリスト

- [ ] ~100 req/日 → Lambda + Bedrock (Serverless) — $50-150/月
- [ ] ~1,000 req/日 → ECS Fargate + Bedrock (Hybrid) — $300-800/月
- [ ] EC2 Spot Instances優先（最大90%削減）
- [ ] Bedrock Batch API使用（50%割引）
- [ ] Prompt Caching有効化（30-90%削減）
- [ ] DynamoDB TTLで古い監査結果自動削除
- [ ] AWS Budgets月額予算設定（80%で警告）
- [ ] CloudWatchアラーム: トークン使用量スパイク検知

## 実験結果（Results）

### スメルの蔓延度（RQ-1）

著者らの分析によると、856ツール中97.1%に少なくとも1つのスメルが検出された（論文Table 1より）。最も多かったスメルは**Unclear Purpose**で56%のツールに該当する。公式サーバとコミュニティサーバの間で品質差は見られなかったと報告されている。

### 記述拡張の効果（RQ-2）

全コンポーネントを拡張した場合の効果（論文Table 3より）:

| 指標 | 変化量 | 備考 |
|------|--------|------|
| タスク成功率 | +5.85pp（中央値） | 統計的に有意 |
| 評価者スコア | +15.12%（平均） | 品質改善 |
| 実行ステップ数 | +67.46%（中央値） | コスト増加 |
| 性能退行ケース | 16.67% | 注意が必要 |

### コンポーネント削減実験（RQ-3）

著者らはアブレーション実験により、Examplesコンポーネントの除去がドメイン-モデルの組み合わせ全体で統計的に有意な性能低下を引き起こさなかったことを報告している。これは、PurposeとGuidelinesに集中した「コンパクト版」記述が、完全拡張版と同等の信頼性を維持できることを示唆している。

## 実運用への応用（Practical Applications）

本研究の知見はMCPツール開発者に以下の実践的示唆を与える。

1. **優先すべきコンポーネント**: PurposeとGuidelinesが基盤的価値を提供する。過剰なトークンオーバーヘッドなしにエージェントの精度を改善できる
2. **コンテキスト依存の拡張**: ドメインとモデルの組み合わせによって最適な記述レベルが異なる。著者らのTool Description Routerがこの選択を自動化する
3. **CI/CDへの統合**: 公開されたスメルスキャナをMCPサーバのデプロイパイプラインに組み込むことで、品質基準を下回るツール記述のデプロイを防止できる
4. **Zenn記事との関連**: Zenn記事で解説されている「セマンティック明確性」原則（原則3）の定量的検証として位置づけられる。docstringの品質がエージェント性能に直結するという主張を、大規模実証で裏付けている

## 関連研究（Related Work）

- **ToolACE（Liu et al., 2024）**: function calling学習データの自動生成パイプライン。ツール記述の品質向上とは異なるアプローチで、モデル側の能力を強化する
- **MCPAgentBench（Liu et al., 2025）**: MCPツール使用のベンチマーク。本研究の記述品質分析と補完的な関係にあり、ツール選択精度の評価基盤を提供する
- **EasyTool（Yuan et al., 2024）**: ツール記述の簡潔化によるエージェント改善。本研究とは逆のアプローチ（拡張 vs 簡潔化）だが、品質の重要性という点で一致する

## まとめと今後の展望

本論文は、MCPツール記述の品質問題を初めて体系的に分析し、97.1%のツールに品質問題が存在するという深刻な現状を明らかにした。6コンポーネントフレームワークとLLM-as-Jury評価手法により、記述品質の定量的評価が可能になった。一方で、記述拡張には精度-コストのトレードオフがあり、コンテキスト依存の最適化が必要であることも示されている。MCPエコシステムの成熟に伴い、ツール記述の品質保証は今後ますます重要になると考えられる。

## 参考文献

- **arXiv**: [https://arxiv.org/abs/2602.14878](https://arxiv.org/abs/2602.14878)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/c1f033224797db](https://zenn.dev/0h_n0/articles/c1f033224797db)
