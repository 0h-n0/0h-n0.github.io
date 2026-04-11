---
layout: post
title: "ICML 2024論文解説: LLMCompiler — LLMの並列Function Callingを最適化するコンパイラフレームワーク"
description: "古典的コンパイラ最適化の原理をLLMのツール呼び出しに応用し、最大3.7倍の低レイテンシと6.7倍のコスト削減を実現したLLMCompilerを解説"
categories: [blog, paper, conference]
tags: [function-calling, parallel-execution, LLM, tool-use, compiler, openai, claude, gemini, python]
date: 2026-04-12 09:00:00 +0900
source_type: conference
conference: ICML 2024
arxiv_id: "2312.04511"
source_url: https://arxiv.org/abs/2312.04511
zenn_article: a1b896060efa28
zenn_url: https://zenn.dev/0h_n0/articles/a1b896060efa28
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [LLMCompiler: An LLM Compiler for Parallel Function Calling](https://arxiv.org/abs/2312.04511) の解説記事です。

## 論文概要（Abstract）

LLMCompilerは、LLM（大規模言語モデル）が外部関数を呼び出す際の逐次実行ボトルネックを解消するために、古典的コンパイラの命令レベル並列性（ILP: Instruction-Level Parallelism）の原理を応用したフレームワークである。著者らは、Function Calling Planner・Task Fetching Unit・Executorの3コンポーネントから成るアーキテクチャにより、ReActベースラインと比較して最大3.7倍のレイテンシ削減、6.7倍のコスト削減、約9%の精度向上を報告している（論文Table 1, Table 2より）。

この記事は [Zenn記事: Function Calling実装パターン2026](https://zenn.dev/0h_n0/articles/a1b896060efa28) の深掘りです。

## 情報源

- **会議名**: ICML 2024（International Conference on Machine Learning）
- **arXiv ID**: 2312.04511
- **URL**: [https://arxiv.org/abs/2312.04511](https://arxiv.org/abs/2312.04511)
- **著者**: Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W. Mahoney, Kurt Keutzer, Amir Gholami（UC Berkeley）
- **コード**: [https://github.com/SqueezeAILab/LLMCompiler](https://github.com/SqueezeAILab/LLMCompiler)

## カンファレンス情報

ICML（International Conference on Machine Learning）は機械学習分野の最高峰国際会議の1つであり、採択率は通常25%前後である。LLMCompilerはICML 2024のメインカンファレンスに採択された。

## 背景と動機（Background & Motivation）

LLMのFunction Calling（ツール呼び出し）は、2024年以降のAIエージェント開発における基盤技術として定着している。しかし、従来のReActパラダイム（Yao et al., 2022）では、LLMが1ステップずつ「思考→行動→観察」を繰り返すため、以下の問題が生じていた。

1. **レイテンシの蓄積**: 独立した関数呼び出しであっても逐次実行されるため、各ツールの実行時間が加算される
2. **コストの増大**: 各ステップでLLMの推論が必要となり、入出力トークン数が累積的に増加する
3. **精度の低下**: 中間ステップでのLLM推論が増えるほど、エラーが伝搬・蓄積するリスクが高まる

著者らは、この問題がCPUの命令スケジューリングと構造的に類似していることに着目した。コンパイラが独立した命令を検出して並列実行するように、LLMの関数呼び出しでも依存関係のないタスクを自動検出し並列ディスパッチすることで、上記3つの問題を同時に解決できると主張している。

## 主要な貢献（Key Contributions）

- **貢献1**: LLMの逐次的Function Callingを並列化するフレームワーク「LLMCompiler」の提案。古典的コンパイラのDAG（有向非巡回グラフ）ベース命令スケジューリングをLLMのツール呼び出し計画に応用した
- **貢献2**: Function Calling Planner・Task Fetching Unit・Executorの3コンポーネントから成るアーキテクチャにより、プランニングとツール実行を分離し、LLM呼び出し回数を削減
- **貢献3**: ReActと比較して、複数ベンチマークで最大3.7倍のレイテンシ削減、6.7倍のコスト削減、約9%の精度向上を実証

## 技術的詳細（Technical Details）

### アーキテクチャ概要

LLMCompilerは3つのコンポーネントで構成される。

```mermaid
graph LR
    A[ユーザークエリ] --> B[Function Calling Planner]
    B --> C[Task Fetching Unit]
    C --> D[Executor]
    D --> E[結果集約]
    E --> F[最終応答生成]
```

#### 1. Function Calling Planner

Plannerはユーザーのクエリとツール定義を受け取り、関数呼び出しの実行計画をDAG形式で一括生成する。ReActとの最大の違いは、**1回のLLM推論で全体の実行計画を生成する**点にある。

具体的には、Plannerの出力は以下のような構造化テキストである。

```
1. get_weather(city="東京") 
2. get_weather(city="大阪")
3. get_exchange_rate(from="USD", to="JPY")
4. join(1, 2, 3)
```

ここで各行がタスクノードを表し、`join`は依存関係を示す。タスク1, 2, 3は互いに独立しているため並列実行可能であり、タスク4はそれらすべての完了を待つ。

この計画生成は、コンパイラにおけるDAGスケジューリングと等価である。

$$
G = (V, E)
$$

ここで、
- $V$: 関数呼び出しタスクの集合
- $E$: タスク間の依存関係（エッジ $(v_i, v_j) \in E$ は $v_j$ が $v_i$ の出力を入力として必要とすることを示す）

独立したタスク（入次数が0のノード）は即座に並列ディスパッチ可能である。

#### 2. Task Fetching Unit

Task Fetching Unitは、Plannerが生成したDAGを監視し、依存関係が充足されたタスクを検出してExecutorに渡す。著者らによれば、このコンポーネントにはLLM推論は不要であり、純粋にプログラム的なDAGトラバーサルで実装されている。

$$
\text{Ready}(t) = \{ v \in V \mid \forall (u, v) \in E, \; u \in \text{Completed}(t) \}
$$

ここで $\text{Ready}(t)$ は時刻 $t$ に実行可能なタスク集合、$\text{Completed}(t)$ は時刻 $t$ までに完了したタスク集合を表す。

#### 3. Executor

Executorは、Task Fetching Unitから受け取ったタスクを並列に実行する。各タスクは外部APIやツールを実際に呼び出し、結果を返却する。全タスク完了後、最終的な結果集約と応答生成が行われる。

### アルゴリズム

以下にLLMCompilerの全体フローを擬似コードで示す。

```python
from dataclasses import dataclass
from typing import Callable
import asyncio


@dataclass
class Task:
    """DAG内の個別タスク"""
    id: int
    func: Callable
    args: dict
    dependencies: list[int]  # 依存するタスクIDのリスト
    result: dict | None = None


async def llm_compiler(query: str, tools: list[dict], llm: Callable) -> str:
    """LLMCompilerの全体フロー

    Args:
        query: ユーザーのクエリ
        tools: 利用可能なツール定義のリスト
        llm: LLM推論関数

    Returns:
        最終応答テキスト
    """
    # Phase 1: Planner（1回のLLM呼び出しで実行計画を生成）
    plan = llm(
        prompt=f"Query: {query}\nTools: {tools}\n"
               "Generate a DAG of function calls.",
    )
    dag: list[Task] = parse_plan_to_dag(plan)

    # Phase 2: Task Fetching + Execution（LLM呼び出し不要）
    completed: dict[int, dict] = {}
    while len(completed) < len(dag):
        # 依存関係が充足されたタスクを検出
        ready = [
            t for t in dag
            if t.id not in completed
            and all(dep in completed for dep in t.dependencies)
        ]
        # 並列実行
        results = await asyncio.gather(
            *[execute_task(t, completed) for t in ready]
        )
        for task, result in zip(ready, results):
            completed[task.id] = result

    # Phase 3: 結果集約（1回のLLM呼び出し）
    final_response = llm(
        prompt=f"Query: {query}\nResults: {completed}\n"
               "Generate a final answer.",
    )
    return final_response
```

**ReActとの根本的な違い**: ReActでは各ステップでLLMを呼び出すため、$N$個のツール呼び出しに対して最低$2N+1$回のLLM推論が必要になる（各ステップの思考+行動+最終応答）。LLMCompilerでは、Plannerで1回、最終応答生成で1回の計2回で済む。この差がコスト削減の主因である。

## 実装のポイント（Implementation）

著者らは以下の実装上の注意点を挙げている。

1. **Plannerのプロンプト設計**: ツール間の依存関係を正確に推論させるために、Few-shotの実行計画例をプロンプトに含める必要がある。著者らは3-5個の例が十分と報告している
2. **動的リプランニング**: 一部のタスクが失敗した場合、Plannerを再呼び出しして残りのタスクを再計画する機構が実装されている。これはコンパイラの例外ハンドリングに相当する
3. **`join`の柔軟性**: 依存タスクの結果を引数として受け取る`$`記法（例: `get_detail(id=$1)`で、タスク1の結果を引数に渡す）をサポートしている
4. **オープンソースモデル対応**: LLaMA-2ベースのモデルでもPlannerとして機能することが検証されている。ただし、GPT-4と比較するとDAG生成の品質は劣る

## 実験結果（Results）

著者らは以下のベンチマークで評価を行い、結果を報告している（論文Table 1, Table 2より）。

### HotpotQA（マルチホップ質問応答）

| 手法 | 精度 (EM) | レイテンシ | コスト（トークン数） |
|------|----------|-----------|-------------------|
| ReAct | 38.9% | 1.0x（基準） | 1.0x（基準） |
| LLMCompiler (GPT-4) | 44.1% | 0.27x（3.7倍高速） | 0.15x（6.7倍削減） |

### Movie Recommendation

| 手法 | 精度 | レイテンシ | コスト |
|------|------|-----------|-------|
| ReAct | 62.0% | 1.0x | 1.0x |
| LLMCompiler (GPT-4) | 67.5% | 0.39x（2.6倍高速） | 0.26x（3.8倍削減） |

著者らによれば、レイテンシ削減は主に並列ディスパッチによるものであり、コスト削減はLLM推論回数の削減（ReActの$2N+1$回から2回）による。精度向上は、中間推論ステップの削減によりエラー伝搬が抑制されたためと分析されている。

### 制約と限界

著者らは以下の制約も報告している。

- **プランニング品質への依存**: Plannerが不正確なDAGを生成した場合、全体の精度が低下する。特にツール間の依存関係が複雑な場合に誤りが生じやすい
- **DFSDT（ToolLLM）との比較**: 複雑なマルチステップ推論では、DFSDTの探索的アプローチがLLMCompilerの一括計画より有利な場合がある
- **ストリーミング非対応**: 論文ではバッチ処理を前提としており、ストリーミング応答中の動的リプランニングは扱っていない

## 実運用への応用（Practical Applications）

LLMCompilerのアーキテクチャは、Zenn記事で解説されている並列Function Callingの実装パターンと直接的に対応する。

- **OpenAIの`parallel_tool_calls`パラメータ**: LLMCompilerのPlannerに相当する機能がAPI側に組み込まれている。ただし、依存関係の検出はモデルの判断に依存する
- **`asyncio.gather`パターン**: Zenn記事で示されたExecutor側の並列実行パターンは、LLMCompilerのExecutorコンポーネントと同等である
- **エラーリカバリ**: LLMCompilerの動的リプランニングは、Zenn記事のリトライ+LLMフィードバックパターンの上位概念として位置づけられる

産業応用としては、複数の外部APIを並列に呼び出す必要があるカスタマーサポートBot、マルチソース情報検索、データパイプラインオーケストレーション等が想定される。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

LLMCompilerのPlannerをLLM APIで、Executor部分をLambdaで実装するアーキテクチャを示す。

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $50-150 | Lambda + Bedrock + DynamoDB |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $300-800 | Lambda + ECS Fargate + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,000-5,000 | EKS + Karpenter + EC2 Spot |

**Small構成の詳細** (月額$50-150):
- **Lambda**: Planner呼び出し + Executor並列実行、1GB RAM、60秒タイムアウト ($20/月)
- **Bedrock**: Claude 3.5 Haiku（Planner推論）、Prompt Caching有効 ($80/月)
- **DynamoDB**: DAGキャッシュ、On-Demand ($10/月)
- **Step Functions**: DAGオーケストレーション ($5/月)

**コスト削減テクニック**:
- LLMCompilerの構造上、LLM呼び出しが2回で済むため、ReActベースと比較してBedrock費用が最大6.7倍削減される
- Prompt Caching有効化でPlanner呼び出しの入力トークンコストを30-90%削減
- DynamoDBでDAGキャッシュすることで、同一クエリパターンの再計画を回避

**コスト試算の注意事項**:
- 上記は2026年4月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です
- 実際のコストはトラフィックパターン、リージョン、バースト使用量により変動します
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

**Small構成 (Serverless): Lambda + Bedrock + Step Functions**

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "llmcompiler-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["ap-northeast-1a", "ap-northeast-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]

  enable_nat_gateway   = false
  enable_dns_hostnames = true
}

resource "aws_iam_role" "lambda_planner" {
  name = "llmcompiler-planner-role"

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
  role = aws_iam_role.lambda_planner.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
      Resource = "arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.claude-3-5-haiku*"
    }]
  })
}

resource "aws_lambda_function" "planner" {
  filename      = "planner.zip"
  function_name = "llmcompiler-planner"
  role          = aws_iam_role.lambda_planner.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 60
  memory_size   = 1024

  environment {
    variables = {
      BEDROCK_MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"
      DYNAMODB_TABLE   = aws_dynamodb_table.dag_cache.name
    }
  }
}

resource "aws_dynamodb_table" "dag_cache" {
  name         = "llmcompiler-dag-cache"
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

resource "aws_cloudwatch_metric_alarm" "planner_cost" {
  alarm_name          = "llmcompiler-planner-cost"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = 3600
  statistic           = "Sum"
  threshold           = 100000
  alarm_description   = "Planner Lambda実行時間異常"
  dimensions = {
    FunctionName = aws_lambda_function.planner.function_name
  }
}
```

### セキュリティベストプラクティス

- **IAMロール**: Bedrock InvokeModelのみ許可（最小権限）
- **ネットワーク**: Lambda VPC内配置、パブリックアクセス不可
- **シークレット**: API キーはSecrets Manager経由、環境変数ハードコード禁止
- **暗号化**: DynamoDB KMS暗号化、S3バケット暗号化
- **監査**: CloudTrail有効化

### 運用・監視設定

**CloudWatch Logs Insights クエリ**:

```sql
fields @timestamp, planner_latency_ms, executor_latency_ms, total_tools
| stats avg(planner_latency_ms) as avg_plan,
        avg(executor_latency_ms) as avg_exec,
        avg(total_tools) as avg_tools
  by bin(1h)
| filter total_tools > 1
```

**CloudWatch アラーム（コスト重視）**:

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_alarm(
    AlarmName='llmcompiler-bedrock-token-spike',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='TokenUsage',
    Namespace='Custom/LLMCompiler',
    Period=3600,
    Statistic='Sum',
    Threshold=500000,
    AlarmDescription='Bedrockトークン使用量異常（Planner再計画ループの可能性）'
)
```

### コスト最適化チェックリスト

**アーキテクチャ選択**:
- [ ] ~100 req/日 → Lambda + Step Functions (Serverless) - $50-150/月
- [ ] ~1,000 req/日 → ECS Fargate + ElastiCache (Hybrid) - $300-800/月
- [ ] 10,000+ req/日 → EKS + Spot Instances (Container) - $2,000-5,000/月

**LLMCompiler固有の最適化**:
- [ ] DAGキャッシュ: 同一パターンのクエリでPlanner再実行を回避
- [ ] Planner推論をHaikuモデルで実行（コスト削減、精度トレードオフ検証済み）
- [ ] Executor並列度制限: API レート制限に合わせてconcurrency上限を設定
- [ ] 動的リプランニング回数上限: 最大3回（著者らの推奨）

**リソース最適化**:
- [ ] EC2 Spot Instances優先（最大90%削減）
- [ ] Reserved Instances 1年コミット（72%削減）
- [ ] Lambda メモリサイズ最適化（Planner: 1024MB推奨）
- [ ] ECS/EKS アイドル時スケールダウン

**監視・アラート**:
- [ ] AWS Budgets 月額予算設定
- [ ] CloudWatch トークン使用量スパイク検知
- [ ] Cost Anomaly Detection有効化
- [ ] Planner失敗率モニタリング（DAG生成エラー検知）

## 関連研究（Related Work）

- **ReAct (Yao et al., 2022)**: 思考と行動を交互に行うパラダイム。LLMCompilerはReActの逐次性を解消する上位アーキテクチャとして位置づけられる
- **ToolLLM / DFSDT (Qin et al., 2023)**: 16,000以上のAPIに対する深さ優先探索ベースの計画手法。複雑な依存関係がある場合にLLMCompilerより有利な場合がある
- **HuggingGPT / JARVIS (Shen et al., 2023)**: タスク分解+モデルオーケストレーションの先行研究。並列サブタスク実行の概念はLLMCompilerと共通するが、コンパイラ最適化の形式的な枠組みは持たない

## まとめと今後の展望

LLMCompilerは、古典的コンパイラ技術をLLMのFunction Callingに応用することで、逐次実行の3つの課題（レイテンシ・コスト・精度）を同時に改善した。著者らの報告によれば、DAGベースの一括計画生成により、LLM推論回数を$2N+1$回から2回に削減できる点が最大の貢献である。

今後の研究方向としては、ストリーミング応答との統合、より複雑な条件分岐（if-else）を含むDAGの生成、およびオープンソースモデルでのPlanner品質向上が挙げられる。実務的には、OpenAI・Anthropic・Googleが提供するネイティブの並列Function Calling機能と、LLMCompilerのような外部フレームワークのどちらが適切かを、ツール数と依存関係の複雑さに応じて選択することが重要である。

## 参考文献

- **Conference URL**: [https://proceedings.mlr.press/v235/kim24y.html](https://proceedings.mlr.press/v235/kim24y.html)
- **arXiv**: [https://arxiv.org/abs/2312.04511](https://arxiv.org/abs/2312.04511)
- **Code**: [https://github.com/SqueezeAILab/LLMCompiler](https://github.com/SqueezeAILab/LLMCompiler)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/a1b896060efa28](https://zenn.dev/0h_n0/articles/a1b896060efa28)
