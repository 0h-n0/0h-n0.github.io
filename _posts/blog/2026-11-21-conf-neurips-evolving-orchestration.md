---
layout: post
title: "NeurIPS 2025論文解説: Multi-Agent Collaboration via Evolving Orchestration"
description: "中央オーケストレータが強化学習で最適化されたルーティングポリシーでLLMエージェントを動的に指揮するPuppeteerフレームワークの解説"
categories: [blog, paper, conference]
tags: [multi-agent, LLM, reinforcement-learning, orchestration, autogen, agent, python, microsoft]
date: 2026-11-21 09:00:00 +0900
source_type: conference
conference: "NeurIPS 2025"
source_url: https://arxiv.org/abs/2505.19591
zenn_article: c5509c723c2670
zenn_url: https://zenn.dev/0h_n0/articles/c5509c723c2670
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## 論文概要（Abstract）

本記事は [https://arxiv.org/abs/2505.19591](https://arxiv.org/abs/2505.19591)（NeurIPS 2025）の解説記事です。

マルチエージェントLLMシステムにおいて、事前定義された静的なトポロジー（チェーン型・ツリー型・メッシュ型）はタスクの複雑化やエージェント数の増加に伴い調整オーバーヘッドが増大し、性能が劣化するという課題がある。著者らは、中央オーケストレータ（puppeteer）が強化学習で獲得したポリシーに基づきエージェント（puppet）を動的にルーティングする「Puppeteer」フレームワークを提案している。閉ドメイン（算術推論・多肢選択）と開ドメイン（ソフトウェア要件定義・概念生成）の双方で、ベースライン手法と比較して平均スコアの向上と計算コストの削減を報告している。

この記事は [Zenn記事: AutoGenからAgent Frameworkへ：自律エージェント移行実践ガイド](https://zenn.dev/0h_n0/articles/c5509c723c2670) の深掘りです。AutoGenやAgent Frameworkが提供する静的オーケストレーションパターン（Sequential、DAG等）に対して、本論文はRL駆動の動的オーケストレーションという異なるアプローチを提示しており、フレームワーク選択時の設計判断に有用な知見を提供する。

## 情報源

- **会議名**: NeurIPS 2025（Advances in Neural Information Processing Systems 38, Main Conference Track）
- **年**: 2025
- **URL**: [https://arxiv.org/abs/2505.19591](https://arxiv.org/abs/2505.19591)
- **Proceedings**: [NeurIPS 2025 Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f1320d2e2842169c6fc89dcbd80e94d0-Abstract-Conference.html)
- **著者**: Yufan Dang（清華大学）, Chen Qian（上海交通大学）, Xueheng Luo, Jingru Fan, Zihao Xie, Ruijie Shi, Weize Chen, Cheng Yang, Xiaoyin Che, Ye Tian, Xuantang Xiong, Lei Han, Zhiyuan Liu, Maosong Sun
- **コード**: [https://github.com/OpenBMB/ChatDev/tree/puppeteer](https://github.com/OpenBMB/ChatDev/tree/puppeteer)

## カンファレンス情報

NeurIPS（Conference on Neural Information Processing Systems）は機械学習・人工知能分野における最も影響力の高い国際会議の一つであり、採択率は例年20%前後と競争率が高い。本論文はMain Conference Trackとして採択されている。著者らは清華大学・上海交通大学を中心とした研究グループであり、ChatDevプロジェクト（LLMエージェントによるソフトウェア開発）の延長線上にある研究と位置づけられる。

## 背景と動機（Background & Motivation）

LLMベースのマルチエージェントシステム（MAS）は、単一モデルでは困難な多面的タスクへの対応手段として注目されている。しかし、著者らは既存のMASが抱える以下の課題を指摘している。

**静的トポロジーの限界**: AutoGenのRoundRobinGroupChatやLangGraphの固定グラフのように、エージェント間の通信パターンが事前に決定されている場合、タスクの性質に応じた柔軟な調整ができない。著者らは、50ノードのメッシュ構造MASが数百行のコード生成に最大10時間を要した事例を挙げている。

**スケーラビリティ問題**: エージェント数が増加するにつれて、冗長な計算、非効率な通信、集団的問題解決能力の低下が生じると著者らは主張している。

**研究課題**: 著者らはこの背景から「動的オーケストレーションは協調効果と計算効率を同時に最大化できるか？」という問いを設定し、2つの技術的革新を提案している。

1. **動的オーケストレーション**: 中央オーケストレータがタスクの現在状態に基づいてエージェントをステップごとにルーティングする
2. **適応的進化**: 完了タスクからのフィードバックでオーケストレータのポリシーをRLで更新し、強いエージェント軌跡を優先し弱い軌跡を刈り込む

## 主要な貢献（Key Contributions）

- **Puppeteerパラダイム**: 中央集権型オーケストレータがLLMエージェント群を動的に指揮するフレームワークの提案。エージェント選択とエージェント内部処理を分離することで、エージェントを再訓練せずに適応性とスケーラビリティを向上させる設計
- **逐次決定過程としての定式化**: マルチエージェント協調をMDP（マルコフ決定過程）として定式化し、REINFORCEアルゴリズムでオーケストレータポリシーを最適化する手法
- **動的トポロジーの創発**: 訓練を通じて、タスクに応じたコンパクトで循環的な推論構造が自動的に創発されることの実証
- **効率性**: ベースライン手法と比較して、性能向上と同時にトークン消費量の削減を達成（著者らの報告による）

## 技術的詳細（Technical Details）

### エージェントとMASの形式的定義

著者らはLLMエージェントを三つ組 $a = (m, r, t)$ として定義している：

- $m$: 基盤モデル（例: Llama-3.1-8B, GPT-4-Turbo）
- $r$: 推論パターンまたはプロンプト戦略（例: critique, reflect, planning）
- $t$: ツールセット（例: search_arxiv, run_python）

エージェント空間は $\mathcal{A} = \{(m, r, t)\}$ であり、マルチエージェントシステムは有向グラフ $G = (V, E)$ としてモデル化される。各ノードはエージェント、各辺は中間コンテキストの伝達を表す。

### Puppeteerの逐次決定過程

各タイムステップ $t$ において、オーケストレータはグローバル状態 $S_t$ とタスク仕様 $\tau$ に基づき、活性化するエージェント $a_t$ を選択する。

**エージェント選択（論文Eq. 1）**:

$$
a_t \sim \pi(S_t, \tau) = \mathbb{P}(a \mid S_t, \tau)
$$

ここで $\pi$ はオーケストレータのポリシーであり、状態とタスクからエージェントの確率分布への写像である。

**エージェント出力と状態更新（論文Eq. 2）**:

$$
o_t = f_{a_t}(s_t(a_t), S_t), \quad S_{t+1} = \Phi(S_t, o_t)
$$

ここで、
- $f_{a_t}$: 選択されたエージェント $a_t$ の推論関数
- $s_t(a_t)$: エージェント $a_t$ に渡される $S_t$ のスライス
- $\Phi$: 状態遷移関数
- $o_t$: エージェントの出力

著者らはこの過程がマルコフ性を持つと仮定している（論文Eq. 3）：

$$
\mathbb{P}(a_{t+1} \mid S_0, \dots, S_{t+1}, \tau) = \mathbb{P}(a_{t+1} \mid S_{t+1}, \tau)
$$

終端条件は、指定されたTerminatorエージェントが選択されるか、リソース予算が尽きた場合である。最終出力は $o^* = \Phi(S_T, o_T)$ として得られる。

### 強化学習によるポリシー最適化

オーケストレータのポリシーはREINFORCEアルゴリズムで最適化される。

**目的関数と勾配（論文Eq. 5）**:

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\big[R(\tau)\big], \quad \nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{n=1}^{N}\left(\sum_{t=1}^{T}\nabla_\theta \log \pi_\theta(a_t \mid S_t)\right) \cdot R(\tau)
$$

ここで $\tau = \{S_0, a_0, o_0, S_1, \dots, S_T, a_T, o_T\}$ は軌跡、$N$ はサンプルサイズ、$\theta$ はポリシーパラメータである。

**報酬関数（論文Eq. 6）**:

$$
R_t = \begin{cases} r - \lambda \cdot C_T, & \text{if } t = T \\ \gamma \cdot R_{t+1} - \lambda \cdot C_t, & \text{if } t < T \end{cases}
$$

ステップコストは以下で定義される：

$$
C_t = F \cdot \log\left(1 + \frac{t}{\varphi}\right)
$$

ここで、
- $r$: 最終報酬（正解のあるタスクでは0または1、開放ドメインでは$[0, 1]$の連続値）
- $\lambda$: 精度とコストのトレードオフを制御するハイパーパラメータ
- $\gamma$: 割引率（$\gamma = 0.99$）
- $F$: FLOPsまたはトークンベースのコスト指標
- $\varphi$: 最大ステップ予算

$\lambda$ を大きくするとコスト削減がより重視され、小さくすると精度が優先される。

### 直列化とトポロジー創発

著者らは協調トポロジーの探索空間が組合せ爆発するため、グラフを推論シーケンスに「展開」（unfold）する手法を採用している。トポロジカル順序に従って問題依存関係を直列化し、エピソード完了後に有向グラフに「折り畳む」（fold）ことで、暗黙的に協調トポロジーが構築される。

訓練を通じて、以下の2つの動態が観察されたと著者らは報告している：

1. **コンパクト化**: グラフ密度が上昇し、通信が少数のハブエージェントに集中する
2. **循環性の増加**: サイクルがより頻繁に出現し、中間結果の再循環と再帰的批評が促進される

### エージェント構成

論文Table 2より、使用されるエージェントは以下の2カテゴリに分類される：

**ツール使用エージェント（5種）**: read_file, search_arxiv, search_bing, access_website, run_python

**推論エージェント（8種）**: reasoning, critique, reflect, question, summarize, conclude, modify, planning

## 実装のポイント（Implementation）

著者らの実装に関して、以下の詳細が論文中で報告されている。

**ポリシーモデル**: NVIDIA Llama-3.1-Nemotron-70B-Reward-HFをベースとした変種が使用されている。ポリシーヘッドの構造や状態のシリアライズ方法については論文中に詳細な記述がない。

**ハイパーパラメータ**:
- エピソード長: 4
- 並列探索: 最大3
- $\lambda = 0.1$, $\gamma = 0.99$
- 訓練サンプル数: 200

**計算資源**（論文Table 3より）: NVIDIA A800 GPU 8枚、ピークメモリ28.8–78.4 GB/GPU、訓練時間2–6時間。訓練はオンラインで推論と交互に実行されるため、訓練のみのGPU時間を分離することは困難と著者らは述べている。

**再現上の注意点**:
- 論文Eq. 5では軌跡全体のリターン $R(\tau)$ を使用しているが、Eq. 6ではステップごとのリターン $R_t$ を定義しており、どちらが勾配計算に使用されるか明示されていない
- MASの定義ではDAG（有向非巡回グラフ）を前提としているが、実験結果ではサイクルを含むグラフが創発されており、形式的な整合性の説明がない
- 結果は単一ランの報告であり、エラーバー・シード数・統計的有意性検定は提示されていない

## Production Deployment Guide

本論文のPuppeteerフレームワークを本番環境でマルチエージェントオーケストレータとしてデプロイする場合のAWS構成を示す。

### AWS実装パターン（コスト最適化重視）

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $80-200 | Lambda + Bedrock + DynamoDB |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $400-1,000 | Lambda + ECS Fargate + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $2,500-6,000 | EKS + Karpenter + EC2 Spot |

**Small構成の詳細**（月額$80-200）:
- **Lambda**: オーケストレータロジック。1GB RAM, 60秒タイムアウト ($30/月)
- **Bedrock**: エージェント推論。Claude 3.5 Haiku, Prompt Caching有効 ($100/月)
- **DynamoDB**: 状態管理 $S_t$ の永続化。On-Demand ($15/月)
- **Step Functions**: エージェントルーティングの状態機械 ($10/月)
- **CloudWatch**: 基本監視 ($5/月)

**コスト削減テクニック**: Bedrock Batch APIで50%削減、Prompt Cachingで30-90%削減、Terminatorメカニズムでエージェント呼び出し回数を制限。

**注意**: 上記は2026年11月時点のAWS東京リージョン料金に基づく概算値。最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認のこと。

### Terraformインフラコード

**Small構成（Serverless）: Lambda + Step Functions + Bedrock**

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "puppeteer-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["ap-northeast-1a", "ap-northeast-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]

  enable_nat_gateway   = false
  enable_dns_hostnames = true
}

resource "aws_iam_role" "orchestrator_lambda" {
  name = "puppeteer-orchestrator-role"

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
  role = aws_iam_role.orchestrator_lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
      Resource = "arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.claude-3-5-haiku*"
    }]
  })
}

resource "aws_lambda_function" "orchestrator" {
  filename      = "orchestrator.zip"
  function_name = "puppeteer-orchestrator"
  role          = aws_iam_role.orchestrator_lambda.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 120
  memory_size   = 1024

  environment {
    variables = {
      BEDROCK_MODEL_ID    = "anthropic.claude-3-5-haiku-20241022-v1:0"
      STATE_TABLE         = aws_dynamodb_table.agent_state.name
      MAX_STEPS           = "4"
      COST_LAMBDA         = "0.1"
    }
  }
}

resource "aws_dynamodb_table" "agent_state" {
  name         = "puppeteer-agent-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "session_id"
  range_key    = "step_t"

  attribute {
    name = "session_id"
    type = "S"
  }
  attribute {
    name = "step_t"
    type = "N"
  }

  ttl {
    attribute_name = "expire_at"
    enabled        = true
  }
}

resource "aws_cloudwatch_metric_alarm" "orchestrator_cost" {
  alarm_name          = "puppeteer-cost-spike"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = 3600
  statistic           = "Sum"
  threshold           = 200000
  alarm_description   = "オーケストレータ実行時間異常（コスト急増の可能性）"
  dimensions = {
    FunctionName = aws_lambda_function.orchestrator.function_name
  }
}
```

### 運用・監視設定

**CloudWatch Logs Insights クエリ**:

```sql
-- オーケストレータのステップ数・エージェント選択パターン分析
fields @timestamp, session_id, step_t, selected_agent, token_count
| stats count(*) as invocations, avg(token_count) as avg_tokens by selected_agent
| sort invocations desc
```

**X-Ray トレーシング**:

```python
from aws_xray_sdk.core import xray_recorder, patch_all

patch_all()

@xray_recorder.capture('puppeteer_step')
def orchestrate_step(state: dict, task: str, step_t: int) -> dict:
    xray_recorder.put_annotation('step', step_t)
    xray_recorder.put_annotation('agent', state.get('selected_agent', 'unknown'))
    xray_recorder.put_metadata('token_count', state.get('tokens', 0))
    return execute_agent(state, task)
```

### コスト最適化チェックリスト

**アーキテクチャ選択**:
- [ ] ~100 req/日 → Lambda + Step Functions + Bedrock（$80-200/月）
- [ ] ~1,000 req/日 → ECS Fargate + Bedrock（$400-1,000/月）
- [ ] 10,000+ req/日 → EKS + Spot Instances（$2,500-6,000/月）

**リソース・LLMコスト最適化**:
- [ ] Spot Instances優先（最大90%削減）/ Reserved Instances 1年コミット（最大72%削減）
- [ ] Bedrock Batch API（50%割引）/ Prompt Caching（30-90%削減）
- [ ] モデル選択: 推論エージェントにはHaiku、批評エージェントにはSonnet
- [ ] Terminatorによる早期終了でトークン消費を制限
- [ ] AWS Budgets + Cost Anomaly Detection + 日次コストレポート

## 実験結果（Results）

### 実験設定

著者らは2つのエージェントサブスペースを定義している：

- **Mimas**（小規模モデル群）: Qwen-2.5-7B/14B, Llama-3.1-8B, Llama-3.2-3B, Mistral-7B/Nemo-12B
- **Titan**（大規模モデル群）: GPT-4-Turbo, GPT-4o-Mini, Gemini-1.5-Pro/Flash, Claude-3-Sonnet/Haiku, Qwen-2.5-72B, Llama-3.1-405B

ベースラインはSelf-Refine（単一エージェント反復改善）、AFlow（MCTSベースのワークフロー探索）、MacNet（静的DAGトポロジー）、EvoAgent（進化的MAS生成）。

評価データセットは閉ドメイン（GSM-Hard, MMLU-Pro）と開ドメイン（SRDD, CommonGen-Hard）の4種。

### 主要結果（論文Table 1より）

**Titanサブスペース**:

| 手法 | GSM-Hard | MMLU-Pro | SRDD | CommonGen-Hard | 平均 |
|------|----------|----------|------|----------------|------|
| Llama-3.1-405B（単体） | 0.1350 | 0.7600 | 0.6061 | 0.8116 | 0.5781 |
| AFlow | 0.5400 | 0.7500 | 0.6478 | 0.8218 | 0.6899 |
| MacNet | 0.2905 | 0.4800 | 0.4228 | 0.8817 | 0.5187 |
| Puppeteer（初期化） | 0.6600 | 0.7400 | 0.6191 | 0.7381 | 0.6893 |
| Puppeteer（進化後） | 0.7000 | 0.8300 | 0.7637 | 0.7987 | **0.7731** |

**Mimasサブスペース**:

| 手法 | GSM-Hard | MMLU-Pro | SRDD | CommonGen-Hard | 平均 |
|------|----------|----------|------|----------------|------|
| AFlow | 0.2900 | 0.5000 | 0.6362 | 0.7194 | 0.5364 |
| Puppeteer（初期化） | 0.5600 | 0.5700 | 0.6653 | 0.7139 | 0.6273 |
| Puppeteer（進化後） | 0.5400 | 0.6300 | 0.6266 | 0.7333 | **0.6324** |

著者らはPuppeteerの進化後の平均スコアが両サブスペースで全ベースラインを上回ったと報告している。

**結果の注意点**:
- Titanサブスペースにおいて、個別タスクではAFlow（GSM-Hard: 0.7500）やMacNet（CommonGen-Hard: 0.8817）がPuppeteerの進化後スコアを上回るケースがある
- Mimasサブスペースでは、Puppeteerの進化後にGSM-Hard（0.5600→0.5400）とSRDD（0.6653→0.6266）で初期化時より性能が低下しており、平均の向上はMMLU-ProとCommonGen-Hardの改善による
- 結果は単一ランの報告であり、エラーバーや統計的有意性検定は提示されていない。サンプル数（200）も限定的であるため、再現性の検証が必要

### 効率性分析

著者らは訓練を通じてトークン消費量が減少する傾向を報告している。

- **Titan**: オーケストレータがTerminatorをより早期に選択するようになり、推論チェーンが短縮される
- **Mimas**: 主にコストの低いエージェントを選好するようになるが、チェーン長はほぼ一定。著者らはこれを「弱いエージェントは信頼性を維持するためにより長い推論チェーンを必要とする」と説明している

## 実運用への応用（Practical Applications）

本論文のPuppeteerフレームワークは、AutoGenやAgent Frameworkの静的オーケストレーション（Sequential, DAG）とは異なる、動的ルーティングの設計思想を提供する。

**AutoGen/Agent Frameworkとの関連**:
- Agent Frameworkの`WorkflowBuilder`は`add_edge`でエッジを明示的に定義する静的アプローチ。Puppeteerはエッジを学習で獲得する動的アプローチ
- Agent Frameworkの`SequentialBuilder`はラウンドロビン的な固定順序。Puppeteerはタスク依存の可変順序
- 両者は排他的ではなく、静的ワークフローの中にPuppeteer的な動的ルーティングを組み込むハイブリッド設計が考えられる

**適用が有効なシナリオ**:
- タスクの性質が事前に予測困難で、必要なエージェント構成がタスクごとに異なる場合
- エージェント数が多く（10以上）、全組み合わせの事前定義が非現実的な場合

**制約と限界**:
- 訓練に200サンプル・2-6時間のGPU計算が必要（論文Table 3より）
- エージェントセットが固定で、推論時の動的追加・削除に対応していない
- エージェント間の誤調整や欺瞞的合意が生じる可能性がある（著者ら自身が指摘）
- REINFORCEの高分散性に対するベースライン項が論文中に記載されていない

## 関連研究（Related Work）

- **ChatDev** (Qian et al., 2023): 同じ研究グループによるLLMエージェントでのソフトウェア開発フレームワーク。ウォーターフォール型の固定トポロジーを使用
- **AutoGen** (Wu et al., 2023): Microsoftの会話型マルチエージェントフレームワーク。本論文が対比する静的トポロジーの代表例
- **AFlow** (Zhang et al., 2024): MCTSでコード空間上のワークフローを探索する手法。本論文のベースラインの一つで、個別タスクではPuppeteerを上回るケースがある
- **MacNet**: 静的DAGトポロジーでマルチエージェント協調を行う手法。CommonGen-Hardで最高スコア（0.8817）を記録しているが、平均ではPuppeteerに劣る
- **GPTSwarm / ADAS**: 最適化可能なグラフや探索ベースのアーキテクチャ設計。Puppeteerとは異なり、探索空間の定義に人手の設計が必要

## まとめと今後の展望

本論文は、マルチエージェントLLMシステムにおける静的トポロジーの限界に対して、RL駆動の動的オーケストレーション（Puppeteer）という解法を提示している。平均スコアの改善と計算コスト削減を同時に達成したと著者らは報告しているが、個別タスクでの勝敗は混在しており、単一ラン・統計検定なしという実験上の制約がある。

AutoGenからAgent Frameworkへの移行を検討する実務者にとって、本論文は「静的vs動的オーケストレーション」の設計判断に関する学術的な根拠を提供する。Agent Frameworkの`WorkflowBuilder`による明示的エッジ定義と、Puppeteerの学習ベースルーティングはトレードオフの関係にあり、タスクの予測可能性とエージェント数に応じた使い分けが推奨される。

## 参考文献

- **Conference URL**: [https://proceedings.neurips.cc/paper_files/paper/2025/hash/f1320d2e2842169c6fc89dcbd80e94d0-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f1320d2e2842169c6fc89dcbd80e94d0-Abstract-Conference.html)
- **arXiv**: [https://arxiv.org/abs/2505.19591](https://arxiv.org/abs/2505.19591)
- **Code**: [https://github.com/OpenBMB/ChatDev/tree/puppeteer](https://github.com/OpenBMB/ChatDev/tree/puppeteer)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/c5509c723c2670](https://zenn.dev/0h_n0/articles/c5509c723c2670)
