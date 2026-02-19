---
layout: post
title: "ACL 2025論文解説: MAIN-RAG — マルチエージェント協調フィルタリングでRAGの検索ノイズを解消する"
description: "ACL 2025採択のMAIN-RAG論文を詳細解説。3エージェント協調フィルタリングと適応的閾値調整でRAG精度2-11%向上を実現"
categories: [blog, paper, conference]
tags: [RAG, multi-agent, LLM, retrieval, NLP, llamaindex, filtering]
date: 2026-02-19 23:45:00 +0900
source_type: conference
conference: "ACL 2025"
source_url: https://aclanthology.org/2025.acl-long.131/
zenn_article: 62e946539206db
zenn_url: https://zenn.dev/0h_n0/articles/62e946539206db
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## 論文概要（Abstract）

MAIN-RAGは、RAGパイプラインにおける検索ドキュメントのノイズ問題を、複数LLMエージェントの協調フィルタリングで解決する**訓練不要（training-free）**のフレームワークである。Predictor（予測器）、Judge（評価器）、Final-Predictor（最終予測器）の3エージェントが協調して検索結果の関連性を評価・スコアリングし、適応的閾値（Adaptive Judge Bar）で動的にフィルタリングする。4つのQAベンチマークで既存RAG手法に対して2-11%の精度改善を達成し、ACL 2025（Long Paper）に採択された。

この記事は [Zenn記事: LlamaIndex v0.14実践ガイド：AgentWorkflowで本番RAGを構築する](https://zenn.dev/0h_n0/articles/62e946539206db) の深掘りです。

## 情報源

- **会議名**: ACL 2025（Association for Computational Linguistics 第63回年次大会）
- **年**: 2025
- **URL**: https://aclanthology.org/2025.acl-long.131/
- **著者**: Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh, Menghai Pan, Chin-Chia Michael Yeh, Guanchu Wang, Mingzhi Hu, Zhichao Xu, Yan Zheng, Mahashweta Das, Na Zou
- **arXiv ID**: 2501.00332
- **発表形式**: Long Paper (pp. 2607-2622)

## カンファレンス情報

**ACLについて**:
- ACL（Association for Computational Linguistics）は自然言語処理（NLP）分野の最高峰国際会議の1つ
- Long Paper採択率は通常20-25%程度（非常に競争率が高い）
- 2025年はオーストリア・ウィーンで開催
- RAGとエージェントの統合は2025年のホットトピックの1つ

## 技術的詳細（Technical Details）

### 問題定義: RAGの検索ノイズ問題

RAGの根本的な課題は、ベクトル検索で取得したtop-kドキュメントの中に**クエリと無関係なドキュメント（ノイズ）が含まれる**ことである。ノイズドキュメントはLLMの回答品質を直接的に劣化させる。

$$
\text{Quality}(\text{response}) \propto \frac{|\mathcal{D}_{\text{relevant}}|}{|\mathcal{D}_{\text{retrieved}}|}
$$

ここで、
- $\mathcal{D}_{\text{relevant}}$: 検索結果中のクエリに関連するドキュメント集合
- $\mathcal{D}_{\text{retrieved}}$: 検索で取得した全ドキュメント集合

つまり、検索結果の**精度（precision）**が低いほど回答品質が低下する。この問題はtop-kの$k$を大きくするほど深刻化する。

### 3エージェントアーキテクチャ

MAIN-RAGは以下の3つの専門エージェントで構成される。

```
Query + Retrieved Documents
         │
    ┌────▼────┐
    │ Agent-1  │  Predictor: 各ドキュメントから予備回答生成
    │(Predictor)│  → Doc-Query-Answer Triplet作成
    └────┬────┘
         │
    ┌────▼────┐
    │ Agent-2  │  Judge: Tripletの関連性評価
    │ (Judge)  │  → Yes/No判定 + log確率スコア
    └────┬────┘
         │
    ┌────▼────┐  Adaptive Judge Bar (τq)
    │ Filtering│  → スコア分布に基づく動的閾値
    └────┬────┘
         │
    ┌────▼────┐
    │ Agent-3  │  Final-Predictor: フィルタ済みドキュメントで最終回答
    │(Final)   │
    └─────────┘
```

**Agent-1 (Predictor): 予備回答生成**

各検索ドキュメント$d_i$に対して、クエリ$q$への予備回答$a_i$を生成する。

$$
a_i = \text{LLM}_1(q, d_i) \quad \forall d_i \in \mathcal{D}_{\text{retrieved}}
$$

この段階の目的は、ドキュメントが回答に寄与できるかどうかを「実際に回答させて確認する」こと。関連性の低いドキュメントからは的外れな回答が生成される。

**Agent-2 (Judge): 関連性スコアリング**

Predictor が生成した (Document, Query, Answer) トリプルを受け取り、ドキュメントがクエリの回答を支持しているかを "Yes"/"No" で判定する。

$$
\text{score}(d_i) = \log p(\text{"Yes"} | q, d_i, a_i) - \log p(\text{"No"} | q, d_i, a_i)
$$

ここで、
- $p(\text{"Yes"})$: LLMが"Yes"トークンを生成する確率
- $p(\text{"No"})$: LLMが"No"トークンを生成する確率

スコアが正の値なら関連性が高い、負の値なら関連性が低いと判定。この**対数確率差**をスコアとして使用することで、単純なバイナリ判定よりも細粒度の関連性評価が可能になる。

**Adaptive Judge Bar (適応的閾値): τ_q**

固定閾値ではなく、各クエリのスコア分布に基づいて動的に閾値を調整する。

$$
\tau_q = \bar{s}_q - n \cdot \sigma_q
$$

ここで、
- $\bar{s}_q$: クエリ$q$に対する全ドキュメントスコアの平均
- $\sigma_q$: スコアの標準偏差
- $n$: ハイパーパラメータ（唯一の調整パラメータ）

スコアが$\tau_q$以上のドキュメントのみを保持し、残りをフィルタリングする。

$$
\mathcal{D}_{\text{filtered}} = \{ d_i \mid \text{score}(d_i) \geq \tau_q \}
$$

**なぜ適応的か:**
- スコアが全体的に高い場合（関連ドキュメントが多い）: $\bar{s}_q$が高くなり、閾値も上昇→より厳格にフィルタ
- スコアが全体的に低い場合（関連ドキュメントが少ない）: $\bar{s}_q$が低くなり、閾値も低下→緩やかにフィルタ

**Agent-3 (Final-Predictor): 最終回答生成**

フィルタリング後のドキュメントをスコア降順に並べ、Final-Predictorが最終回答を生成する。

$$
\text{answer} = \text{LLM}_3(q, \text{sort}(\mathcal{D}_{\text{filtered}}, \text{score}, \text{desc}))
$$

ドキュメントの順序（降順 vs 昇順）が回答品質に影響を与えることがablation studyで確認されている。

### LlamaIndex AgentWorkflowとの対応

MAIN-RAGの3エージェント構成は、Zenn記事で紹介されているLlamaIndex v0.14のAgentWorkflowで直接実装可能である。

```python
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent

# Agent-1: Predictor
predictor_agent = FunctionAgent(
    name="Predictor",
    description="各検索ドキュメントから予備回答を生成する",
    tools=[generate_preliminary_answer],
    system_prompt="ドキュメントを参照してクエリに回答してください。",
    can_handoff_to=["Judge"],
)

# Agent-2: Judge
judge_agent = FunctionAgent(
    name="Judge",
    description="Doc-Query-Answer Tripletの関連性を評価する",
    tools=[evaluate_relevance, compute_score, apply_adaptive_threshold],
    system_prompt="ドキュメントがクエリの回答を支持しているか評価してください。",
    can_handoff_to=["FinalPredictor"],
)

# Agent-3: Final-Predictor
final_predictor = FunctionAgent(
    name="FinalPredictor",
    description="フィルタリング済みドキュメントから最終回答を生成する",
    tools=[generate_final_answer],
    system_prompt="高品質なドキュメントのみを使用して回答してください。",
    can_handoff_to=[],
)

# MAIN-RAG Workflow
workflow = AgentWorkflow(
    agents=[predictor_agent, judge_agent, final_predictor],
    root_agent="Predictor",
)

response = await workflow.run(user_msg="LlamaIndexのAgentWorkflowとは？")
```

## 実装のポイント（Implementation）

### 対数確率スコアの計算

MAIN-RAGの核心はJudgeエージェントの対数確率スコア計算である。これはHuggingFaceのtransformersライブラリで以下のように実装できる。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_relevance_score(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    query: str,
    document: str,
    answer: str,
) -> float:
    """Judge Agentの関連性スコア計算

    Args:
        model: 言語モデル
        tokenizer: トークナイザ
        query: ユーザークエリ
        document: 検索ドキュメント
        answer: Predictorの予備回答

    Returns:
        関連性スコア（正: 関連, 負: 非関連）
    """
    prompt = f"""以下のドキュメントは、クエリへの回答を支持していますか？

クエリ: {query}
ドキュメント: {document}
回答: {answer}

判定（Yes/No）:"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # 最後のトークンのlogits

    # Yes/Noのトークンidを取得
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    # 対数確率差
    log_prob_yes = torch.log_softmax(logits, dim=-1)[yes_id].item()
    log_prob_no = torch.log_softmax(logits, dim=-1)[no_id].item()

    return log_prob_yes - log_prob_no
```

### 適応的閾値の実装

```python
import numpy as np

def adaptive_judge_bar(
    scores: list[float],
    n: float = 1.0,
) -> float:
    """適応的閾値を計算

    Args:
        scores: 各ドキュメントの関連性スコアリスト
        n: 標準偏差の倍率（唯一のハイパーパラメータ）

    Returns:
        フィルタリング閾値 τ_q
    """
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    threshold = mean_score - n * std_score
    return threshold

def filter_documents(
    documents: list[dict],
    scores: list[float],
    n: float = 1.0,
) -> list[dict]:
    """適応的閾値でドキュメントをフィルタリング"""
    threshold = adaptive_judge_bar(scores, n)

    # 閾値以上のドキュメントをスコア降順でソート
    filtered = [
        (doc, score)
        for doc, score in zip(documents, scores)
        if score >= threshold
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in filtered]
```

### 主要ハイパーパラメータ

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| $n$（閾値倍率） | 1.0 | 唯一の調整パラメータ。大きいほど多くのドキュメントを保持 |
| top-k（検索数） | 10 | 初期検索のドキュメント数 |
| ドキュメント順序 | 降順 | スコアが高い順に並べる（ablation studyで確認） |

**訓練不要の利点:**
- ファインチューニング不要 → 導入コストが低い
- 任意のLLMバックエンドに適用可能（Mistral-7B, Llama3-8B, etc.）
- ハイパーパラメータが$n$の1つだけ → チューニングが容易

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $80-180 | Lambda + Bedrock + OpenSearch Serverless |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $400-1,000 | ECS Fargate + Bedrock + OpenSearch |
| **Large** | 300,000+ (10,000/日) | Container | $2,500-6,000 | EKS + SageMaker Endpoint + OpenSearch |

**Small構成の詳細** (月額$80-180):
- **Lambda**: 1GB RAM, 90秒タイムアウト ($20/月) — 3エージェント直列実行
- **Bedrock**: Claude 3.5 Haiku ($100/月) — Predictor/Judge/Final-Predictorで3回呼び出し
- **OpenSearch Serverless**: ベクトル検索 ($40/月) — top-k検索用
- **CloudWatch**: 基本監視 ($5/月)

**MAIN-RAG特有のコスト考慮**:
- 1クエリあたりLLM呼び出し回数: top-k × 2 (Predictor + Judge) + 1 (Final-Predictor)
- top-k=10の場合: 21回のLLM呼び出し/クエリ
- コスト削減: top-kを5に減らす → 11回（約50%削減）

**コスト試算の注意事項**:
- 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値
- LLM呼び出し回数がtop-kに比例するため、top-k設定がコストに直結
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

**Small構成 (Serverless): Lambda + Bedrock + OpenSearch Serverless**

```hcl
# --- OpenSearch Serverless（ベクトル検索） ---
resource "aws_opensearchserverless_collection" "rag_vectors" {
  name = "main-rag-vectors"
  type = "VECTORSEARCH"
}

resource "aws_opensearchserverless_security_policy" "encryption" {
  name = "main-rag-encryption"
  type = "encryption"
  policy = jsonencode({
    Rules = [{
      ResourceType = "collection"
      Resource      = ["collection/main-rag-vectors"]
    }]
    AWSOwnedKey = true
  })
}

# --- Lambda関数（MAIN-RAG 3エージェント実行） ---
resource "aws_lambda_function" "main_rag" {
  filename      = "main_rag.zip"
  function_name = "main-rag-handler"
  role          = aws_iam_role.main_rag_lambda.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 90
  memory_size   = 1024

  environment {
    variables = {
      BEDROCK_MODEL_ID     = "anthropic.claude-3-5-haiku-20241022-v1:0"
      OPENSEARCH_ENDPOINT  = aws_opensearchserverless_collection.rag_vectors.collection_endpoint
      TOP_K                = "10"
      ADAPTIVE_N           = "1.0"
    }
  }
}

# --- CloudWatch アラーム（エージェント呼び出し回数監視） ---
resource "aws_cloudwatch_metric_alarm" "agent_calls" {
  alarm_name          = "main-rag-llm-calls-spike"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = 3600
  statistic           = "Sum"
  threshold           = 200000  # 200秒/時間超過でアラート
  alarm_description   = "MAIN-RAG LLM呼び出し過多（コスト急増）"

  dimensions = {
    FunctionName = aws_lambda_function.main_rag.function_name
  }
}

# --- IAMロール ---
resource "aws_iam_role" "main_rag_lambda" {
  name = "main-rag-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}
```

### 運用・監視設定

**CloudWatch Logs Insights — フィルタリング効率の監視**:
```sql
fields @timestamp, query_id, docs_retrieved, docs_after_filter, filter_ratio
| stats avg(filter_ratio) as avg_filter_ratio,
        min(docs_after_filter) as min_docs_kept,
        max(docs_after_filter) as max_docs_kept
  by bin(1h)
| filter avg_filter_ratio < 0.3  -- 70%以上フィルタされている場合は検索品質に問題
```

### コスト最適化チェックリスト

- [ ] top-kを5-10に設定（LLM呼び出し回数の直接制御）
- [ ] Bedrock Prompt Caching有効化（Judge Agentのプロンプト固定部分）
- [ ] 閾値$n$を1.0に設定（論文推奨のデフォルト値）
- [ ] ドキュメントスコアのキャッシュ（同一ドキュメントの再評価回避）
- [ ] OpenSearch Serverlessの最小容量設定
- [ ] AWS Budgets月額予算設定

## 実験結果（Results）

**データセット:**
- TriviaQA-unfiltered (11,313テストクエリ)
- PopQA long-tail subset (1,399クエリ)
- ARC-Challenge
- ALCE-ASQA

**主要結果:**

| データセット | MAIN-RAG (Mistral-7B) | MAIN-RAG (Llama3-8B) | Best Baseline |
|-------------|----------------------|----------------------|---------------|
| **TriviaQA** | 71.0% | **74.1%** | 73.1% |
| **PopQA** | 58.9% | **64.0%** | 61.8% |
| **ARC-C** | 58.9% | **61.9%** | 57.6% |
| **ASQA (em)** | 35.7% | **39.2%** | 37.1% |

**Ablation Study結果:**

| 条件 | TriviaQA (Mistral) | PopQA (Llama3) |
|------|-------------------|----------------|
| **MAIN-RAG (default, 降順)** | **71.0%** | **64.0%** |
| 昇順ソート | 70.2% | 63.5% |
| フィルタなし | 69.1% | 61.2% |
| Judge Agent除去 | 68.3% | 60.5% |

**分析ポイント**:
- Llama3-8BがMistral-7Bを一貫して上回る → Judgeの品質がバックエンドLLMに依存
- PopQA（long-tail）で最大の改善（+2.2%） → ノイズが多い検索結果での効果が顕著
- ドキュメント降順ソートが約0.5-0.8%の精度寄与 → 順序が回答品質に影響
- フィルタリングなしとの差が1.9-2.8% → 適応的閾値の効果が明確

## 実運用への応用（Practical Applications）

### Zenn記事のAgentic Retrievalとの関係

MAIN-RAGの3エージェント構成は、Zenn記事のAgentic Retrieval（Stage 1-2）の実装パターンとして直接活用できる。

| MAIN-RAG | LlamaIndex v0.14 |
|----------|-----------------|
| Agent-1 (Predictor) | Retriever Agent（検索実行） |
| Agent-2 (Judge) | Grader Agent（品質評価）→ Rerankerの代替 |
| Agent-3 (Final-Predictor) | Synthesizer Agent（回答生成） |
| Adaptive Judge Bar | カスタムフィルタリングロジック |

### 導入時の注意点

1. **LLM呼び出しコスト**: 1クエリあたり$2k+1$回のLLM呼び出し（$k$=top-k）。Bedrock Haikiなど低コストモデルの使用を推奨
2. **レイテンシ**: 3エージェント直列実行のため、Naive RAGの3-5倍。Agent-1とAgent-2の並列化で改善可能
3. **バックエンドLLMの選択**: Llama3-8Bが最良の結果。SageMaker Endpointでのセルフホスティングでコスト削減

## 関連研究（Related Work）

- **Self-RAG**: LLM自体が反省トークンを生成して検索品質を自己評価する手法。訓練が必要だがMAIN-RAGは訓練不要
- **CRAG (Corrective RAG)**: 検索結果を3段階評価し、不十分な場合にWeb再検索する手法。MAIN-RAGのJudge Agentと相補的
- **Agentic RAG Survey (2501.15228)**: MAIN-RAGの位置づけはMulti-Agent RAGのGrader/Filterパターンに該当

## まとめと今後の展望

MAIN-RAGは「訓練不要・ハイパーパラメータ1つ」という実用性の高い設計で、RAGの検索ノイズ問題に対する効果的な解決策を提示した。LlamaIndex v0.14のAgentWorkflowを用いてPredictor→Judge→Final-Predictorの3エージェント構成を実装することで、既存のRAGパイプラインの精度を2-11%向上させることが期待できる。

今後の課題として、(1) Agent-1とAgent-2の並列化によるレイテンシ削減、(2) マルチホップ推論タスクへの拡張、(3) 閾値$n$の自動最適化が挙げられる。

## 参考文献

- **Conference URL**: https://aclanthology.org/2025.acl-long.131/
- **arXiv**: https://arxiv.org/abs/2501.00332
- **Related Zenn article**: https://zenn.dev/0h_n0/articles/62e946539206db
