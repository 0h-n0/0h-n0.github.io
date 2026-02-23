---
layout: post
title: "AWS公式解説: Amazon Bedrockのコスト最適化戦略 — 7つの手法で推論コストを最大90%削減"
description: "AWS公式ブログが提示するBedrock推論コスト最適化の7戦略を技術的に解説"
categories: [blog, tech_blog]
tags: [AWS, Bedrock, cost-optimization, LLM, prompt-routing, prompt-caching, batch-inference, rag]
date: 2026-02-23 11:00:00 +0900
source_type: tech_blog
source_domain: aws.amazon.com
source_url: https://aws.amazon.com/blogs/machine-learning/effective-cost-optimization-strategies-for-amazon-bedrock/
zenn_article: f5fa165860f5e8
zenn_url: https://zenn.dev/0h_n0/articles/f5fa165860f5e8
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

本記事は [https://aws.amazon.com/blogs/machine-learning/effective-cost-optimization-strategies-for-amazon-bedrock/](https://aws.amazon.com/blogs/machine-learning/effective-cost-optimization-strategies-for-amazon-bedrock/) の解説記事です。

この記事は [Zenn記事: Bedrock Intelligent Prompt Routingで社内RAGコスト最大60%削減](https://zenn.dev/0h_n0/articles/f5fa165860f5e8) の深掘りです。

AWSの公式ブログ「Effective cost optimization strategies for Amazon Bedrock」は、Amazon Bedrockの推論コストを削減するための包括的な戦略を解説している。Intelligent Prompt Routing、Prompt Caching、Batch Inference、Cross-Region Inference、Model Distillation、Provisioned Throughput、On-Demand料金モデルの7つの手法を、ユースケース別に整理している。

## 情報源

- **種別**: 企業テックブログ（AWS Machine Learning Blog）
- **URL**: [https://aws.amazon.com/blogs/machine-learning/effective-cost-optimization-strategies-for-amazon-bedrock/](https://aws.amazon.com/blogs/machine-learning/effective-cost-optimization-strategies-for-amazon-bedrock/)
- **組織**: Amazon Web Services
- **発表日**: 2025年（GA発表に伴う更新版）

## 技術的背景（Technical Background）

LLMの推論コストは、入力トークン数と出力トークン数に比例する従量課金モデルが一般的である。Amazon Bedrockでは、2026年2月時点でClaude 3.5 Sonnet v2の入力が$3/MTok、出力が$15/MTokであり、大規模RAGシステムでは月額数千ドル規模のコストが発生しうる。

このコスト課題に対して、AWSは複数の最適化レイヤーを提供している。Zenn記事で紹介されている3層戦略（IPR + Cross-Region + Distillation）は、このブログで解説されている戦略のサブセットに該当する。

## 7つのコスト最適化戦略

### 戦略1: Intelligent Prompt Routing（IPR）

**概要**: 同一モデルファミリー内で、クエリの複雑さに応じて高性能モデル（例: Claude 3.5 Sonnet）と軽量モデル（例: Claude 3.5 Haiku）を自動切替する。

**AWS公式の報告値**:
- Anthropicファミリーで平均63.6%のコスト削減（AWSの内部ベンチマーク）
- 87%のプロンプトがHaikuにルーティング
- ルーターのP90レイテンシオーバーヘッドは約85ms

**適用場面**: 同一ファミリー内で精度-コストのトレードオフが明確なワークロード

```python
import boto3

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# IPRはmodelIdにルーターARNを指定するだけで有効化
response = bedrock_runtime.converse(
    modelId="arn:aws:bedrock:us-east-1::default-prompt-router/anthropic.claude",
    messages=[{"role": "user", "content": [{"text": query}]}],
    inferenceConfig={"maxTokens": 1024, "temperature": 0.1},
)

# ルーティング結果を取得
routed_model = response.get("trace", {}).get("promptRouter", {}).get("invokedModelId")
```

### 戦略2: Prompt Caching

**概要**: 繰り返し使用されるプロンプトプレフィックス（システムプロンプト、コンテキスト文書等）をキャッシュし、再計算を省略する。

**AWSの公式ドキュメントによる報告値**:
- 最大90%のコスト削減（キャッシュヒット時）
- 最大85%のレイテンシ削減

**適用条件**:
- システムプロンプトが長い（1,024トークン以上が目安）
- 同一プレフィックスが繰り返し使用される
- RAGのコンテキスト部分が一定期間固定される場合

**キャッシュの仕組み**: キャッシュされたトークンは通常の入力トークン料金より低いレートで課金される。キャッシュの有効期間（TTL）は5分間で、その間に同一プレフィックスのリクエストが来ればキャッシュヒットとなる。

### 戦略3: Batch Inference

**概要**: 非リアルタイム処理（レポート生成、データ分類、大量文書要約等）をバッチジョブとして一括実行する。

**AWSの公式ドキュメントによる報告値**:
- On-Demand価格と比較して50%のコスト削減

**適用場面**:
- リアルタイム性が不要なバルク処理
- 大量のドキュメント分類・要約
- 定期的なレポート生成（日次・週次）

```python
bedrock = boto3.client("bedrock", region_name="us-east-1")

# バッチ推論ジョブの作成
response = bedrock.create_model_invocation_job(
    jobName="daily-report-generation",
    modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
    roleArn="arn:aws:iam::123456789012:role/BedrockBatchRole",
    inputDataConfig={
        "s3InputDataConfig": {
            "s3Uri": "s3://my-bucket/batch-input/",
            "s3InputFormat": "JSONL",
        }
    },
    outputDataConfig={
        "s3OutputDataConfig": {
            "s3Uri": "s3://my-bucket/batch-output/"
        }
    },
)
```

### 戦略4: Cross-Region Inference

**概要**: リクエストを複数リージョンに自動分散し、スロットリングを軽減する。追加のルーティングコストは不要。

**AWSの公式ドキュメントによる報告値**:
- Geographic Profile: 追加コストなし（同一リージョン料金を適用）
- Global Profile: 約10%のコスト削減

**適用場面**: ピーク時のスロットリングが問題となるワークロード

```python
# Cross-Region Inferenceは "us." プレフィックスで有効化
cross_region_model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

response = bedrock_runtime.converse(
    modelId=cross_region_model_id,
    messages=[{"role": "user", "content": [{"text": query}]}],
)
```

### 戦略5: Model Distillation

**概要**: 高性能な教師モデル（例: Sonnet）の知識を軽量な生徒モデル（例: Haiku）に蒸留し、ドメイン特化の高精度モデルを作成する。

**AWSの公式ドキュメントによる報告値**:
- 最大75%のコスト削減
- 最大500%の推論速度向上
- 精度低下は2%未満（RAGユースケース）

**適用条件**:
- ドメイン特化のワークロードで、十分な学習データ（5,000件以上の高品質回答ログ）がある場合
- Layer 1（IPR）を導入し効果測定後に検討すべき

### 戦略6: Provisioned Throughput

**概要**: 事前に推論キャパシティを確保し、安定したスループットを保証する。コミットメント期間に応じて割引が適用される。

**適用場面**: トラフィック量が予測可能で、安定した応答時間が必要なワークロード

### 戦略7: On-Demand料金モデルの最適活用

**概要**: 使った分だけ支払うデフォルトの料金モデル。トラフィック量が少ない場合や変動が大きい場合に最適。

**コスト見積もり**:

| モデル | 入力 ($/1M tokens) | 出力 ($/1M tokens) | 用途 |
|--------|-------------------|-------------------|------|
| Claude 3.5 Sonnet v2 | $3.00 | $15.00 | 複雑な推論 |
| Claude 3.5 Haiku | $0.80 | $4.00 | 簡易タスク |
| Claude Haiku 4.5 | — | — | 次世代軽量モデル |

*上記はAWS公式料金ページ（2026年2月参照）に基づく概算値です。最新料金は[AWS料金計算ツール](https://calculator.aws/)で確認してください。*

## 戦略の組み合わせパターン

AWSブログでは、これらの戦略を単独ではなく組み合わせて使うことを推奨している。Zenn記事の3層戦略は、以下の組み合わせに対応する。

```
┌──────────────────────────────────────────────────┐
│  Layer 3: Model Distillation (最大75%削減)        │
├──────────────────────────────────────────────────┤
│  Layer 2: Cross-Region Inference (スループット向上) │
├──────────────────────────────────────────────────┤
│  Layer 1: IPR (最大63.6%削減)                     │
└──────────────────────────────────────────────────┘
  + Prompt Caching (最大90%削減) を全レイヤーに適用
  + Batch Inference (50%削減) を非リアルタイム処理に適用
```

**組み合わせの制約**:
- IPRとPrompt Cachingは併用可能
- Cross-Region InferenceとPrompt Cachingは併用可能（2025年以降サポート）
- Batch InferenceはIPRと併用不可（バッチは単一モデル指定）
- Provisioned ThroughputはIPRと併用不可

## パフォーマンス最適化（Performance）

### コスト削減効果の積算

各戦略のコスト削減効果は乗算的に効くため、組み合わせの効果は大きい。

**計算例**: 月額$10,000のBedrock利用料に対して：

| 適用戦略 | 削減率 | 適用後コスト |
|---------|-------|------------|
| ベースライン | 0% | $10,000 |
| + IPR (63.6%) | 63.6% | $3,640 |
| + Prompt Caching (50%) | 81.8% | $1,820 |
| + Batch (該当分30%に50%適用) | 84.5% | $1,547 |

*削減率は概算であり、実際のワークロードに依存します。IPRの63.6%はAWSの内部ベンチマークによる報告値です。*

### レイテンシへの影響

- IPR: +約85ms（ルーティング判定オーバーヘッド）
- Prompt Caching: -85%（キャッシュヒット時のTTFT削減）
- Cross-Region: ±変動（リージョン間通信のオーバーヘッドと負荷分散の効果が相殺）

## 運用での学び（Production Lessons）

### 段階的な導入推奨

AWSブログでは、以下の順序での段階的導入を推奨している：

1. **Phase 1**: On-Demand + Prompt Cachingでベースラインを確立
2. **Phase 2**: IPRを導入し、コスト削減効果を計測（1-2週間）
3. **Phase 3**: Cross-Region Inferenceでスロットリング対策
4. **Phase 4**: 十分なログが蓄積されたらModel Distillationを検討
5. **Phase 5**: トラフィックが安定したらProvisioned Throughputを評価

### モニタリング項目

- **ルーティング比率**: Sonnet/Haikuの振り分け比率（CloudWatchカスタムメトリクス）
- **キャッシュヒット率**: Prompt Cachingのヒット率（5分TTL内のリクエスト集中度に依存）
- **スロットリング率**: ThrottlingException の発生頻度
- **品質スコア**: ルーティング後の応答品質（ドメイン固有の評価指標で計測）

## 学術研究との関連（Academic Connection）

本ブログで紹介されている戦略は、以下の学術研究と関連する：

- **IPR**: RouteLLM（Ong et al., 2024）の商用実装。閾値ベースのルーティングの最適性はDekoninck et al.（2024）により証明されている
- **Prompt Caching**: RAGCache（Jin et al., 2024; arXiv 2401.02038）のKVキャッシュ再利用アイデアと関連
- **Model Distillation**: MiniLLM（Gu et al., 2024; ICLR 2024）の逆KLDによるオンポリシー蒸留と同様のアプローチ

## まとめと実践への示唆

AWSブログは、Bedrockのコスト最適化を7つの戦略として体系的に整理している。Zenn記事の3層戦略（IPR + Cross-Region + Distillation）に加えて、Prompt CachingとBatch Inferenceを組み合わせることで、理論的にはさらなるコスト削減が可能であるとAWSは説明している。

重要なのは、すべての戦略を一度に導入するのではなく、Phase 1からPhase 5へと段階的に進めることである。各フェーズでコスト削減効果を計測し、次のフェーズの導入判断を行うデータドリブンなアプローチが推奨されている。

## 参考文献

- **Blog URL**: [https://aws.amazon.com/blogs/machine-learning/effective-cost-optimization-strategies-for-amazon-bedrock/](https://aws.amazon.com/blogs/machine-learning/effective-cost-optimization-strategies-for-amazon-bedrock/)
- **Amazon Bedrock Cost Optimization**: [https://aws.amazon.com/bedrock/cost-optimization/](https://aws.amazon.com/bedrock/cost-optimization/)
- **Bedrock Pricing**: [https://aws.amazon.com/bedrock/pricing/](https://aws.amazon.com/bedrock/pricing/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/f5fa165860f5e8](https://zenn.dev/0h_n0/articles/f5fa165860f5e8)
