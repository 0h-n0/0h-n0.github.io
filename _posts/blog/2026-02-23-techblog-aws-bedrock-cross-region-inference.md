---
layout: post
title: "AWS公式解説: Amazon Bedrock Cross-Region Inferenceでスロットリングを解消しスループットを向上させる"
description: "AWS公式ブログが解説するCross-Region Inferenceの仕組み・設定方法・コスト最適化を技術的に解説"
categories: [blog, tech_blog]
tags: [AWS, Bedrock, cross-region-inference, LLM, cost-optimization, latency, rag]
date: 2026-02-23 12:00:00 +0900
source_type: tech_blog
source_domain: aws.amazon.com
source_url: https://aws.amazon.com/blogs/machine-learning/getting-started-with-cross-region-inference-in-amazon-bedrock/
zenn_article: f5fa165860f5e8
zenn_url: https://zenn.dev/0h_n0/articles/f5fa165860f5e8
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

本記事は [https://aws.amazon.com/blogs/machine-learning/getting-started-with-cross-region-inference-in-amazon-bedrock/](https://aws.amazon.com/blogs/machine-learning/getting-started-with-cross-region-inference-in-amazon-bedrock/) の解説記事です。

この記事は [Zenn記事: Bedrock Intelligent Prompt Routingで社内RAGコスト最大60%削減](https://zenn.dev/0h_n0/articles/f5fa165860f5e8) の深掘りです。

AWSの公式ブログ「Getting started with cross-region inference in Amazon Bedrock」は、Amazon Bedrockのリクエストを複数のAWSリージョンに自動分散し、単一リージョンの容量制限によるスロットリングを解消する機能であるCross-Region Inferenceの仕組み、設定方法、およびコスト面での考慮事項を解説している。

## 情報源

- **種別**: 企業テックブログ（AWS Machine Learning Blog）
- **URL**: [https://aws.amazon.com/blogs/machine-learning/getting-started-with-cross-region-inference-in-amazon-bedrock/](https://aws.amazon.com/blogs/machine-learning/getting-started-with-cross-region-inference-in-amazon-bedrock/)
- **組織**: Amazon Web Services
- **発表日**: 2024年（Cross-Region Inference GA発表時）

## 技術的背景（Technical Background）

### スロットリング問題

Amazon Bedrockのファンデーションモデルには、リージョンごとにトークン/分（TPM）とリクエスト/分（RPM）のクォータが設定されている。社内RAGシステムのように業務時間帯にリクエストが集中するワークロードでは、単一リージョンのクォータを超過し、`ThrottlingException` が頻発することがある。

従来の対策としては以下が考えられていた：
- **リトライ+指数バックオフ**: レイテンシが増大する
- **Provisioned Throughput**: コストが固定化される
- **マルチリージョンの手動実装**: アプリケーション側のロジックが複雑化

Cross-Region Inferenceは、これらの問題をAWSマネージドで解決するアプローチである。

### 関連する学術研究

LLMサービングにおける地理的分散推論は、DistServe（Zhong et al., 2024）やSplitwise（Patel et al., 2024）などのシステム研究で提案されている。Cross-Region InferenceはこれらのアイデアをAWSのマネージドサービスとして商用実装したものと位置づけられる。

## 実装アーキテクチャ（Architecture）

### 2種類のCross-Region Inference Profile

AWSは2種類のCross-Region Inference Profileを提供している。

**1. Geographic Profile（地域内分散）**

特定の地理的領域内のリージョン間でリクエストを分散する。

| Profile | 対象リージョン | 用途 |
|---------|-------------|------|
| `us.*` | us-east-1, us-east-2, us-west-2 | 米国内限定 |
| `eu.*` | eu-central-1, eu-west-1, eu-west-3 | EU内限定 |
| `ap.*` | ap-northeast-1, ap-southeast-1, ap-southeast-2 | APAC内限定 |

- **コスト**: 元のリージョンの料金がそのまま適用（追加コストなし）
- **データレジデンシー**: 指定した地域内にデータが留まるため、GDPRなどの規制に対応可能

**2. Global Profile（全リージョン分散）**

全AWSリージョン横断でリクエストを分散する。

- **コスト**: Geographic Profileと比較して約10%のコスト削減（AWSの公式ドキュメントによる）
- **データレジデンシー**: データが複数地域に移動するため、規制要件が緩い場合に使用

### 有効化の方法

Cross-Region Inferenceの有効化は、モデルIDにリージョンプレフィックスを付加するだけで完了する。

```python
import boto3

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# 通常のモデル呼び出し（単一リージョン）
single_region_model = "anthropic.claude-3-5-sonnet-20241022-v2:0"

# Geographic Cross-Region Inference（US内分散）
cross_region_model = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

# Global Cross-Region Inference（全リージョン分散）
global_model = "anthropic.claude-3-5-sonnet-20241022-v2:0"  # 別途設定が必要

response = bedrock_runtime.converse(
    modelId=cross_region_model,  # "us." プレフィックスで有効化
    messages=[
        {
            "role": "user",
            "content": [{"text": "社内の有給休暇日数を教えてください"}],
        }
    ],
    inferenceConfig={"maxTokens": 1024, "temperature": 0.1},
)
```

### AWS Consoleでの設定

AWS Consoleでは、Bedrockの「Cross-region inference」ページからInference Profileを作成・管理できる。プログラムによる設定が不要な場合は、コンソールからの有効化も可能である。

## Prompt Cachingとの併用

AWSの公式ドキュメントによると、2025年以降、Prompt CachingはCross-Region Inferenceと併用可能となっている。これにより、キャッシュヒット時のコスト削減（最大90%）とスロットリング解消を同時に実現できる。

```python
# Prompt Caching + Cross-Region Inference の併用
response = bedrock_runtime.converse(
    modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    system=[
        {
            "text": long_system_prompt,
            # Prompt Cachingは自動的に適用される
            # 1,024トークン以上のプレフィックスが対象
        }
    ],
    messages=[{"role": "user", "content": [{"text": query}]}],
)
```

**注意点**: キャッシュはリージョン間で共有されない可能性がある。Cross-Region Inferenceでリクエストが異なるリージョンにルーティングされた場合、キャッシュヒット率が低下する可能性がある。AWSの公式ドキュメントでこの動作を確認することを推奨する。

## パフォーマンス最適化（Performance）

### スループットの改善

Cross-Region Inferenceの主要な効果はスループットの向上である。単一リージョンでクォータに達していたワークロードが、複数リージョンに分散されることで全体のTPM/RPMが実質的に増加する。

**定量的な効果**:
- AWSの公式ドキュメントでは、3リージョン分散時にスループットが「最大3倍に向上」する可能性が示唆されている
- ただし実際の効果はモデルの各リージョンでの利用可能容量に依存する

### レイテンシへの影響

リージョン間のネットワークレイテンシ（通常数ms〜数十ms）が追加される。ただしLLMの推論レイテンシ（数百ms〜数秒）と比較して相対的に小さいため、実用上の影響は限定的である。

**トレードオフの数式的理解**:

$$
L_{\text{total}} = L_{\text{network}} + L_{\text{inference}}
$$

ここで、
- $L_{\text{network}}$: リージョン間ネットワークレイテンシ（5-50ms）
- $L_{\text{inference}}$: LLM推論レイテンシ（500-5000ms）

$L_{\text{network}} \ll L_{\text{inference}}$ であるため、Cross-Region Inferenceによるレイテンシ増加は相対的に無視できる水準である。

## 運用での学び（Production Lessons）

### データレジデンシー要件の確認

Cross-Region Inferenceを使用する前に、以下を確認する：

1. **データ保護規制**: GDPR（EU）、個人情報保護法（日本）などの規制がデータの地理的移動を制限していないか
2. **社内ポリシー**: 社内のデータガバナンスポリシーがリージョン間データ移動を許可しているか
3. **契約要件**: 顧客との契約でデータ保管場所が指定されていないか

Geographic Profileで地域を限定すれば、多くのデータレジデンシー要件に対応可能である。

### モニタリング

Cross-Region Inference使用時のモニタリング項目：

```python
import boto3

cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")

# スロットリング率の監視
cloudwatch.put_metric_alarm(
    AlarmName="bedrock-throttling-rate",
    ComparisonOperator="GreaterThanThreshold",
    EvaluationPeriods=3,
    MetricName="ThrottledCount",
    Namespace="AWS/Bedrock",
    Period=300,
    Statistic="Sum",
    Threshold=10,
    AlarmDescription="Bedrockスロットリング発生",
)
```

### Intelligent Prompt Routingとの組み合わせ

Zenn記事で紹介されている3層戦略の文脈では、Cross-Region Inference（Layer 2）はIPR（Layer 1）の上位レイヤーとして機能する。IPRでHaikuにルーティングされたリクエストも、Cross-Region Inferenceにより複数リージョンに分散される。

```
ユーザーリクエスト
    │
    ▼
[Layer 1: IPR]
    ├─ 簡易クエリ → Haiku
    │      │
    │      ▼
    │  [Layer 2: Cross-Region]
    │      ├─ us-east-1
    │      ├─ us-east-2
    │      └─ us-west-2
    │
    └─ 複雑クエリ → Sonnet
           │
           ▼
       [Layer 2: Cross-Region]
           ├─ us-east-1
           ├─ us-east-2
           └─ us-west-2
```

## 学術研究との関連（Academic Connection）

- **DistServe**（Zhong et al., 2024）: プリフィルとデコードの分離によるLLMサービング最適化。Cross-Region Inferenceの内部実装で類似のアイデアが使われている可能性がある
- **Orca**（Microsoft, 2022）: バッチスケジューリングによるLLM推論の効率化。リージョン間のバッチ統合に関連
- **vLLM**（Kwon et al., 2023; SOSP 2023）: PagedAttentionによるKVキャッシュ管理。Cross-Region Inferenceにおけるキャッシュ戦略の基盤技術

## まとめと実践への示唆

Cross-Region Inferenceは、モデルIDにリージョンプレフィックスを付加するだけで有効化できる手軽さが大きな利点である。Geographic Profileでは追加コストなしでスロットリングを解消でき、Global Profileではさらに約10%のコスト削減が期待できるとAWSの公式ドキュメントは説明している。

社内RAGシステムでは、まずGeographic Profileで自社のデータレジデンシー要件を満たしつつスロットリングを解消し、要件が許す場合はGlobal Profileに移行してコスト削減を図るのが推奨される段階的アプローチである。

## 参考文献

- **Blog URL**: [https://aws.amazon.com/blogs/machine-learning/getting-started-with-cross-region-inference-in-amazon-bedrock/](https://aws.amazon.com/blogs/machine-learning/getting-started-with-cross-region-inference-in-amazon-bedrock/)
- **Cross-Region Inference Documentation**: [https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)
- **Global Cross-Region Inference**: [https://docs.aws.amazon.com/bedrock/latest/userguide/global-cross-region-inference.html](https://docs.aws.amazon.com/bedrock/latest/userguide/global-cross-region-inference.html)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/f5fa165860f5e8](https://zenn.dev/0h_n0/articles/f5fa165860f5e8)
