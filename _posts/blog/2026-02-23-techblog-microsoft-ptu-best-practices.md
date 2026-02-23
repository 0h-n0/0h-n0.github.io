---
layout: post
title: "Microsoft解説: Azure OpenAI PTU（Provisioned Throughput Units）ベストプラクティス — 容量計画からスピルオーバーまで"
description: "Azure OpenAI PTUの容量計画・スピルオーバー戦略・コスト最適化のベストプラクティスを公式ガイドに基づき解説"
categories: [blog, tech_blog]
tags: [Azure, OpenAI, PTU, provisioned throughput, cost optimization, infrastructure]
date: 2026-02-23 12:00:00 +0900
source_type: tech_blog
source_domain: techcommunity.microsoft.com
source_url: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/best-practice-guidance-for-ptu/4152133
zenn_article: 838465e8c756eb
zenn_url: https://zenn.dev/0h_n0/articles/838465e8c756eb
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Mastering PTU Management for LLMs and GenAI: Optimizing Azure OpenAI for Peak Performance and Cost Efficiency](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/best-practice-guidance-for-ptu/4152133)（Microsoft Azure AI Foundry Blog）の解説記事です。

## ブログ概要（Summary）

Microsoft Azure AI Foundryチームが公開したこのブログ記事は、Azure OpenAI ServiceのPTU（Provisioned Throughput Units）を本番環境で適切に管理するためのベストプラクティスを網羅的に示したガイドである。PTUの容量見積もり方法、スピルオーバー（PTU→Pay-as-you-go自動切り替え）戦略、セマンティックキャッシュによるコスト削減、バッチ処理の最適化、監視・アラートの設定まで、PTU運用の全フェーズをカバーしている。

この記事は [Zenn記事: Azure OpenAI負荷分散設計：API ManagementとPTUスピルオーバーで可用性99.9%を実現する](https://zenn.dev/0h_n0/articles/838465e8c756eb) の深掘りです。

## 情報源

- **種別**: 企業テックブログ（Microsoft Azure AI Foundry Blog）
- **URL**: [https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/best-practice-guidance-for-ptu/4152133](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/best-practice-guidance-for-ptu/4152133)
- **組織**: Microsoft Azure AI Foundry チーム
- **発表日**: 2024年（随時更新）

## 技術的背景（Technical Background）

Azure OpenAI Serviceには大きく2つの課金モデルがある。**Standard（Pay-as-you-go / PAYG）** はトークン単位の従量課金、**PTU（Provisioned Throughput Units）** は予約済みスループットの月額固定課金である。

PTUの利点は**一貫した低レイテンシ**と**予測可能なコスト**であるが、未使用容量もコストが発生するというトレードオフがある。この「使い切れなければ無駄」「足りなければスロットリング」のバランスを取るのがPTU運用の核心的課題であり、本ブログはその解決策を体系的に提示している。

学術的には、この問題はクラウドコンピューティングにおける**容量計画（Capacity Planning）** と **リソースプロビジョニング**の変種として理解できる。LLM固有の要素として、リクエスト間のトークン数分散が大きい（ショートクエリからロングドキュメント処理まで）ため、従来のRPSベースの容量計画よりも複雑な見積もりが必要になる。

## 実装アーキテクチャ（Architecture）

### PTU容量計画プロセス

ブログでは以下のステップによるPTU容量見積もりを推奨している。

**ステップ1: トラフィックパターンの分析**

```
分析項目:
├─ 時間帯別リクエスト量（ピーク/オフピーク比）
├─ プロンプトトークン数の分布（P50, P95, P99）
├─ 出力トークン数の分布
├─ リクエスト間の到着間隔
└─ 曜日・季節変動
```

**ステップ2: Azure OpenAI Capacity Calculatorでの見積もり**

ブログでは [Azure OpenAI Capacity Calculator](https://oai.azure.com/portal/calculator) を使用した見積もりを推奨している。入力パラメータとして、モデル種別、平均プロンプト長、平均出力長、同時リクエスト数を指定する。

**ステップ3: バッファ率の適用**

見積もったPTU量にバッファ率を加えて最終的なプロビジョニング量を決定する。ブログでは明示的なバッファ率は示されていないが、「ピーク時のパフォーマンス低下への許容度」に基づいて決定すべきと述べている。

### スピルオーバー構成

PTUの容量を超過した際に自動的にPAYGエンドポイントへフォールバックするスピルオーバー構成が、ブログの核心的推奨事項である。

```
[定常状態]
リクエスト → API Management → PTU (Priority 1)
                                ↓ 処理完了

[PTU容量超過時]
リクエスト → API Management → PTU (Priority 1) → 429
                  ↓ Circuit Break
                  → PAYG (Priority 2) → 処理完了

[深夜帯（低トラフィック）]
リクエスト → API Management → PTU (Priority 1)
                                ↓ 処理完了
                                （PAYGは使用されない = コスト発生なし）
```

ブログでは、この構成を「最小限のPTU過剰プロビジョニングでサービス品質を維持し、大幅なコスト削減を実現する」と評価している。

### GenAI Gatewayの負荷分散戦略

ブログでは、API ManagementのGenAI Gateway機能を通じた3つの負荷分散戦略を紹介している。

| 戦略 | 説明 | PTUとの関係 |
|------|------|------------|
| **Round-robin** | 均等配分 | PTUの利用率が下がるため非推奨 |
| **Weighted** | 重み付き配分 | PTUに高い重みを設定可能 |
| **Priority-based** | 優先度順 | PTU最優先→PAYGフォールバック（推奨） |

**Priority-based構成が推奨される理由**: PTUは使い切らなければコストが無駄になるため、全リクエストをまずPTUに送り、容量超過時のみPAYGを使用するのが最もコスト効率が高い。

### トークンレート制限ポリシー

ブログでは、消費者ごとのトークン使用量を制御する`azure-openai-token-limit`ポリシーの使用を推奨している。

```xml
<azure-openai-token-limit
    counter-key="@(context.Subscription.Id)"
    tokens-per-minute="20000"
    estimate-prompt-tokens="true"
    remaining-tokens-variable-name="remainingTokens" />
```

`estimate-prompt-tokens="true"`を有効にすると、API Management側でプロンプトのトークン数を事前推定し、PTUの容量を超過するリクエストをバックエンドに送る前にブロックする。これにより、PTUクォータの無駄消費を防止できる。

### Provisioned-Managed Utilization V2メトリクス

PTUの利用率を監視するための核心的メトリクスとして、ブログでは**Provisioned-Managed Utilization V2**を紹介している。このAzure Monitorメトリクスは1分間隔でデプロイメントの利用率を測定し、容量超過やリソース不足の早期検知を可能にする。

```
利用率 < 80%  → PTUが過剰プロビジョニング（コスト無駄）
利用率 80-95% → 適切な範囲
利用率 > 95%  → スロットリングリスク（PAYG スピルオーバー発生）
利用率 100%   → PTU容量飽和（429エラー頻発）
```

### セマンティックキャッシュ

ブログでは、API ManagementのGenAI Gatewayが提供するセマンティックキャッシュ機能を紹介している。過去のプロンプトと意味的に類似するリクエストに対してキャッシュ済み応答を返すことで、PTUへの呼び出し回数を削減する。

```xml
<inbound>
    <llm-semantic-cache-lookup
        score-threshold="0.9"
        embeddings-backend-id="embeddings-backend"
        embeddings-backend-auth="system-assigned" />
</inbound>
<outbound>
    <llm-semantic-cache-store duration="3600" />
</outbound>
```

ブログでは「セマンティックに類似するプロンプトの応答を保存・再利用することで、トークン使用量を削減し、レスポンスタイムを改善しつつコストを下げる」と評価している。

ただし、セマンティックキャッシュの利用にはAzure Managed Redis（RediSearch対応）が必要であり、`score-threshold`を低く設定しすぎると不適切なキャッシュ応答が返るリスクがある。

### バッチ処理の最適化

ブログでは、非リアルタイム処理をPTUの低利用時間帯にスケジューリングする「バッチ処理最適化」を推奨している。PTUは24時間365日固定料金であるため、深夜帯などの低トラフィック時間帯にバッチジョブを実行することで、PTUの投資対効果を最大化できる。

## パフォーマンス最適化（Performance）

### PTUの性能特性

ブログに明示的なベンチマーク数値は記載されていないが、PTUとPAYGの性能差について以下の特性が示されている。

- **PTU**: 予約済み容量のため、一貫した低レイテンシを提供。リクエストが容量内であれば、PAYGよりも安定したレスポンスタイムが期待できる。
- **PAYG**: 空き容量に依存するベストエフォート型。ピーク時にはレイテンシが増加する可能性がある。

### 監視すべきメトリクス

ブログでは以下のメトリクスの監視を推奨している。

| メトリクス | 説明 | アラート閾値 |
|-----------|------|------------|
| **Azure OpenAI Requests** | ステータスコード別のAPIコール数 | 429比率 > 5% |
| **Token Usage** | プロンプト・出力トークン消費量 | PTU容量の90%超過 |
| **Time to Response** | 最初のトークンまでのレイテンシ | P95 > 3秒 |
| **Provisioned-Managed Utilization V2** | PTU利用率 | > 95%で警告 |

### チャージバック（費用配賦）

ブログでは、`azure-openai-emit-token-metric`ポリシーによるトークンメトリクスの収集を、部門別コスト配賦の基盤として推奨している。Application Insightsへのメトリクス送信により、ユーザーID・クライアントIP・API識別子でのディメンション分割が可能になる。

## 運用での学び（Production Lessons）

### 429エラーへの2段階対応

ブログでは、429 (Too Many Requests)応答への対応として2つの戦略を示している。

1. **トラフィックリダイレクション**: 余剰需要を代替デプロイメントにルーティング（スピルオーバー）
2. **リトライロジック**: 許容可能なレイテンシのシナリオでは指数バックオフで再試行

### 段階的なワークロード増加

ブログでは、PTUデプロイメントへの急激なトラフィック投入を避け、段階的にワークロードを増加させることを推奨している。これは、PTUの内部スケジューリングが急激な負荷変動に対して最適化に時間を要する可能性があるためと推察される。

### Circuit Breakerの動的トリップ

ブログでは、Circuit Breakerの動的トリップ期間機能（`acceptRetryAfter`）を「バックエンドのRetry-Afterヘッダーの値を活用して、正確かつタイムリーなバックエンド復旧を実現する」と評価している。固定のトリップ期間ではなく、バックエンドの実際の回復時間に合わせた制御により、PTUの利用率を最大化する。

## 学術研究との関連（Academic Connection）

- **容量計画理論**: PTUの見積もりは、クラウドコンピューティングにおけるリソースプロビジョニング理論と密接に関連する。特に、M/G/1キューイングモデルでのサーバ容量計算がPTUの見積もりに応用可能である。
- **実運用ワークロード分析** (arXiv:2404.01566): 実LLMサービスのワークロードトレース分析で報告されているバースト係数（ピーク/平均比5〜15倍）は、PTUのバッファ率設定に直接適用可能な数値である。
- **FrugalGPT** (arXiv:2305.05176): 複数プロバイダーへのカスケードルーティングの理論。スピルオーバー構成の学術的背景として参照できる。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

Azure OpenAI PTUに相当するAWSサービスとして、Amazon Bedrock Provisioned Throughputがある。以下にAWSでのPTU相当構成を示す。

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 | On-Demand | $80-200 | Bedrock On-Demand |
| **Medium** | ~30,000 | Provisioned | $1,500-4,000 | Bedrock Provisioned Throughput |
| **Large** | 300,000+ | Hybrid | $5,000-15,000 | Bedrock Provisioned + On-Demand |

**Bedrock Provisioned Throughput（PTU相当）の特徴**:
- 予約済みモデルユニット（Model Units）で課金
- 1時間単位・1ヶ月単位・6ヶ月単位のコミットメント
- 6ヶ月コミットで最大40%割引

**スピルオーバー実装例（Python）**:

```python
import boto3
from botocore.exceptions import ClientError

bedrock_provisioned = boto3.client('bedrock-runtime', region_name='ap-northeast-1')
bedrock_ondemand = boto3.client('bedrock-runtime', region_name='us-east-1')

def invoke_with_spillover(
    prompt: str,
    provisioned_model_arn: str,
    ondemand_model_id: str
) -> dict:
    """PTU→On-Demandスピルオーバー付きBedrock呼び出し

    Args:
        prompt: 入力プロンプト
        provisioned_model_arn: Provisioned Throughputモデル ARN
        ondemand_model_id: On-DemandモデルID（フォールバック）

    Returns:
        Bedrock応答
    """
    try:
        # まずProvisioned Throughputで試行
        response = bedrock_provisioned.invoke_model(
            modelId=provisioned_model_arn,
            body=json.dumps({"prompt": prompt})
        )
        return json.loads(response['body'].read())
    except ClientError as e:
        if e.response['Error']['Code'] == 'ThrottlingException':
            # Provisioned容量超過時、On-Demandにフォールバック
            response = bedrock_ondemand.invoke_model(
                modelId=ondemand_model_id,
                body=json.dumps({"prompt": prompt})
            )
            return json.loads(response['body'].read())
        raise
```

**コスト試算の注意事項**:
- 上記は2026年2月時点のAWS料金に基づく概算値です
- Bedrock Provisioned Throughputの料金はモデル・コミットメント期間により変動します
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### コスト最適化チェックリスト

- [ ] Bedrock Provisioned Throughput利用率を監視（80-95%が適切）
- [ ] On-Demandへのスピルオーバー率を追跡（10%以下が目標）
- [ ] 6ヶ月コミットメントで最大40%割引適用
- [ ] 非リアルタイム処理はBatch API使用で50%削減
- [ ] Prompt Caching有効化で30-90%削減
- [ ] CloudWatch利用率メトリクスでアラート設定
- [ ] AWS Budgets月額予算設定
- [ ] Cost Anomaly Detection有効化

## まとめと実践への示唆

Microsoft Azure AI Foundryチームの本ブログは、PTU運用の全フェーズ（容量計画→プロビジョニング→スピルオーバー→監視→最適化）を体系的にカバーした実用的なガイドである。

特に重要な知見は以下の3点である：
1. **Priority-basedスピルオーバー**がPTUコスト最適化の定番パターンであること
2. **Provisioned-Managed Utilization V2**メトリクスがPTU運用の核心的KPIであること
3. **セマンティックキャッシュ**と**バッチ処理最適化**により、PTU利用率をさらに向上できること

Zenn記事で解説したPTU+PAYGのスピルオーバー構成は、本ブログで推奨されているベストプラクティスに基づいている。PTUの容量計画やCircuit Breakerパラメータのチューニングにおいて、本ブログの指針が具体的な判断基準を提供する。

## 参考文献

- **Blog URL**: [https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/best-practice-guidance-for-ptu/4152133](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/best-practice-guidance-for-ptu/4152133)
- **Related**: [Azure OpenAI Capacity Calculator](https://oai.azure.com/portal/calculator)
- **Related**: [Provisioned throughput unit costs and billing](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/provisioned-throughput-onboarding)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/838465e8c756eb](https://zenn.dev/0h_n0/articles/838465e8c756eb)

---

:::message
この記事はAI（Claude Code）により自動生成されました。内容の正確性については元のMicrosoft公式ブログと照合していますが、最新情報は公式ドキュメントをご確認ください。
:::
