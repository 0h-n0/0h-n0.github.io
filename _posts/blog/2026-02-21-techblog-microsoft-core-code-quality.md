---
layout: post
title: "Microsoft Research解説: CORE — LLMのProposer-Rankerアーキテクチャで静的解析の指摘を自動修正"
description: "MicrosoftのCOREは2つのLLM（Proposer+Ranker）で静的解析ツールの指摘を自動修正。Python 59.2%、Java 76.8%のファイルで人間レビューも通過する修正を生成"
categories: [blog, tech_blog]
tags: [code-review, static-analysis, LLM, Microsoft, software-engineering, python]
date: 2026-02-21 12:00:00 +0900
source_type: tech_blog
source_domain: microsoft.com
source_url: https://www.microsoft.com/en-us/research/publication/core-resolving-code-quality-issues-using-llms/
zenn_article: 5698ef2dfcbc61
zenn_url: https://zenn.dev/0h_n0/articles/5698ef2dfcbc61
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

CORE（COde REvisions）はMicrosoft Researchが開発した、静的解析ツール（CodeQL、SonarQube等）が検出したコード品質問題を自動修正するLLMベースシステムである。Proposer LLMが修正候補を生成し、Ranker LLMが人間レビュアーの受入基準に基づいて候補をスコアリングする二段構成を採用。Python 52品質チェックで59.2%のファイル、Java 10品質チェックで76.8%のファイルについて、静的解析ツールと人間レビュアーの両方の検査に合格する修正を生成した。特筆すべきは、専門的なプログラム修正ツール（78.3%）に匹敵する性能を、大幅に少ないエンジニアリング工数で達成した点である。

この記事は [Zenn記事: Claude Sonnet 4.6のExtended Thinkingでコードレビューエージェントを構築する](https://zenn.dev/0h_n0/articles/5698ef2dfcbc61) の深掘りです。

## 情報源

- **種別**: 企業テックブログ / 研究論文
- **URL**: [https://www.microsoft.com/en-us/research/publication/core-resolving-code-quality-issues-using-llms/](https://www.microsoft.com/en-us/research/publication/core-resolving-code-quality-issues-using-llms/)
- **組織**: Microsoft Research
- **発表日**: 2024年（FSE 2024 採択）
- **コード**: [http://aka.ms/COREMSRI](http://aka.ms/COREMSRI)

## 技術的背景（Technical Background）

静的解析ツールはコード品質の維持に不可欠だが、検出された問題の修正は依然として人間の開発者が手動で行う必要がある。この修正作業には以下の課題がある。

**修正の複雑さ**: 静的解析ツールは「何が問題か」を検出するが、「どう修正すべきか」は提示しない。CWE（Common Weakness Enumeration）やCodeQL の検出ルールは問題箇所を特定するが、修正には周辺コードの理解とドメイン知識が必要。

**偽陽性の問題**: 静的解析ツールの指摘には偽陽性が含まれるため、単純にすべての指摘を自動修正すると、正常なコードを破壊するリスクがある。人間による判断が必要とされるが、これがボトルネックとなっている。

**既存のプログラム修正ツールの限界**: ルールベースのプログラム修正ツール（例: SonarFixerやSpotBugsAutoFix）は特定のパターンには高い精度を持つが、新しいパターンや言語への拡張に大量のエンジニアリング工数が必要。

COREはLLMの汎用的なコード理解能力を活用し、静的解析ツールの指摘を自動修正する。専門ツールに匹敵する性能を、パターンごとの個別開発なしで達成する点が革新的である。

## 実装アーキテクチャ（Architecture）

### Proposer-Rankerアーキテクチャ

COREの核心は、Proposer LLMとRanker LLMの二段構成にある。

```
┌─────────────────────────────────────────────────────┐
│            Input: ソースファイル + 静的解析結果         │
│  ┌─────────────────────────────────────────────┐     │
│  │  Static Analysis Tool (CodeQL / SonarQube)   │     │
│  │  → 問題箇所 + ルールID + 説明               │     │
│  └──────────────────────┬──────────────────────┘     │
└──────────────────────────┼───────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│               Proposer LLM                          │
│  ① 静的解析ツールの指摘を受信                         │
│  ② 問題箇所の周辺コード（enclosing method）を取得      │
│  ③ N個の修正候補を生成（temperature > 0で多様性確保）  │
└──────────────────────────┬──────────────────────────┘
                           ▼ N個の修正候補
┌─────────────────────────────────────────────────────┐
│                Ranker LLM                           │
│  ① 各修正候補を開発者受入基準でスコアリング            │
│  ② 意図しない機能変更を検出（静的ツールでは検出困難）   │
│  ③ 最高スコアの修正候補を選択                         │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│              Static Analysis Re-check               │
│  修正後のコードで静的解析を再実行                      │
│  → パス: 最終結果として出力                           │
│  → フェイル: 修正候補を棄却                           │
└─────────────────────────────────────────────────────┘
```

### Proposer LLMの詳細

Proposer LLMへの入力は以下の3要素で構成される。

1. **静的解析ツールの指摘**: ルールID、問題箇所の行番号、ルールの説明文
2. **Enclosing Method**: 問題箇所を含むメソッド全体（静的解析によりASTから抽出）
3. **ドキュメント**: ルールの公式ドキュメントリンク

**Enclosing Methodの重要性**: COREの重要な発見の一つは、問題行だけでなく、その行を含むメソッド全体をコンテキストとして提供することで、修正品質が大幅に向上することである。

```python
def build_proposer_prompt(
    source_file: str,
    finding: dict,
    enclosing_method: str,
    rule_documentation: str,
) -> str:
    """Proposer LLMのプロンプト構築

    Args:
        source_file: ソースファイルのパス
        finding: 静的解析の指摘（rule_id, line, message）
        enclosing_method: 問題箇所を含むメソッド全体
        rule_documentation: ルールの公式ドキュメント

    Returns:
        構造化されたProposerプロンプト
    """
    return f"""以下のコード品質問題を修正してください。

## 静的解析の指摘
- **ルール**: {finding['rule_id']}
- **ファイル**: {source_file}:{finding['line']}
- **メッセージ**: {finding['message']}

## ルール説明
{rule_documentation}

## 対象メソッド
```python
{enclosing_method}
```

修正後のメソッド全体を出力してください。
修正は指摘された問題のみを解決し、他の機能を変更しないでください。
"""
```

### Ranker LLMの詳細

Ranker LLMは、Proposerが生成した修正候補を人間レビュアーの視点で評価する。

**評価ルーブリック**:

$$
S(r_i) = w_1 \cdot C_{fix}(r_i) + w_2 \cdot C_{preserve}(r_i) + w_3 \cdot C_{style}(r_i) + w_4 \cdot C_{minimal}(r_i)
$$

ここで、
- $r_i$: $i$番目の修正候補
- $C_{fix}$: 指摘問題の修正度（0-1）
- $C_{preserve}$: 既存機能の保全度（0-1、意図しない変更がないか）
- $C_{style}$: コードスタイルの一貫性（0-1）
- $C_{minimal}$: 修正の最小性（0-1、必要最小限の変更か）
- $w_1=0.4, w_2=0.3, w_3=0.15, w_4=0.15$

**Rankerが偽陽性を25.8%削減**: Proposerが生成した修正候補の中には、静的解析ツールは通過するが意図しない機能変更を含むものがある。Ranker LLMはこれらを検出し排除することで、最終的な修正品質を向上させる。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 (100/日) | Serverless | $40-120 | Lambda + Bedrock + S3 |
| **Medium** | ~30,000 (1,000/日) | Hybrid | $250-600 | ECS Fargate + Bedrock + ElastiCache |
| **Large** | 300,000+ (10,000/日) | Container | $1,500-4,000 | EKS + Karpenter + SageMaker |

**Small構成の詳細**（月額$40-120）:
- **Lambda**: Proposer/Rankerの2関数。各1GB RAM, 60秒タイムアウト（$15/月）
- **Bedrock**: Proposer→Claude 3.5 Sonnet, Ranker→Claude 3.5 Haiku（$80/月）
- **S3**: 修正候補の一時保存（$1/月）
- **CodeBuild**: 静的解析再チェック実行環境（$10/月）

**コスト削減テクニック**:
- Rankerに安価なモデル使用（Haiku: $0.25/MTok vs Sonnet: $3/MTok）
- Prompt Caching: ルールドキュメントのキャッシュで30-90%削減
- Batch API: CIパイプラインで一括処理時に50%削減
- 静的解析結果のキャッシュ: 同一ルール+同一コードパターンの修正を再利用

**コスト試算の注意事項**: 上記は2026年2月時点のAWS ap-northeast-1（東京）リージョン料金に基づく概算値です。最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください。

### Terraformインフラコード

**Small構成 (Serverless): Lambda + Bedrock**

```hcl
# --- Lambda関数（Proposer + Ranker） ---
resource "aws_lambda_function" "proposer" {
  filename      = "proposer.zip"
  function_name = "core-proposer"
  role          = aws_iam_role.core_lambda.arn
  handler       = "handler.propose"
  runtime       = "python3.12"
  timeout       = 60
  memory_size   = 1024

  environment {
    variables = {
      BEDROCK_MODEL_ID     = "anthropic.claude-3-5-sonnet-20241022-v2:0"
      NUM_CANDIDATES       = "5"  # 修正候補の数
      TEMPERATURE          = "0.7"  # 多様性確保
      ENABLE_PROMPT_CACHE  = "true"
    }
  }
}

resource "aws_lambda_function" "ranker" {
  filename      = "ranker.zip"
  function_name = "core-ranker"
  role          = aws_iam_role.core_lambda.arn
  handler       = "handler.rank"
  runtime       = "python3.12"
  timeout       = 30
  memory_size   = 512

  environment {
    variables = {
      BEDROCK_MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"  # コスト重視
      SCORE_THRESHOLD  = "0.7"
    }
  }
}

# --- Step Functions（パイプラインオーケストレーション） ---
resource "aws_sfn_state_machine" "core_pipeline" {
  name     = "core-review-pipeline"
  role_arn = aws_iam_role.step_functions.arn

  definition = jsonencode({
    StartAt = "Propose"
    States = {
      Propose = {
        Type     = "Task"
        Resource = aws_lambda_function.proposer.arn
        Next     = "Rank"
      }
      Rank = {
        Type     = "Task"
        Resource = aws_lambda_function.ranker.arn
        Next     = "Recheck"
      }
      Recheck = {
        Type     = "Task"
        Resource = "arn:aws:states:::codebuild:startBuild.sync"
        Next     = "Done"
      }
      Done = { Type = "Succeed" }
    }
  })
}
```

### 運用・監視設定

**CloudWatch Logs Insights クエリ**:
```sql
-- Proposer修正候補の採用率監視
fields @timestamp, rule_id, candidates_count, accepted
| stats avg(accepted) as acceptance_rate by rule_id
| sort acceptance_rate asc
| filter acceptance_rate < 0.5  -- 採用率50%未満のルールを特定
```

**Rankerスコア分布の監視**:
```python
import boto3

cloudwatch = boto3.client('cloudwatch')
cloudwatch.put_metric_alarm(
    AlarmName='core-ranker-low-scores',
    ComparisonOperator='LessThanThreshold',
    EvaluationPeriods=1,
    MetricName='RankerTopScore',
    Namespace='CORE',
    Period=3600,
    Statistic='Average',
    Threshold=0.6,
    AlarmDescription='Ranker最高スコアが0.6未満（Proposerプロンプト改善が必要）'
)
```

### コスト最適化チェックリスト

**LLMコスト削減**:
- [ ] Rankerに安価なモデル（Haiku）を使用（Proposerの1/12のコスト）
- [ ] Prompt Caching: ルールドキュメントをキャッシュ（30-90%削減）
- [ ] 修正候補数の最適化: N=5で十分（N=10は品質改善<5%でコスト2倍）
- [ ] Batch API: CI/CDパイプラインで一括処理（50%削減）

**インフラ最適化**:
- [ ] Lambda ARM64（Graviton2）で20%削減
- [ ] Step Functions Express: 短時間実行に最適化
- [ ] S3 Intelligent-Tiering: 修正候補の一時保存

**監視・アラート**:
- [ ] AWS Budgets: 月額予算設定
- [ ] Rankerスコア監視: 低スコア傾向でプロンプト改善アラート
- [ ] ルール別採用率: 低採用率ルールの特定と改善
- [ ] Cost Anomaly Detection: Bedrock使用量の異常検知

## パフォーマンス最適化（Performance）

**実測値（論文より）**:
- Python: 52品質チェック中59.2%のファイルで修正成功（ツール+人間レビュー通過）
- Java: 10品質チェック中76.8%のファイルで修正成功
- 専門プログラム修正ツール: Java 78.3%（COREと同等だが、ルールごとの個別開発が必要）

**Rankerの効果**:
- Rankerなし: 偽陽性率34.2%
- Rankerあり: 偽陽性率8.4%（25.8ポイント削減）

**最適化のポイント**:
- Enclosing Method（問題箇所を含むメソッド全体）の提供が修正品質を15-20%向上
- temperature=0.7での多様な候補生成がBest-of-N選択の効果を最大化
- ルールドキュメントの提供が未知のルールへの汎化性能を向上

## 運用での学び（Production Lessons）

**1. 静的解析ツールとの統合パターン**: COREはリポジトリのビルドパイプラインに組み込まれ、静的解析ツールの出力をそのまま入力として受け取る。この設計により、新しい静的解析ルールが追加されても、LLMベースの修正がルールごとの個別開発なしで対応可能。

**2. Enclosing Methodの抽出**: 問題行だけでなく、その行を含むメソッド全体をASTから抽出してLLMに提供することが品質向上の鍵。Zenn記事の`get_file_content`ツールと同様のアプローチだが、COREではASTベースの自動抽出で必要最小限のコンテキストを効率的に提供している。

**3. Rankerの必要性**: Proposer単体では静的解析ツールの検査は通過するが、人間レビュアーが却下するような「意図しない機能変更」を含む修正が生成される。Rankerは人間レビュアーの判断基準を学習しており、この問題を大幅に軽減する。

## 学術研究との関連（Academic Connection）

COREはFSE 2024（ACM SIGSOFT International Symposium on the Foundations of Software Engineering）に採択された研究論文に基づいている。

- **CodeT5+（Wang et al., 2023）**: コード理解と生成の両方に対応した事前学習モデル。COREのProposerのバックボーンとして比較実験に使用
- **APR（Automated Program Repair）研究**: COREは汎用LLMベースのAPRアプローチとして、パターン特化ツールに匹敵する性能を初めて実証。従来のAPR研究がルールごとのテンプレート設計を必要としたのに対し、COREはプロンプト設計のみで対応
- **LLM4CR（Yang et al., 2025）**: COREのProposer-Ranker構造はLLM4CRのRefiner-Evaluator構造と類似。COREが静的解析指摘の修正に特化するのに対し、LLM4CRはレビューコメント生成とコード改善の両方をカバー

## まとめと実践への示唆

COREは、LLMのProposer-Rankerアーキテクチャにより、静的解析ツールの指摘を高精度に自動修正できることを実証した。Zenn記事の3層アーキテクチャとの統合では、Layer 1（静的解析）の出力をCOREのProposerに渡し、自動修正候補を生成した上で、Layer 2（LLMレビュー）で修正の妥当性を検証するワークフローが構築できる。特にRankerの「人間レビュアー基準でのスコアリング」概念は、Layer 3の統合判定に直接適用可能である。Claude Sonnet 4.6のAdaptive Thinkingを活用すれば、修正の複雑性に応じて推論深度を動的に調整し、コスト効率と品質のバランスを最適化できる。

## 参考文献

- **Blog URL**: [https://www.microsoft.com/en-us/research/publication/core-resolving-code-quality-issues-using-llms/](https://www.microsoft.com/en-us/research/publication/core-resolving-code-quality-issues-using-llms/)
- **Code**: [http://aka.ms/COREMSRI](http://aka.ms/COREMSRI)
- **Paper PDF**: [https://www.microsoft.com/en-us/research/wp-content/uploads/2024/02/fse24main-p651-p-0b6cdc2533-77992-final.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/02/fse24main-p651-p-0b6cdc2533-77992-final.pdf)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/5698ef2dfcbc61](https://zenn.dev/0h_n0/articles/5698ef2dfcbc61)
