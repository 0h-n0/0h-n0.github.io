---
layout: post
title: "調査レポート解説: LangChain State of AI Agents 2025 — 1,340名調査に見るエージェント本番運用の実態"
description: "57.3%が本番運用中、89%がオブザーバビリティ導入済み。1,340名の開発者・ビジネスリーダー調査からエージェント運用の課題と評価戦略を読み解く"
categories: [blog, techblog, survey]
tags: [LLM-agent, production, observability, evaluation, LangChain]
date: 2026-03-21 09:00:00 +0900
source_type: techblog
source_url: https://www.langchain.com/state-of-agent-engineering
zenn_article: 61d5ada98ddbb6
zenn_url: https://zenn.dev/0h_n0/articles/61d5ada98ddbb6
target_audience: "修士学生レベル"
---

## レポート概要

**State of AI Agents**（LangChain, 2025年12月公開）は、1,340名のエンジニア・プロダクトマネージャー・経営層を対象にした、AIエージェントの本番運用実態に関する大規模調査レポートである。調査期間は2025年11月18日〜12月2日。テクノロジー業界（63%）を中心に、金融（10%）、ヘルスケア（6%）、教育（4%）など多岐にわたる産業から回答を得ている。

主要な発見として、回答者の**57.3%がエージェントを本番運用中**であり、さらに30.4%が具体的なデプロイ計画を持って開発中であることが報告された。前年調査の51%から6.3ポイント上昇し、エージェント技術が実験段階から本番運用段階へ移行していることを示す。

- **レポート**: [State of AI Agents](https://www.langchain.com/state-of-agent-engineering)

## 技術的背景

### エージェント技術の成熟度

LLMエージェントは2023年のAutoGPT登場以降急速に発展したが、本番環境での安定運用には多くの課題が残る。本調査は、プロンプティングベースのエージェントから、ReAct・CoT・Reflexionなどの構造化されたエージェントパターンへの移行期における実態を捉えている。

企業規模による差異が顕著であり、1万人以上の大企業では67%が本番運用中（24%が開発中）であるのに対し、100人未満の組織では50%が本番運用中（36%が開発中）である。大企業は組織的なガバナンスと既存インフラの存在により、パイロットから本番への移行が速い傾向にある。

### ユースケース分布

全体でのトップユースケースは以下の通りである。

| 順位 | ユースケース | 割合 |
|------|------------|------|
| 1 | カスタマーサービス | 26.5% |
| 2 | リサーチ・データ分析 | 24.4% |
| 3 | 社内ワークフロー自動化 | 18.0% |

大企業（1万人以上）では社内生産性向上（26.8%）がトップに立ち、カスタマーサービス（24.7%）、リサーチ（22.2%）が続く。これは大企業が社内ツールの最適化に注力している段階であることを示唆する。

## オブザーバビリティと評価の実態

### オブザーバビリティ: 89%が導入済み

本調査の注目すべき発見は、回答者の**89%が何らかのオブザーバビリティ**を実装済みであり、62%が詳細なトレーシングを導入していることである。本番運用中の組織に限定すると、94%がオブザーバビリティを導入し、71.5%がフルトレーシングを実施している。

この高い導入率は、エージェントの非決定的な動作特性から、デバッグと品質管理にトレースが不可欠であることを反映している。LangSmithのようなトレーシングプラットフォームが普及したことも要因の一つである。

### 評価（Evals）: オブザーバビリティとのギャップ

一方で、体系的な評価の導入はオブザーバビリティほど進んでいない。

| 評価手法 | 導入率 |
|---------|--------|
| オフライン評価（テストスイート） | 52.4% |
| オンライン評価（本番モニタリング） | 37.3% |
| 評価未実施 | 29.5% |

本番運用中の組織に限定すると、オンライン評価実施率は44.8%に上昇し、未評価率は22.8%に低下する。オフライン・オンライン両方を実施している組織は約25%にとどまる。

評価方法論としては、**人間レビュー（59.8%）**と**LLM-as-Judge（53.3%）**が上位を占める。LLM-as-Judgeの急速な普及は、人間評価のスケーラビリティの限界を補完する手段として定着しつつあることを示す。

## 本番運用の障壁

### 品質が最大の課題

本番運用における最大の障壁は**品質問題（32%）**であり、レイテンシ（20%）を大きく上回る。大企業（2,000人以上）では**セキュリティ（24.9%）**がレイテンシを上回り、2番目の障壁として浮上する。

ここでの「品質問題」にはハルシネーション、応答の一貫性欠如、エッジケースでの予期しない動作が含まれる。ReAct+CoTエージェントのようなマルチステップ推論パイプラインでは、各ステップのエラーが累積するため、品質管理は特に困難になる。

### モデル選択と微調整

- 回答者の**66.7%以上がOpenAI GPTモデル**を使用
- **75%以上が複数モデル**を併用
- **約33%がオープンソース/自社ホストモデル**を運用
- **43%がモデルの微調整**を実施（57%は未実施）

複数モデルの併用率の高さは、用途やコストに応じたモデル使い分け戦略の一般化を示す。

## 学術研究との関連

本調査の知見は、エージェント研究の複数の方向性を裏付ける。

**プロセス報酬モデル（AgentPRM等）との関連**: 品質が最大障壁であり、かつオブザーバビリティが広く導入されている現状は、ステップレベルの品質評価（PRM）の需要が高いことを示唆する。トレースデータはPRM訓練のためのロールアウトデータとして活用可能である。

**Reflexionとの関連**: 29.5%が評価未実施という状況で、Reflexionのような自己反省メカニズムは、人間の介入なしに品質を改善する手段として有望である。本番運用中組織の44.8%がオンライン評価を実施している事実は、リアルタイムフィードバックループの基盤が整いつつあることを意味する。

**自己修復メカニズムとの関連**: レイテンシ（20%）とセキュリティ（24.9%）が障壁として挙がっている点は、障害時の自動リカバリ（PALADINの障害事例バンク等）の実用的需要を裏付ける。

**LLM-as-Judgeの普及**: 53.3%がLLM-as-Judgeを採用しているデータは、自動評価研究の実用化が進んでいることの直接的な証拠である。

## 運用での学び

本調査から抽出される実運用上の教訓は以下の通りである。

1. **オブザーバビリティ・ファースト**: エージェント開発では、機能実装と同時にトレーシング基盤を構築すべきである。94%の本番運用組織が導入済みという数字は、これがオプションではなく必須であることを示す
2. **評価のギャップを埋める**: オブザーバビリティと評価の導入率差（89% vs 52.4%）は改善の余地を示す。オフライン評価の自動化とLLM-as-Judgeの組み合わせが現実的な対策である
3. **品質に投資する**: 品質問題（32%）がコスト・レイテンシを上回る最大障壁であり、ガードレール・入力検証・出力フィルタリングへの投資が優先される
4. **マルチモデル戦略**: 75%以上が複数モデルを使い分けており、単一モデルへの依存はリスクとなる

## まとめ

LangChainの調査は、LLMエージェントが実験段階を超えて本番運用に移行していることを定量的に示す。57.3%の本番運用率、89%のオブザーバビリティ導入率は、技術の成熟を反映している。一方で、品質管理と体系的評価の導入には依然として大きなギャップがあり、プロセス報酬モデル、自己修復メカニズム、LLM-as-Judge評価といった研究課題が実運用の需要と直結していることが確認できる。

---

## Production Deployment Guide

### 概要

本調査の知見に基づく、エージェント本番運用のためのオブザーバビリティ・評価基盤のAWS構成。

### アーキテクチャ

```
[Agent Application (ECS/Lambda)]
    ├── [LangSmith / OpenTelemetry] → トレース収集
    ├── [CloudWatch Logs] → 構造化ログ
    └── [Bedrock / OpenAI API] → モデル推論
          ↓
[Kinesis Data Firehose] → [S3 Data Lake]
          ↓
[Athena / QuickSight] → ダッシュボード・分析
          ↓
[Lambda (Eval Runner)] → オフライン評価自動実行
```

### 構成パターン

**Small（チーム5名以下、エージェント1-2本）**: LangSmithの無料枠 + CloudWatch Logs。手動レビューとLLM-as-Judge評価をJupyter Notebookで実行。月額約$100。

**Medium（チーム10-20名、エージェント5-10本）**: LangSmith有料プラン + EventBridge Schedulerによる定期評価ジョブ。S3にトレースデータを蓄積し、Athenaでアドホック分析。月額約$1,500。

**Large（組織規模、エージェント20本以上）**: OpenTelemetry + Grafana Tempo/Jaeger自前構築。MLflowでの評価パイプライン管理。QuickSightでの経営向けダッシュボード。月額約$5,000。

### Terraform構成例

```hcl
# 評価ジョブの定期実行
resource "aws_scheduler_schedule" "agent_eval" {
  name       = "agent-eval-daily"
  group_name = "agent-monitoring"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression = "cron(0 3 * * ? *)"  # 毎日 03:00 UTC

  target {
    arn      = aws_lambda_function.eval_runner.arn
    role_arn = aws_iam_role.scheduler.arn

    input = jsonencode({
      eval_type    = "offline"
      sample_size  = 100
      judge_model  = "claude-sonnet-4-6"
      metrics      = ["task_success", "hallucination_rate", "latency_p95"]
    })
  }
}

# トレースデータのS3保存
resource "aws_kinesis_firehose_delivery_stream" "traces" {
  name        = "agent-traces"
  destination = "extended_s3"

  extended_s3_configuration {
    role_arn   = aws_iam_role.firehose.arn
    bucket_arn = aws_s3_bucket.trace_data.arn
    prefix     = "traces/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/"

    buffering_size     = 64
    buffering_interval = 300
    compression_format = "GZIP"
  }
}

# アラート: 品質劣化検知
resource "aws_cloudwatch_metric_alarm" "quality_degradation" {
  alarm_name          = "agent-quality-degradation"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 3
  metric_name         = "TaskSuccessRate"
  namespace           = "AgentMetrics"
  period              = 3600
  statistic           = "Average"
  threshold           = 0.7
  alarm_description   = "タスク成功率が70%を下回った場合にアラート"
  alarm_actions       = [aws_sns_topic.agent_alerts.arn]
}
```

### モニタリング

```yaml
# 調査データに基づく重点メトリクス
metrics:
  - name: task_success_rate
    description: "タスク成功率（品質: 最大障壁32%への対策）"
    target: ">= 0.85"
  - name: hallucination_rate
    description: "ハルシネーション率"
    target: "< 0.05"
  - name: p95_latency_ms
    description: "P95レイテンシ（レイテンシ: 2番目の障壁20%）"
    target: "< 10000"
  - name: eval_coverage
    description: "評価カバレッジ率（調査では52.4%がオフライン評価実施）"
    target: ">= 0.80"
  - name: trace_completeness
    description: "トレース完全性（調査では71.5%がフルトレーシング）"
    target: ">= 0.95"
```

### コスト最適化

- **トレースサンプリング**: 全リクエストのトレースは高コスト。本番では10-20%のサンプリングで十分な品質モニタリングが可能
- **評価バッチ実行**: リアルタイム評価ではなく日次バッチでLLM-as-Judge評価を実行し、API呼び出しコストを削減
- **S3ライフサイクル**: トレースデータは30日後にGlacierへ移行し、90日後に削除
- **モデル使い分け**: 調査結果の通り75%以上が複数モデルを併用。評価にはClaude Sonnet、本番推論にはGPT-4o-miniのように用途別に最適化
