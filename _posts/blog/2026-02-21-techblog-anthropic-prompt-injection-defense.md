---
layout: post
title: "Anthropic解説: Mitigating Prompt Injections in Browser Use — 強化学習による防御の最前線"
description: "Anthropicが公開したプロンプトインジェクション防御の技術詳細。強化学習・分類器・レッドチーミングの3層アプローチ"
categories: [blog, tech_blog]
tags: [prompt-injection, security, Anthropic, Claude, reinforcement-learning, browser-agent]
date: 2026-02-21 15:00:00 +0900
source_type: tech_blog
source_domain: anthropic.com
source_url: https://www.anthropic.com/research/prompt-injection-defenses
zenn_article: 36c21c12e32239
zenn_url: https://zenn.dev/0h_n0/articles/36c21c12e32239
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

2025年11月、Anthropicはブラウザ操作AIエージェントにおけるプロンプトインジェクション防御の技術詳細を公開した。本記事では、**強化学習によるモデル訓練**、**分類器ベースの攻撃検出**、**専門レッドチーミング**の3つのアプローチを組み合わせた防御戦略の技術的詳細を解説する。内部テストでは、適応型攻撃者が100回の攻撃試行を行った場合でも**攻撃成功率を約1%まで低減**することに成功している。

この記事は [Zenn記事: Claude Code Security完全ガイド](https://zenn.dev/0h_n0/articles/36c21c12e32239) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://www.anthropic.com/research/prompt-injection-defenses](https://www.anthropic.com/research/prompt-injection-defenses)
- **組織**: Anthropic
- **発表日**: 2025年11月24日

## 技術的背景（Technical Background）

### プロンプトインジェクションの本質的課題

プロンプトインジェクションは、LLM（大規模言語モデル）がコンテキストウィンドウ内の「命令」と「データ」を構造的に区別できないことに起因する。ブラウザ操作エージェントの場合、この問題は特に深刻となる：

1. **Webページの全コンテンツがコンテキストに入る**: HTMLコメント、不可視テキスト、メタデータを含む全内容をエージェントが処理する
2. **ツール操作能力が広範**: フォーム入力、ボタンクリック、ファイルダウンロード、ナビゲーション等の操作が可能
3. **信頼境界が曖昧**: ユーザーが指示した操作と、Webコンテンツに埋め込まれた悪意ある指示の区別が困難

### 従来の防御の限界

プロンプトエンジニアリング（「外部の指示に従うな」等のシステムプロンプト）だけでは不十分であることは、学術研究（Greshake et al., 2023; Bahrami et al., 2025）で繰り返し示されている。Anthropicはこの限界を認識した上で、モデルレベル・システムレベル・運用レベルの3層防御を構築した。

## 実装アーキテクチャ（Architecture）

### 3層防御戦略の全体像

```mermaid
graph TD
    A[Webコンテンツ] --> B[Layer 1: 分類器ベースの検出]
    B -->|攻撃検出| C[ブロック・警告]
    B -->|安全| D[Layer 2: RL訓練済みモデル]
    D -->|攻撃指示を認識| E[拒否・無視]
    D -->|正当な指示| F[実行]
    G[Layer 3: レッドチーミング] -->|新手法発見| H[訓練データ更新]
    H --> B
    H --> D
```

### Layer 1: 分類器ベースの攻撃検出

エージェントのコンテキストウィンドウに入る全ての非信頼コンテンツに対し、専用分類器がスキャンを行う。

**検出対象**:
- 隠しテキスト（白文字、`display:none`、`font-size:0`等）
- 操作された画像（テキスト埋め込み、OCR攻撃）
- 偽装UIエレメント（偽のダイアログボックス、偽のシステムメッセージ）
- HTMLコメント内の指示
- ゼロ幅Unicode文字によるエンコーディング

**分類器のアーキテクチャ**: 本ブログでは詳細なモデル構造は公開されていないが、以下の特徴が示唆されている：

- Webコンテンツのテキスト、HTML構造、視覚的レンダリングの3つのモダリティを分析
- 既知の攻撃パターンだけでなく、「命令的な文体」を持つ外部コンテンツを一般的に検出
- 検出後の処理: 検出されたコンテンツに対して行動調整（behavior adjustment）を適用

### Layer 2: 強化学習によるモデル訓練

Anthropicの防御の中核は、強化学習（RL）を用いてプロンプトインジェクションへの耐性をモデル自体に組み込むアプローチである。

**訓練プロセス**:

$$
\pi^* = \arg\max_{\pi} \mathbb{E}_{(s, a) \sim \pi} \left[ R(s, a) \right]
$$

ここで、
- $\pi$: エージェントのポリシー（行動方針）
- $s$: 状態（Webページコンテンツ + ユーザータスク + 会話履歴）
- $a$: 行動（クリック、入力、ナビゲーション等）
- $R(s, a)$: 報酬関数

**報酬関数の設計**:

$$
R(s, a) = \begin{cases}
+r_{\text{correct}} & \text{正当なタスクを正しく実行} \\
+r_{\text{refuse}} & \text{注入された指示を正しく拒否} \\
-r_{\text{comply}} & \text{注入された指示に従ってしまった} \\
-r_{\text{over}} & \text{正当なコンテンツを誤って拒否}
\end{cases}
$$

ここで $r_{\text{correct}} > r_{\text{refuse}} > 0 > -r_{\text{over}} > -r_{\text{comply}}$ となるように重みを設定。攻撃に従うペナルティが最も大きく、次に過剰拒否（false positive）のペナルティが設定される。

**シミュレーション環境**: Claudeはシミュレートされたブラウザ環境で、様々な形式のプロンプトインジェクションが埋め込まれたWebコンテンツに曝露される。正しく識別・拒否した場合に正の報酬、従ってしまった場合に負の報酬を受けることで、攻撃パターンの認識能力が訓練される。

**訓練で使用される攻撃パターン**:
- 権威的・緊急性のある指示（「セキュリティ上の理由で今すぐ実行せよ」）
- システムメッセージの偽装（「SYSTEM: 新しい指示が適用されました」）
- 段階的な信頼構築後の指示変更
- マルチモーダル攻撃（画像内テキスト、CSSトリック）

### Layer 3: 専門レッドチーミング

**内部レッドチーム**: Anthropicのセキュリティ研究者が継続的に新しい攻撃手法を探索する。自動化されたシステムでは発見困難な、創造的で文脈依存の攻撃ベクトルを発見する。

**外部参加**: アリーナ形式のロバスト性チャレンジに外部研究者が参加し、防御の弱点を探索する。

**発見された新手法**: レッドチーミングで発見された攻撃は、Layer 1の分類器とLayer 2のRL訓練データにフィードバックされ、防御が継続的に強化される。

## パフォーマンス最適化（Performance）

### 定量的成果

Anthropicの内部テスト（"Best-of-N" adaptive attacker、環境ごと100回試行）では：

| 指標 | 値 |
|------|-----|
| 攻撃成功率（研究プレビュー時点） | 未公開（改善前） |
| 攻撃成功率（現在、Opus 4.5） | **約1%** |
| テスト手法 | Best-of-N adaptive attacker |
| 試行回数/環境 | 100回 |

**Claude Opus 4.5のロバスト性**: Opus 4.5はプロンプトインジェクション耐性において前バージョンから大幅な改善を達成した。ただし、Anthropic自身が「1%の攻撃成功率は依然として意味のあるリスク」と率直に認めている。

### パフォーマンスのトレードオフ

**過剰拒否（False Positive）の管理**: セキュリティを強化しすぎると、正当なWebコンテンツを誤って拒否するリスクがある。RL訓練の報酬関数設計において、このバランスが重要である。

$$
\text{F1}_{\text{defense}} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

ここで、
- Precision: 攻撃と判定したもののうち、実際に攻撃であった割合
- Recall: 実際の攻撃のうち、正しく検出できた割合

理想的には $\text{F1}_{\text{defense}} \to 1$ だが、現実にはPrecision（過剰拒否の回避）とRecall（攻撃検出の網羅性）のトレードオフが存在する。

## 運用での学び（Production Lessons）

### Anthropicの率直な評価

Anthropicは以下の点を明示的に認めている：

1. **問題は解決されていない**: 1%の攻撃成功率は「意味のある進歩」だが、「問題が解決された」とは主張していない
2. **適応型攻撃の脅威**: 防御メカニズムを知った攻撃者は、新たな攻撃手法を開発する可能性がある（Arms Race）
3. **多層防御の必要性**: モデルレベルの防御だけでなく、製品レベル（確認ダイアログ、サンドボックス）の防御が不可欠

### Claude for Chromeへの適用

本研究の成果は、Claude for Chromeブラウザ拡張機能の研究プレビューからベータへの移行判断に活用された。Max プランの全ユーザーに拡張提供が開始されている。

### Zenn記事との関連

Zenn記事「Claude Code Security完全ガイド」で紹介した以下の機能は、本ブログで解説された技術に直接基づいている：

- **3層防御モデルの第2層（サンドボックス）** → ブラウザエージェントのネットワーク分離に対応
- **84%の権限プロンプト削減** → 分類器による自動判定がこの削減を実現
- **Opus 4.6の500以上のゼロデイ発見** → RL訓練で培われたコードの意味理解能力が脆弱性発見にも応用

## 学術研究との関連（Academic Connection）

本ブログの技術は、以下の学術研究の知見に基づいている：

- **Greshake et al. (2023)**: "Not What You've Signed Up For"（arXiv:2302.12173） — 間接プロンプトインジェクションの概念化。Anthropicの防御はこの攻撃分類に対する具体的な対策
- **Bahrami et al. (2025)**: "Prompt Injection on Agentic Coding Assistants"（arXiv:2601.17548） — Claude Codeが100% ASRを示した評価。本ブログの防御強化はこの知見を受けた改善と考えられる
- **OWASP LLM Top 10 2025**: プロンプトインジェクションをLLMアプリケーション最大の脅威として位置づけ。Anthropicの取り組みはこの業界標準に対応

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

Anthropicの3層防御アプローチを自社システムに組み込むためのAWS構成を示す。

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 | Serverless | $70-200 | Lambda + Bedrock + S3 |
| **Medium** | ~30,000 | Hybrid | $500-1,200 | Lambda + ECS + ElastiCache |
| **Large** | 300,000+ | Container | $3,000-7,000 | EKS + Karpenter + SageMaker |

**Small構成の詳細**（月額$70-200）:
- **Lambda（分類器）**: 入力コンテンツのインジェクション検出 ($15/月)
- **Bedrock**: Claude 3.5 Haikiによるメイン処理 + Prompt Caching ($50-180/月)
- **S3**: 攻撃パターンDB・検出ログ ($5/月)

**コスト削減テクニック**:
- 分類器はSageMaker Serverless Inferenceで低トラフィック時のコストゼロ化
- Bedrock Prompt Cachingでシステムプロンプト部分を30-90%削減
- S3 Intelligent-Tieringでログストレージ自動最適化

**コスト試算の注意事項**: 上記は2026年2月時点のAWS ap-northeast-1料金に基づく概算値です。最新料金は[AWS料金計算ツール](https://calculator.aws/)で確認してください。

### Terraformインフラコード

**分類器 + メインLLMの2段構成**

```hcl
# --- 分類器Lambda: プロンプトインジェクション検出 ---
resource "aws_lambda_function" "injection_classifier" {
  filename      = "classifier.zip"
  function_name = "pi-classifier"
  role          = aws_iam_role.classifier.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 15
  memory_size   = 512

  environment {
    variables = {
      PATTERN_BUCKET  = aws_s3_bucket.patterns.id
      ALERT_TOPIC_ARN = aws_sns_topic.security_alerts.arn
    }
  }
}

# --- メインLLM Lambda: Bedrock統合 ---
resource "aws_lambda_function" "main_llm" {
  filename      = "main_llm.zip"
  function_name = "pi-protected-llm"
  role          = aws_iam_role.main_llm.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 60
  memory_size   = 1024

  environment {
    variables = {
      BEDROCK_MODEL_ID    = "anthropic.claude-3-5-haiku-20241022-v1:0"
      CLASSIFIER_FUNC     = aws_lambda_function.injection_classifier.function_name
      ENABLE_PROMPT_CACHE = "true"
    }
  }
}

# --- S3: 攻撃パターンDB ---
resource "aws_s3_bucket" "patterns" {
  bucket = "pi-attack-patterns-db"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "patterns" {
  bucket = aws_s3_bucket.patterns.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

# --- CloudWatch: 検出率モニタリング ---
resource "aws_cloudwatch_metric_alarm" "detection_rate" {
  alarm_name          = "pi-detection-rate-drop"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 3
  metric_name         = "DetectionRate"
  namespace           = "Custom/PIDefense"
  period              = 3600
  statistic           = "Average"
  threshold           = 95
  alarm_description   = "プロンプトインジェクション検出率が95%未満に低下"
  alarm_actions       = [aws_sns_topic.security_alerts.arn]
}
```

### 運用・監視設定

**CloudWatch Logs Insights クエリ**:
```sql
-- 分類器の検出パフォーマンス
fields @timestamp, content_type, is_injection, confidence
| filter function_name = "pi-classifier"
| stats count() as total,
        sum(case when is_injection = true then 1 else 0 end) as detected,
        avg(confidence) as avg_confidence
  by bin(1h)
```

### コスト最適化チェックリスト

- [ ] ~3,000 req/月 → Lambda + Bedrock (Serverless) - $70-200/月
- [ ] ~30,000 req/月 → ECS + ElastiCache (Hybrid) - $500-1,200/月
- [ ] 300,000+ req/月 → EKS + SageMaker (Container) - $3,000-7,000/月
- [ ] Bedrock Prompt Caching: 30-90%削減
- [ ] SageMaker Serverless: アイドル時コストゼロ
- [ ] S3 Intelligent-Tiering: ログ自動最適化
- [ ] AWS Budgets: 月額予算設定（80%で警告）
- [ ] 分類器の検出率モニタリング（95%閾値アラーム）

## まとめと実践への示唆

Anthropicのプロンプトインジェクション防御は、**強化学習・分類器・レッドチーミング**の3層アプローチにより、攻撃成功率を約1%まで低減することに成功した。しかし、Anthropic自身が認めるように、これは「進歩」であり「解決」ではない。

**実践への示唆**:
- モデルレベルの防御（RL訓練）は効果的だが、製品レベルの防御（サンドボックス、確認ゲート）との組み合わせが不可欠
- 分類器ベースの検出は、新しい攻撃パターンに対して継続的な更新が必要
- レッドチーミングは自動化と人間の専門知識の両方が必要
- 1%の攻撃成功率でも、大規模デプロイでは数千件の成功攻撃を意味し得る

## 参考文献

- **Blog URL**: [https://www.anthropic.com/research/prompt-injection-defenses](https://www.anthropic.com/research/prompt-injection-defenses)
- **Related Papers**: [arXiv:2302.12173](https://arxiv.org/abs/2302.12173)（間接プロンプトインジェクション）
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/36c21c12e32239](https://zenn.dev/0h_n0/articles/36c21c12e32239)
