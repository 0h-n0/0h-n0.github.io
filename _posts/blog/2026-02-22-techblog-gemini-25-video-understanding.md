---
layout: post
title: "Google解説: Gemini 2.5の動画理解 — VideoMME 85.2%達成のマルチモーダル技術詳細"
description: "Google Developers Blogが発表したGemini 2.5の動画理解能力の詳細解説。VideoMMEベンチマーク85.2%達成、最大6時間の動画処理、200万トークンコンテキスト対応の技術を分析する。"
categories: [blog, tech_blog]
tags: [gemini, video-understanding, multimodal, google, benchmark, ai]
date: 2026-02-22 11:00:00 +0900
source_type: tech_blog
source_domain: developers.googleblog.com
source_url: https://developers.googleblog.com/en/gemini-2-5-video-understanding/
zenn_article: 3d32da8cfe0ac1
zenn_url: https://zenn.dev/0h_n0/articles/3d32da8cfe0ac1
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Advancing the frontier of video understanding with Gemini 2.5 — Google Developers Blog](https://developers.googleblog.com/en/gemini-2-5-video-understanding/) の解説記事です。

## ブログ概要（Summary）

Google Developers Blogが2025年5月9日に公開した本記事では、Gemini 2.5 Proの動画理解能力について詳細に報告している。著者らは Anirudh Baddepudi（プロダクトマネージャー）、Antoine Yang（研究科学者）、Mario Lučić（研究科学者）である。ブログによると、Gemini 2.5 ProはVideoMMEベンチマークで85.2%を達成し、GPT-4.1を上回る性能を示したとされている。特に、最大256フレーム処理（1fps、線形サンプリング）と200万トークンコンテキストの組み合わせにより、低解像度設定で約6時間の動画処理が可能になったと報告されている。

この記事は [Zenn記事: Gemini 2.0マルチモーダルAPI実践ガイド：画像・音声・動画をPythonで統合処理する](https://zenn.dev/0h_n0/articles/3d32da8cfe0ac1) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://developers.googleblog.com/en/gemini-2-5-video-understanding/](https://developers.googleblog.com/en/gemini-2-5-video-understanding/)
- **組織**: Google DeepMind / Google Developers
- **発表日**: 2025年5月9日

## 技術的背景（Technical Background）

動画理解はマルチモーダルAIにおける重要な課題である。テキストや画像と異なり、動画は時間方向の情報を含むため、処理に必要なトークン数が桁違いに大きくなる。Gemini 1.0では32Kトークン（約1分40秒の動画）、Gemini 1.5では100万トークン（約1時間の動画）が上限であった。

Gemini 2.5ではコンテキストウィンドウが200万トークンに拡張され、以下の動画処理容量が実現されたとブログで報告されている：

| 解像度設定 | トークン/秒 | 処理可能時間 | トークン消費（1時間） |
|-----------|------------|-------------|-------------------|
| デフォルト | ~300 | ~1.8時間 | ~1,080,000 |
| 低解像度 | ~100 | ~5.6時間 | ~360,000 |

この進化により、映画全編や長時間の講義録画を1回のAPI呼び出しで分析することが技術的に可能になった。

### VideoMMEベンチマーク

VideoMMEは2024年にCVPR 2025に採択された動画理解の包括的ベンチマークであり、以下の特徴を持つ：

- 900本の手動収集・キュレーション動画
- 6つの主要ビジュアルドメイン、30サブフィールド
- 短時間（11秒）〜長時間（1時間）の動画を網羅
- 字幕の有無による条件設定

ブログによると、Gemini 2.5 ProはVideoMMEで以下の性能を達成している（ブログ掲載値）：

| モデル | VideoMME精度 |
|--------|------------|
| Gemini 2.5 Pro | **85.2%** |
| GPT-4.1 | 記載なし（Gemini 2.5 Proが上回るとの記述） |
| Gemini 1.5 Pro | ~75.0% (arXiv:2403.05530より) |

## 実装アーキテクチャ（Architecture）

ブログで報告されている Gemini 2.5の動画理解アーキテクチャの特徴は以下の通りである。

### フレームサンプリング戦略

Gemini 2.5 Proは最大256フレームを処理する。ブログによると、1fpsの線形サンプリングが基本設定であり、256フレーム（約4分17秒の動画に相当）を超える場合は均等間隔でサンプリングされる。

フレームサンプリングの計算：

$$
\text{sampling\_interval} = \max\left(1, \left\lfloor \frac{T_{\text{video}}}{N_{\text{max}}} \right\rfloor\right)
$$

ここで、
- $T_{\text{video}}$: 動画の総フレーム数
- $N_{\text{max}}$: 最大処理フレーム数（256）
- 結果として得られるサンプリング間隔でフレームを等間隔抽出

### マルチモーダル統合

ブログでは「natively multimodal model can use audio-visual information seamlessly with code and other data formats」と記述されている。これは、動画の映像フレームと音声トラックが同一のTransformerコンテキスト内で統合的に処理されることを意味する。

```python
from google import genai
from google.genai import types

def advanced_video_analysis(
    video_path: str,
    analysis_type: str = "comprehensive",
) -> str:
    """Gemini 2.5の動画理解機能を活用した高度な分析

    Args:
        video_path: 動画ファイルパス
        analysis_type: 分析タイプ（comprehensive/moments/summary）

    Returns:
        分析結果テキスト
    """
    client = genai.Client()
    video_file = client.files.upload(file=video_path)

    prompts = {
        "comprehensive": (
            "この動画を詳細に分析してください:\n"
            "1. 主要なシーンとその説明\n"
            "2. 重要な瞬間のタイムスタンプ\n"
            "3. 音声の文字起こし\n"
            "4. 全体のストーリーラインの要約"
        ),
        "moments": (
            "この動画の中で重要な瞬間を特定し、"
            "それぞれのタイムスタンプと説明を提供してください。"
        ),
        "summary": (
            "この動画の内容を3-5文で要約してください。"
        ),
    }

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[video_file, prompts[analysis_type]],
    )
    return response.text
```

### 低解像度処理によるコスト最適化

```python
def analyze_video_low_resolution(
    video_path: str,
    prompt: str,
) -> str:
    """低解像度設定でコストを1/3に削減した動画分析

    Args:
        video_path: 動画ファイルパス
        prompt: 分析プロンプト

    Returns:
        分析結果テキスト
    """
    client = genai.Client()
    video_file = client.files.upload(file=video_path)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[video_file, prompt],
        config=types.GenerateContentConfig(
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
        ),
    )
    return response.text
```

## パフォーマンス最適化（Performance）

### ベンチマーク別の性能詳細

ブログで報告されている Gemini 2.5 Proの動画理解ベンチマーク結果を整理する。

| ベンチマーク | タスク | Gemini 2.5 Pro | 特記事項 |
|-------------|-------|----------------|---------|
| VideoMME | 総合動画QA | 85.2% | GPT-4.1を上回ると報告 |
| YouCook2 | 料理動画のdense captioning | 特化モデル相当 | ファインチューニングなし |
| QVHighlights | モーメント検索 | 特化モデル相当 | テンポラルグラウンディング |

### 動画長別の処理パフォーマンス

ブログの記述に基づく処理容量の整理：

| 動画長 | デフォルト解像度トークン数 | 低解像度トークン数 | API推定コスト（Flash） |
|--------|----------------------|------------------|---------------------|
| 1分 | ~18,000 | ~6,000 | ~$0.002 |
| 10分 | ~180,000 | ~60,000 | ~$0.018 |
| 1時間 | ~1,080,000 | ~360,000 | ~$0.108 |
| 6時間 | 超過（200万トークン超） | ~2,160,000 | ~$0.648 |

### チューニング手法

ブログで紹介されているユースケースに基づく最適化アプローチ：

1. **動画→学習アプリケーション変換**: 動画を分析し、p5.jsアニメーション等のインタラクティブアプリに変換。動画のビジュアル要素をコードとして再構成する
2. **正確なモーメント検出**: 音声・映像の手がかりを統合して特定の瞬間を特定。従来の動画処理システムより高い精度をブログは報告している
3. **YouTube動画の直接分析**: APIおよびGoogle AI Studio経由でYouTube動画のURLを直接入力可能

## 運用での学び（Production Lessons）

ブログの記述から推測される本番運用での考慮事項：

1. **リクエストあたりの動画数制限**: Gemini 2.5以降では、1リクエストあたり最大10本の動画をアップロード可能。複数動画の比較分析がシングルリクエストで実現できる

2. **Files APIのアップロード待ち**: 大容量動画のアップロード後、処理状態が`ACTIVE`になるまで待機が必要。ブログでは明記されていないが、数百MBの動画では数分かかる場合がある

3. **音声+映像の統合分析**: 動画の音声トラックと映像フレームを同時に分析するため、音声のみ・映像のみの分析より情報量が多い。ただし、トークン消費量も増加する

4. **モデル選択の指針**: ブログでは「Gemini 2.5 Pro achieves state-of-the-art performance」と報告しているが、コスト効率を重視する場合はGemini 2.5 Flash（低コスト）が推奨される

## 学術研究との関連（Academic Connection）

このブログで報告されている技術は、以下の学術研究の延長線上にある：

- **Gemini 1.0 (arXiv:2312.11805)**: ネイティブマルチモーダルアーキテクチャの基礎。モダリティ別エンコーダ→Transformerデコーダの設計
- **Gemini 1.5 (arXiv:2403.05530)**: Sparse MoE + 100万トークンコンテキスト。ロングコンテキスト動画処理の基盤技術
- **Video-MME (arXiv:2405.21075)**: CVPR 2025採択のベンチマーク。Gemini 2.5 Proの評価に使用されている

ブログで報告されているVideoMME 85.2%は、Gemini 1.5 Proの約75.0%（arXiv:2403.05530より）から約10ポイントの向上であり、2.5世代でのアーキテクチャ改善（200万トークンコンテキスト、改良されたビジョンエンコーダ等）の効果を示している。

## Production Deployment Guide

### AWS実装パターン（コスト最適化重視）

Gemini 2.5の動画理解APIをAWSバックエンドで運用するパターンを示す。

**トラフィック量別の推奨構成**:

| 規模 | 月間リクエスト | 推奨構成 | 月額コスト | 主要サービス |
|------|--------------|---------|-----------|------------|
| **Small** | ~3,000 | Serverless | $80-200 | Step Functions + Lambda + S3 |
| **Medium** | ~30,000 | Hybrid | $400-1,200 | ECS Fargate + S3 + SQS |
| **Large** | 300,000+ | Container | $2,500-6,000 | EKS + S3 + SQS + ElastiCache |

**Small構成の詳細** (月額$80-200):
- **Step Functions**: 動画処理の非同期ワークフロー管理 ($5/月)
- **Lambda**: Gemini API呼び出し、結果処理 ($25/月)
- **S3**: 動画ファイル保存、30日ライフサイクル ($20/月)
- **Gemini API**: Flash使用、低解像度設定 ($100/月)
- **CloudWatch**: 基本監視 ($10/月)

**コスト削減テクニック**:
- 低解像度設定（`MEDIA_RESOLUTION_LOW`）でトークン消費を1/3に削減
- 動画クリッピング（`start_offset`/`end_offset`）で必要区間のみ処理
- 結果キャッシュで同一動画への重複リクエスト排除
- Flashモデル優先（Proの約1/10コスト）

**コスト試算の注意事項**:
- 上記は2026年2月時点のGemini API料金とAWS ap-northeast-1リージョン料金に基づく概算値です
- 動画処理のGemini API費用はトークン消費量に比例し、長時間動画ほど高額になる
- 最新料金は [AWS料金計算ツール](https://calculator.aws/) で確認してください

### Terraformインフラコード

**動画分析用Step Functions構成**

```hcl
# --- Step Functions（動画処理ワークフロー） ---
resource "aws_sfn_state_machine" "video_pipeline" {
  name     = "gemini-video-analysis-pipeline"
  role_arn = aws_iam_role.sfn_role.arn

  definition = jsonencode({
    Comment = "Gemini 2.5動画分析パイプライン"
    StartAt = "UploadVideo"
    States = {
      UploadVideo = {
        Type     = "Task"
        Resource = aws_lambda_function.upload_handler.arn
        Next     = "AnalyzeVideo"
      }
      AnalyzeVideo = {
        Type     = "Task"
        Resource = aws_lambda_function.analyze_handler.arn
        TimeoutSeconds = 900
        Retry = [{
          ErrorEquals     = ["States.TaskFailed"]
          IntervalSeconds = 30
          MaxAttempts     = 3
          BackoffRate     = 2.0
        }]
        Next = "SaveResults"
      }
      SaveResults = {
        Type     = "Task"
        Resource = aws_lambda_function.save_handler.arn
        End      = true
      }
    }
  })
}

# --- Lambda: 動画アップロード ---
resource "aws_lambda_function" "upload_handler" {
  function_name = "gemini-video-upload"
  role          = aws_iam_role.lambda_role.arn
  handler       = "upload.handler"
  runtime       = "python3.12"
  timeout       = 300
  memory_size   = 512
}

# --- Lambda: Gemini API呼び出し ---
resource "aws_lambda_function" "analyze_handler" {
  function_name = "gemini-video-analyze"
  role          = aws_iam_role.lambda_role.arn
  handler       = "analyze.handler"
  runtime       = "python3.12"
  timeout       = 900
  memory_size   = 2048

  environment {
    variables = {
      GEMINI_MODEL_ID    = "gemini-2.5-flash"
      MEDIA_RESOLUTION   = "LOW"
    }
  }
}
```

### 運用・監視設定

```sql
-- CloudWatch Logs Insights: 動画処理レイテンシ分析
fields @timestamp, video_duration_sec, processing_time_ms, token_count
| stats avg(processing_time_ms) as avg_latency,
        pct(processing_time_ms, 95) as p95_latency
  by bin(video_duration_sec, 60) as duration_bucket
| sort duration_bucket
```

### コスト最適化チェックリスト

- [ ] 動画入力: `media_resolution=LOW` でトークン消費を1/3に削減
- [ ] 動画クリッピング: 必要区間のみ処理
- [ ] Flash モデル優先（Proの約1/10コスト）
- [ ] Step Functionsで非同期処理（API制限回避）
- [ ] S3ライフサイクル: 処理済み動画を30日で自動削除
- [ ] 結果キャッシュ: DynamoDBで重複リクエスト排除
- [ ] AWS Budgets: 月額予算設定

## まとめと実践への示唆

Google Developers Blogで報告されている Gemini 2.5の動画理解能力は、VideoMME 85.2%というベンチマーク性能と、最大6時間の動画処理容量を組み合わせたものである。Zenn記事で解説されている動画理解APIの実装パターンは、このブログで報告されている能力に基づいている。

開発者にとっての実践的な示唆は以下の通りである：
- 低解像度設定でコストを1/3に削減できるため、精度要件が厳密でない用途ではFlash + LOW解像度が推奨される
- 1リクエストあたり最大10本の動画を処理可能なため、複数動画の比較分析がシングルリクエストで実現可能
- YouTube動画の直接分析がAPI経由で可能であり、URL入力のみで分析を開始できる

## 参考文献

- **Blog URL**: [https://developers.googleblog.com/en/gemini-2-5-video-understanding/](https://developers.googleblog.com/en/gemini-2-5-video-understanding/)
- **Related Papers**: [https://arxiv.org/abs/2405.21075](https://arxiv.org/abs/2405.21075) (Video-MME)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/3d32da8cfe0ac1](https://zenn.dev/0h_n0/articles/3d32da8cfe0ac1)
