---
layout: post
title: "Google Developers Blog解説: Gemini 2.5の動画理解 — ネイティブマルチモーダルによる映像×音声×コード統合処理"
description: "Google公式ブログが解説するGemini 2.5の動画理解能力を技術的に分析。1fps処理、media_resolution制御、6時間動画の一括処理を詳解する"
categories: [blog, tech_blog]
tags: [gemini, video-understanding, multimodal, google, api]
date: 2026-02-23 11:00:00 +0900
source_type: tech_blog
source_domain: developers.googleblog.com
source_url: https://developers.googleblog.com/en/gemini-2-5-video-understanding/
zenn_article: df5295d69a456f
zenn_url: https://zenn.dev/0h_n0/articles/df5295d69a456f
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Advancing the frontier of video understanding with Gemini 2.5（Google Developers Blog、2025年5月9日）](https://developers.googleblog.com/en/gemini-2-5-video-understanding/) の解説記事です。

## ブログ概要（Summary）

Google Developers Blog が2025年5月に公開した本記事は、Gemini 2.5 の動画理解能力に焦点を当てた技術解説である。Gemini 2.5 は「ネイティブマルチモーダルモデルが音声・映像情報をコード等のデータフォーマットとシームレスに統合処理できる初めての事例」と位置づけられている。1fps のフレームサンプリング、`media_resolution` パラメータによるトークン消費制御、2Mトークンコンテキストでの約6時間動画の一括処理など、実装に直結する技術的詳細が記載されている。

この記事は [Zenn記事: Gemini 3.1 Pro マルチモーダルAPI実践ガイド：画像・音声・動画をPythonで統合処理する](https://zenn.dev/0h_n0/articles/df5295d69a456f) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://developers.googleblog.com/en/gemini-2-5-video-understanding/](https://developers.googleblog.com/en/gemini-2-5-video-understanding/)
- **組織**: Google Developers（Anirudh Baddepudi, Antoine Yang, Mario Lučić）
- **発表日**: 2025年5月9日

## 技術的背景（Technical Background）

動画理解は、マルチモーダル AI における最も計算コストの高いタスクの一つである。動画は画像（フレーム列）＋音声＋時間構造という3つの情報軸を持ち、それぞれを独立に処理してから統合する従来のアプローチには以下の限界があった。

**従来アプローチの課題**:
- **2段階パイプライン**: 動画→フレーム抽出→画像認識→テキスト要約、という多段階処理では中間表現での情報損失が不可避
- **音声の独立処理**: 映像と音声を別々のモデルで処理すると、「画面に映っている人物が発話している内容」のような映像-音声間の対応関係を活用できない
- **時間構造の無視**: 個別フレームの認識では、動画特有の時間的な因果関係や推移を捉えられない

Gemini 2.5 はネイティブマルチモーダル Transformer として、これらの情報を統一されたトークンシーケンスとして処理することで、上記の課題に対処している。

## 実装アーキテクチャ（Architecture）

### フレームサンプリングとトークン化

ブログの記述によると、Gemini 2.5 は動画を以下のプロセスでトークン化する。

**1. フレームサンプリング**: デフォルトで1fps（1秒あたり1フレーム）の線形サブサンプリングを適用。最大256フレーム（約4分17秒）まで処理可能。長尺動画評価（1H-VideoQA）では最大7,200フレーム（2時間）まで拡張される。

$$
N_{\text{frames}} = \min\left(\lfloor T \times \text{fps} \rfloor, N_{\max}\right)
$$

ここで、$T$ は動画の長さ（秒）、$\text{fps}$ はサンプリングレート（デフォルト1.0）、$N_{\max}$ は最大フレーム数（256または7,200）である。

**2. フレームごとのトークン化**: 各フレームは `media_resolution` パラメータに応じてトークンに変換される。

| media_resolution | トークン数/フレーム | 用途 |
|-----------------|-------------------|------|
| HIGH | 約1,120 | OCR、細部認識 |
| MEDIUM（デフォルト） | 約560 | 一般的な動画理解 |
| LOW | 約280 | シーン全体の把握 |

**3. 音声トークンの統合**: 音声トラックは約32トークン/秒でエンコードされ、対応するフレームのトークン列と**インターリーブ**される。

### トークン消費量の計算

10分の動画を処理する場合のトークン消費量は以下の通り。

$$
N_{\text{total}} = N_{\text{frames}} \times N_{\text{tokens/frame}} + T \times R_{\text{audio}}
$$

| 解像度 | フレームトークン | 音声トークン | 合計 | API コスト (Pro $2/M) |
|--------|---------------|------------|------|---------------------|
| HIGH | 600 × 1,120 = 672,000 | 600 × 32 = 19,200 | 691,200 | $1.38 |
| MEDIUM | 600 × 560 = 336,000 | 19,200 | 355,200 | $0.71 |
| LOW | 600 × 280 = 168,000 | 19,200 | 187,200 | $0.37 |

ここで、10分 = 600秒、1fps で600フレーム。

### 6時間動画の処理

ブログでは、`media_resolution=LOW` を使用することで Gemini 2.5 Pro が約6時間の動画を2Mトークンのコンテキストウィンドウ内で処理できると報告されている。

$$
T_{\max} = \frac{2,000,000}{280 \times 1 + 32} = \frac{2,000,000}{312} \approx 6,410 \text{秒} \approx 1\text{時間}47\text{分}
$$

この計算は1fpsの場合であり、ブログで言及されている「約6時間」はサブサンプリングレートの調整（例: 0.3fps）による可能性がある。

$$
T_{\max}^{0.3\text{fps}} = \frac{2,000,000}{280 \times 0.3 + 32} \approx \frac{2,000,000}{116} \approx 17,241 \text{秒} \approx 4\text{時間}47\text{分}
$$

### YouTube URL 直接入力

ブログによると、Gemini 2.5 は YouTube の公開動画 URL を直接入力として受け付ける。この機能により、Files API を経由せずに動画を処理できる。

```python
from google import genai
from google.genai import types


def analyze_youtube_video(url: str, prompt: str) -> str:
    """YouTube動画を直接分析する

    Args:
        url: YouTube公開動画のURL
        prompt: 分析に関する質問・指示

    Returns:
        分析結果のテキスト

    Note:
        公開動画のみ対応。限定公開・非公開動画では
        InvalidArgument エラーが発生する。
    """
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            types.Part(file_data=types.FileData(file_uri=url)),
            prompt,
        ],
    )
    return response.text
```

## パフォーマンス最適化（Performance）

### ベンチマーク結果

ブログで報告されているベンチマーク結果（原文の表記に基づく）。

| ベンチマーク | タスク | Gemini 2.5 Pro | Gemini 2.5 Pro (LOW) |
|-------------|-------|---------------|---------------------|
| VideoMME | 動画QA | 85.2% | 84.7% |
| EgoTempo | 時間推論 | LLM-based accuracy | - |
| QVHighlights | モーメント検索 | R1@0.5 | - |
| YouCook2 | 密な字幕生成 | CIDEr score | - |

VideoMME の結果が示す通り、`media_resolution=LOW` を使用してもデフォルト解像度との性能差はわずか0.5ポイントである。トークン消費量が半分になることを考えると、コスト効率の面で LOW 解像度の使用が有利な場面は多い。

### 解像度とコストのトレードオフ

```python
from google import genai
from google.genai import types


def analyze_video_cost_optimized(
    video_path: str,
    prompt: str,
    cost_priority: bool = True,
) -> str:
    """コスト最適化された動画分析

    Args:
        video_path: 動画ファイルパス
        prompt: 分析指示
        cost_priority: Trueならコスト優先（LOW解像度）

    Returns:
        分析結果テキスト
    """
    client = genai.Client()
    video_file = client.files.upload(file=video_path)

    # ACTIVE状態を待機
    import time
    while video_file.state.name == "PROCESSING":
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)

    resolution = (
        types.MediaResolution.MEDIA_RESOLUTION_LOW
        if cost_priority
        else types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
    )

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[video_file, prompt],
        config=types.GenerateContentConfig(
            media_resolution=resolution,
        ),
    )
    return response.text
```

## 運用での学び（Production Lessons）

### ユースケース別の最適設定

ブログでは以下の4つのユースケースが紹介されている。それぞれに適した設定を分析する。

**1. インタラクティブ学習アプリケーション生成**

教育動画からインタラクティブなWebアプリケーションの仕様書を自動生成するユースケース。動画の内容理解に加え、コード生成能力が必要。

- 推奨設定: `media_resolution=MEDIUM`, `thinking_level=HIGH`
- 理由: 教育コンテンツの正確な理解と、仕様書としての論理的整合性の両方が必要

**2. アニメーション生成（p5.js）**

動画の内容に基づいてp5.jsアニメーションを生成するタスク。時間的な整合性（temporal consistency）が重要。

- 推奨設定: `media_resolution=HIGH`, `thinking_level=HIGH`
- 理由: フレーム間の視覚的な変化を正確に捉える必要がある

**3. モーメント検索（Moment Retrieval）**

動画内の特定のシーンやイベントを、自然言語クエリで検索するタスク。音声と映像の同期的な理解が求められる。

- 推奨設定: `media_resolution=LOW`, `thinking_level=MEDIUM`
- 理由: 全体的なシーン把握が重要であり、細部の解像度は不要

**4. 時間推論（Temporal Reasoning）**

動画内のイベントの順序、因果関係、カウンティングなどの時間的推論タスク。

- 推奨設定: `media_resolution=MEDIUM`, `thinking_level=HIGH`
- 理由: 時間構造の正確な把握と論理的推論が必要

### 実装時の注意点

**Files API のポーリング間隔**: 大きな動画ファイル（数GB）のアップロード後、ACTIVE 状態になるまで数分かかる場合がある。ポーリング間隔は5秒程度が推奨される。

**レート制限への対応**: 動画処理はトークン消費量が大きいため、短時間に多数のリクエストを送信するとレート制限に達しやすい。指数バックオフ付きリトライが必須。

**音声トラックの活用**: 動画処理では映像だけでなく音声トラックも自動的に処理される。音声を無視したい場合でも、音声トークン分のコストが発生する点に注意。

## 学術研究との関連（Academic Connection）

このブログで紹介されている Gemini 2.5 の動画理解能力は、以下の学術研究の成果を実用化したものと位置づけられる。

- **Gemini 1.0 (arXiv:2312.11805)**: ネイティブマルチモーダル Transformer のアーキテクチャ基盤。動画を画像フレーム＋音声のインターリーブシーケンスとして処理する設計。
- **Gemini 1.5 (arXiv:2403.05530)**: 10Mトークンのコンテキスト長拡張。長尺動画の一括処理を実現。
- **VideoQA 研究**: EgoSchema (Mangalam et al., 2023)、VideoMME (Fu et al., 2024) などのベンチマークが、動画理解モデルの評価基盤として使用されている。
- **Temporal Grounding**: QVHighlights (Lei et al., 2021) に代表される、動画内の時間的位置を特定する研究が、モーメント検索機能の基盤技術となっている。

## まとめと実践への示唆

本ブログは、Gemini 2.5 の動画理解が「映像+音声+コードの統合処理」として初めてネイティブに実現されたことを報告している。Zenn記事で解説されている Gemini 3.1 Pro API の動画処理機能は、この Gemini 2.5 の成果を直接継承したものである。

実装者にとっての重要な示唆は以下の3点である。

1. **media_resolution=LOW でも性能低下はわずか（0.5ポイント）**: コスト優先の場合は積極的に LOW を使用すべき
2. **YouTube URL 直接入力**: Files API を経由しない処理が可能であり、パイプラインの簡素化に寄与する
3. **音声トラックの自動処理**: 映像と音声を同時に理解する能力を活用し、文字起こし+映像分析の統合タスクが効率的に実行可能

## 参考文献

- **Blog URL**: [https://developers.googleblog.com/en/gemini-2-5-video-understanding/](https://developers.googleblog.com/en/gemini-2-5-video-understanding/)
- **Related Papers**: Gemini 1.0 ([arXiv:2312.11805](https://arxiv.org/abs/2312.11805)), Gemini 1.5 ([arXiv:2403.05530](https://arxiv.org/abs/2403.05530))
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/df5295d69a456f](https://zenn.dev/0h_n0/articles/df5295d69a456f)
