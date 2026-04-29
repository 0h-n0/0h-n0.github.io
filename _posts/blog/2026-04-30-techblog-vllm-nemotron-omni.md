---
layout: post
title: "vLLM公式ブログ解説: Nemotron 3 Nano Omniの高効率マルチモーダル推論デプロイ"
description: "vLLM公式ブログからNemotron 3 Nano Omniの推論サーバー構築・EVS設定・スループット最適化の実践手法を詳細解説"
categories: [blog, tech_blog]
tags: [vllm, nvidia, nemotron, multimodal, inference, deployment, agent]
date: 2026-04-30 13:00:00 +0900
source_type: tech_blog
source_domain: vllm.ai
source_url: https://vllm.ai/blog/nemotron-omni
zenn_article: fabaf781f4158d
zenn_url: https://zenn.dev/0h_n0/articles/fabaf781f4158d
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Run Highly Efficient Multimodal Agentic AI with NVIDIA Nemotron 3 Nano Omni Using vLLM](https://vllm.ai/blog/nemotron-omni) の解説記事です。

## ブログ概要（Summary）

vLLMプロジェクトが2026年4月28日に公開したこの公式ブログでは、vLLM v0.20.0によるNemotron 3 Nano Omniの推論サーバー構築方法、マルチモーダル入力の処理パイプライン、Efficient Video Sampling（EVS）の設定、量子化バリアントの選択指針が実践的な形で解説されている。ブログでは、固定インタラクティビティ閾値での比較として動画推論で約9.2倍、マルチドキュメント推論で約7.4倍のスループット向上が報告されている。OpenAI互換APIとしてのデプロイにより、既存のLLMクライアントコードからの移行が容易である点も強調されている。

この記事は [Zenn記事: Nemotron 3 Nano Omniで構築するマルチモーダルAIエージェント実践ガイド](https://zenn.dev/0h_n0/articles/fabaf781f4158d) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: https://vllm.ai/blog/nemotron-omni
- **組織**: vLLMプロジェクト
- **発表日**: 2026年4月28日

## 技術的背景（Technical Background）

vLLM（Very Large Language Model serving engine）は、PagedAttentionをコアとする高スループットLLM推論エンジンである。連続バッチング（continuous batching）により、複数のリクエストを効率的にスケジューリングし、GPU利用率を最大化する。v0.20.0でNemotron 3 Nano Omniの公式サポートが追加され、テキスト・画像・動画・音声の4モダリティ入力をOpenAI互換APIとして提供可能となった。

従来、マルチモーダルモデルの推論サーバー構築には、各モダリティのエンコーダのロード方法、入力前処理パイプラインの構築、出力パーサーの実装など、フレームワーク固有の知識が必要であった。vLLMのNemotron 3 Nano Omniサポートでは、これらがOpenAI互換APIの背後に隠蔽され、`--model`パラメータを指定するだけでマルチモーダル推論が利用可能となっている。

## 実装アーキテクチャ（Architecture）

### サーバー起動パラメータの詳細

vLLMブログで示されているサーバー起動コマンドを分解して解説する。

```bash
pip install "vllm[audio]==0.20.0"

python3 -m vllm.entrypoints.openai.api_server \
    --model "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16" \
    --served-model-name nemotron \
    --trust-remote-code \
    --dtype auto \
    --host 0.0.0.0 \
    --port 5000 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --media-io-kwargs '{"video":{"num_frames":512,"fps":1}}' \
    --video-pruning-rate 0.5 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3
```

各パラメータの技術的意味を以下にまとめる。

| パラメータ | 値 | 技術的意味 |
|---|---|---|
| `--model` | nvidia/Nemotron-...BF16 | HuggingFaceモデルID。BF16/FP8/NVFP4から選択 |
| `--trust-remote-code` | — | モデル固有のカスタムコード実行を許可。Mamba-2カーネル等に必要 |
| `--tensor-parallel-size` | 1 | GPU分割数。H100 80GBならTP=1でBF16が動作 |
| `--max-model-len` | 131072 | コンテキスト長の上限。VRAM消費とトレードオフ |
| `--media-io-kwargs` | JSON | ビデオ入力の設定。num_frames: 抽出フレーム数、fps: サンプリングレート |
| `--video-pruning-rate` | 0.5 | EVSのプルーニング率。0.0（高精度）〜0.7（高速）で調整 |
| `--enable-auto-tool-choice` | — | ツール呼び出しの自動検出を有効化 |
| `--tool-call-parser` | qwen3_coder | ツール呼び出しフォーマットのパーサー |
| `--reasoning-parser` | nemotron_v3 | 推論過程（Chain of Thought）のパーサー |

### Efficient Video Sampling（EVS）の動作原理

ブログでは「EVSにより長時間動画の処理が可能になる」と説明されている。技術的には、以下のパイプラインで動画入力が処理される。

1. **フレーム抽出**: 動画から`num_frames`（デフォルト512）フレームを`fps`（デフォルト1）で抽出
2. **Conv3Dチューブレット埋め込み**: 連続フレームを3D畳み込みで融合し、ビジョントークン数を約50%削減
3. **EVSプルーニング**: 静的なフレーム間のトークンを動的に間引き。`video-pruning-rate`で制御

`--video-pruning-rate` は推論時に調整可能であり、ユースケースに応じた精度・速度トレードオフを実現する。ブログでは明示的な数値は示されていないが、Zenn記事で紹介した以下の指針が適用される。

| pruning rate | 特性 | 推奨ユースケース |
|---|---|---|
| 0.0 | 間引きなし | 微細な動作認識 |
| 0.3 | 軽い間引き | プレゼンテーション分析 |
| 0.5 | バランス（デフォルト） | 会議録画、一般動画 |
| 0.7 | 強い間引き | 監視カメラ、バッチ処理 |

### マルチモーダル入力の処理フロー

vLLMのNemotron 3 Nano Omniバックエンドでは、OpenAI互換APIの`content`フィールドに複数のモダリティを混在させることができる。ブログで示されているクライアントコードは以下の通りである。

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="null")

resp = client.chat.completions.create(
    model="nemotron",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about GPUs."},
    ],
    temperature=1,
    max_tokens=1024,
)
print("Reasoning:", resp.choices[0].message.reasoning,
      "\nContent:", resp.choices[0].message.content)
```

ブログで注目すべき点は、レスポンスオブジェクトに`.message.reasoning`フィールドが含まれることである。`--reasoning-parser nemotron_v3`を指定すると、モデルの思考過程（Chain of Thought）が自動的にパースされ、`reasoning`フィールドに格納される。これにより、推論過程の透明性を確保しつつ、最終回答のみを`.message.content`から取得できる。

### マルチモーダル入力のコード例

ブログでは明示的なマルチモーダルのコード例は示されていないが、vLLMのOpenAI互換APIのドキュメントに基づき、以下の形式が使用可能である。

```python
from openai import OpenAI
import base64
from pathlib import Path

client = OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="null")

image_b64 = base64.b64encode(Path("document.png").read_bytes()).decode()

resp = client.chat.completions.create(
    model="nemotron",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
                {
                    "type": "text",
                    "text": "この文書の内容を要約してください",
                },
            ],
        }
    ],
    temperature=0.2,
    max_tokens=4096,
)
print(resp.choices[0].message.content)
```

動画・音声入力も同様の形式で、`video_url`や`audio_url`タイプを使用してBase64エンコードされたデータを渡すことができる。

## パフォーマンス最適化（Performance）

### スループット比較

ブログでは、固定インタラクティビティ閾値（per-userトークンレートを一定に保持）での比較として以下の数値が報告されている。

| ユースケース | スループット向上 | 意味 |
|---|---|---|
| 動画推論 | 約9.2倍 | 他のオープンオムニモデルと比較した実効システム容量 |
| マルチドキュメント推論 | 約7.4倍 | 同上 |

ブログでは「20%高いマルチモーダルインテリジェンス」という表現も使われている。具体的なトークン/秒やレイテンシの数値は示されていないが、システムレベルでの実効容量が大幅に向上していることが強調されている。

### TPとバッチサイズのトレードオフ

vLLMブログでは直接的な詳細は示されていないが、Zenn記事で解説した以下のトレードオフが適用される。

$$
\text{Throughput} \propto \frac{\text{BS}}{\text{TP}} \quad (\text{大バッチ・少GPU分割がスループット最大化})
$$

$$
\text{Latency} \propto \frac{1}{\text{TP}} \quad (\text{GPU分割数増加がレイテンシ削減})
$$

ここで、BS（Batch Size）は`--max-num-seqs`、TP（Tensor Parallelism）は`--tensor-parallel-size`に対応する。

**レイテンシ重視の構成:**
```bash
--tensor-parallel-size 2 --max-num-seqs 4
```

**スループット重視の構成:**
```bash
--tensor-parallel-size 1 --max-num-seqs 32 --max-model-len 65536
```

GPU間通信のオーバーヘッドにより、TPを上げるとスループットは低下する傾向がある。大量バッチ処理ではTP=1でBSを上げるほうが効率的な場合が多い。

### 対応GPU

ブログで明示的に言及されているGPUは以下の通りである。

| GPU | VRAM | 推奨量子化 | 用途 |
|---|---|---|---|
| B200 | 192GB | BF16 | 最大スループット |
| H100 / H200 | 80GB / 141GB | BF16 / FP8 | 本番環境 |
| A100 | 40GB / 80GB | FP8 | 本番環境（コスト重視） |
| L40S | 48GB | FP8 | 中規模本番 |
| DGX Spark | — | FP8 / NVFP4 | 開発・プロトタイピング |
| RTX 6000 | 48GB | NVFP4 | ローカル開発 |

## 運用での学び（Production Lessons）

### エージェントシステムでの位置づけ

vLLMブログでは、「エージェントシステムにおいて、Nemotron 3 Nano Omniはマルチモーダル知覚と文脈のサブエージェントとして機能する」と説明されている。

この設計パターンでは、Nemotron 3 Nano Omniが以下の役割を担う。

1. **入力の統一処理**: テキスト・画像・動画・音声を単一の表現空間に変換
2. **文脈の構造化**: マルチモーダル入力から構造化された情報（JSON、表、要約）を生成
3. **推論エージェントへの橋渡し**: 構造化された情報をルーターやオーケストレーターに渡す

ブログでは「軽量なアーキテクチャにより、他のモデルと効率的に並行動作できる」と強調されており、マルチエージェント構成での知覚専用モデルとしての利用が推奨されている。

### ツール呼び出しの統合

ブログで紹介されている`--enable-auto-tool-choice`と`--tool-call-parser qwen3_coder`の組み合わせにより、Nemotron 3 Nano Omniはツール呼び出しを自動的に検出・パースできる。これは、マルチモーダル入力を受け取った後、必要に応じて外部ツール（検索API、データベースクエリ等）を呼び出すエージェントワークフローを構築する際に利用される。

### デプロイ環境の選択

ブログでは、以下のデプロイオプションが紹介されている。

| 環境 | 特徴 | 推奨用途 |
|---|---|---|
| vLLM直接デプロイ | 高スループット連続バッチング | 本番APIサーバー |
| NVIDIA Brev Launchable | ワンクリックデプロイ | プロトタイピング・検証 |
| Jupyter Cookbook | 対話的な実験環境 | 開発・実験 |

## 学術研究との関連（Academic Connection）

### vLLMのPagedAttention

vLLMの基盤技術であるPagedAttention（Kwon et al., 2023）は、KVキャッシュをOS仮想メモリのページングと同様に管理することで、メモリの断片化を防ぎGPU利用率を向上させる手法である。Nemotron 3 Nano OmniのAttention層（全体の約6%）のKVキャッシュ管理にこの技術が適用されている。

Mamba SSM層はKVキャッシュを使用しないため、PagedAttentionの恩恵を受ける部分は限定的であるが、少数のAttention層のKVキャッシュを効率的に管理することで、連続バッチングにおけるメモリ効率がさらに向上している。

### マルチモーダル推論の最適化

vLLMのマルチモーダル処理パイプライン（BaseMultiModalProcessor）は、チャンク化されたプリフィルとプレフィックスキャッシングを組み合わせて、マルチモーダル入力のトークン化と前処理を効率化している。同一システムプロンプトを使用する複数リクエストでは、プレフィックスキャッシングにより初回以降の処理が高速化される。

## まとめと実践への示唆

vLLMブログは、Nemotron 3 Nano Omniの実践的なデプロイ手順と最適化手法を提供する重要な1次情報源である。OpenAI互換APIによるシームレスな統合、EVSによる動画処理の効率化、reasoningフィールドによる推論過程の透明化という3つの実用的な特徴が、Zenn記事で紹介したエージェント構築の技術的基盤を形成している。ブログで報告されている動画推論9.2倍・ドキュメント推論7.4倍のスループット向上は、マルチモーダルエージェントの本番運用を経済的に現実的なものにしている。

## 参考文献

- **Blog URL**: https://vllm.ai/blog/nemotron-omni
- **vLLM Documentation**: https://docs.vllm.ai/en/stable/
- **Related Papers**: PagedAttention (https://arxiv.org/abs/2309.06180)
- **Related Zenn article**: https://zenn.dev/0h_n0/articles/fabaf781f4158d
- **NVIDIA Nemotron Models**: https://huggingface.co/nvidia
