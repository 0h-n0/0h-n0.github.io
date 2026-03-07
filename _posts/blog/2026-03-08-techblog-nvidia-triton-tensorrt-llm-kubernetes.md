---
layout: post
title: "NVIDIA技術ブログ解説: Triton + TensorRT-LLMによるKubernetes上のLLMスケーリング"
description: "NVIDIA Triton Inference ServerとTensorRT-LLMをKubernetesで運用し、HPAによる自動スケーリングとqueue-to-compute比メトリクスで推論負荷に追従する構成を解説"
categories: [blog, tech_blog]
tags: [NVIDIA, Triton, TensorRT-LLM, Kubernetes, GPU, LLM, autoscaling, HPA, Prometheus]
date: 2026-03-08 09:10:00 +0900
source_type: tech_blog
source_domain: developer.nvidia.com
source_url: https://developer.nvidia.com/blog/scaling-llms-with-nvidia-triton-and-nvidia-tensorrt-llm-using-kubernetes/
zenn_article: 3a91fb8a02cdc4
zenn_url: https://zenn.dev/0h_n0/articles/3a91fb8a02cdc4
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [Scaling LLMs with NVIDIA Triton and NVIDIA TensorRT-LLM Using Kubernetes](https://developer.nvidia.com/blog/scaling-llms-with-nvidia-triton-and-nvidia-tensorrt-llm-using-kubernetes/)（NVIDIA Developer Blog、2024年10月22日公開、2025年3月18日最終更新）の解説記事です。

## ブログ概要（Summary）

NVIDIAのMaggie Zhang氏らが公開したこの技術ブログは、NVIDIA Triton Inference Server（現NVIDIA Dynamo Triton）とTensorRT-LLMをKubernetes上で運用し、Prometheusメトリクスに基づくHPA（Horizontal Pod Autoscaler）で推論Pod数を自動調整する構成を解説している。特徴的なのは「queue-to-compute ratio」というカスタムメトリクスを導入し、推論リクエストのキュー待ち時間と実計算時間の比率でスケーリング判断を行う点である。

この記事は [Zenn記事: Ollama本番運用ガイド：Kubernetes・認証・監視で構築するオンプレLLM基盤](https://zenn.dev/0h_n0/articles/3a91fb8a02cdc4) の深掘りです。Zenn記事ではOllamaのHPAをCPU/メモリ使用率で設定しているが、本ブログではLLM推論に特化したカスタムメトリクスによる、より精度の高い自動スケーリング手法を提案している。

## 情報源

- **種別**: 企業テックブログ（NVIDIA Developer）
- **URL**: [https://developer.nvidia.com/blog/scaling-llms-with-nvidia-triton-and-nvidia-tensorrt-llm-using-kubernetes/](https://developer.nvidia.com/blog/scaling-llms-with-nvidia-triton-and-nvidia-tensorrt-llm-using-kubernetes/)
- **組織**: NVIDIA
- **著者**: Maggie Zhang, J Wyman, Indrajit Maloji Bhosale, Wenhan Tan
- **発表日**: 2024年10月22日（最終更新: 2025年3月18日）

## 技術的背景（Technical Background）

LLM推論のスケーリングには、従来のWebサービスとは異なる課題がある。LLMの推論はPrefill（プロンプト処理）とDecode（トークン生成）の2フェーズで構成され、リクエストごとの処理時間が出力トークン数に比例して大きく変動する。CPU使用率やメモリ使用率といった汎用メトリクスでは、推論リクエストのキューイング状態を正確に把握できない。

Zenn記事ではOllamaのHPAをCPU使用率70%・メモリ使用率80%で設定しているが、これはGPU上で動作するLLM推論の負荷を間接的にしか反映しない。NVIDIAのブログでは、Triton Inference Serverが提供する推論固有のメトリクスを活用することで、この問題に対処している。

## 実装アーキテクチャ（Architecture）

### 3層構成

ブログで解説されているアーキテクチャは以下の3コンポーネントで構成される。

```mermaid
graph TB
    Client[クライアント] -->|HTTP/gRPC| Triton[Triton Inference Server<br>Port 8000/8001]
    Triton -->|TensorRT Engine| TRT[TensorRT-LLM<br>最適化エンジン]
    TRT --> GPU[NVIDIA GPU<br>A10G/H100]
    Triton -->|Port 8002| Metrics[/metrics<br>Prometheusエンドポイント]
    Metrics --> Prom[Prometheus]
    Prom --> Adapter[Prometheus Adapter]
    Adapter --> HPA[HPA<br>queue-to-compute ratio]
    HPA -->|Pod数調整| Triton
```

| コンポーネント | 役割 | ポート |
|-------------|------|-------|
| Triton Inference Server | 推論サーバー（モデルサービング） | 8000 (HTTP), 8001 (gRPC), 8002 (metrics) |
| TensorRT-LLM | モデル最適化（量子化、カーネル融合） | - |
| Prometheus | メトリクス収集 | 9090 |
| Prometheus Adapter | HPAへのメトリクス変換 | 443 |
| HPA | Pod自動スケーリング | - |

### モデル準備ワークフロー

ブログでは、モデルのデプロイまでに以下のステップが必要とされている。

1. **HuggingFaceからモデルチェックポイントをダウンロード**
2. **TensorRTエンジンファイルを生成**（NGC Containerイメージを使用）
3. **Tensor Parallelism（TP）/Pipeline Parallelism（PP）の設定**
4. **カスタムDockerイメージの作成**（最適化済みエンジンを含む）
5. **Helmチャートでデプロイ**

この前処理ステップはOllamaと大きく異なる。Ollamaでは`models.pull`でモデル名を指定するだけでダウンロード・最適化が自動で行われるが、Triton+TensorRT-LLMではモデルの最適化を明示的に行う必要がある。その分、推論性能は大幅に向上する。

### Kubernetesデプロイメント構成

ブログで紹介されているデプロイメントの主要設定は以下の通りである。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-llm-server
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: triton
          image: custom-triton-tensorrt-llm:latest
          ports:
            - containerPort: 8000
              name: http
            - containerPort: 8001
              name: grpc
            - containerPort: 8002
              name: metrics
          resources:
            limits:
              nvidia.com/gpu: 1
              ephemeral-storage: "50Gi"
          readinessProbe:
            httpGet:
              path: /v2/health/ready
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 10
```

**Zenn記事のOllamaデプロイとの比較**:

| 項目 | Triton+TensorRT-LLM | Ollama |
|------|---------------------|--------|
| ポート数 | 3（HTTP/gRPC/metrics） | 1（11434） |
| メトリクス | 内蔵（/metrics） | ollama-metrics Exporter必要 |
| ヘルスチェック | `/v2/health/ready` | `/` on 11434 |
| モデル最適化 | TensorRTエンジン事前生成 | ランタイム自動最適化 |
| 起動時間 | 2-5分（エンジンロード） | 30秒-2分（モデルロード） |

## パフォーマンス最適化（Performance）

### queue-to-compute ratioによるスケーリング

ブログの核心は、LLM推論に特化したカスタムメトリクス「queue-to-compute ratio」である。

$$
R_{\text{queue-to-compute}} = \frac{T_{\text{queue}}}{T_{\text{compute}}}
$$

ここで、$T_{\text{queue}}$はリクエストがキューで待機した時間、$T_{\text{compute}}$は実際の推論処理にかかった時間である。この比率が1を超えると、リクエストの待ち時間が処理時間を上回っており、スケールアウトが必要であることを示す。

**PrometheusRuleでの計算**:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: triton-queue-compute-ratio
spec:
  groups:
    - name: triton.rules
      rules:
        - record: triton:queue_to_compute_ratio
          expr: |
            rate(nv_inference_queue_duration_us_sum[1m])
            / rate(nv_inference_compute_infer_duration_us_sum[1m])
```

**HPAの設定**:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: triton-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: triton-llm-server
  minReplicas: 1
  maxReplicas: 4
  metrics:
    - type: Pods
      pods:
        metric:
          name: triton_queue_to_compute_ratio
        target:
          type: AverageValue
          averageValue: "1000m"  # 比率1.0をターゲット
```

### Zenn記事のOllama HPAとの比較

Zenn記事ではOllamaのHPAをCPU/メモリ使用率で設定しているが、このアプローチには以下の制約がある。

| メトリクス | 利点 | 制約 |
|-----------|------|------|
| CPU使用率 | 汎用的、追加設定不要 | GPU推論負荷を反映しない |
| メモリ使用率 | 汎用的 | LLMはモデルロード時に一定値で安定 |
| queue-to-compute ratio | 推論負荷を直接反映 | Prometheus Adapter設定が必要 |

OllamaではPrometheusメトリクスが内蔵されていないため、ollama-metricsを使って以下の近似的なスケーリングメトリクスを構築できる。

$$
R_{\text{ollama}} \approx \frac{\text{rate}(\text{ollama\_request\_duration\_seconds\_sum}[5m])}{\text{rate}(\text{ollama\_request\_duration\_seconds\_count}[5m])}
$$

この値はリクエストあたりの平均処理時間を示し、閾値を超えた場合にスケールアウトのトリガーとして利用できる。

### TensorRT-LLMの最適化技術

ブログで紹介されているTensorRT-LLMの最適化技術は以下の通りである。

| 最適化技術 | 効果 | Ollamaとの比較 |
|-----------|------|---------------|
| カーネル融合 | 複数GPU演算を1カーネルに統合 | Ollamaはllama.cppベースで限定的 |
| 量子化（INT8/FP8） | メモリ使用量50-75%削減 | Ollamaも量子化対応（GGUF形式） |
| In-Flight Batching | リクエスト単位の動的バッチング | Ollamaは`OLLAMA_NUM_PARALLEL`で制御 |
| PagedAttention | KVキャッシュのページング管理 | Ollamaも0.17以降で対応 |
| Tensor Parallelism | 複数GPU間でのモデル分割 | Ollamaは非対応（単一GPU） |

### 自動スケーリングの実証

ブログでは、クライアントレプリカを1から10に増加させた際の動作が示されている。

1. **リクエスト増加**: クライアント10台からの同時リクエスト
2. **queue-to-compute比が上昇**: キュー待ち時間が増大
3. **HPAがスケールアウト**: サーバーPodが1から4に増加
4. **比率が正常化**: 追加Podにリクエストが分散
5. **リクエスト減少後**: サーバーPodが1に戻る

## 運用での学び（Production Lessons）

### GPU Feature DiscoveryとDCGM Exporter

ブログでは、Kubernetes上でGPUを管理するための3つのコンポーネントが紹介されている。

1. **NVIDIA Device Plugin**: GPUノードの検出と`nvidia.com/gpu`リソースの公開
2. **GPU Feature Discovery**: GPUの性能特性（アーキテクチャ、メモリ量）のラベル付け
3. **DCGM Exporter**: GPU使用率・温度・電力のPrometheusメトリクス提供

これらはZenn記事で解説されているNVIDIA GPU Operatorに含まれるコンポーネントであり、Ollamaの運用でもそのまま活用できる。特にDCGM Exporterは、Zenn記事の監視アーキテクチャ図に含まれている`DCGM_FI_DEV_GPU_UTIL`や`DCGM_FI_DEV_GPU_TEMP`メトリクスの提供元である。

### PodMonitorによるメトリクス自動検出

Triton Inference Serverのメトリクスは、PodMonitor CRDで自動検出される。

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: triton-podmonitor
spec:
  selector:
    matchLabels:
      app: triton-llm-server
  podMetricsEndpoints:
    - port: metrics
      interval: 6s
      path: /metrics
```

6秒間隔のスクレイプは、Zenn記事のollama-metricsの15秒間隔より短い。LLM推論のバースト的な負荷変動を捉えるには、短い間隔が望ましいが、Prometheusのストレージ消費とのトレードオフを考慮する必要がある。

### エンタープライズLLMサービングの選択基準

ブログの構成を踏まえると、LLMサービングフレームワークの選択は以下の基準で判断できる。

| 要件 | 推奨フレームワーク | 理由 |
|------|-------------------|------|
| 簡単セットアップ | Ollama | モデル名指定のみ、Helm Chart完備 |
| 最大スループット | Triton + TensorRT-LLM | カーネル融合・TP対応、HPA統合 |
| 中間的な選択 | vLLM | PagedAttention、OpenAI互換API |
| エンタープライズサポート | Red Hat AI Inference Server | 商用サポート付きvLLMバリアント |

## 学術研究との関連（Academic Connection）

Triton Inference Serverのバッチング戦略は、Yu et al.の "[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)"（OSDI 2022）で提案されたIteration-Level Schedulingに基づいている。従来のリクエストレベルバッチングでは、短いリクエストが長いリクエストの完了を待つ「head-of-line blocking」が発生するが、イテレーションレベルのスケジューリングにより各デコードステップで新しいリクエストを挿入できる。

TensorRT-LLMのPagedAttention実装は、Kwon et al.の "[Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)"（SOSP 2023）を基盤としている。

## まとめと実践への示唆

NVIDIAのこのブログは、エンタープライズ規模のLLMサービングにおけるKubernetes自動スケーリングの設計パターンを示している。Zenn記事でOllamaの本番運用を検討するエンジニアにとっての主要な示唆は以下の通りである。

1. **queue-to-compute ratioは汎用的な概念**であり、ollama-metricsの`ollama_request_duration_seconds`を使って類似のメトリクスを構築できる
2. **Tritonの3ポート構成**（HTTP/gRPC/metrics）は、Ollamaの単一ポート構成と比較してメトリクス収集が容易だが、ollama-metricsのSidecarパターンで同等の監視が実現可能
3. **TensorRT-LLMの事前最適化**はセットアップコストが高いが、スループットが重要な大規模環境では検討に値する。Ollamaが適する中小規模環境との使い分けが重要

## 参考文献

- **Blog URL**: [https://developer.nvidia.com/blog/scaling-llms-with-nvidia-triton-and-nvidia-tensorrt-llm-using-kubernetes/](https://developer.nvidia.com/blog/scaling-llms-with-nvidia-triton-and-nvidia-tensorrt-llm-using-kubernetes/)
- **Related Papers**: [Orca: OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu), [PagedAttention: SOSP 2023](https://arxiv.org/abs/2309.06180)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/3a91fb8a02cdc4](https://zenn.dev/0h_n0/articles/3a91fb8a02cdc4)
