---
layout: post
title: "BAIR Blog解説: The Shift from Models to Compound AI Systems — モノリシックモデルから複合AIへの転換"
description: "UC Berkeley BAIRが提唱するCompound AI Systemsの概念。AlphaCode 2、Medprompt等の事例と設計課題を詳細解説"
categories: [blog, tech_blog]
tags: [compound-ai, architecture, LLM, optimization, mlops, ai, devops]
date: 2026-03-14 12:00:00 +0900
source_type: tech_blog
source_domain: bair.berkeley.edu
source_url: https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/
zenn_article: 7b88993fccf7f8
zenn_url: https://zenn.dev/0h_n0/articles/7b88993fccf7f8
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [The Shift from Models to Compound AI Systems](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)（BAIR Blog、2024年2月18日公開）の解説記事です。

## ブログ概要（Summary）

UC Berkeley AIリサーチ（BAIR）グループが2024年2月に公開した、Compound AI Systemsの概念を提唱するブログ記事である。著者のMatei Zaharia（Databricks共同創業者）らは、最先端のAI成果がもはや単一モデルではなく**複数コンポーネントの協調システム**によって達成されていると主張している。AlphaCode 2、AlphaGeometry、Medprompt等の具体例を挙げ、なぜCompound AIシステムが優位なのか、そしてその設計・最適化・運用における技術的課題を整理している。

この記事は [Zenn記事: AIソフトウェアアーキテクチャ2026年版：MLOps・LLMOps・AgentOpsの実践設計](https://zenn.dev/0h_n0/articles/7b88993fccf7f8) の深掘りです。

## 情報源

- **種別**: 大学研究グループブログ
- **URL**: [https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)
- **組織**: Berkeley Artificial Intelligence Research (BAIR)
- **著者**: Matei Zaharia, Omar Khattab, Lingjiao Chen, Jared Quincy Davis, Heather Miller, Chris Potts, James Zou, Michael Carbin, Jonathan Frankle, Naveen Rao, Ali Ghodsi
- **発表日**: 2024年2月18日

## 技術的背景（Technical Background）

2023年までのAI開発は「より大きなモデルを訓練すれば性能が向上する」というスケーリング法則に支配されていた。しかし、著者らはこの前提に疑問を呈し、**エンジニアリングによるシステム的な改善**が、モデルスケーリングよりも迅速かつ効率的に性能を向上させるケースが増えていることを指摘している。

## Compound AI Systemsの定義

著者らはCompound AI Systemsを以下のように定義している。

> "A system that tackles AI tasks using multiple interacting components, including multiple calls to models, retrievers, or external tools."

単一のモノリシックモデルとは対照的に、**複数のモデル・検索エンジン・外部ツール・コード実行環境**が相互作用するシステムである。

## なぜCompound AI Systemsが優位なのか

著者らは4つの理由を挙げている。

### 1. スケーリングよりエンジニアリングが速い

モデルの再訓練には数ヶ月と膨大なコストがかかる。一方、Compound AIシステムのエンジニアリング的改善は数日〜数週間で実現可能である。

著者らが挙げた例として、コーディングベンチマークで30%のベースライン性能を、サンプリングとテスト実行の組み合わせ（Compound AIアプローチ）により80%まで向上させたケースがある。

### 2. 動的な知識統合

訓練済みモデルの知識は静的であり、最新情報を反映できない。検索コンポーネント（RAG）を組み込むことで、リアルタイムの情報アクセスとユーザー固有データへのアクセス制御が可能になる。

### 3. 制御性と信頼性の向上

システムレベルで出力フィルタリング、事実確認（citationベースの検証）、ハルシネーション検出を実装できる。単一モデルの内部動作に依存するよりも、外部コンポーネントによる品質保証が信頼性を高める。

### 4. パフォーマンス要件の多様性

アプリケーションによってコスト・レイテンシ・品質のバランスが異なる。GitHub Copilotのように「入念にチューニングされた小型モデル」が最適な場合もあれば、複雑な推論タスクで高コストを許容する場合もある。

## 代表的なCompound AI Systemの事例

著者らが紹介している具体的なシステムとその構成を以下に整理する。

| システム | コンポーネント構成 | 達成性能 |
|---------|----------------|---------|
| **AlphaCode 2** | ファインチューンLLM + コード実行 + クラスタリング | コーディングコンテスト85パーセンタイル |
| **AlphaGeometry** | ファインチューンLLM + 記号数学エンジン | 国際数学オリンピック銀-金メダルレベル |
| **Medprompt** | GPT-4 + 最近傍探索 + Chain-of-Thought | 専用Med-PaLMモデルを上回る |
| **Gemini MMLU** | LLM + CoT@32推論 | 90.04%（GPT-4 5-shotの86.4%を上回る） |
| **RAGシステム** | LLM + 検索コンポーネント | 企業・検索で広く展開 |

**AlphaCode 2**の構成は特に示唆的である。単一のコード生成モデルではなく、以下のパイプラインで構成されている。

```mermaid
graph LR
    A[問題入力] --> B[コード生成 LLM × N回]
    B --> C[コード実行・テスト]
    C --> D[結果クラスタリング]
    D --> E[最良候補選択]
    E --> F[提出コード]
```

この「大量サンプリング→実行フィルタリング→選択」というパターンは、単一モデルの1回の推論では到達不可能な品質を、システムレベルの設計で実現している。

### Medpromptの設計

著者らの紹介によると、MicrosoftのMedpromptは以下の構成で医療QAベンチマークで専用モデルを上回っている。

1. **GPT-4**: 汎用LLM（医療ドメインでの追加訓練なし）
2. **最近傍探索**: 質問に類似した過去の事例を検索
3. **Chain-of-Thought**: 検索事例を元にステップバイステップで推論
4. **Ensemble**: 複数回推論して多数決

$$
\text{Medprompt}(q) = \text{majority\_vote}\left(\left\{f_{CoT}(q, \text{NN}(q, \mathcal{D}))_i\right\}_{i=1}^{K}\right)
$$

ここで、
- $q$: 医療質問
- $\mathcal{D}$: 事例データベース
- $\text{NN}(q, \mathcal{D})$: $q$の最近傍事例
- $f_{CoT}$: Chain-of-Thought推論
- $K$: アンサンブル回数

## 技術的課題

### 設計空間の爆発

Compound AIシステムでは、コンポーネント選択、リソース配分（レイテンシ・コスト予算の配分）、テクニックの組み合わせといった膨大な設計決定が必要になる。

### 非微分可能コンポーネントの最適化

検索エンジンやコード実行環境は微分可能でないため、勾配ベースのエンドツーエンド最適化が適用できない。著者らは**DSPy**を解決策として紹介している。DSPyはLLMの「言語能力」を活用してプロンプトとパラメータを自動チューニングするフレームワークである。

### コスト最適化

著者らは**FrugalGPT**（Chen et al., 2023）を紹介している。FrugalGPTはモデルカスケード（安いモデル→必要に応じて高いモデルにエスカレーション）により、「品質4%改善」または「コスト90%削減」を達成したと報告されている。

### オブザーバビリティ

Compound AIシステムでは入力ごとに実行パスが異なるため、従来の単純なメトリクス監視では不十分である。LangSmith、Phoenix Traces等のトレーシングツールにより、中間出力の品質やデータパイプラインとの相関を追跡する必要がある。

## 実装アーキテクチャ（Architecture）

### 開発パラダイムの分類

著者らはCompound AIシステムの開発アプローチを以下の3つに分類している。

**1. フレームワークアプローチ**: LangChain、LlamaIndex等のツールでコンポーネントを手動で接続
**2. エンドツーエンド最適化**: DSPyのようなコンパイラでシステム全体を自動最適化
**3. 制約ベース**: Guardrails、LMQL、SGLangでLLM出力を構造化

### DSPyによる自動最適化

DSPyは「自然言語のシグネチャ」でAIタスクを記述し、プロンプト生成・few-shot選択・パイプライン構成を自動化する。

```python
import dspy

# タスク定義（シグネチャ）
class QAWithRetrieval(dspy.Module):
    """質問に対して検索ベースで回答するモジュール"""

    def __init__(self) -> None:
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        """検索→推論のパイプライン"""
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)
```

DSPyのコンパイラは、少数の評価データからプロンプトとパイプラインパラメータを自動最適化する。

## パフォーマンス最適化（Performance）

著者らが紹介した各システムの性能向上をまとめる。

| システム | ベースライン（単一モデル） | Compound AI | 改善 |
|---------|----------------------|------------|------|
| Gemini MMLU | 86.4%（GPT-4 5-shot） | 90.04%（CoT@32） | +3.64% |
| Medprompt | Med-PaLM 2（専用モデル） | GPT-4 + RAG + CoT | 汎用モデルが専用モデルを上回る |
| AlphaCode 2 | 単一コード生成 | 生成+実行+選択 | 85パーセンタイル達成 |
| FrugalGPT | GPT-4のみ | カスケード | コスト90%削減 or 品質4%改善 |

## 運用での学び（Production Lessons）

著者らのブログから読み取れる運用上の知見を以下に整理する。

1. **Compound AIシステムはモデルが改善されても有効**: 「より良いモデルが出ればシステム的な工夫は不要になる」という議論に対し、著者らはモデル改善とシステム改善は**相補的**であると主張。モデルが改善されても、検索・実行・アンサンブルの組み合わせはさらなる性能向上をもたらす
2. **設計決定の文書化が不可欠**: コンポーネント選択の理由、トレードオフの判断基準を記録しないと、後からのデバッグや最適化が困難になる
3. **オブザーバビリティへの投資が早期に必要**: トレーシングツールの導入は後回しにされがちだが、システムが複雑化してからの導入はコストが高い

## 学術研究との関連（Academic Connection）

- **DSPy**（Khattab et al., Stanford NLP、2023）: Compound AIシステムのプログラミングモデルとコンパイラ。著者Omar Khattabは本ブログ記事の共著者でもある
- **FrugalGPT**（Chen et al., Stanford, 2023）: LLMカスケードによるコスト最適化の先行研究
- **RouteLLM**（Ong et al., UC Berkeley/Anyscale, ICLR 2025）: 嗜好データからのルーター学習。FrugalGPTの発展形

## まとめと実践への示唆

BAIRグループの著者らは、2024年以降のAI開発が「モデルからシステムへ」のパラダイムシフトの途上にあると主張している。この転換は、MLOps → LLMOps → AgentOps の進化と軌を一にしており、Zenn記事で解説した8つの構成要素（Data Estate〜Governance）はCompound AIシステムの各レイヤーに対応する。実務的には、まずRAGパイプラインやモデルルーティングといった「確立されたパターン」から導入し、段階的にシステムの複雑度を上げることが推奨される。

## 参考文献

- **Blog URL**: [https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)
- **DSPy**: [https://github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
- **FrugalGPT**: Chen et al., 2023
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/7b88993fccf7f8](https://zenn.dev/0h_n0/articles/7b88993fccf7f8)
