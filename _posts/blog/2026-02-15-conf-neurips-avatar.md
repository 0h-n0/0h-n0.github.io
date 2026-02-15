---
layout: post
title: "カンファレンス論文解説: AvaTaR - LLMエージェントのツール使用最適化"
description: "NeurIPS 2024採択論文：対比推論によるエージェント性能改善手法の詳細解説"
categories: [blog, conference, neurips]
tags: [LLM, agent, tool-use, optimization, contrastive-learning]
date: 2026-02-15 12:00:00 +0900
source_type: conference
conference: NeurIPS 2024
source_url: https://arxiv.org/abs/2406.11200
zenn_article: a32342e48355ae
zenn_url: https://zenn.dev/0h_n0/articles/a32342e48355ae
target_audience: "修士学生レベル"
---

# カンファレンス論文解説: AvaTaR - LLMエージェントのツール使用最適化

## 論文概要

**AvaTaR: Optimizing LLM Agents for Tool Usage via Contrastive Reasoning** (NeurIPS 2024) は、Stanford University と Amazon によるLLMエージェントのツール使用能力を **対比推論** で最適化する手法の提案論文です。従来のプロンプトエンジニアリングが試行錯誤的だったのに対し、AvaTaRは **成功例と失敗例の対比分析** により、体系的にエージェント性能を改善します。

**著者**: Shirley Wu et al. (Stanford University, Amazon)

**主要な貢献**:
- **Actor-Comparator アーキテクチャ**: エージェント（Actor）と評価者（Comparator）の2 LLM構成
- **対比推論メカニズム**: 成功例と失敗例を比較し、体系的なエラーパターンを特定
- **7タスクで検証**: マルチモーダル検索（4タスク）とQA（3タスク）で平均14%性能向上
- **DSPy統合**: DSPyフレームワークに `AvatarModule` として統合済み

**実験結果の核心**:
- **Hit@1メトリック**: 検索タスクで平均14%向上、QAタスクで13%向上
- **強い汎化性能**: 新規タスクへの転移学習で性能維持
- **自動化**: 人手によるプロンプト設計が不要

---

## 技術的背景

### LLMエージェントのツール使用課題

LLMエージェントがツール（API、データベース、検索エンジン）を効果的に使用するには、以下の課題があります:

1. **ツール選択**: 複数のツールから適切なものを選ぶ
2. **パラメータ構築**: ツールに渡す引数を正しく生成
3. **結果統合**: ツールの出力を最終回答に統合
4. **エラー処理**: ツール失敗時のリカバリ戦略

**従来手法の限界**:
- **Few-shot prompting**: 例示が少ないと汎化しない
- **手動最適化**: タスクごとにプロンプトを調整（時間コスト）
- **試行錯誤的**: 体系的な改善戦略がない

### AvaTaRのアプローチ

AvaTaRは **Contrastive Reasoning** （対比推論）を導入し、以下を自動化:

```
1. 正例・負例のサンプリング（成功/失敗タスク）
2. 対比分析（何が違うのか？）
3. 改善指示生成（どう直すべきか？）
4. Actor LLMへのフィードバック
```

**キーアイデア**: 成功例だけでなく **失敗例** からの学習が効果的。

---

## Actor-Comparator アーキテクチャ

### 2 LLM構成

AvaTaRは以下の2つのLLMで構成されます:

#### 1. Actor LLM

**役割**: 実際のタスクを実行するエージェント

**入力**:
- タスク記述（例: "Find papers about transformers"）
- 利用可能ツールリスト（例: `search_arxiv`, `search_semantic_scholar`）
- Comparatorからの改善指示

**出力**:
- ツール呼び出しシーケンス
- 最終回答

**実装例**:
```python
class ActorLLM:
    def __init__(self, model="gpt-4", tools=[]):
        self.model = model
        self.tools = tools
        self.feedback_history = []

    def execute(self, task, feedback=None):
        # システムプロンプト構築
        system_prompt = f"""
        You are an agent with access to these tools:
        {self.format_tools(self.tools)}

        {feedback if feedback else ""}

        Execute the task step-by-step:
        """

        # タスク実行
        response = self.model.generate(
            system=system_prompt,
            user=task
        )

        # ツール呼び出し解析・実行
        actions = self.parse_actions(response)
        results = self.run_tools(actions)

        return {
            "actions": actions,
            "results": results,
            "final_answer": self.extract_answer(results)
        }
```

#### 2. Comparator LLM

**役割**: Actor の成功例・失敗例を対比分析し、改善指示を生成

**入力**:
- 成功タスク群（$\mathcal{D}^+$）
- 失敗タスク群（$\mathcal{D}^-$）
- Actor の実行トレース（ツール呼び出し履歴）

**出力**:
- 体系的エラーパターンの特定
- 改善指示（具体的なプロンプト）

**実装例**:
```python
class ComparatorLLM:
    def __init__(self, model="gpt-4-turbo"):
        self.model = model

    def analyze(self, positive_samples, negative_samples):
        # 対比分析プロンプト
        prompt = f"""
        Compare the following successful and failed task executions:

        SUCCESSFUL CASES:
        {self.format_samples(positive_samples)}

        FAILED CASES:
        {self.format_samples(negative_samples)}

        Identify systematic differences:
        1. What tools are used differently?
        2. What parameters are incorrect in failures?
        3. What retrieval strategies work better?

        Provide concrete guidance for improving the agent.
        """

        feedback = self.model.generate(prompt)
        return feedback
```

---

## 対比推論メカニズム

### アルゴリズムフロー

```
Input: 訓練データ D = {(task_i, ground_truth_i)}
Output: 最適化されたActor LLM

1. Initialize Actor with base prompt
2. For iteration t = 1 to T:
    a. Sample positive set D^+ (Actor成功)
    b. Sample negative set D^- (Actor失敗)
    c. Comparator.analyze(D^+, D^-)
    d. feedback_t = Comparator.generate_feedback()
    e. Actor.update_prompt(feedback_t)
    f. Evaluate Actor on validation set
    g. If performance plateaus, stop
3. Return optimized Actor
```

### 正例・負例のサンプリング戦略

**正例 $\mathcal{D}^+$**: Actor が正解を出力したタスク

```python
def sample_positive(dataset, actor, k=10):
    positive_samples = []
    for task, ground_truth in dataset:
        result = actor.execute(task)
        if is_correct(result["final_answer"], ground_truth):
            positive_samples.append({
                "task": task,
                "trace": result["actions"],
                "answer": result["final_answer"]
            })
    return random.sample(positive_samples, min(k, len(positive_samples)))
```

**負例 $\mathcal{D}^-$**: Actor が誤答を出力したタスク

```python
def sample_negative(dataset, actor, k=10):
    negative_samples = []
    for task, ground_truth in dataset:
        result = actor.execute(task)
        if not is_correct(result["final_answer"], ground_truth):
            negative_samples.append({
                "task": task,
                "trace": result["actions"],
                "answer": result["final_answer"],
                "error_type": classify_error(result, ground_truth)
            })
    return random.sample(negative_samples, min(k, len(negative_samples)))
```

### 対比分析の実例

**タスク**: "Find papers about Vision Transformers published in 2023"

#### 成功例の分析

```
Positive Case 1:
  Actions: [
    search_arxiv(query="Vision Transformer", year=2023),
    filter_by_citations(threshold=50),
    summarize_papers()
  ]
  Result: Correct (5 relevant papers)

Positive Case 2:
  Actions: [
    search_semantic_scholar(query="ViT image classification", year=2023),
    rank_by_relevance(),
    extract_metadata()
  ]
  Result: Correct (7 relevant papers)
```

**パターン**: 検索クエリが具体的、年フィルタを正しく使用

#### 失敗例の分析

```
Negative Case 1:
  Actions: [
    search_arxiv(query="transformer"),  # ← too broad
    summarize_papers()
  ]
  Result: Incorrect (included NLP papers)

Negative Case 2:
  Actions: [
    search_semantic_scholar(query="Vision Transformer 2023")  # ← year in query, not filter
  ]
  Result: Incorrect (query string matching failed)
```

**エラーパターン**:
1. クエリが曖昧（"transformer" vs "Vision Transformer"）
2. 年フィルタをクエリ文字列に含める（正しくはパラメータで指定）

### Comparatorの改善指示生成

上記の対比から、Comparatorは以下のフィードバックを生成:

```
FEEDBACK:

When searching for papers:
1. Use specific query terms (e.g., "Vision Transformer" not "transformer")
2. Always use year parameter (e.g., year=2023) instead of including year in query string
3. Apply citation threshold (threshold=50) to filter high-quality papers
4. Prefer search_arxiv for academic papers; search_semantic_scholar for broader coverage

Common mistakes to avoid:
- Broad queries that retrieve irrelevant results
- Mixing query parameters with query text
- Skipping result filtering steps
```

このフィードバックをActorのシステムプロンプトに追加します。

---

## 実験結果

### 評価タスク

論文は7つのタスクで評価しました:

#### マルチモーダル検索タスク（4つ）

1. **Zero-shot ImageNet**: 画像分類（ツール: CLIP, ResNet）
2. **Winoground**: 視覚的推論（ツール: VQA, BLIP）
3. **Flickr30k**: 画像-テキストマッチング
4. **COCO Captions**: キャプション生成

#### QAタスク（3つ）

5. **HotpotQA**: マルチホップ推論（ツール: Wikipedia API）
6. **FEVER**: 事実検証（ツール: Google Search, Wikidata）
7. **StrategyQA**: 戦略的推論

### 性能比較

#### マルチモーダル検索タスク（Hit@1）

| 手法 | ImageNet | Winoground | Flickr30k | COCO | 平均 |
|------|----------|-----------|-----------|------|------|
| Few-shot (GPT-4) | 68.3% | 52.1% | 71.5% | 65.9% | 64.5% |
| AutoGPT | 72.1% | 54.8% | 73.2% | 68.3% | 67.1% |
| ReAct | 74.5% | 57.3% | 75.8% | 70.2% | 69.5% |
| **AvaTaR** | **82.7%** | **64.9%** | **84.1%** | **78.5%** | **77.6%** |

**改善率**: 平均 +14% (69.5% → 77.6%)

#### QAタスク（Exact Match）

| 手法 | HotpotQA | FEVER | StrategyQA | 平均 |
|------|----------|-------|-----------|------|
| Few-shot | 45.2% | 62.8% | 58.3% | 55.4% |
| Chain-of-Thought | 51.3% | 67.1% | 61.2% | 59.9% |
| ReAct | 54.7% | 69.5% | 63.8% | 62.7% |
| **AvaTaR** | **61.2%** | **76.3%** | **70.1%** | **69.2%** |

**改善率**: 平均 +13% (62.7% → 69.2%)

### アブレーション研究

#### Comparatorの効果

| 構成 | HotpotQA | 改善率 |
|------|----------|--------|
| Actor のみ（Few-shot） | 45.2% | ベースライン |
| Actor + 手動フィードバック | 52.8% | +7.6% |
| Actor + **Comparator（対比なし）** | 56.3% | +11.1% |
| Actor + **Comparator（対比あり）** | **61.2%** | **+16.0%** |

**知見**: 対比推論が重要（単なるフィードバックより5%向上）

#### 反復回数の影響

| 反復回数 | COCO (Hit@1) | 訓練時間 |
|---------|-------------|----------|
| 1回 | 72.1% | 5 min |
| 3回 | 76.8% | 15 min |
| **5回** | **78.5%** | **25 min** |
| 10回 | 78.7% | 50 min |

**知見**: 5回で収束、10回以上は過学習リスク

---

## 実装のポイント

### DSPy統合

AvaTaRは[DSPy](https://github.com/stanfordnlp/dspy)フレームワークに統合されています:

```python
import dspy
from dspy.avatar import AvatarModule, AvatarOptimizer

# 1. タスク定義
class PaperSearch(dspy.Signature):
    """Search for academic papers"""
    query = dspy.InputField()
    papers = dspy.OutputField(desc="List of relevant papers")

# 2. Actorモジュール
actor = AvatarModule(PaperSearch)

# 3. 訓練データ
train_data = [
    dspy.Example(query="Vision Transformers 2023", papers=["paper1", "paper2"]),
    dspy.Example(query="BERT fine-tuning", papers=["paper3", "paper4"]),
    # ...
]

# 4. Optimizer設定
optimizer = AvatarOptimizer(
    metric=hit_at_k,
    num_iterations=5,
    positive_sample_size=10,
    negative_sample_size=10
)

# 5. 最適化実行
optimized_actor = optimizer.compile(actor, trainset=train_data)

# 6. 推論
result = optimized_actor(query="Diffusion Models")
print(result.papers)
```

### カスタムツールの定義

```python
from dspy.avatar import Tool

# ツール定義
search_arxiv = Tool(
    name="search_arxiv",
    description="Search arXiv papers by query and year",
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "year": {"type": "integer", "description": "Publication year"},
        "max_results": {"type": "integer", "default": 10}
    },
    function=lambda query, year, max_results=10: arxiv_api_call(query, year, max_results)
)

# ActorにツールをバインドHome
actor = AvatarModule(PaperSearch, tools=[search_arxiv, filter_by_citations])
```

---

## Claude Codeスキルへの応用

### 自己改善スキルの設計

AvaTaRの対比推論パターンは、Claude Codeスキルの品質向上に応用できます:

#### 1. 評価ファーストの開発

```markdown
# SKILL.md

## 評価タスク

以下のタスクで80%以上の成功率を目標:
- Task 1: PDFから表を抽出 (10サンプル)
- Task 2: フォームフィールド自動入力 (10サンプル)
- Task 3: 複数PDF結合 (10サンプル)

## 対比分析

成功例と失敗例を比較し、体系的なエラーパターンを特定:
- 失敗パターン1: パスが相対パスの場合にエラー → 絶対パス変換を追加
- 失敗パターン2: 大きなPDF (>100ページ) でメモリ不足 → ストリーミング処理に変更
```

#### 2. フィードバックループの実装

```python
# scripts/evaluate_skill.py
def evaluate_skill(skill_name, test_cases):
    positive_cases = []
    negative_cases = []

    for task, expected_output in test_cases:
        result = claude_code.run_skill(skill_name, task)

        if is_correct(result, expected_output):
            positive_cases.append({"task": task, "trace": result["trace"]})
        else:
            negative_cases.append({
                "task": task,
                "trace": result["trace"],
                "error": result["error"]
            })

    # 対比分析
    feedback = analyze_contrast(positive_cases, negative_cases)

    # SKILL.mdに追記
    with open(f".claude/skills/{skill_name}/SKILL.md", "a") as f:
        f.write(f"\n## Lessons Learned\n{feedback}\n")

    return {
        "success_rate": len(positive_cases) / len(test_cases),
        "feedback": feedback
    }
```

#### 3. 反復改善ワークフロー

```bash
# hooks/post-skill-execution
#!/bin/bash
SKILL_NAME=$1
RESULT=$2

# 失敗をログに記録
if [ "$RESULT" != "success" ]; then
    echo "{\"skill\": \"$SKILL_NAME\", \"task\": \"$TASK\", \"error\": \"$ERROR\"}" \
        >> .claude/logs/failures.jsonl
fi

# 100件の失敗が溜まったら対比分析
FAILURE_COUNT=$(wc -l < .claude/logs/failures.jsonl)
if [ "$FAILURE_COUNT" -ge 100 ]; then
    python scripts/analyze_failures.py
    > .claude/logs/failures.jsonl  # ログクリア
fi
```

---

## 限界と今後の課題

### 訓練データ依存

**問題**: 訓練データに含まれないツール組み合わせは最適化されない。

**例**:
```
訓練: search_arxiv + filter_by_citations → 最適化済み
新規: search_arxiv + extract_code → 未最適化
```

**対策**: 多様なツール組み合わせを訓練データに含める

### ComparatorのLLM依存

**問題**: Comparatorの分析品質がLLM性能に依存。

**例**:
- GPT-4 Comparator: 詳細な分析、具体的な改善提案
- GPT-3.5 Comparator: 表面的な分析、曖昧な提案

**対策**: 強力なLLM（GPT-4, Claude Opus）をComparatorに使用

### 反復コスト

**問題**: 各反復でComparatorがLLMを呼び出し、コストが増加。

**コスト試算**:
```
反復回数: 5回
サンプル数: 正例10件 + 負例10件 = 20件
Comparatorトークン: 約5000トークン/反復
総コスト: 5回 × 5000トークン × $0.03/1K = $0.75
```

**対策**:
- 初期反復のみComparator使用（後半は手動調整）
- キャッシュで同一サンプルの再計算を回避

---

## まとめ

### 本論文の貢献

1. **対比推論フレームワーク**: 成功例・失敗例から体系的に学習
2. **Actor-Comparator分離**: エージェント実行と評価を分離し、反復改善を自動化
3. **7タスクで実証**: 平均14%の性能向上を達成
4. **DSPy統合**: 実装コストを大幅削減

### Claude Codeスキル開発者へのアクションアイテム

- **評価タスクを定義**: スキルの成功/失敗を明確に判定
- **失敗ログを収集**: 体系的なエラーパターン分析の材料
- **対比分析を手動実行**: 成功例と失敗例を比較し、改善指示を SKILL.md に追記
- **反復評価**: スキル更新後に再評価し、性能改善を確認

### 次のステップ

- GitHub実装: [zou-group/avatar](https://github.com/zou-group/avatar)
- DSPy統合: [DSPy AvatarOptimizer](https://dspy-docs.vercel.app/docs/deep-dive/optimizers/avatar)
- 関連Zenn記事: [Claude Codeスキル作成完全ガイド](https://zenn.dev/0h_n0/articles/a32342e48355ae)

---

## 参考文献

- 論文: [AvaTaR: Optimizing LLM Agents for Tool Usage via Contrastive Reasoning](https://arxiv.org/abs/2406.11200)
- NeurIPS 2024 Poster: [Virtual Conference](https://neurips.cc/virtual/2024/poster/95465)
- コード: [GitHub Repository](https://github.com/zou-group/avatar)
- DSPy統合: [DSPy Documentation](https://dspy-docs.vercel.app/)
