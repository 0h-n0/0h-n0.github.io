---
layout: post
title: "論文解説: Large Language Models as Optimizers（OPRO）"
description: "自然言語でプロンプト最適化を実現するOPRO手法の詳細解説 - LLMがLLMを最適化する革新的アプローチ"
categories: [blog, paper, arxiv]
tags: [prompt-engineering, llm, optimization, meta-learning, claude]
date: 2026-02-15 09:00:00 +0900
source_type: arxiv
arxiv_id: 2309.03409
source_url: https://arxiv.org/abs/2309.03409
zenn_article: 21f1740dc0ddd9
zenn_url: https://zenn.dev/0h_n0/articles/21f1740dc0ddd9
target_audience: "修士学生レベル"
---

## 論文概要

**タイトル**: Large Language Models as Optimizers
**著者**: Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, Xinyun Chen (Google DeepMind)
**発表**: ICLR 2024
**arXiv ID**: 2309.03409
**提出日**: 2023年9月7日（最終改訂: 2024年4月15日）

本論文は、**OPRO（Optimization by PROmpting）**という革新的な手法を提案しています。OPROは、最適化問題を自然言語で記述し、LLM自身に解を生成させる反復的アプローチです。従来の勾配ベース最適化が不要で、**LLMがLLMを最適化する**メタ学習パラダイムを実現します。

特筆すべき成果として、OPROで最適化されたプロンプトは人間が設計したプロンプトを**GSM8Kで最大8%、Big-Bench Hardで最大50%上回る**性能を達成しました。

---

## 背景と動機

### プロンプト最適化の課題

LLMの性能はプロンプト設計に大きく依存しますが、従来の最適化手法には以下の限界があります：

1. **勾配ベース最適化の適用困難**
   - LLMの内部パラメータが非公開（GPT-4等）
   - 離散的なトークン空間での最適化の難しさ
   - 数式的な勾配定義が不明確

2. **人間による試行錯誤の非効率性**
   - ドメイン知識と直感に依存
   - 時間とコストが膨大
   - 再現性・体系化が困難

3. **既存の自動化手法の制約**
   - タスク特化型で汎用性に欠ける
   - ホワイトボックスモデル前提（内部構造へのアクセス必要）
   - 微調整やAPIコストが高い

### OPROの核心的アイデア

**「最適化器としてのLLM」** - LLMは数学的勾配を計算しなくても、自然言語で記述された最適化問題を理解し、過去の試行履歴から改善案を生成できる能力を持ちます。

---

## 主要な貢献

### 1. 自然言語による最適化フレームワーク

OPROは最適化問題全体を自然言語で記述可能にします：

```
Optimization Problem (in natural language):
"Find a prompt that maximizes accuracy on GSM8K dataset."

Solution Space:
All possible text strings (prompts)

Objective Function:
Accuracy(prompt, dataset)
```

### 2. メタプロンプトによる反復改善

LLMに「過去の試行と性能」を入力し、次の候補解を生成させる仕組み：

```python
meta_prompt = f"""
Previous prompts and their scores:
1. "Let's think step by step" → Accuracy: 72.3%
2. "Solve this carefully" → Accuracy: 68.5%
3. "Break down the problem" → Accuracy: 75.1%

Generate a new prompt that improves upon these results.
"""

new_prompt = llm.generate(meta_prompt)
# Output: "Let's solve this problem by carefully breaking it down into steps"
```

### 3. 勾配フリー最適化の実現

従来の最適化手法との比較：

| 手法 | 勾配計算 | モデルアクセス | 汎用性 |
|------|---------|--------------|--------|
| **SGD** | 必要 | ホワイトボックス | 高 |
| **強化学習** | Policy gradient | グレーボックス | 中 |
| **進化的アルゴリズム** | 不要 | ブラックボックス | 中 |
| **OPRO** | **不要** | **ブラックボックス** | **高** |

### 4. 実証された性能向上

主要ベンチマークでの結果：

- **GSM8K（数学的推論）**: +8% accuracy over human-designed prompts
- **Big-Bench Hard**: +50% on specific tasks
- **線形回帰**: 従来の進化的アルゴリズムと同等性能
- **巡回セールスマン問題**: 近似解を効率的に発見

---

## 技術的詳細

### OPROアルゴリズムの数理的定式化

最適化問題を以下のように定義します：

$$
\theta^* = \arg\max_{\theta \in \Theta} f(\theta)
$$

ここで：
- $\theta$: 解候補（プロンプト文字列）
- $\Theta$: 解空間（すべての可能なテキスト）
- $f(\theta)$: 目的関数（例: 精度、報酬）

**OPROの反復プロセス**：

$$
\theta_{t+1} \sim p_{LLM}(\cdot \mid \mathcal{M}_t)
$$

ここで $\mathcal{M}_t$ はメタプロンプト（過去の履歴を含む）：

$$
\mathcal{M}_t = \text{Instruction} \oplus \{(\theta_1, f(\theta_1)), \ldots, (\theta_t, f(\theta_t))\}
$$

### アルゴリズムの実装

```python
from typing import List, Tuple, Callable
import numpy as np

class OPRO:
    def __init__(
        self,
        scorer_llm,  # 評価用LLM（例: GPT-3.5）
        optimizer_llm,  # 最適化用LLM（例: PaLM 2）
        objective_fn: Callable[[str], float],
        n_iterations: int = 8,
        batch_size: int = 8
    ):
        self.scorer_llm = scorer_llm
        self.optimizer_llm = optimizer_llm
        self.objective_fn = objective_fn
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.history: List[Tuple[str, float]] = []

    def construct_meta_prompt(self, instruction: str) -> str:
        """過去の履歴を含むメタプロンプトを生成"""
        meta_prompt = f"{instruction}\n\n"
        meta_prompt += "Previously generated prompts and their scores:\n"

        # スコアでソートして上位を表示
        sorted_history = sorted(self.history, key=lambda x: x[1], reverse=True)
        for i, (prompt, score) in enumerate(sorted_history[:20], 1):
            meta_prompt += f"{i}. \"{prompt}\" → Score: {score:.2%}\n"

        meta_prompt += "\nGenerate a new instruction that improves upon these results."
        return meta_prompt

    def optimize(self, instruction: str) -> str:
        """OPROメインループ"""
        for iteration in range(self.n_iterations):
            # メタプロンプト構築
            meta_prompt = self.construct_meta_prompt(instruction)

            # バッチで新しい候補プロンプトを生成
            new_prompts = []
            for _ in range(self.batch_size):
                candidate = self.optimizer_llm.generate(
                    meta_prompt,
                    temperature=1.0  # 多様性を確保
                )
                new_prompts.append(candidate)

            # 各候補を評価
            for prompt in new_prompts:
                score = self.objective_fn(prompt)
                self.history.append((prompt, score))
                print(f"Iteration {iteration+1}: \"{prompt}\" → {score:.2%}")

        # 最良のプロンプトを返す
        best_prompt, best_score = max(self.history, key=lambda x: x[1])
        return best_prompt

# 使用例
def evaluate_prompt_on_gsm8k(prompt: str) -> float:
    """GSM8Kでプロンプトを評価する関数（擬似実装）"""
    # 実際にはデータセット全体で精度を計算
    test_cases = load_gsm8k_test_set()
    correct = 0
    for question, answer in test_cases:
        response = scorer_llm.generate(f"{prompt}\n\nQuestion: {question}")
        if extract_answer(response) == answer:
            correct += 1
    return correct / len(test_cases)

# 実行
opro = OPRO(
    scorer_llm=GPT35(),
    optimizer_llm=PaLM2(),
    objective_fn=evaluate_prompt_on_gsm8k,
    n_iterations=8,
    batch_size=8
)

best_prompt = opro.optimize(
    instruction="Generate an instruction for solving math word problems."
)
print(f"\nBest prompt: {best_prompt}")
```

### メタプロンプトの設計パターン

論文では、以下の2つのメタプロンプト形式を比較検討しています：

**1. スコアを昇順で提示（Ascending）**

```
Previously tried:
- "Calculate carefully" → 65%
- "Solve step by step" → 70%
- "Think before answering" → 78%

Generate a better instruction.
```

**2. スコアを降順で提示（Descending）**

```
Previously tried:
- "Think before answering" → 78%
- "Solve step by step" → 70%
- "Calculate carefully" → 65%

Generate a better instruction.
```

**実験結果**: Ascending形式の方が性能が高い傾向。理由は、LLMが最後に見た情報（最高スコア）に注目しやすいため。

---

## 実装のポイント

### 1. 温度パラメータの調整

```python
# 多様性を確保するため高めの温度設定
optimizer_llm.generate(meta_prompt, temperature=1.0)

# 評価時は決定論的に
scorer_llm.generate(prompt, temperature=0.0)
```

### 2. 早期停止条件

```python
def should_stop(history: List[Tuple[str, float]], patience: int = 3) -> bool:
    """改善が停滞したら停止"""
    if len(history) < patience:
        return False

    recent_scores = [score for _, score in history[-patience:]]
    best_recent = max(recent_scores)
    best_overall = max(score for _, score in history)

    # 最近の改善がわずかなら停止
    return best_recent < best_overall + 0.01
```

### 3. 履歴のフィルタリング

すべての履歴を含めるとコンテキストが肥大化するため、上位K個のみ保持：

```python
def get_top_k_history(history: List[Tuple[str, float]], k: int = 20) -> List[Tuple[str, float]]:
    """スコア上位K個と最新の履歴を保持"""
    sorted_by_score = sorted(history, key=lambda x: x[1], reverse=True)[:k]
    recent = history[-5:]  # 最新5件も含める

    # 重複を除去
    seen = set()
    filtered = []
    for item in sorted_by_score + recent:
        if item[0] not in seen:
            filtered.append(item)
            seen.add(item[0])

    return filtered
```

---

## 実験結果

### GSM8K（Grade School Math）での性能

| プロンプト | 設計者 | 精度 |
|----------|--------|------|
| "Let's think step by step" | Human | 71.8% |
| "Take a deep breath and work on this problem step-by-step" | OPRO | **79.4%** (+7.6%) |

**OPROの発見したプロンプトの特徴**：
- 具体的な行動指示（"Take a deep breath"）
- 段階的アプローチの明示（"step-by-step"）
- ポジティブなトーン

### Big-Bench Hardでの結果

| タスク | ベースライン | OPRO | 改善率 |
|--------|------------|------|--------|
| Logical Deduction | 32.5% | **48.7%** | +50% |
| Sports Understanding | 72.3% | **79.1%** | +9.4% |

### 線形回帰問題

目的関数: $f(x) = w^T x + b$（$w, b$を推定）

```python
def linear_regression_objective(solution_str: str) -> float:
    """文字列からw, bをパースし、MSEを計算"""
    w, b = parse_solution(solution_str)
    predictions = X @ w + b
    mse = np.mean((predictions - y) ** 2)
    return -mse  # 最大化問題に変換

# OPROで最適化
opro = OPRO(optimizer_llm=PaLM2(), objective_fn=linear_regression_objective)
best_solution = opro.optimize("Find the weights w and bias b that minimize MSE.")
```

**結果**: 進化的アルゴリズム（CMA-ES）とほぼ同等の性能（誤差率2%以内）。

---

## 実運用への応用

### 1. Claude Code CLAUDE.mdの最適化

OPROをClaude CodeのCLAUDE.md最適化に応用する例：

```python
def evaluate_claude_md(claude_md_content: str) -> float:
    """CLAUDE.mdの性能を評価"""
    # テストタスクセット（過去10タスク）を実行
    test_tasks = load_test_tasks()
    success_rate = 0

    for task in test_tasks:
        # CLAUDE.mdを適用してClaude Codeを実行
        result = run_claude_code(task, claude_md=claude_md_content)
        if result.is_correct():
            success_rate += 1

    return success_rate / len(test_tasks)

# OPROで最適化
opro = OPRO(
    scorer_llm=ClaudeCode(),
    optimizer_llm=GPT4(),
    objective_fn=evaluate_claude_md,
    n_iterations=10
)

optimized_claude_md = opro.optimize(
    "Generate a CLAUDE.md file that maximizes coding task success rate."
)
```

**期待される成果**（論文のGSM8K結果から類推）：
- 初期CLAUDE.md: 72% success rate
- OPRO最適化後: 79% success rate (+7%)

### 2. ドメイン特化プロンプトの自動生成

```python
# 例: SQL生成タスク用プロンプト最適化
def evaluate_sql_prompt(prompt: str) -> float:
    sql_tasks = load_spider_dataset()  # Text-to-SQL benchmark
    correct = 0
    for nl_query, gold_sql in sql_tasks:
        generated_sql = llm.generate(f"{prompt}\n\nQuery: {nl_query}")
        if sql_equivalent(generated_sql, gold_sql):
            correct += 1
    return correct / len(sql_tasks)

opro = OPRO(optimizer_llm=GPT4(), objective_fn=evaluate_sql_prompt)
best_sql_prompt = opro.optimize("Generate SQL queries from natural language.")
```

### 3. 継続的プロンプト改善パイプライン

```python
class ContinuousOPRO:
    def __init__(self, production_llm, optimizer_llm):
        self.production_llm = production_llm
        self.optimizer_llm = optimizer_llm
        self.current_prompt = "Initial prompt"
        self.performance_log = []

    def log_production_performance(self, user_feedback: float):
        """本番環境でのフィードバックを記録"""
        self.performance_log.append((self.current_prompt, user_feedback))

    def weekly_optimization(self):
        """週次でプロンプトを再最適化"""
        opro = OPRO(
            optimizer_llm=self.optimizer_llm,
            objective_fn=lambda p: np.mean([score for prompt, score in self.performance_log if prompt == p])
        )
        # 過去の履歴を初期化
        opro.history = self.performance_log
        # 1イテレーションだけ実行（既存履歴を活用）
        new_prompt = opro.optimize("Improve the current prompt", n_iterations=1)
        self.current_prompt = new_prompt
        return new_prompt
```

---

## 関連研究

### プロンプト最適化手法の比較

| 手法 | アプローチ | 強み | 弱み |
|------|----------|------|------|
| **AutoPrompt** | 勾配ベース | 理論的保証 | ホワイトボックス必要 |
| **APE** | Few-shot生成 | シンプル | 反復改善なし |
| **RLPrompt** | 強化学習 | 探索効率高 | 報酬設計が困難 |
| **OPRO** | LLM反復生成 | **ブラックボックス可**, **自然言語** | LLM APIコスト |

### LLMによるメタ学習

OPROは以下の研究領域の交差点に位置します：

1. **In-Context Learning**: LLMが文脈から学習する能力を活用
2. **Self-Improvement**: LLMが自己のプロンプトを改善
3. **Neural Architecture Search**: 探索空間が離散的な最適化問題

類似手法：
- **Constitutional AI（Anthropic）**: LLMが自身の応答を改善
- **Self-Refine**: LLMが出力を反復的に洗練

---

## 限界と今後の課題

### 1. 計算コストの問題

OPROは各イテレーションで複数のLLM呼び出しが必要：

$$
\text{Total API Calls} = n_{\text{iterations}} \times n_{\text{batch}} \times (1 + n_{\text{eval samples}})
$$

例: 8イテレーション × 8バッチ × 100評価サンプル = **6,400回のAPI呼び出し**

**対策**：
- 評価サンプルをサブサンプリング（100 → 20）
- キャッシュの活用
- より小さいモデルでの事前探索

### 2. 局所最適解への収束

LLMは過去の高スコアプロンプトの変種を生成しがちで、探索が停滞：

```python
# 多様性を強制する仕組み
def add_exploration_bonus(prompt: str, history: List[str]) -> float:
    """既存プロンプトとの類似度にペナルティ"""
    similarity_penalty = max(
        cosine_similarity(embed(prompt), embed(p)) for p in history
    )
    return base_score - 0.1 * similarity_penalty
```

### 3. 評価関数の設計

目的関数が不完全だと、過適合したプロンプトが生成される：

```python
# 悪い例: 訓練データのみで評価
def bad_objective(prompt):
    return accuracy_on_train_set(prompt)  # 過適合リスク

# 良い例: ホールドアウトセットで評価
def good_objective(prompt):
    return accuracy_on_validation_set(prompt)
```

### 4. プロンプトの解釈可能性

OPROが生成したプロンプトが人間には理解しにくい場合がある：

```
Generated prompt: "Breathe deeply, consider each number's relationship to its neighbors,
and trace the logical flow backwards from the answer while keeping track of intermediate
steps in a mental ledger."
```

**対策**: メタプロンプトに制約を追加：

```
"Generate a concise (max 20 words) and natural-sounding instruction."
```

---

## まとめ

### 主要な貢献の要約

1. **自然言語による最適化**: 数式・勾配不要で最適化問題を記述・解決
2. **メタプロンプティング**: LLMの反復的改善能力を活用
3. **実証された性能**: GSM8K +8%, Big-Bench Hard +50%
4. **汎用性**: 線形回帰からTSPまで多様な問題に適用可能

### 実装上の重要ポイント

- メタプロンプトの履歴はスコア昇順で提示
- 温度パラメータを調整して探索と活用をバランス
- 計算コスト削減のため評価サンプルをサブサンプリング
- 多様性を維持するための探索ボーナス

### Claude Codeへの示唆

Arizeの研究（+5-10%精度向上）とOPROを組み合わせることで、CLAUDE.mdの自動最適化が可能になります：

1. 過去のタスク実行履歴を収集
2. OPROでCLAUDE.mdを反復改善
3. A/Bテストで効果を検証
4. 週次で最適化を実行

これにより、**人間の試行錯誤なしでプロンプトを継続的に改善**できるパイプラインが構築できます。

---

## 参考文献

- [arXiv: 2309.03409](https://arxiv.org/abs/2309.03409)
- [GitHub: google-deepmind/opro](https://github.com/google-deepmind/opro)
- [ICLR 2024 Paper Page](https://openreview.net/forum?id=Bb4VGOWELI)

---

## Zenn記事との関連

この論文解説は、Zenn記事「[Claude Codeプロンプト管理術](https://zenn.dev/0h_n0/articles/21f1740dc0ddd9)」で紹介したプロンプト最適化の理論的基盤を提供します。OPROの手法を理解することで、CLAUDE.mdの自動最適化パイプライン構築に役立ちます。

:::message
本記事は修士学生レベルを想定し、数式・実装の詳細を含みます。OPRO論文の完全な理解には、元論文の熟読を推奨します。
:::
