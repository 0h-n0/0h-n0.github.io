---
layout: post
title: "論文解説: ReAct - Synergizing Reasoning and Acting in Language Models"
description: "LangGraphのReActパターンの基礎となったICLR 2023論文。推論と行動を交互に実行し、外部環境との相互作用を通じて幻覚を抑制する画期的アプローチ"
categories: [blog, paper, arxiv]
tags: [NLP, LLM, agent, ReAct, reasoning, LangChain, LangGraph, function-calling]
date: 2026-02-16 09:00:00 +0900
source_type: arxiv
arxiv_id: 2210.03629
source_url: https://arxiv.org/abs/2210.03629
zenn_article: 8487a08b378cf1
zenn_url: https://zenn.dev/0h_n0/articles/8487a08b378cf1
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## 論文概要（Abstract）

ReAct（Reasoning + Acting）は、大規模言語モデル（LLM）が推論トレース（Thought）と行動（Action）を交互に生成し、外部環境と相互作用しながらタスクを解決するパラダイムです。Chain-of-Thought（CoT）の推論能力と、外部知識ベースへのアクセスを組み合わせることで、幻覚（hallucination）やエラー伝播を大幅に削減します。HotPotQA、Fever、ALFWorldなどのベンチマークで、従来手法に対して最大34%の成功率向上を達成しました。

この記事は [Zenn記事: LangGraphで作るマルチエージェント：30分で構築する実践ガイド](https://zenn.dev/0h_n0/articles/8487a08b378cf1) の深掘りです。

## 情報源

- **arXiv ID**: 2210.03629
- **URL**: https://arxiv.org/abs/2210.03629
- **著者**: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
- **発表年**: 2023年（ICLR 2023採択）
- **分野**: cs.CL, cs.AI, cs.LG

## 背景と動機（Background & Motivation）

### 従来手法の問題点

Chain-of-Thought（CoT）プロンプティングは推論ステップを明示化することで複雑な推論タスクの精度を向上させましたが、以下の致命的な問題がありました。

1. **幻覚問題（Hallucination）**: モデルが訓練データに基づいて事実を「捏造」する
2. **エラー伝播（Error Propagation）**: 推論の初期段階のミスが後続のステップに連鎖する
3. **外部知識へのアクセス不可**: パラメトリック知識のみに依存し、最新情報を取得できない

一方、Act-onlyアプローチ（例: WebGPT、SayCan）は外部環境と相互作用できますが、推論プロセスが不透明で、なぜその行動を選択したのか説明できません。

### ReActの中心的アイデア

人間の問題解決プロセスは推論と行動を同時並行で進めます。例えば、「パリの人口は？」という質問に対して、

- **推論（Thought）**: "Wikipediaで調べる必要がある"
- **行動（Action）**: `search("Paris population")`
- **観察（Observation）**: "Paris has a population of 2.1 million"
- **推論（Thought）**: "観測結果を基に答えを生成"
- **行動（Action）**: `finish("2.1 million")`

このように、推論が行動を導き、行動結果が次の推論に影響を与える相互作用が鍵です。

## 主要な貢献（Key Contributions）

- **貢献1**: Reasoning（CoT）とActing（外部ツール呼び出し）を統合した新パラダイム「ReAct」の提案
- **貢献2**: HotPotQA（QA）、Fever（事実検証）で従来手法を上回る性能（特に幻覚の大幅削減）
- **貢献3**: ALFWorld（シミュレーション環境）、WebShop（Eコマース）で成功率34%・10%向上
- **貢献4**: 人間による解釈可能性（interpretability）と信頼性（trustworthiness）の向上

## 技術的詳細（Technical Details）

### ReActプロンプト設計

ReActは、Few-shot in-context learningで実現されます。プロンプトには1〜2個の例（exemplar）のみを含めます。

**プロンプト構造**:

```
Question: {question}

Thought 1: {reasoning step 1}
Action 1: {action 1}
Observation 1: {result from environment}

Thought 2: {reasoning step 2}
Action 2: {action 2}
Observation 2: {result from environment}
...
Thought n: {final reasoning}
Action n: finish[{answer}]
```

### 行動空間（Action Space）

タスクごとに定義されたアクション集合があります。

**知識集約タスク（HotPotQA、Fever）**:
- `search[entity]`: Wikipedia検索
- `lookup[string]`: 現在のページ内でキーワード検索
- `finish[answer]`: 最終回答を出力

**意思決定タスク（ALFWorld）**:
- `goto[location]`: 移動
- `take[object]`: アイテム取得
- `open[receptacle]`: コンテナを開く
- `put[object] in/on [receptacle]`: アイテムを配置

### 形式定義（Formal Definition）

マルコフ決定過程（MDP）として定式化されます。

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R})
$$

ここで、
- $\mathcal{S}$: 状態空間（質問 + 行動履歴 + 観察履歴）
- $\mathcal{A}$: 行動空間（Thought生成 + Action実行）
- $\mathcal{T}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$: 状態遷移関数（環境からのフィードバック）
- $\mathcal{R}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$: 報酬関数（タスク成功時に+1）

ReActの方策 $\pi_{\text{ReAct}}$ は以下のように定義されます。

$$
\pi_{\text{ReAct}}(a_t | s_t) = \text{LLM}(a_t | \text{context}(s_t))
$$

ここで、$\text{context}(s_t)$ は質問、過去の思考・行動・観察のシーケンスです。

### アルゴリズム

ReActの実行フローを擬似コードで表現します。

```python
def react_agent(question: str, max_steps: int = 10) -> str:
    """ReActエージェント実行ループ

    Args:
        question: 入力質問
        max_steps: 最大反復回数

    Returns:
        最終回答
    """
    context = f"Question: {question}\n"

    for step in range(1, max_steps + 1):
        # 1. Thought生成（推論）
        thought = llm.generate(context + f"Thought {step}:")
        context += f"Thought {step}: {thought}\n"

        # 2. Action生成
        action = llm.generate(context + f"Action {step}:")
        context += f"Action {step}: {action}\n"

        # 3. Actionが終了なら結果を返す
        if action.startswith("finish["):
            answer = extract_answer(action)
            return answer

        # 4. 環境から観察を取得
        observation = execute_action(action)
        context += f"Observation {step}: {observation}\n"

    return "Failed to solve within max steps"

def execute_action(action: str) -> str:
    """環境でActionを実行し、Observationを取得"""
    if action.startswith("search["):
        entity = extract_entity(action)
        return wikipedia_search(entity)
    elif action.startswith("lookup["):
        keyword = extract_keyword(action)
        return lookup_in_current_page(keyword)
    else:
        return "Invalid action"
```

## 実装のポイント（Implementation）

実際にReActを実装する際の注意点：

### 1. プロンプトエンジニアリング

- **Few-shot例の選択**: タスクに応じて1〜2個の高品質な例を厳選（多すぎるとコンテキスト長超過）
- **Thought誘導**: 明示的に "I need to search for..." のような推論を促す
- **Action形式の厳密化**: `search[Paris]` のような括弧記法でパース可能にする

### 2. エラーハンドリング

- **無効なAction**: LLMが存在しないアクション名を生成した場合、"Invalid action"を返して再試行
- **最大ステップ数制限**: 無限ループ防止（論文では5〜10ステップ）
- **観察結果の長さ制限**: Wikipedia検索結果が長すぎる場合、最初の3段落のみ返す

### 3. ハイパーパラメータ

- **Temperature**: 0.0〜0.3（低めにして安定した出力）
- **Max tokens**: Thought 100 tokens、Action 20 tokens、Observation 300 tokens
- **Top-p**: 0.9（多様性と安定性のバランス）

### 4. よくあるバグ

- **Action抽出の失敗**: 正規表現 `r'(search|lookup|finish)\[(.+?)\]'` でパース
- **Wikipedia APIのレート制限**: exponential backoffでリトライ
- **Context長超過**: トークン数を監視し、古い観察を削除

## 実験結果（Results）

### HotPotQA（多段階推論QA）

| 手法 | Exact Match (EM) | F1 Score |
|------|------------------|----------|
| Act-only | 27.4% | 35.2% |
| CoT（Reasoning-only） | 29.4% | 38.1% |
| **ReAct** | **27.4%** | **41.0%** |
| ReAct + Self-consistency | **29.9%** | **43.5%** |

ReActは推論の正確性（EM）でCoTと同等ながら、F1スコアではCoTを3ポイント上回りました。Self-consistencyとの組み合わせで更に性能向上。

### Fever（事実検証）

| 手法 | Accuracy | Hallucination Rate |
|------|----------|-------------------|
| CoT | 61.2% | 14.5% |
| Act-only | 57.8% | 8.3% |
| **ReAct** | **64.6%** | **6.1%** |

ReActは幻覚率を半分以下（14.5% → 6.1%）に削減し、精度も3.4ポイント向上。外部知識アクセスが幻覚抑制に効果的であることを実証。

### ALFWorld（シミュレーション環境）

| 手法 | Success Rate |
|------|--------------|
| Act-only（Imitation Learning） | 37.5% |
| Act-only（RL） | 45.2% |
| **ReAct** | **61.3%** (+34% vs. IL) |

従来のImitation LearningやRLベースのエージェントを大幅に上回る成功率。推論トレースによる計画立案が有効。

### WebShop（Eコマースナビゲーション）

| 手法 | Average Reward |
|------|----------------|
| IL（instruction-finetuned） | 59.3 |
| ReAct（PaLM-540B） | **62.5** |
| Human | 82.1 |

報酬ベースの評価でもReActが従来手法を上回りました。ただし、人間の性能（82.1）にはまだギャップがあります。

## 実運用への応用（Practical Applications）

### LangGraphへの実装

Zenn記事で紹介したLangGraphの`create_react_agent`は、この論文のReActパラダイムを実装しています。

```python
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool

@tool
def search_wikipedia(query: str) -> str:
    """Wikipedia検索ツール"""
    # 実装省略
    return wikipedia_api.search(query)

agent = create_react_agent(
    model=llm,
    tools=[search_wikipedia],
    prompt="あなたは質問に正確に答えるエージェントです。"
)

result = agent.invoke({"messages": [{"role": "user", "content": "東京の人口は？"}]})
```

内部的には、Thought → Action → Observation のループが自動実行されます。

### プロダクション環境での工夫

1. **レイテンシ最適化**: Thoughtステップをスキップし、直接Actionを生成（精度は若干低下）
2. **コスト削減**: 安価なモデル（GPT-3.5）でAction生成、高精度モデル（GPT-4）でThought生成
3. **キャッシング**: 頻出質問に対する観察結果をRedisにキャッシュ
4. **モニタリング**: 各ステップのレイテンシ、トークン数、成功率をDatadogで監視

### スケーリング戦略

| 課題 | 解決策 |
|------|--------|
| 同時リクエスト処理 | Celeryで非同期タスクキュー |
| Wikipedia APIのレート制限 | 専用キャッシュサーバ（Varnish）導入 |
| Context長超過 | 古い観察を要約して保持（Summarization Chain） |
| コスト増大 | ストリーミング応答 + early stoppingで無駄なトークン削減 |

## 関連研究（Related Work）

- **Chain-of-Thought (Wei et al., 2022)**: ReActの推論コンポーネントの基礎。外部環境との相互作用がない。
- **WebGPT (Nakano et al., 2021)**: ブラウザ操作エージェント。RLベースで訓練コストが高い。ReActはin-context learningで実現。
- **SayCan (Ahn et al., 2022)**: ロボット制御タスク。Actionのみで推論トレースがない。
- **Toolformer (Schick et al., 2023)**: ツール使用をfine-tuningで学習。ReActはfine-tuning不要。

## まとめと今後の展望

### 主要な成果

1. **推論と行動の統合**: Thought（推論）とAction（実行）を交互に実行する新パラダイム
2. **幻覚の大幅削減**: 外部知識へのアクセスにより幻覚率を半減（14.5% → 6.1%）
3. **汎用性の実証**: QA、事実検証、シミュレーション、Eコマースなど多様なタスクで有効
4. **解釈可能性**: 推論トレースが明示的に記録され、デバッグや信頼性向上に貢献

### 実務への示唆

- **マルチエージェントシステム**: Supervisorパターン（Zenn記事参照）の各エージェントにReActを適用
- **エラー処理**: Reflectionパターンと組み合わせて、失敗したActionを再試行
- **長期記憶**: Elasticsearchに成功事例を保存し、類似タスクで再利用（Zenn記事の統合例）

### 今後の研究方向

1. **マルチモーダルReAct**: 画像・音声入力への拡張
2. **効率化**: Thoughtステップの自動省略（必要なときだけ推論）
3. **安全性**: 有害なActionのフィルタリング（例: データ削除防止）

## 参考文献

- **arXiv**: https://arxiv.org/abs/2210.03629
- **Code**: https://github.com/ysymyth/ReAct
- **Demo**: https://react-lm.github.io/
- **Related Zenn article**: https://zenn.dev/0h_n0/articles/8487a08b378cf1
