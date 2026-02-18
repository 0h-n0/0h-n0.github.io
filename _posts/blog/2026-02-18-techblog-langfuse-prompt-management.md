---
layout: post
title: "Langfuse公式解説: OSSプロンプト管理基盤とA/Bテスト実装の技術詳細"
description: "Langfuseのプロンプトバージョニング・ラベル機能・A/Bテスト・オブザーバビリティ統合を公式ドキュメントベースで詳解"
categories: [blog, tech_blog]
tags: [LLMOps, Langfuse, prompt-management, A/B-testing, observability, llm]
date: 2026-02-18 21:30:00 +0900
source_type: tech_blog
source_domain: langfuse.com
source_url: https://langfuse.com/docs/prompt-management/overview
zenn_article: 9fc2f8c4a420e4
zenn_url: https://zenn.dev/0h_n0/articles/9fc2f8c4a420e4
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## ブログ概要（Summary）

Langfuseは、LLMアプリケーションのためのOSS（MITライセンス）オブザーバビリティ・評価基盤である。プロンプトのバージョン管理・ラベルベースのA/Bテスト・リアルタイムトレーシング・自動評価を統合的に提供する。GitHub Stars 19,000超、Y Combinator W23出身で、OpenTelemetry・LangChain・OpenAI SDK・LiteLLMなど主要フレームワークとの統合をサポートしている。

この記事は [Zenn記事: LLMプロンプト管理CI/CD：Langfuse×LaunchDarklyでA/Bテストと安全ロールアウト](https://zenn.dev/0h_n0/articles/9fc2f8c4a420e4) の深掘りです。

## 情報源

- **種別**: 企業テックブログ / OSSドキュメント
- **URL**: [https://langfuse.com/docs/prompt-management/overview](https://langfuse.com/docs/prompt-management/overview)
- **組織**: Langfuse（Y Combinator W23）
- **発表日**: 2025-2026（継続的更新）

## 技術的背景（Technical Background）

LLMアプリケーションの本番運用において、プロンプトはビジネスロジックそのものである。しかし、プロンプトをソースコードに埋め込む従来のアプローチには以下の問題がある:

1. **デプロイ結合**: プロンプト変更のたびにアプリケーションの再デプロイが必要
2. **バージョン不明**: 本番で稼働しているプロンプトの正確なバージョンが追跡困難
3. **品質測定困難**: プロンプト変更の効果を定量的に測定する手段がない
4. **協業の壁**: プロンプトエンジニアがコードリポジトリにアクセスする必要がある

Langfuseはこれらの課題を、プロンプト管理とオブザーバビリティの統合プラットフォームとして解決する。Zenn記事ではLangfuseのラベル機能によるA/Bテスト実装を紹介したが、本記事ではその技術的基盤を公式ドキュメントベースで詳解する。

## 実装アーキテクチャ（Architecture）

### プロンプト管理のデータモデル

Langfuseのプロンプト管理は、以下の概念で構成される:

```
Prompt
├── name: "summarizer"          # プロンプト名（一意）
├── type: "text" | "chat"       # プロンプト種別
├── versions: [                 # 線形バージョニング
│   ├── version 1: { prompt: "..." }
│   ├── version 2: { prompt: "..." }
│   └── version 3: { prompt: "..." }  # 最新
│ ]
└── labels: {                   # ラベル→バージョンのマッピング
    "production": version 3,
    "staging": version 2,
    "prod-a": version 2,      # A/Bテスト用
    "prod-b": version 3       # A/Bテスト用
  }
```

**線形バージョニング**: 同名のプロンプトを作成すると自動的にバージョン番号がインクリメントされる。ブランチモデルではなく線形モデルを採用しているため、バージョンの前後関係が明確。

**ラベルシステム**: バージョンにラベル（文字列タグ）を付与することで、アプリケーションは「バージョン番号」ではなく「ラベル」でプロンプトを取得できる。これにより、デプロイなしでのプロンプト切り替えが実現する。

### SDK統合パターン

#### Python SDK

```python
from langfuse import Langfuse

langfuse = Langfuse()

# プロンプト作成（バージョン自動インクリメント）
langfuse.create_prompt(
    name="movie-critic",
    type="text",
    prompt="As a {{criticlevel}} movie critic, do you like {{movie}}?",
    labels=["production"]
)

# チャット型プロンプト作成
langfuse.create_prompt(
    name="summarizer",
    type="chat",
    prompt=[
        {"role": "system", "content": "あなたは技術記事の要約エキスパートです。{{style}}"},
        {"role": "user", "content": "以下の記事を要約してください: {{article}}"}
    ],
    labels=["staging"]
)

# プロンプト取得（productionラベル版）
prompt = langfuse.get_prompt("movie-critic")
compiled = prompt.compile(criticlevel="expert", movie="Dune 2")
# → "As a expert movie critic, do you like Dune 2?"
```

#### TypeScript SDK

```typescript
import Langfuse from "langfuse";

const langfuse = new Langfuse();

// プロンプト取得
const prompt = await langfuse.prompt.get("movie-critic");
const compiled = prompt.compile({
  criticlevel: "expert",
  movie: "Dune 2"
});
```

### キャッシュ戦略

Langfuse SDKは**クライアントサイドキャッシュ**を実装しており、プロンプト取得時のレイテンシを最小化する:

- サーバーサイド: CDNキャッシュによる高速配信
- クライアントサイド: SDK内メモリキャッシュ。`get_prompt()`呼び出しはメモリ読み取りと同等速度
- TTL（Time-to-Live）: デフォルト60秒。本番環境では`cache_ttl_seconds`で調整可能

```python
# キャッシュTTLの設定
prompt = langfuse.get_prompt("summarizer", cache_ttl_seconds=300)
```

この設計により、Langfuseサーバーがダウンしてもキャッシュからプロンプトを提供でき、本番環境の可用性に影響しない。

## パフォーマンス最適化（Performance）

### A/Bテスト実装の詳細

Zenn記事で紹介したA/Bテストの技術的実装を深掘りする。

```python
import random
from langfuse import Langfuse
from openai import OpenAI

langfuse = Langfuse()
openai_client = OpenAI()

def summarize_with_ab_test(article_text: str, trace_id: str) -> dict:
    """A/Bテスト付きの記事要約

    Args:
        article_text: 要約対象のテキスト
        trace_id: Langfuseトレース識別子

    Returns:
        要約結果とバリアント情報
    """
    # ラベルでバリアントを取得
    prompt_a = langfuse.get_prompt("summarizer", label="prod-a")
    prompt_b = langfuse.get_prompt("summarizer", label="prod-b")

    # ランダム選択
    selected = random.choice([prompt_a, prompt_b])

    # テンプレート変数を埋め込み
    compiled = selected.compile(article=article_text)

    # Langfuseトレーシングと紐付けてOpenAI呼び出し
    response = openai_client.chat.completions.create(
        model=compiled.get("model", "gpt-4o"),
        messages=compiled["messages"],
        # Langfuseのprompt linkingで自動トレーシング
        langfuse_prompt=selected
    )

    return {
        "summary": response.choices[0].message.content,
        "variant": selected.label,
        "version": selected.version
    }
```

### トレーシングとメトリクス追跡

Langfuseのトレーシングは各リクエストの以下を自動記録する:

| メトリクス | 説明 | A/Bテストでの用途 |
|-----------|------|------------------|
| レイテンシ | LLM呼び出しの応答時間 | バリアント間のp95比較 |
| トークン使用量 | 入力/出力トークン数 | コスト効率の比較 |
| コスト | USD換算の推論コスト | ROI算出 |
| 品質スコア | LLM-as-a-Judge評価 | 出力品質の定量比較 |
| ユーザーフィードバック | 👍/👎等のフィードバック | 実ユーザー満足度 |

### Promptfoo統合

LangfuseはPromptfooとの直接統合をサポートしており、CI/CDパイプラインでの品質ゲートを構成できる:

```yaml
# promptfooconfig.yaml
prompts:
  - id: langfuse:summarizer:production
  - id: langfuse:summarizer:staging
tests:
  - vars:
      article: "React 19のServer Componentsの新機能..."
    assert:
      - type: llm-rubric
        value: "要約は3-5文で、技術用語を保持している"
```

PromptfooはLangfuseからプロンプトを直接取得し、評価結果をLangfuseにフィードバックする双方向統合が可能。

## 運用での学び（Production Lessons）

### 統計的有意性の確保

Zenn記事で「サンプル数を決めずにn=47で有意差なしと結論してしまった」失敗を紹介した。Langfuseのダッシュボードでは以下が推奨される:

- **最低サンプル数**: 500リクエスト/バリアント
- **有意水準**: p < 0.05
- **効果量**: Cohen's d ≥ 0.2（実用的な差異として意味がある）

### バージョンロールバック

障害発生時のロールバックはラベルの付け替えだけで完了する:

```python
# 緊急ロールバック: production ラベルを前バージョンに戻す
langfuse.create_prompt(
    name="summarizer",
    type="chat",
    prompt=previous_version_content,
    labels=["production"]  # 新バージョンとして作成、productionラベル付与
)
```

コードデプロイ不要で、即座にロールバックが完了する。Zenn記事で紹介したLaunchDarklyのFeature Flagとの二重ロールバック機構が構成できる。

### 監視アラートの設定

Langfuseのメトリクスから以下のアラート条件を構成:

1. **品質スコア低下**: 直近1時間の平均品質スコアが閾値（0.8）を下回った場合
2. **レイテンシ急上昇**: p95レイテンシが前日比150%を超えた場合
3. **コスト異常**: 日次トークン使用量が前週平均の200%を超えた場合

## 学術研究との関連（Academic Connection）

### DSPyとの統合

DSPy（arXiv: 2310.11511）のTeleprompterで最適化されたプロンプトをLangfuseに格納し、バージョン管理する設計パターン:

1. DSPyで`compiled_program.save("v2.3.json")`
2. JSONからプロンプト文字列を抽出
3. `langfuse.create_prompt(name="rag-pipeline", prompt=extracted, labels=["staging"])`
4. Promptfooで品質ゲート通過後、`labels=["production"]`に昇格

### OPROとの統合

OPRO（arXiv: 2309.03409）で生成されたプロンプト候補をLangfuseのA/Bテストバリアントとして設定し、本番トラフィックで検証する。OPROの訓練データとしてLangfuseに蓄積されたスコアデータを還元することで、継続的改善ループが成立する。

### TextGradとの統合

TextGrad（arXiv: 2406.07496）の最適化ループの各ステップをLangfuseのトレースとして記録し、テキスト勾配の品質とプロンプト改善の相関を分析する。

## まとめと実践への示唆

Langfuseは、LLMプロンプト管理に必要な機能（バージョニング・ラベル・A/Bテスト・トレーシング・評価）を単一のOSSプラットフォームで提供する。Zenn記事で紹介した3層防御アーキテクチャの核心を構成するツールであり、DSPy・OPRO・TextGradなどの学術的手法との統合ポイントも豊富。

MITライセンスのセルフホスト版とクラウド版の両方が利用可能で、月間10万リクエスト以下のチームはクラウド版の無料プランで十分にスタートできる。

## 参考文献

- **Blog URL**: [https://langfuse.com/docs/prompt-management/overview](https://langfuse.com/docs/prompt-management/overview)
- **A/B Testing docs**: [https://langfuse.com/docs/prompt-management/features/a-b-testing](https://langfuse.com/docs/prompt-management/features/a-b-testing)
- **GitHub**: [https://github.com/langfuse/langfuse](https://github.com/langfuse/langfuse)（MIT License, 19k+ stars）
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/9fc2f8c4a420e4](https://zenn.dev/0h_n0/articles/9fc2f8c4a420e4)
