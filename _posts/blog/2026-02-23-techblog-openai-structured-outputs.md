---
layout: post
title: "OpenAI公式解説: Structured Outputs in the API — strict modeによる100%スキーマ準拠の実現"
description: "2024年8月発表のOpenAI Structured Outputsの技術詳細を解説。strict modeの動作原理、JSON Schema制約、Function Callingとの統合パターンを分析。"
categories: [blog, tech_blog]
tags: [openai, structured-outputs, function-calling, json-schema, strict-mode, constrained-decoding, api, llm]
date: 2026-02-23 12:00:00 +0900
source_type: tech_blog
source_domain: openai.com
source_url: https://openai.com/index/introducing-structured-outputs-in-the-api/
zenn_article: b2d1df91e5f5de
zenn_url: https://zenn.dev/0h_n0/articles/b2d1df91e5f5de
math: true
mermaid: true
target_audience: "修士学生レベル"
---

本記事は [OpenAI: Introducing Structured Outputs in the API](https://openai.com/index/introducing-structured-outputs-in-the-api/) の解説記事です。

## ブログ概要（Summary）

OpenAIは2024年8月6日、APIにStructured Outputs機能を導入した。`strict: true` を設定することで、モデルの出力が開発者指定のJSON Schemaに100%準拠することが保証される。この機能はFunction Calling（tool use）と `response_format` の両方で利用可能であり、gpt-4o-2024-08-06モデルでの評価で100%のスキーマ準拠率を達成したとOpenAIは報告している。

この記事は [Zenn記事: Function Calling×Structured Outputs実装入門：3社APIで型安全なツール連携を構築する](https://zenn.dev/0h_n0/articles/b2d1df91e5f5de) の深掘りです。

## 情報源

- **種別**: 企業テックブログ
- **URL**: [https://openai.com/index/introducing-structured-outputs-in-the-api/](https://openai.com/index/introducing-structured-outputs-in-the-api/)
- **組織**: OpenAI
- **発表日**: 2024年8月6日

## 技術的背景（Technical Background）

LLMの出力をJSON形式で制御する需要はFunction Callingの普及とともに増加した。従来のJSON Mode（`response_format: {"type": "json_object"}`）では出力がJSON形式であることは保証されるが、特定のキー名やネスト構造までは保証されなかった。

Zenn記事で解説されているように、Function Callingではモデルが返す引数のスキーマ違反がランタイムエラーの主要因となる。OpenAIのStructured Outputsはこの問題を、制約付きデコーディング（constrained decoding）の技術で解決した。

## 実装アーキテクチャ（Architecture）

### 動作原理: 制約付きデコーディング

OpenAIの公式発表によると、Structured Outputsは以下の2つの技術を組み合わせている：

**1. モデルのfine-tuning**:
GPT-4oモデルに対し、複雑なJSON Schemaに準拠した出力を生成するようfine-tuningを実施。これにより、制約なしでも高い準拠率を実現する基盤が構築されている。

**2. 推論時の制約付きデコーディング**:
各デコードステップで、JSON Schemaに違反するトークンをマスクし、有効なトークンのみから選択する。これは文脈自由文法（CFG）ベースの手法であり、outlines論文 (Willard & Louf, 2023) と同系統の技術である。

### strict modeの2つのエントリポイント

**Function Calling（tool use）での使用**:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"],
            "additionalProperties": False  # strict mode必須
        },
        "strict": True  # ← これがStructured Outputs有効化
    }
}]
```

**response_formatでの使用**:

```python
response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "weather_response",
            "schema": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "number"},
                    "condition": {"type": "string"}
                },
                "required": ["temperature", "condition"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)
```

### JSON Schema制約

OpenAIのStrict modeでは、サポートされるJSON Schemaのサブセットに制約がある：

| JSON Schema機能 | サポート状況 | 備考 |
|-----------------|------------|------|
| `type` (string, number, integer, boolean, null, array, object) | 対応 | 全プリミティブ型 |
| `enum` | 対応 | 文字列・数値のenumeration |
| `properties` + `required` | 対応 | 全プロパティが `required` に含まれる必要あり |
| `additionalProperties: false` | **必須** | 全objectに設定が必要 |
| `$ref` / `$defs` | 対応 | 再帰スキーマも一定深度までサポート |
| `anyOf` | 対応 | union型の表現に使用 |
| `oneOf` | 非対応 | `anyOf` で代替 |
| `patternProperties` | 非対応 | 動的キー名には未対応 |
| `if/then/else` | 非対応 | 条件分岐スキーマは未サポート |

この制約は、Zenn記事で指摘されている「`additionalProperties: false` がPydantic v2のデフォルトスキーマに含まれない」問題に直結する。OpenAI公式SDKの `pydantic_function_tool()` ヘルパーはこの制約を自動的に処理する。

### スキーマ処理パイプライン

OpenAIの発表内容から推測される処理パイプライン：

```
開発者のJSON Schema
    ↓
スキーマバリデーション（サポートされるサブセットか確認）
    ↓
CFGへの変換（JSONスキーマ → 文脈自由文法）
    ↓
FSM/PDAの構築（有限状態機械 or プッシュダウンオートマトン）
    ↓
語彙インデックス作成（各状態 → 有効トークン集合）
    ↓
推論時マスキング（デコードステップごとに適用）
```

OpenAIの公式ドキュメントによると、**初回リクエスト時にスキーマの処理が発生**するため、新しいスキーマでの初回レスポンスには追加レイテンシが生じる。同一スキーマの後続リクエストではキャッシュが利用される。

## パフォーマンス最適化（Performance）

OpenAIが公式に報告しているベンチマーク結果：

| 評価指標 | JSON Modeのみ | Structured Outputs (strict) |
|---------|-------------|---------------------------|
| スキーマ準拠率 | ~86% | **100%** |
| 複雑スキーマ準拠率 | ~36% | **100%** |

- **スキーマ準拠率100%**: `strict: true` 設定時、生成されたJSONは構造的にスキーマに100%準拠する。これは確率的改善ではなく、デコーディング制約による構造的保証である
- **初回レイテンシ**: 新しいスキーマの初回処理時に追加レイテンシが発生。OpenAIの公式ドキュメントではスキーマの複雑度に応じて数百ミリ秒〜数秒と示唆されている
- **後続リクエスト**: スキーマキャッシュにより追加オーバーヘッドはほぼゼロ

### safety機能: refusal

Structured Outputsでは、安全性上の理由でモデルが応答を拒否する場合、スキーマに準拠したJSONではなく `refusal` フィールドが返される。開発者はこのフィールドをチェックして拒否を検出できる：

```python
if response.choices[0].message.refusal:
    print(f"拒否: {response.choices[0].message.refusal}")
else:
    print(response.choices[0].message.content)  # スキーマ準拠JSON
```

## 運用での学び（Production Lessons）

OpenAIの公式ドキュメントおよびデベロッパーコミュニティから得られる運用上の知見：

- **スキーマ設計の簡潔さ**: ツール定義数は20個以下が推奨。ツール数が増えるとモデルの選択精度が低下する
- **Optional フィールドの扱い**: strict modeでは全フィールドが `required` に含まれる必要がある。オプショナルなフィールドは `{"anyOf": [{"type": "string"}, {"type": "null"}]}` で表現し、`default: null` とする
- **Responses APIへの移行**: 2025年以降、OpenAIはResponses APIを推奨。Function CallingのStructured Outputsは両APIで利用可能だが、新機能はResponses API優先で提供される

## 学術研究との関連（Academic Connection）

OpenAIのStructured Outputsは、以下の学術研究の成果を実装に統合していると考えられる：

- **Efficient Guided Generation (Willard & Louf, 2023)**: FSMベースの制約付きデコーディングの理論基盤。outlinesライブラリとして公開
- **XGrammar (2024)**: OpenAIがXGrammarを直接使用しているかは不明だが、同系統のCFGベース制約デコーディング技術が基盤にあると推測される
- **Constrained Decoding for Structured NLG (Scholak et al., 2021)**: SQL生成における制約付きデコーディングの先行研究

OpenAIは「モデルのfine-tuning + 推論時制約」のハイブリッドアプローチを採用しており、これは学術研究で提案されている制約付きデコーディング単独のアプローチよりもレイテンシと品質のバランスに優れるとされている。

## まとめと実践への示唆

OpenAIのStructured Outputs発表は、Function Callingの `strict: true` モードの技術的基盤を公式に確立した。Zenn記事で紹介されているPydantic v2でのスキーマ一元管理と組み合わせることで、100%のスキーマ準拠を保証しつつ、マルチプロバイダー対応のツール定義を効率的に実装できる。

制約事項として、サポートされるJSON Schemaのサブセット制約（`additionalProperties: false` 必須、`oneOf` 未対応等）があり、複雑なスキーマを使用する場合はローカルフレームワーク（Outlines, XGrammar）との使い分けが重要となる。

## 参考文献

- **Blog URL**: [https://openai.com/index/introducing-structured-outputs-in-the-api/](https://openai.com/index/introducing-structured-outputs-in-the-api/)
- **API Documentation**: [https://developers.openai.com/api/docs/guides/structured-outputs/](https://developers.openai.com/api/docs/guides/structured-outputs/)
- **Function Calling Guide**: [https://developers.openai.com/api/docs/guides/function-calling/](https://developers.openai.com/api/docs/guides/function-calling/)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/b2d1df91e5f5de](https://zenn.dev/0h_n0/articles/b2d1df91e5f5de)
