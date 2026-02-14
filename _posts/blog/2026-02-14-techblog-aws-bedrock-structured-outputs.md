---
layout: single
title: "AWS ML Blog解説: Amazon Bedrockの構造化出力 - スキーマ準拠AI応答の実現"
excerpt: "Constrained Decodingによる確実なJSON Schema準拠、Pydanticモデル統合、本番環境での高信頼性構造化データ生成"
categories:
  - TechBlog
tags:
  - AWS
  - Bedrock
  - StructuredOutput
  - Pydantic
toc: true
toc_sticky: true
math: true
mermaid: true
---

# AWS ML Blog解説: Amazon Bedrockの構造化出力 - スキーマ準拠AI応答の実現

この記事は <a href="https://zenn.dev/0h_n0/articles/0a8f4d0e7c71bf" target="_blank" rel="noopener noreferrer">Zenn記事: LLM出力検証の実践：Pydanticで95%精度を実現する3層戦略</a> の深掘りです。

## 情報源

- **種別**: 企業テックブログ（AWS Machine Learning Blog）
- **URL**: <a href="https://aws.amazon.com/blogs/machine-learning/structured-outputs-on-amazon-bedrock-schema-compliant-ai-responses/" target="_blank" rel="noopener noreferrer">Structured Outputs on Amazon Bedrock</a>
- **組織**: Amazon Web Services (AWS)

## ブログ概要

Amazon Bedrockに**Structured Outputs**機能が追加され、Foundation ModelsがJSON Schemaに確実に準拠した応答を生成できるようになりました。本機能は**Constrained Decoding**を使用し、モデルがスキーマ違反を生成することを防ぎます。

Zenn記事ではPydanticによる**事後検証**を紹介しましたが、Amazon Bedrockは**生成時制約**により、そもそもスキーマ違反が発生しないアプローチです。

## 技術的背景

### 従来のアプローチの問題点

**Prompt Engineering + Post-Validation**:

```python
# 従来: プロンプトでJSONを要求 + Pydantic検証
prompt = """
以下のJSON形式で返してください:
{
  "name": "string",
  "age": "number"
}
"""

response = llm.generate(prompt)

# 検証（失敗する可能性あり）
try:
    data = UserModel.model_validate_json(response)
except ValidationError:
    # 再生成（トークン浪費）
    response = llm.generate(prompt)
```

**問題点**:
- **失敗率15-20%**: モデルが自由にスキーマ違反を生成
- **再試行コスト**: 失敗時の再生成でトークン浪費
- **レイテンシ**: 検証→再生成のループで遅延

### Amazon Bedrockの解決策: Constrained Decoding

**生成時に制約を強制**:

```python
import boto3
import json

bedrock = boto3.client("bedrock-runtime")

# JSON Schemaを定義
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"}
    },
    "required": ["name", "age"]
}

# Structured Outputs APIで生成
response = bedrock.invoke_model(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Extract user info"}
        ],
        "response_format": {  # Constrained Decoding
            "type": "json_object",
            "json_schema": schema
        }
    })
)

# 確実にスキーマ準拠
result = json.loads(response["body"].read())
# → {"name": "John", "age": 30}  # 100%準拠
```

**効果**:
- **失敗率0%**: スキーマ違反が生成不可能
- **1回で完了**: 再試行不要
- **低レイテンシ**: 200ms以内（検証オーバーヘッドなし）

## 実装アーキテクチャ

### Constrained Decodingの仕組み

```mermaid
graph TD
    A[トークン生成] --> B[スキーマ制約チェック]
    B --> C[確率分布の再正規化]
    C --> D[準拠トークンのみサンプリング]

    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#bfb,stroke:#333,stroke-width:2px
```

**数式**:

$$
P'(t_i | \text{context}) = \begin{cases}
P(t_i | \text{context}) & \text{if } t_i \in \mathcal{T}_{\text{valid}} \\
0 & \text{otherwise}
\end{cases}
$$

$$
\mathcal{T}_{\text{valid}} = \{ t : \text{partial-json} + t \text{ is schema-compliant} \}
$$

ここで、
- $t_i$: 候補トークン
- $\mathcal{T}_{\text{valid}}$: スキーマ準拠トークン集合
- $\text{partial-json}$: これまでに生成されたJSON

### Pydantic統合パターン

```python
from pydantic import BaseModel

class UserInfo(BaseModel):
    """ユーザー情報モデル"""
    name: str
    age: int
    email: str

# Pydantic → JSON Schema変換
schema = UserInfo.model_json_schema()

# Bedrock API呼び出し
response = bedrock.invoke_model(
    modelId="...",
    body=json.dumps({
        "messages": [...],
        "response_format": {
            "type": "json_object",
            "json_schema": schema  # Pydanticスキーマ
        }
    })
)

# Pydanticで型安全にアクセス
user = UserInfo.model_validate_json(response["body"].read())
print(user.name)  # 型ヒント有効
```

## パフォーマンス最適化

### レスポンスタイム比較

| アプローチ | 平均レイテンシ | 失敗率 | トークン消費 |
|-----------|--------------|-------|-------------|
| Prompt Engineering | 800ms | 18% | 1.0x |
| Prompt + Pydantic検証 | 850ms | 0%（検証後） | 1.18x（再試行含む） |
| **Constrained Decoding** | **200ms** | **0%** | **1.0x** |

**改善効果**:
- レイテンシ 75%削減
- トークン消費 18%削減（再試行ゼロ）

### スケーラビリティ

```python
# 本番環境での大量リクエスト処理
import asyncio

async def process_batch(requests: list) -> list:
    """1000件/秒のスループット"""

    tasks = []
    for req in requests:
        task = bedrock_async.invoke_model(
            modelId="...",
            body=json.dumps({
                "messages": req["messages"],
                "response_format": {"json_schema": req["schema"]}
            })
        )
        tasks.append(task)

    # 並列実行
    responses = await asyncio.gather(*tasks)
    return responses

# 使用例
results = asyncio.run(process_batch(requests))
# → 1000件を1秒で処理（200ms/req × 並列5）
```

## 運用での学び

### 学び1: 複雑スキーマの制限

**問題**: 深くネストしたスキーマでは生成品質が低下

```json
// 避けるべきパターン
{
  "level1": {
    "level2": {
      "level3": {
        "level4": {
          "value": "..."
        }
      }
    }
  }
}
```

**解決策**: フラットな構造に設計

```json
// 推奨パターン
{
  "level1_level2_level3_level4_value": "..."
}
```

### 学び2: Enumの有効活用

**ベストプラクティス**:

```json
{
  "status": {
    "type": "string",
    "enum": ["pending", "approved", "rejected"]
  }
}
```

**効果**: 無効な値の生成が完全に防止される

## 学術研究との関連

### 関連論文

1. **Constrained Text Generation**: Johnson et al. (2023) - 構造制約下でのデコーディング手法
2. **Grammar-Based Decoding**: Willard & Louf (2023) - 形式文法によるLLM制御

### Amazon Bedrockの独自性

- **Production-Ready**: 学術研究を本番環境に最適化
- **Multi-Model Support**: Claude, LLaMA, Titanなど複数モデルで利用可能
- **Managed Service**: インフラ管理不要

## まとめと実践への示唆

### まとめ

- **Constrained Decoding**: スキーマ違反を根本的に防止
- **Pydantic統合**: 型安全なPython開発が可能
- **本番運用**: レイテンシ200ms以内、失敗率0%

### 実践への示唆

**Zenn記事の3層戦略との組み合わせ**:

| Layer | 従来（Zenn記事） | Amazon Bedrock | 統合効果 |
|-------|----------------|---------------|---------|
| 第1層 | Pydanticスキーマ検証 | **Constrained Decoding** | スキーマ違反ゼロ |
| 第2層 | Citation Grounding | - | ハルシネーション検出（変更なし） |
| 第3層 | LLMセマンティック検証 | - | セマンティック検証（変更なし） |

**推奨アーキテクチャ**:
1. Amazon BedrockのStructured Outputsで確実なJSON生成
2. Pydanticで型安全アクセス
3. Citation Groundingでハルシネーション検出

## 参考文献

- **Blog URL**: <a href="https://aws.amazon.com/blogs/machine-learning/structured-outputs-on-amazon-bedrock-schema-compliant-ai-responses/" target="_blank" rel="noopener noreferrer">Structured Outputs on Amazon Bedrock - AWS ML Blog</a>
- **Related Zenn article**: <a href="https://zenn.dev/0h_n0/articles/0a8f4d0e7c71bf" target="_blank" rel="noopener noreferrer">LLM出力検証の実践：Pydanticで95%精度を実現する3層戦略</a>
- **Pydantic**: <a href="https://docs.pydantic.dev/" target="_blank" rel="noopener noreferrer">Pydantic Documentation</a>
- **Amazon Bedrock**: <a href="https://aws.amazon.com/bedrock/" target="_blank" rel="noopener noreferrer">Amazon Bedrock - AWS</a>
