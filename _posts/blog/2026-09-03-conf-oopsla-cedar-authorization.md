---
layout: post
title: "OOPSLA 2024論文解説: Cedar - 表現力・高速性・安全性・分析可能性を両立する認可言語"
description: "AgentCore Policyの基盤技術Cedarの原論文を解説。Lean形式検証、SMTベースのポリシー分析、Rego比42-80倍の性能を実現する設計"
categories: [blog, paper, conference]
tags: [Cedar, authorization, formal-verification, OOPSLA, SMT, Lean, aws, bedrock, security]
date: 2026-09-03 09:40:00 +0900
source_type: conference
conference: OOPSLA
source_url: https://arxiv.org/abs/2403.04651
zenn_article: 391fc1f0476f7a
zenn_url: https://zenn.dev/0h_n0/articles/391fc1f0476f7a
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## 論文概要（Abstract）

本記事は [Cedar: A New Language for Expressive, Fast, Safe, and Analyzable Authorization (Extended Version)](https://arxiv.org/abs/2403.04651)（OOPSLA 2024、arXiv: 2403.04651）の解説記事です。

Cedarは、認可判定（Authorization Decision）に特化したポリシー言語であり、表現力・高速性・安全性・分析可能性という4つの設計目標を両立している。Lean 4定理証明器による形式モデル、Rustによるリファレンス実装、SMTソルバーによるポリシー分析を統合し、RBAC・ABAC・ReBACの3つの認可モデルをサポートする。著者らのベンチマークでは、OpenFGAの28.7-35.2倍、Regoの42.8-80.8倍の評価性能を達成したと報告されている。Amazon Bedrock AgentCore PolicyはこのCedarをエージェントのツール認可基盤として採用している。

この記事は [Zenn記事: Bedrock AgentCore Runtime×Gatewayで顧客サポートエージェントを構築しツール認証を設計する](https://zenn.dev/0h_n0/articles/391fc1f0476f7a) の深掘りです。

## 情報源

- **会議名**: OOPSLA 2024（ACM SIGPLAN Conference on Object-Oriented Programming, Systems, Languages, and Applications）
- **年**: 2024
- **URL**: [https://arxiv.org/abs/2403.04651](https://arxiv.org/abs/2403.04651)
- **著者**: Joseph W. Cutler, Craig Disselkoen, Aaron Eline, Shaobo He et al.（15名）
- **分野**: cs.PL（プログラミング言語）

## カンファレンス情報

OOPSLAはACM SIGPLANが主催するプログラミング言語・ソフトウェア工学の主要会議の1つであり、形式手法・型理論・言語設計の分野で高い評価を受けている。Cedarの論文がこの会議に採択されたことは、認可ポリシー言語の設計における形式手法の適用が学術的に評価されたことを意味する。

## 背景と動機（Background & Motivation）

認可（Authorization）は「リクエストを許可するか拒否するか」を判定するセキュリティの基本機能である。従来の認可システムには以下の課題がある。

- **Rego（Open Policy Agent）**: チューリング完全で柔軟だが、ポリシーの停止性が保証されず、形式検証が困難
- **OpenFGA（Google Zanzibar系）**: 関係ベースアクセス制御に特化するが、属性ベースの条件記述に制約がある
- **カスタム実装**: アプリケーション固有のハードコードされた認可ロジックは、監査・分析が困難

著者らはこれらの課題に対し、「言語設計の段階で表現力を意図的に制限することで、高速性・安全性・分析可能性を獲得する」というトレードオフを選択している。

## 技術的詳細（Technical Details）

### Cedarの形式的セマンティクス

Cedarの式評価は小ステップ操作的セマンティクス（small-step operational semantics）として定義されている。評価判定は以下の形式をとる。

$$
\mu, \sigma \vdash e \longrightarrow e'
$$

ここで、
- $\mu$: エンティティストア（エンティティ参照から属性レコードと祖先集合へのマッピング）
- $\sigma$: 認可リクエスト（変数から値へのマッピング）
- $e$: Cedar式
- $e'$: 評価後の式

**寛容なセマンティクス（Forgiving Semantics）**: Cedarの評価は、存在しないエンティティ参照に対する操作（等値比較等）を一般的にエラーとせず成功させる。これにより、スキーマ変更時のポリシーの後方互換性が確保される。

### エンティティモデルとヒエラルキー

Cedarのエンティティは以下の構造を持つ。

```
EntityType::id {
    attributes: { key: value, ... },
    parents: { ParentType::id1, ParentType::id2, ... }
}
```

ヒエラルキーは有向非巡回グラフ（DAG）を形成し、`in`演算子で所属判定を行う。

```
// principal が admin チームに所属しているか
principal in Team::"admin"
```

### 型システム: シングルトン型と静的能力

Cedarの型システムには2つの独創的な機構が含まれている。

**シングルトン型（Singleton Types）**: 真偽値のシングルトン型`True`と`False`を導入し、式が確実にその値に評価されることを表現する。これにより、到達不能なコードの除去と、オプショナル属性アクセス前の存在チェックの検証が可能となる。

**静的能力（Static Capabilities）**: 型判定$\alpha; \Gamma \vdash e : \tau; \varepsilon$において、$\alpha$は式評価開始時に利用可能なオプショナル属性、$\varepsilon$は式が`true`に評価された場合に利用可能と証明される属性を表す。

```
// has でオプショナル属性の存在をチェックしてからアクセス
when { principal has department && principal.department == "support" }
```

上記のパターンで、`has`チェックが$\varepsilon$に`department`を追加し、後続の`principal.department`アクセスが型安全であることを静的に保証する。

### 認可アルゴリズム

認可判定は以下の4ステップで実行される。

1. **ポリシースライシング**: リクエストの`principal`と`resource`のヒエラルキーに基づき、関連するポリシーのみを抽出
2. **ポリシー評価**: 抽出された各ポリシーを評価し、`true`に評価されるポリシーを収集
3. **判定ロジック**: `forbid`ポリシーが1つでもマッチすれば拒否。`forbid`がなく`permit`が1つ以上マッチすれば許可
4. **デフォルト拒否**: いずれのポリシーもマッチしなければ拒否

$$
\text{Decision} = \begin{cases} \text{DENY} & \text{if } \exists p \in \text{forbid}: p = \text{true} \\ \text{ALLOW} & \text{if } \exists p \in \text{permit}: p = \text{true} \\ \text{DENY} & \text{otherwise (default deny)} \end{cases}
$$

この「Forbid Wins + Default Deny」の原則は、Zenn記事のAgentCore GatewayにおけるCedarポリシー評価の基盤となっている。

### 3つの認可モデル

Cedarは単一の言語でRBAC・ABAC・ReBACの3つの認可モデルを自然にサポートする。

**RBAC（ロールベース）**:
```
permit(principal in Team::"support", action, resource);
```

**ABAC（属性ベース）**:
```
permit(principal, action, resource)
when { resource.owner == principal };
```

**ReBAC（関係ベース）**:
```
permit(principal, action == Action::"GetTicket", resource)
when { principal in resource.assigned_agents };
```

Zenn記事のCedarポリシー例（`principal.department == "support"`による部門制御）はABACモデルに該当する。

## Lean形式モデルと安全性証明

### 証明された5つの性質

著者らはCedarのセマンティクスをLean 4で形式モデル化し、以下の5つの性質を証明している。

1. **Forbid trumps permit**: マッチする`forbid`ポリシーが1つでも存在すればリクエストは拒否される
2. **Default deny**: いかなる`permit`ポリシーもマッチしなければリクエストは拒否される
3. **Explicit allow**: 許可には少なくとも1つの`permit`ポリシーの`true`評価が必要
4. **Sound slicing**: スライスの外側のポリシーはリクエストを満たし得ない
5. **Validation soundness**: スキーマに対して型検査を通過したポリシーは、スキーマに準拠するエンティティに対して型不一致や属性欠損によるランタイムエラーを起こさない

### Rustリファレンス実装との整合性

Lean形式モデルとRust実装の整合性は、cargo fuzzを用いた差分ランダムテストとプロパティベーステストで検証されている。著者らは開発中に約24のバグを発見・修正したと報告している。

## SMTベースのポリシー分析

### シンボリックコンパイル

Cedarポリシーは、自動推論のためにSMT論理式にコンパイルされる。エンコーディングは以下の要素で構成される。

- **型エンコーディング$\mathcal{T}$**: プリミティブ型はSMTの対応する型（Bool、BitVec 64、String）に直接マッピング。エンティティはエンティティIDフィールド付きの代数的データ型としてエンコード
- **エンティティストア**: 非解釈関数（Uninterpreted Functions）として表現。属性関数$\text{Entity} \rightarrow \text{Record}$と祖先関数$\text{Entity} \rightarrow \text{Set}(\text{ParentType})$
- **整形式制約**: 量化子なし（ground）の制約で有効なエンティティヒエラルキーを保証

### 健全性・完全性・決定可能性

著者らは、このSMTエンコーディングが以下の3つの性質を満たすことを示している。

- **健全性**: 制約を満たすSMTモデルは、有効なCedar評価に対応する
- **完全性**: 任意の有効なCedar評価は、あるSMTモデルに対応する
- **決定可能性**: 決定可能なSMT理論のみを使用し、全称量化子を含まない

著者らは「非自明なポリシー言語に対するこのような完全なエンコーディングは初めて」と述べている。この健全性と完全性の組み合わせにより、SMT分析の結果が偽陽性・偽陰性なく信頼できることが保証される。

実際のユースケースとして、以下のような分析クエリが実行可能である。

- 「このポリシーセットのもとで、`support`部門以外のユーザーがチケット管理ツールにアクセスできる入力が存在するか？」
- 「ポリシーAを追加した場合、既存のポリシーセットと矛盾が生じるか？」

## 実験結果（Results）

### 評価性能ベンチマーク

著者らは3つのアプリケーション（gdrive、github、TinyTodo）でCedarの評価性能をOpenFGAおよびRegoと比較している。

| 比較対象 | 性能比（Cedarが高速） |
|---|---|
| Cedar vs. OpenFGA | **28.7× - 35.2×** |
| Cedar vs. Rego | **42.8× - 80.8×** |
| ポリシースライシング効果 | **10.0× - 18.0×** |

（論文のベンチマーク結果より）

### SMT分析性能

- **平均エンコード・解決時間**: 75.1 ms
- 3つのサンプルアプリケーションでテスト

この75.1msという分析時間は、CI/CDパイプラインでのポリシー変更検証において実用的な水準である。ただし、リクエスト処理パス上でのリアルタイム分析には適さないため、ポリシー変更時のオフライン検証に使用することが想定されている。

## 実装のポイント（Implementation）

### 設計上のトレードオフ

Cedarが高速性と分析可能性を獲得するために行った表現力の制限は以下の通り。

| 制限 | 理由 | 影響 |
|---|---|---|
| ループ・再帰なし | 線形時間評価を保証 | チューリング完全ではない |
| 有限集合のみ | 無限データ構造を回避 | 無限ストリーム処理は不可 |
| 整数乗算はリテラル定数のみ | SMT理論の決定可能性を確保 | 動的なコスト計算は不可 |
| テンプレートスロットは`?principal`と`?resource`のみ | インデックスの単純性 | 汎用パラメトリックポリシーは不可 |

### AgentCore Policyでの活用

Zenn記事のAgentCore Gatewayでは、以下のCedar機能が活用されている。

1. **ABAC（属性ベース）**: JWTクレーム（department、role等）をエンティティ属性として参照し、ツールアクセスを制御
2. **Forbid Wins**: 特定ツールへのアクセスを絶対に禁止するハードバウンダリの設定
3. **ポリシースライシング**: MCP `tools/list`リクエスト時に、関連ポリシーのみを効率的に評価
4. **部分評価**: 現在のコンテキストで「常に拒否される」ツールを事前に除外し、LLMへの提示対象から除去

## 関連研究（Related Work）

- **Rego / Open Policy Agent** (Styra, 2015-): チューリング完全な汎用ポリシー言語。Cedar論文のベンチマークでは42.8-80.8倍遅いと報告されている。柔軟性が高い反面、ポリシーの停止性や安全性の形式検証が困難
- **OpenFGA / Google Zanzibar** (2019): 関係ベースアクセス制御に特化。高速だがCedarよりも28.7-35.2倍遅い。属性条件の表現力が限定的
- **Progent** (Shi et al., 2025, arXiv: 2504.11703): LLM + SMTソルバーによるエージェント権限制御。CedarのSMTエンコーディングとProgentのZ3ベースポリシー比較は技術的に共通する基盤を持つ

## まとめと今後の展望

Cedar論文は、認可言語の設計において「表現力を意図的に制限することで、高速性・安全性・分析可能性を獲得する」という設計原則を明確にし、Lean形式検証とSMTベースの分析という2つの数学的手法でその正しさを裏付けている。AgentCore Policyがこの言語を採用したことは、エージェントのツール認可という新しいドメインにおいて、形式手法に裏打ちされた認可基盤が実用レベルに達したことを示している。CNCFへの参加により、クラウドネイティブ認可の標準技術としての発展が期待される。

## 参考文献

- **Conference URL / arXiv**: [https://arxiv.org/abs/2403.04651](https://arxiv.org/abs/2403.04651)
- **Code**: [https://github.com/cedar-policy/cedar](https://github.com/cedar-policy/cedar)
- **Cedar公式サイト**: [https://www.cedarpolicy.com/](https://www.cedarpolicy.com/)
- **OOPSLA 2024**: [Proceedings of the ACM on Programming Languages, Vol. 8, OOPSLA1](https://doi.org/10.1145/3649835)
- **Related Zenn article**: [https://zenn.dev/0h_n0/articles/391fc1f0476f7a](https://zenn.dev/0h_n0/articles/391fc1f0476f7a)
