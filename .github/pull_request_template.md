## 記事情報

- **タイトル**: <!-- 記事タイトル -->
- **情報源タイプ**: <!-- arxiv/tech_blog/conference -->
- **情報源URL**: <!-- ソースURL -->
- **関連Zenn記事**: <!-- Zenn記事URL -->

## 自動検証結果

<!-- validate_generated_articles.sh の出力をここに貼る -->

```
✅ Frontmatter validation: PASSED
✅ Math blocks: PAIRED
✅ Word count: XXXX chars
⚠️ Content age: X years old (annotation added/not needed)
```

## 手動レビューチェックリスト

### 視覚的確認（Jekyll Preview）

- [ ] 数式が正しくレンダリングされている（MathJax）
- [ ] Mermaidダイアグラムが表示されている（該当する場合）
- [ ] マークダウンの表、リンク、コードブロックが正しく表示されている

### コンテンツ品質

- [ ] タイトルと内容が一致している
- [ ] 情報源の内容を正確に伝えている
- [ ] コード例（ある場合）が実行可能または擬似コードとして妥当
- [ ] 古いコンテンツには適切な注釈がある

### 構造的整合性

- [ ] 冒頭で約束した項目数と実際のセクション数が一致している（例：「5つのポイント」→5個のセクション）
- [ ] 見出しの番号が順序通り（1, 2, 3...）でスキップがない
- [ ] 箇条書きの番号が順序通りでスキップがない
- [ ] 「次のセクションで説明」などの前方参照に対応するセクションが存在する

### 技術的妥当性

- [ ] 数式の変数定義が明記されている
- [ ] アルゴリズムの説明が論理的
- [ ] 実験結果（ある場合）が正確に引用されている

## Jekyll Preview コマンド

```bash
cd 0h-n0.github.io
bundle exec jekyll serve
# http://localhost:4000/posts/{slug}/ を開いて確認
```

## 関連Issue

<!-- zen-auto-create-article の Issue番号 -->
Closes 0h-n0/zen-auto-create-article#<!-- issue番号 -->
