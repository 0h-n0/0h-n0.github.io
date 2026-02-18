#!/usr/bin/env python3
"""
すべてのブログ記事のfrontmatterに math: true と mermaid: true を追加するスクリプト
"""
import os
import re
from pathlib import Path

def process_frontmatter(file_path):
    """記事ファイルのfrontmatterに math と mermaid を追加"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # frontmatterを抽出
    match = re.match(r'^---\n(.*?)\n---\n(.*)$', content, re.DOTALL)
    if not match:
        print(f"⚠️  Frontmatter not found: {file_path}")
        return False

    frontmatter = match.group(1)
    body = match.group(2)

    # 既に math または mermaid が設定されているかチェック
    has_math = re.search(r'^math:\s*(true|false)', frontmatter, re.MULTILINE)
    has_mermaid = re.search(r'^mermaid:\s*(true|false)', frontmatter, re.MULTILINE)

    modified = False

    # math: true を追加（存在しない場合）
    if not has_math:
        frontmatter += "\nmath: true"
        modified = True

    # mermaid: true を追加（存在しない場合）
    if not has_mermaid:
        frontmatter += "\nmermaid: true"
        modified = True

    if not modified:
        return False

    # ファイルを更新
    new_content = f"---\n{frontmatter}\n---\n{body}"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return True

def main():
    blog_dir = Path("_posts/blog")

    if not blog_dir.exists():
        print(f"❌ Blog directory not found: {blog_dir}")
        return

    markdown_files = list(blog_dir.glob("*.md"))
    print(f"📝 Found {len(markdown_files)} markdown files")

    updated_count = 0
    for md_file in markdown_files:
        if process_frontmatter(md_file):
            print(f"✅ Updated: {md_file.name}")
            updated_count += 1
        else:
            print(f"⏭️  Skipped: {md_file.name}")

    print(f"\n🎉 Updated {updated_count} files out of {len(markdown_files)}")

if __name__ == "__main__":
    main()
