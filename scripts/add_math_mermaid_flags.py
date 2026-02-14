#!/usr/bin/env python3
"""既存記事に math/mermaid フラグを追加"""

from pathlib import Path
import re


def add_flags_to_frontmatter(content: str) -> tuple[str, bool]:
    """
    frontmatterに math/mermaid フラグを追加

    Returns:
        (新しいコンテンツ, 変更があったか)
    """
    lines = content.split('\n')

    if not lines or lines[0].strip() != '---':
        return content, False

    try:
        end_idx = lines[1:].index('---') + 1
    except ValueError:
        return content, False

    frontmatter = lines[1:end_idx]
    body = '\n'.join(lines[end_idx + 1:])

    # math/mermaid フラグが既に存在するかチェック
    has_math = any('math:' in line for line in frontmatter)
    has_mermaid = any('mermaid:' in line for line in frontmatter)

    changed = False
    if not has_math:
        frontmatter.append('math: true')
        changed = True
    if not has_mermaid:
        frontmatter.append('mermaid: true')
        changed = True

    if changed:
        new_content = '---\n' + '\n'.join(frontmatter) + '\n---\n' + body
        return new_content, True
    else:
        return content, False


def main():
    posts_dir = Path('_posts/blog')

    if not posts_dir.exists():
        print(f"Error: {posts_dir} does not exist")
        return

    updated_count = 0
    skipped_count = 0

    for md_file in sorted(posts_dir.glob('*.md')):
        print(f"Processing: {md_file.name}")
        content = md_file.read_text(encoding='utf-8')
        new_content, changed = add_flags_to_frontmatter(content)

        if changed:
            md_file.write_text(new_content, encoding='utf-8')
            print(f"  ✓ Added math/mermaid flags")
            updated_count += 1
        else:
            print(f"  - Already has flags")
            skipped_count += 1

    print(f"\nSummary: {updated_count} updated, {skipped_count} skipped")


if __name__ == '__main__':
    main()
