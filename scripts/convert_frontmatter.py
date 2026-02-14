#!/usr/bin/env python3
"""Convert Chirpy frontmatter to Minimal Mistakes format."""

import re
from pathlib import Path


def convert_frontmatter(content: str) -> str:
    """Convert Chirpy frontmatter to Minimal Mistakes format."""
    lines = content.split('\n')

    if not lines[0].strip() == '---':
        return content  # No frontmatter

    # Find end of frontmatter
    try:
        end_idx = lines[1:].index('---') + 1
    except ValueError:
        return content  # Invalid frontmatter

    frontmatter_lines = lines[1:end_idx]
    body = '\n'.join(lines[end_idx + 1:])

    # Parse frontmatter
    fm_dict = {}
    for line in frontmatter_lines:
        if ':' in line:
            key, value = line.split(':', 1)
            fm_dict[key.strip()] = value.strip()

    # Convert layout
    if fm_dict.get('layout') == 'post':
        fm_dict['layout'] = 'single'

    # Convert description to excerpt
    if 'description' in fm_dict:
        fm_dict['excerpt'] = fm_dict.pop('description')

    # Convert categories from [TechBlog] to YAML array
    categories = fm_dict.get('categories', '')
    if categories.startswith('[') and categories.endswith(']'):
        # Extract category names
        cats = categories.strip('[]').split(',')
        fm_dict['categories'] = [c.strip() for c in cats]

    # Convert tags from [tag1, tag2] to YAML array
    tags = fm_dict.get('tags', '')
    if tags.startswith('[') and tags.endswith(']'):
        # Extract tag names
        tag_list = tags.strip('[]').split(',')
        fm_dict['tags'] = [t.strip() for t in tag_list]

    # Add Minimal Mistakes specific fields
    fm_dict['toc'] = 'true'
    fm_dict['toc_sticky'] = 'true'

    # Reconstruct frontmatter
    new_fm = ['---']

    # Preserve order: layout, title, excerpt, categories, tags, toc, toc_sticky
    order = ['layout', 'title', 'excerpt', 'categories', 'tags', 'toc', 'toc_sticky']

    for key in order:
        if key in fm_dict:
            value = fm_dict[key]
            if isinstance(value, list):
                # YAML array format
                new_fm.append(f'{key}:')
                for item in value:
                    new_fm.append(f'  - {item}')
            else:
                new_fm.append(f'{key}: {value}')

    new_fm.append('---')

    return '\n'.join(new_fm) + '\n' + body


def main():
    """Convert all blog posts."""
    posts_dir = Path('_posts/blog')

    if not posts_dir.exists():
        print(f"Error: {posts_dir} not found")
        return

    converted_count = 0

    for md_file in posts_dir.glob('*.md'):
        print(f"Converting: {md_file.name}")

        content = md_file.read_text(encoding='utf-8')
        new_content = convert_frontmatter(content)

        if new_content != content:
            md_file.write_text(new_content, encoding='utf-8')
            converted_count += 1
            print(f"  ✓ Converted")
        else:
            print(f"  - No changes needed")

    print(f"\nTotal converted: {converted_count} files")


if __name__ == '__main__':
    main()
