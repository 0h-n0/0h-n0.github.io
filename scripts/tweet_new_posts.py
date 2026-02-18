"""
Tweet new blog posts to Twitter/X.

Detects newly added _posts/blog/*.md files in the latest commit,
parses their frontmatter, and posts a tweet for each one.

Usage:
    TWITTER_BEARER_TOKEN=<token> python scripts/tweet_new_posts.py
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

TWEET_API_URL = "https://api.x.com/2/tweets"
SITE_BASE_URL = "https://0h-n0.github.io"
MAX_TWEET_LENGTH = 280
MAX_DESCRIPTION_LENGTH = 110
INTER_TWEET_DELAY = 3


def get_new_post_files() -> list[Path]:
    """Return list of newly added _posts/blog/*.md files in HEAD vs HEAD~1."""
    result = subprocess.run(
        [
            "git", "diff", "HEAD~1", "--name-only", "--diff-filter=A",
            "--", "_posts/blog/*.md",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    paths = [Path(p.strip()) for p in result.stdout.splitlines() if p.strip()]
    return paths


def parse_frontmatter(path: Path) -> dict:
    """Parse YAML frontmatter from a Jekyll post file."""
    content = path.read_text(encoding="utf-8")
    if not content.startswith("---"):
        return {}
    # Extract between the first two '---' delimiters
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        return yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}


def build_slug(file_path: Path) -> str:
    """Convert YYYY-MM-DD-slug.md filename to Jekyll permalink slug."""
    stem = file_path.stem  # e.g. '2026-02-18-paper-textgrad-2406-07496'
    slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", stem)
    return slug


def build_tweet_text(title: str, description: str, url: str, tags: list[str]) -> str:
    """Assemble tweet text, trimming description if necessary to stay within 280 chars."""
    hashtags = " ".join(f"#{t}" for t in tags[:3])

    def assemble(desc: str) -> str:
        parts = [f"📝 {title}", "", desc, "", f"👉 {url}"]
        if hashtags:
            parts += ["", hashtags]
        return "\n".join(parts)

    text = assemble(description)
    if len(text) <= MAX_TWEET_LENGTH:
        return text

    # Trim description until it fits
    trimmed = description[:MAX_DESCRIPTION_LENGTH]
    text = assemble(trimmed + "…")
    while len(text) > MAX_TWEET_LENGTH and len(trimmed) > 0:
        trimmed = trimmed[:-1]
        text = assemble(trimmed + "…")

    return text


def post_tweet(text: str, bearer_token: str) -> None:
    """Post a single tweet via X API v2."""
    response = requests.post(
        TWEET_API_URL,
        headers={
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
        },
        json={"text": text},
        timeout=30,
    )
    response.raise_for_status()


def main() -> None:
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        print("Error: TWITTER_BEARER_TOKEN is not set.", file=sys.stderr)
        sys.exit(1)

    new_files = get_new_post_files()
    if not new_files:
        print("No new posts detected. Exiting.")
        return

    print(f"Found {len(new_files)} new post(s).")
    errors: list[str] = []

    for i, file_path in enumerate(new_files):
        if i > 0:
            time.sleep(INTER_TWEET_DELAY)

        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping.", file=sys.stderr)
            continue

        fm = parse_frontmatter(file_path)
        title = fm.get("title", file_path.stem)
        description = fm.get("description", "")
        tags = fm.get("tags", [])

        slug = build_slug(file_path)
        url = f"{SITE_BASE_URL}/posts/{slug}/"

        tweet_text = build_tweet_text(title, description, url, tags)

        print(f"Tweeting: {file_path.name}")
        print(f"  chars={len(tweet_text)}")

        try:
            post_tweet(tweet_text, bearer_token)
            print("  -> OK")
        except requests.HTTPError as exc:
            msg = f"Failed to tweet {file_path.name}: {exc} | body={exc.response.text}"
            print(msg, file=sys.stderr)
            errors.append(msg)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
