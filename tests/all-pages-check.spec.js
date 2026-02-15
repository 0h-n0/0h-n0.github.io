const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

// sitemap.xmlからURL一覧を取得
function getSitemapUrls() {
  const sitemapPath = path.join(__dirname, '../_site/sitemap.xml');
  if (!fs.existsSync(sitemapPath)) {
    console.warn('⚠️ sitemap.xml not found. Build site first with `bundle exec jekyll build`');
    return [];
  }

  const sitemapContent = fs.readFileSync(sitemapPath, 'utf-8');
  const urlMatches = sitemapContent.matchAll(/<loc>(.*?)<\/loc>/g);
  const urls = Array.from(urlMatches, m => m[1]);

  // sitemap.xmlは既に完全なURL（http://localhost:4000/... または https://0h-n0.github.io/...）を含むのでそのまま返す
  return urls;
}

// ページタイプを判定
function getPageType(url) {
  // 完全なURLからパス部分を抽出（http://localhost:4000/posts/... → /posts/...）
  const urlObj = new URL(url);
  const pathname = urlObj.pathname;

  if (pathname === '/') return 'home';
  if (pathname.startsWith('/posts/')) return 'post';
  if (pathname.startsWith('/categories/')) return 'category';
  if (pathname.startsWith('/tags/')) return 'tag';
  if (pathname.startsWith('/year-archive')) return 'archives';
  return 'page';
}

test.describe('全ページ表示確認', () => {
  const urls = getSitemapUrls();

  if (urls.length === 0) {
    test.skip('sitemap.xmlが見つかりません。`bundle exec jekyll build`を実行してください。');
    return;
  }

  // リソースエラー検出
  test.beforeEach(async ({ page }) => {
    page.on('response', response => {
      if (response.status() >= 400) {
        console.error(`❌ リソースエラー: ${response.url()} - ${response.status()}`);
      }
    });
  });

  for (const url of urls) {
    const pageType = getPageType(url);

    test(`${pageType}: ${url}`, async ({ page }) => {
      // ページロード成功（HTTP 200）
      const response = await page.goto(url);
      expect(response.status()).toBe(200);

      await page.waitForLoadState('networkidle');

      // ページタイプごとの必須要素確認
      switch (pageType) {
        case 'post':
          // 記事タイトル
          await expect(page.locator('h1[data-toc-skip]')).toBeVisible();
          // 記事本文（divタグに限定）
          await expect(page.locator('div.post-content').first()).toBeVisible();
          // カテゴリ・タグエリア
          await expect(page.locator('div.post-tail-wrapper').first()).toBeVisible();
          break;

        case 'home':
          // 記事リスト
          await expect(page.locator('#post-list')).toBeVisible();
          const postPreviews = page.locator('.post-preview.card-style');
          expect(await postPreviews.count()).toBeGreaterThan(0);
          break;

        case 'category':
        case 'tag':
          // カテゴリ/タグカード
          await expect(page.locator('.card.categories, .list-group')).toBeVisible();
          break;

        case 'archives':
          // アーカイブリスト
          await expect(page.locator('#archives')).toBeVisible();
          break;

        default:
          // 汎用ページ（最低限の確認 - 最初の見出しを確認）
          await expect(page.locator('h1, h2').first()).toBeVisible();
      }

      // 共通要素: フッター
      await expect(page.locator('footer')).toBeVisible();
    });
  }
});

// パフォーマンスチェック（サンプル10件）
test('主要ページのパフォーマンス確認', async ({ page }) => {
  const urls = getSitemapUrls().slice(0, 10);
  const slowPages = [];

  for (const url of urls) {
    const startTime = Date.now();
    await page.goto(url);
    await page.waitForLoadState('networkidle');
    const loadTime = Date.now() - startTime;

    console.log(`⏱️  ${url}: ${loadTime}ms`);

    if (loadTime > 5000) {
      slowPages.push({ url, loadTime });
    }
  }

  if (slowPages.length > 0) {
    console.warn('⚠️  5秒以上かかったページ:', slowPages);
  }

  expect(slowPages.length).toBeLessThan(3); // 10件中3件未満が許容範囲
});
