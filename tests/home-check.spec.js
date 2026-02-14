// ホームページの表示確認
const { test, expect } = require('@playwright/test');

test.describe('ホームページの表示確認', () => {

  test('ホームページのスクリーンショット撮影', async ({ page }) => {
    await page.goto('http://localhost:4000');
    await page.waitForLoadState('networkidle');

    // 全体のスクリーンショット
    await page.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/home-full.png',
      fullPage: true
    });

    // メインコンテンツエリアのみ
    const mainContent = page.locator('#post-list');
    await mainContent.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/home-post-list.png'
    });

    console.log('✅ ホームページのスクリーンショット保存完了');
  });

  test('記事カードの要素確認', async ({ page }) => {
    await page.goto('http://localhost:4000');
    await page.waitForLoadState('networkidle');

    const firstPost = page.locator('.post-preview.card-style').first();
    await expect(firstPost).toBeVisible();

    // 絵文字が表示されているか
    const emoji = firstPost.locator('.post-emoji');
    await expect(emoji).toBeVisible();
    const emojiText = await emoji.textContent();
    console.log('📌 記事タイプ絵文字:', emojiText);

    // タイトルが表示されているか
    const title = firstPost.locator('h1.post-title a');
    await expect(title).toBeVisible();
    const titleText = await title.textContent();
    console.log('📄 タイトル:', titleText);

    // カテゴリバッジが表示されているか
    const categories = firstPost.locator('.category-badge');
    const categoryCount = await categories.count();
    console.log('📁 カテゴリ数:', categoryCount);

    // タグバッジが表示されているか
    const tags = firstPost.locator('.tag-badge');
    const tagCount = await tags.count();
    console.log('🏷️  タグ数:', tagCount);

    // ホバーエフェクトのテスト
    const boundingBox = await firstPost.boundingBox();
    if (boundingBox) {
      await page.mouse.move(boundingBox.x + 50, boundingBox.y + 50);
      await page.waitForTimeout(500);

      // ホバー状態のスクリーンショット
      await firstPost.screenshot({
        path: '/home/relu/misc/zen-auto-create-article/temp/home-card-hover.png'
      });
      console.log('✅ ホバーエフェクトのスクリーンショット保存');
    }
  });
});
