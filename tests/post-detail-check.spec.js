// 記事詳細ページの UI/UX 検証
const { test, expect } = require('@playwright/test');

test.describe('記事詳細ページの表示確認', () => {

  test('記事タイトルに絵文字が表示されている', async ({ page }) => {
    await page.goto('http://localhost:4000');
    await page.waitForLoadState('networkidle');

    // 最初の記事をクリック
    await page.locator('#post-list .post-preview h1.post-title a').first().click();
    await page.waitForLoadState('networkidle');

    // タイトルに絵文字が含まれているか確認
    const titleEmoji = page.locator('h1[data-toc-skip] .post-emoji');
    await expect(titleEmoji).toBeVisible();
    const emojiText = await titleEmoji.textContent();
    console.log('📌 記事タイトル絵文字:', emojiText);

    // 絵文字が空でないことを確認
    expect(emojiText?.trim()).toBeTruthy();
  });

  test('カテゴリがバッジ化されている', async ({ page }) => {
    await page.goto('http://localhost:4000');
    await page.waitForLoadState('networkidle');

    // 最初の記事をクリック
    await page.locator('#post-list .post-preview h1.post-title a').first().click();
    await page.waitForLoadState('networkidle');

    // カテゴリバッジが表示されているか確認
    const categoryBadges = page.locator('.post-tail-wrapper .post-meta .category-badge');
    const count = await categoryBadges.count();

    if (count > 0) {
      console.log('📁 カテゴリバッジ数:', count);
      await expect(categoryBadges.first()).toBeVisible();

      // スタイルが適用されているか確認
      const bgColor = await categoryBadges.first().evaluate(el =>
        window.getComputedStyle(el).backgroundColor
      );
      console.log('🎨 カテゴリバッジ背景色:', bgColor);
      expect(bgColor).not.toBe('rgba(0, 0, 0, 0)'); // 透明でないことを確認
    } else {
      console.log('⚠️  この記事にはカテゴリがありません');
    }
  });

  test('タグがバッジ化されている', async ({ page }) => {
    await page.goto('http://localhost:4000');
    await page.waitForLoadState('networkidle');

    // 最初の記事をクリック
    await page.locator('#post-list .post-preview h1.post-title a').first().click();
    await page.waitForLoadState('networkidle');

    // タグバッジが表示されているか確認
    const tagBadges = page.locator('.post-tail-wrapper .post-tags .tag-badge');
    const count = await tagBadges.count();

    if (count > 0) {
      console.log('🏷️  タグバッジ数:', count);
      await expect(tagBadges.first()).toBeVisible();

      // スタイルが適用されているか確認
      const bgColor = await tagBadges.first().evaluate(el =>
        window.getComputedStyle(el).backgroundColor
      );
      console.log('🎨 タグバッジ背景色:', bgColor);
      expect(bgColor).not.toBe('rgba(0, 0, 0, 0)'); // 透明でないことを確認
    } else {
      console.log('⚠️  この記事にはタグがありません');
    }
  });

  test('関連記事カードに絵文字とバッジが表示されている', async ({ page }) => {
    await page.goto('http://localhost:4000');
    await page.waitForLoadState('networkidle');

    // 最初の記事をクリック
    await page.locator('#post-list .post-preview h1.post-title a').first().click();
    await page.waitForLoadState('networkidle');

    // ページの一番下までスクロール（関連記事セクションを表示）
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(500);

    // 関連記事カードが存在するか確認
    const relatedCards = page.locator('#related-posts .related-post-card');
    const cardCount = await relatedCards.count();

    if (cardCount > 0) {
      console.log('📚 関連記事カード数:', cardCount);

      // 最初のカードを検証
      const firstCard = relatedCards.first();

      // 絵文字が表示されているか
      const emoji = firstCard.locator('h3 .post-emoji');
      await expect(emoji).toBeVisible();
      const emojiText = await emoji.textContent();
      console.log('📌 関連記事絵文字:', emojiText);

      // カテゴリバッジが表示されているか（存在する場合）
      const categoryBadge = firstCard.locator('.category-badge');
      const hasCategoryBadge = await categoryBadge.count() > 0;
      console.log('📁 カテゴリバッジあり:', hasCategoryBadge);

      // タグバッジが表示されているか（存在する場合）
      const tagBadge = firstCard.locator('.tag-badge');
      const hasTagBadge = await tagBadge.count() > 0;
      console.log('🏷️  タグバッジあり:', hasTagBadge);

      // スクリーンショット撮影
      await firstCard.screenshot({
        path: '/home/relu/misc/zen-auto-create-article/temp/related-post-card.png'
      });
      console.log('✅ 関連記事カードのスクリーンショット保存');
    } else {
      console.log('⚠️  関連記事がありません');
    }
  });

  test('記事詳細ページ全体のスクリーンショット', async ({ page }) => {
    await page.goto('http://localhost:4000');
    await page.waitForLoadState('networkidle');

    // 最初の記事をクリック
    await page.locator('#post-list .post-preview h1.post-title a').first().click();
    await page.waitForLoadState('networkidle');

    // 全体のスクリーンショット
    await page.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/post-detail-full.png',
      fullPage: true
    });

    // タイトルエリアのスクリーンショット
    const titleArea = page.locator('h1[data-toc-skip]');
    await titleArea.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/post-detail-title.png'
    });

    // カテゴリ・タグエリアのスクリーンショット
    const tailWrapper = page.locator('.post-tail-wrapper');
    await tailWrapper.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/post-detail-tail.png'
    });

    console.log('✅ 記事詳細ページのスクリーンショット保存完了');
  });
});
