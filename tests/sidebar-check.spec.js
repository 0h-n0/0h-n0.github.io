// Playwright テスト: 右サイドバーの日付表示確認
const { test, expect } = require('@playwright/test');

test.describe('右サイドバーの表示確認', () => {

  test.beforeEach(async ({ page }) => {
    // ローカルサーバーにアクセス
    await page.goto('http://localhost:4000');
    // ページが完全に読み込まれるまで待機
    await page.waitForLoadState('networkidle');
  });

  test('「最近の更新」セクションが表示されている', async ({ page }) => {
    // サイドバーが存在することを確認
    const sidebar = page.locator('#access-lastmod');
    await expect(sidebar).toBeVisible();

    // 見出しが「最近の更新」であることを確認
    const heading = sidebar.locator('.panel-heading');
    await expect(heading).toContainText('最近の更新');
  });

  test('記事タイトルと日付が表示されている', async ({ page }) => {
    const sidebar = page.locator('#access-lastmod');

    // 最初の記事アイテムを取得
    const firstItem = sidebar.locator('li').first();
    await expect(firstItem).toBeVisible();

    // タイトルリンクが存在することを確認
    const titleLink = firstItem.locator('a');
    await expect(titleLink).toBeVisible();
    const titleText = await titleLink.textContent();
    console.log('📄 記事タイトル:', titleText);

    // 日付要素が存在することを確認
    const dateElement = firstItem.locator('.post-date');
    await expect(dateElement).toBeVisible();
    const dateText = await dateElement.textContent();
    console.log('📅 日付:', dateText?.trim());

    // 日付が空でないことを確認
    expect(dateText?.trim()).not.toBe('');
  });

  test('複数の記事が表示されている', async ({ page }) => {
    const sidebar = page.locator('#access-lastmod');
    const items = sidebar.locator('li');

    // 少なくとも1つ以上の記事があることを確認
    const count = await items.count();
    console.log(`📊 表示されている記事数: ${count}`);
    expect(count).toBeGreaterThan(0);

    // 各記事に日付が含まれていることを確認
    for (let i = 0; i < Math.min(count, 5); i++) {
      const item = items.nth(i);
      const dateElement = item.locator('.post-date');
      await expect(dateElement).toBeVisible();
      const dateText = await dateElement.textContent();
      console.log(`記事 ${i + 1} の日付: ${dateText?.trim()}`);
    }
  });

  test('スタイルが正しく適用されている', async ({ page }) => {
    const sidebar = page.locator('#access-lastmod');
    const firstItem = sidebar.locator('li').first();

    // タイトルのフォントウェイトを確認
    const titleLink = firstItem.locator('a');
    const fontWeight = await titleLink.evaluate(el =>
      window.getComputedStyle(el).fontWeight
    );
    console.log('タイトルのフォントウェイト:', fontWeight);

    // 日付のフォントサイズを確認
    const dateElement = firstItem.locator('.post-date');
    const fontSize = await dateElement.evaluate(el =>
      window.getComputedStyle(el).fontSize
    );
    console.log('日付のフォントサイズ:', fontSize);

    // マージンボトムを確認
    const marginBottom = await firstItem.evaluate(el =>
      window.getComputedStyle(el).marginBottom
    );
    console.log('記事間の余白:', marginBottom);
  });

  test('スクリーンショットを撮影', async ({ page }) => {
    // サイドバー全体のスクリーンショット
    const sidebar = page.locator('#panel-wrapper');
    await sidebar.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/playwright-sidebar.png'
    });

    // 「最近の更新」セクションのみのスクリーンショット
    const recentUpdates = page.locator('#access-lastmod');
    await recentUpdates.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/playwright-recent-updates.png'
    });

    console.log('✅ スクリーンショットを保存しました');
  });

  test('日付フォーマットの確認', async ({ page }) => {
    const sidebar = page.locator('#access-lastmod');
    const firstItem = sidebar.locator('li').first();
    const dateElement = firstItem.locator('.post-date em');

    // data-ts 属性（タイムスタンプ）が存在することを確認
    const hasTimestamp = await dateElement.evaluate(el => el.hasAttribute('data-ts'));
    console.log('タイムスタンプ属性の有無:', hasTimestamp);

    if (hasTimestamp) {
      const timestamp = await dateElement.getAttribute('data-ts');
      console.log('タイムスタンプ値:', timestamp);

      // timeago クラスが存在することを確認
      const hasTimeagoClass = await dateElement.evaluate(el =>
        el.classList.contains('timeago')
      );
      console.log('timeago クラスの有無:', hasTimeagoClass);
    }
  });
});
