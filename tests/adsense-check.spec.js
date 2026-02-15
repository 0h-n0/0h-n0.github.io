const { test, expect } = require('@playwright/test');

test.describe('Google AdSense 配置確認（本番ビルド時のみ）', () => {

  test('AdSenseスクリプトが読み込まれている', async ({ page }) => {
    await page.goto('http://localhost:4000');
    await page.waitForLoadState('networkidle');

    // 最初の記事をクリック
    await page.locator('#post-list .post-preview h1.post-title a').first().click();
    await page.waitForLoadState('networkidle');

    // AdSenseスクリプトの存在確認
    const adScriptExists = await page.evaluate(() => {
      return typeof window.adsbygoogle !== 'undefined';
    });

    console.log(`📜 AdSenseスクリプト読み込み: ${adScriptExists ? '✅' : '❌'}`);
    expect(adScriptExists).toBe(true);
  });

  test('広告がページレイアウトを崩していない', async ({ page }) => {
    await page.goto('http://localhost:4000');
    await page.waitForLoadState('networkidle');

    // 最初の記事をクリック
    await page.locator('#post-list .post-preview h1.post-title a').first().click();
    await page.waitForLoadState('networkidle');

    // メインコンテンツの幅が適切か確認（divタグの.post-contentに限定）
    const postContent = page.locator('div.post-content').first();
    const contentWidth = await postContent.evaluate(el => el.offsetWidth);

    console.log(`📐 記事本文の幅: ${contentWidth}px`);
    expect(contentWidth).toBeGreaterThan(300); // 最小幅確保
    expect(contentWidth).toBeLessThan(1200); // 最大幅を超えない
  });

  test('AdSenseスクリプトのエラーがない', async ({ page }) => {
    const consoleErrors = [];

    // コンソールエラーを収集
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto('http://localhost:4000');
    await page.waitForLoadState('networkidle');

    // 最初の記事をクリック
    await page.locator('#post-list .post-preview h1.post-title a').first().click();
    await page.waitForLoadState('networkidle');

    // AdSense関連のエラーをフィルタリング
    const adsenseErrors = consoleErrors.filter(err =>
      err.includes('adsbygoogle') || err.includes('googlesyndication')
    );

    if (adsenseErrors.length > 0) {
      console.error('❌ AdSense関連のエラー:', adsenseErrors);
    } else {
      console.log('✅ AdSense関連のエラーなし');
    }

    expect(adsenseErrors.length).toBe(0);
  });
});
