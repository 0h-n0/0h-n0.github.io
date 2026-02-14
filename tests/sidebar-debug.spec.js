// デバッグ用: 日付表示の詳細確認
const { test, expect } = require('@playwright/test');

test('日付表示のCSS詳細確認', async ({ page }) => {
  await page.goto('http://localhost:4000');
  await page.waitForLoadState('networkidle');

  const sidebar = page.locator('#access-lastmod');
  const firstItem = sidebar.locator('li').first();
  const dateElement = firstItem.locator('.post-date');

  // HTMLを出力
  const html = await firstItem.innerHTML();
  console.log('=== HTML構造 ===');
  console.log(html);

  // CSS詳細を取得
  const styles = await dateElement.evaluate(el => {
    const computed = window.getComputedStyle(el);
    return {
      display: computed.display,
      visibility: computed.visibility,
      opacity: computed.opacity,
      color: computed.color,
      fontSize: computed.fontSize,
      position: computed.position,
      marginTop: computed.marginTop,
      marginBottom: computed.marginBottom,
      paddingLeft: computed.paddingLeft,
      height: computed.height,
      overflow: computed.overflow,
      backgroundColor: computed.backgroundColor
    };
  });

  console.log('\n=== 日付要素のCSS ===');
  console.log(JSON.stringify(styles, null, 2));

  // 親要素(li)のスタイルも確認
  const liStyles = await firstItem.evaluate(el => {
    const computed = window.getComputedStyle(el);
    return {
      height: computed.height,
      overflow: computed.overflow,
      display: computed.display,
      marginBottom: computed.marginBottom
    };
  });

  console.log('\n=== 親要素(li)のCSS ===');
  console.log(JSON.stringify(liStyles, null, 2));

  // スクリーンショット（日付要素の境界を強調）
  await page.evaluate(() => {
    const dates = document.querySelectorAll('#access-lastmod .post-date');
    dates.forEach(date => {
      date.style.border = '2px solid red';
      date.style.backgroundColor = 'yellow';
    });
  });

  await page.screenshot({
    path: '/home/relu/misc/zen-auto-create-article/temp/debug-highlighted.png',
    fullPage: true
  });

  console.log('\n✅ デバッグ用スクリーンショット保存（日付を赤枠・黄背景で強調）');
});
