const { test } = require('@playwright/test');

test('最終確認用スクリーンショット', async ({ page }) => {
  await page.goto('http://localhost:4000');
  await page.waitForLoadState('networkidle');
  
  // 通常表示のスクリーンショット
  const sidebar = page.locator('#panel-wrapper');
  await sidebar.screenshot({
    path: '/home/relu/misc/zen-auto-create-article/temp/final-sidebar.png'
  });
  
  // 「最近の更新」セクションのみ
  const recentUpdates = page.locator('#access-lastmod');
  await recentUpdates.screenshot({
    path: '/home/relu/misc/zen-auto-create-article/temp/final-recent-updates.png'
  });
  
  console.log('✅ 最終確認用スクリーンショット保存完了');
});
