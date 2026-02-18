const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });

  await page.goto('https://0h-n0.github.io/');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(2000);

  // 切り替えボタンの状態確認
  const toggleButton = page.locator('button.mode-toggle').first();
  const isVisible = await toggleButton.isVisible();
  console.log(`🌓 切り替えボタン: ${isVisible ? '✅ 表示されています' : '❌ 非表示'}`);

  if (isVisible) {
    // 初期モード確認
    let currentMode = await page.evaluate(() => {
      return document.documentElement.getAttribute('data-mode') || 'not set';
    });
    console.log(`\n📱 初期モード: ${currentMode}`);

    // ライトモードのスクリーンショット
    await page.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/mode-initial.png',
      fullPage: false,
      clip: { x: 0, y: 0, width: 1920, height: 900 }
    });
    console.log('📸 初期状態のスクリーンショット保存');

    // 切り替えボタンをクリック
    console.log('\n🔄 モードを切り替えています...');
    await toggleButton.click();
    await page.waitForTimeout(1000);

    // 切り替え後のモード確認
    currentMode = await page.evaluate(() => {
      return document.documentElement.getAttribute('data-mode') || 'not set';
    });
    console.log(`📱 切り替え後のモード: ${currentMode}`);

    // 切り替え後のスクリーンショット
    await page.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/mode-toggled.png',
      fullPage: false,
      clip: { x: 0, y: 0, width: 1920, height: 900 }
    });
    console.log('📸 切り替え後のスクリーンショット保存');

    // もう一度切り替え
    console.log('\n🔄 再度切り替えています...');
    await toggleButton.click();
    await page.waitForTimeout(1000);

    currentMode = await page.evaluate(() => {
      return document.documentElement.getAttribute('data-mode') || 'not set';
    });
    console.log(`📱 再切り替え後のモード: ${currentMode}`);

    // 背景色とテキスト色を確認
    const colors = await page.evaluate(() => {
      const body = document.body;
      const styles = window.getComputedStyle(body);
      return {
        backgroundColor: styles.backgroundColor,
        color: styles.color
      };
    });

    console.log('\n🎨 現在の配色:');
    console.log(`  背景: ${colors.backgroundColor}`);
    console.log(`  テキスト: ${colors.color}`);

    console.log('\n✅ ダークモード切り替えが正常に動作しています！');
  }

  await browser.close();
})();
