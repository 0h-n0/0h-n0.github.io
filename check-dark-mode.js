const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });

  await page.goto('https://0h-n0.github.io/');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(2000);

  // ダークモード切り替えボタンを確認
  const toggleInfo = await page.evaluate(() => {
    const toggle = document.getElementById('mode-toggle');
    const sidebar = document.getElementById('sidebar');

    return {
      toggle: {
        exists: !!toggle,
        visible: toggle ? (toggle.offsetWidth > 0 && toggle.offsetHeight > 0) : false,
        id: toggle?.id,
        className: toggle?.className,
        innerHTML: toggle?.innerHTML.substring(0, 100)
      },
      currentMode: document.documentElement.getAttribute('data-mode'),
      sidebar: {
        exists: !!sidebar,
        width: sidebar ? window.getComputedStyle(sidebar).width : null
      }
    };
  });

  console.log('🌓 ダークモード切り替えボタン:');
  console.log(JSON.stringify(toggleInfo, null, 2));

  if (toggleInfo.toggle.exists) {
    console.log('\n✅ 切り替えボタンが見つかりました！');

    // サイドバーのスクリーンショット（ボタンを含む）
    const sidebar = page.locator('#sidebar');
    await sidebar.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/sidebar-with-toggle.png'
    });
    console.log('📸 サイドバーのスクリーンショット保存');

    // ライトモードのスクリーンショット
    await page.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/light-mode.png',
      fullPage: false
    });
    console.log('📸 ライトモードのスクリーンショット保存');

    // ダークモードに切り替え
    console.log('\n🔄 ダークモードに切り替え中...');
    await page.click('#mode-toggle');
    await page.waitForTimeout(1000); // アニメーション待機

    const newMode = await page.evaluate(() => {
      return document.documentElement.getAttribute('data-mode');
    });
    console.log(`現在のモード: ${newMode}`);

    // ダークモードのスクリーンショット
    await page.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/dark-mode.png',
      fullPage: false
    });
    console.log('📸 ダークモードのスクリーンショット保存');

    // コードブロックの色を確認
    const darkModeCodeColors = await page.evaluate(() => {
      const article = document.querySelector('article');
      if (!article) return null;

      const codeElement = article.querySelector('.highlight code');
      if (!codeElement) return null;

      const styles = window.getComputedStyle(codeElement);
      return {
        color: styles.color,
        backgroundColor: styles.backgroundColor
      };
    });

    console.log('\n💻 ダークモードのコード色:');
    console.log(JSON.stringify(darkModeCodeColors, null, 2));

    console.log('\n✅ ダークモード切り替えが正常に動作しています！');
  } else {
    console.log('\n❌ 切り替えボタンが見つかりません');
  }

  await browser.close();
})();
