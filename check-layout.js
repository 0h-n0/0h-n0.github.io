const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });

  const ARTICLE_URL = 'https://0h-n0.github.io/posts/techblog-aws-bedrock-structured-outputs/';

  await page.goto(ARTICLE_URL);
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(2000);

  // レイアウト構造を確認
  const layoutInfo = await page.evaluate(() => {
    const body = document.body;
    const sidebar = document.getElementById('sidebar');
    const topbar = document.getElementById('topbar-wrapper');
    const mainWrapper = document.getElementById('main-wrapper');
    const coreWrapper = document.getElementById('core-wrapper');
    const panelWrapper = document.getElementById('panel-wrapper');

    const getElementInfo = (el, name) => {
      if (!el) return { name, exists: false };

      const rect = el.getBoundingClientRect();
      const styles = window.getComputedStyle(el);

      return {
        name,
        exists: true,
        rect: {
          top: Math.round(rect.top),
          left: Math.round(rect.left),
          right: Math.round(rect.right),
          bottom: Math.round(rect.bottom),
          width: Math.round(rect.width),
          height: Math.round(rect.height)
        },
        styles: {
          display: styles.display,
          position: styles.position,
          float: styles.float,
          width: styles.width,
          marginLeft: styles.marginLeft,
          marginRight: styles.marginRight
        }
      };
    };

    return {
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      body: getElementInfo(body, 'body'),
      sidebar: getElementInfo(sidebar, 'sidebar'),
      topbar: getElementInfo(topbar, 'topbar'),
      mainWrapper: getElementInfo(mainWrapper, 'main-wrapper'),
      coreWrapper: getElementInfo(coreWrapper, 'core-wrapper'),
      panelWrapper: getElementInfo(panelWrapper, 'panel-wrapper')
    };
  });

  console.log('📐 レイアウト情報:');
  console.log(JSON.stringify(layoutInfo, null, 2));

  // サイドバーが正しい位置にあるか確認
  if (layoutInfo.panelWrapper.exists) {
    const panel = layoutInfo.panelWrapper.rect;
    console.log(`\n📊 右サイドバー (panel-wrapper) の位置:`);
    console.log(`  Left: ${panel.left}px`);
    console.log(`  Width: ${panel.width}px`);
    console.log(`  Viewport width: ${layoutInfo.viewport.width}px`);

    if (panel.left > layoutInfo.viewport.width) {
      console.log('  ❌ 警告: サイドバーが画面の右外に配置されています！');
    } else if (panel.left + panel.width > layoutInfo.viewport.width) {
      console.log('  ⚠️  警告: サイドバーの一部が画面外にはみ出しています');
    } else {
      console.log('  ✅ サイドバーは画面内に表示されています');
    }
  }

  // スクリーンショット（デスクトップビュー）
  await page.screenshot({
    path: '/home/relu/misc/zen-auto-create-article/temp/layout-desktop.png',
    fullPage: false
  });

  // モバイルビューで確認
  await page.setViewportSize({ width: 768, height: 1024 });
  await page.waitForTimeout(1000);

  const mobileLayout = await page.evaluate(() => {
    const panelWrapper = document.getElementById('panel-wrapper');
    if (!panelWrapper) return null;

    const rect = panelWrapper.getBoundingClientRect();
    const styles = window.getComputedStyle(panelWrapper);

    return {
      visible: rect.width > 0 && rect.height > 0,
      display: styles.display,
      position: styles.position,
      width: styles.width,
      rect: {
        left: Math.round(rect.left),
        width: Math.round(rect.width)
      }
    };
  });

  console.log('\n📱 モバイルビュー (768px):');
  console.log(JSON.stringify(mobileLayout, null, 2));

  await page.screenshot({
    path: '/home/relu/misc/zen-auto-create-article/temp/layout-mobile.png',
    fullPage: false
  });

  console.log('\n✅ レイアウトチェック完了');

  await browser.close();
})();
