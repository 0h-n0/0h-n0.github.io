const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });

  await page.goto('https://0h-n0.github.io/');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(2000);

  // サイドバーの全要素を確認
  const sidebarContent = await page.evaluate(() => {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return { error: 'Sidebar not found' };

    // すべてのbutton要素を探す
    const buttons = Array.from(sidebar.querySelectorAll('button'));

    // すべてのaタグで"mode"や"theme"を含むものを探す
    const links = Array.from(sidebar.querySelectorAll('a'));

    // id や class に "mode" または "theme" を含む要素
    const modeElements = Array.from(document.querySelectorAll('[id*="mode"], [class*="mode"], [id*="theme"], [class*="theme"]'));

    return {
      buttons: buttons.map(btn => ({
        id: btn.id,
        className: btn.className,
        text: btn.textContent.trim(),
        innerHTML: btn.innerHTML.substring(0, 100)
      })),
      links: links.filter(a => a.textContent.toLowerCase().includes('mode') || a.textContent.toLowerCase().includes('theme')).map(a => ({
        id: a.id,
        className: a.className,
        text: a.textContent.trim(),
        href: a.href
      })),
      modeElements: modeElements.map(el => ({
        tagName: el.tagName,
        id: el.id,
        className: el.className,
        text: el.textContent.trim().substring(0, 50)
      })),
      sidebarHTML: sidebar.innerHTML.substring(0, 500)
    };
  });

  console.log('🔍 サイドバーの内容:');
  console.log(JSON.stringify(sidebarContent, null, 2));

  // スクリーンショット
  await page.screenshot({
    path: '/home/relu/misc/zen-auto-create-article/temp/sidebar-full.png',
    fullPage: true
  });

  await browser.close();
})();
