const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });

  const ARTICLE_URL = 'https://0h-n0.github.io/posts/techblog-aws-bedrock-structured-outputs/';

  console.log('🌐 ページを開いています:', ARTICLE_URL);
  await page.goto(ARTICLE_URL);
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(3000); // Mermaid読み込み待機

  // 右サイドバーの確認
  const sidebarInfo = await page.evaluate(() => {
    const panelWrapper = document.querySelector('#panel-wrapper');
    const accessLastmod = document.querySelector('#access-lastmod');
    const accessTags = document.querySelector('#access-tags');

    return {
      panelWrapper: {
        exists: !!panelWrapper,
        visible: panelWrapper ? (panelWrapper.offsetWidth > 0 && panelWrapper.offsetHeight > 0) : false,
        display: panelWrapper ? window.getComputedStyle(panelWrapper).display : null,
        position: panelWrapper ? window.getComputedStyle(panelWrapper).position : null,
        width: panelWrapper ? window.getComputedStyle(panelWrapper).width : null
      },
      accessLastmod: {
        exists: !!accessLastmod,
        count: accessLastmod ? accessLastmod.querySelectorAll('li').length : 0
      },
      accessTags: {
        exists: !!accessTags,
        count: accessTags ? accessTags.querySelectorAll('a').length : 0
      }
    };
  });

  console.log('📊 右サイドバーの状態:');
  console.log(JSON.stringify(sidebarInfo, null, 2));

  // サイドバーのスクリーンショット
  const panelWrapper = page.locator('#panel-wrapper');
  if (await panelWrapper.count() > 0) {
    await panelWrapper.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/sidebar-check.png'
    });
    console.log('✅ サイドバーのスクリーンショット保存');
  } else {
    console.log('❌ サイドバーが見つかりません');
  }

  // Mermaid図の確認
  const mermaidInfo = await page.evaluate(() => {
    const mermaidDivs = document.querySelectorAll('.mermaid, [data-mermaid-type]');
    const svgs = document.querySelectorAll('svg[id^="mermaid"]');

    const info = {
      mermaidDivs: mermaidDivs.length,
      mermaidSvgs: svgs.length,
      mermaidLoaded: typeof window.mermaid !== 'undefined',
      details: []
    };

    mermaidDivs.forEach((div, i) => {
      const styles = window.getComputedStyle(div);
      info.details.push({
        index: i,
        tagName: div.tagName,
        className: div.className,
        display: styles.display,
        visibility: styles.visibility,
        width: styles.width,
        height: styles.height,
        hasError: div.querySelector('.error') !== null,
        innerHTML: div.innerHTML.substring(0, 100)
      });
    });

    return info;
  });

  console.log('\n📈 Mermaid図の状態:');
  console.log(JSON.stringify(mermaidInfo, null, 2));

  // Mermaid図のスクリーンショット
  const mermaidElements = page.locator('.mermaid, [data-mermaid-type]');
  const mermaidCount = await mermaidElements.count();

  if (mermaidCount > 0) {
    console.log(`\n📊 Mermaid図: ${mermaidCount}個見つかりました`);

    for (let i = 0; i < Math.min(mermaidCount, 3); i++) {
      await mermaidElements.nth(i).screenshot({
        path: `/home/relu/misc/zen-auto-create-article/temp/mermaid-${i + 1}.png`
      });
      console.log(`✅ Mermaid図 ${i + 1} のスクリーンショット保存`);
    }
  } else {
    console.log('❌ Mermaid図が見つかりません');
  }

  // ページ全体のスクリーンショット
  await page.screenshot({
    path: '/home/relu/misc/zen-auto-create-article/temp/page-full-check.png',
    fullPage: true
  });

  console.log('\n✅ 検証完了！');

  await browser.close();
})();
