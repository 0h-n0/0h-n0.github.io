const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });

  const ARTICLE_URL = 'https://0h-n0.github.io/posts/techblog-aws-bedrock-structured-outputs/';

  await page.goto(ARTICLE_URL);
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(3000);

  // Mermaidコードブロックの状態を確認
  const codeBlockInfo = await page.evaluate(() => {
    const mermaidCodeBlocks = document.querySelectorAll('code.language-mermaid, pre code.language-mermaid');
    const preMermaid = document.querySelectorAll('pre.mermaid');

    return {
      languageMermaidCount: mermaidCodeBlocks.length,
      preMermaidCount: preMermaid.length,
      details: Array.from(mermaidCodeBlocks).map((block, i) => ({
        index: i,
        parentTag: block.parentElement.tagName,
        parentClass: block.parentElement.className,
        content: block.textContent.substring(0, 100)
      }))
    };
  });

  console.log('📋 Mermaidコードブロック:');
  console.log(JSON.stringify(codeBlockInfo, null, 2));

  // Mermaid変数の状態
  const mermaidState = await page.evaluate(() => {
    if (typeof mermaid === 'undefined') {
      return { loaded: false };
    }

    return {
      loaded: true,
      initialized: typeof mermaid.initialize === 'function',
      initCalled: typeof mermaid.init === 'function',
      config: mermaid.mermaidAPI?.getConfig?.() || {},
      version: mermaid.version
    };
  });

  console.log('\n🎨 Mermaid状態:');
  console.log(JSON.stringify(mermaidState, null, 2));

  // 手動でMermaidを初期化してみる
  const manualInitResult = await page.evaluate(async () => {
    if (typeof mermaid === 'undefined') {
      return 'Mermaid not loaded';
    }

    try {
      // コードブロックを.mermaidクラスに変換
      const mermaidCodeBlocks = document.querySelectorAll('code.language-mermaid');
      let converted = 0;

      mermaidCodeBlocks.forEach((block) => {
        const pre = block.parentElement;
        const code = block.textContent;
        const newPre = document.createElement('pre');
        newPre.className = 'mermaid';
        newPre.textContent = code;
        pre.parentElement.insertBefore(newPre, pre.nextSibling);
        converted++;
      });

      // Mermaidを初期化
      mermaid.initialize({
        startOnLoad: false,
        theme: 'default',
        logLevel: 'debug'
      });

      // 手動でレンダリング
      await mermaid.run({
        querySelector: '.mermaid'
      });

      return {
        success: true,
        converted,
        rendered: document.querySelectorAll('svg[id^="mermaid"]').length
      };
    } catch (e) {
      return {
        success: false,
        error: e.message,
        stack: e.stack
      };
    }
  });

  console.log('\n🔧 手動初期化結果:');
  console.log(JSON.stringify(manualInitResult, null, 2));

  // レンダリング後のスクリーンショット
  await page.waitForTimeout(2000);

  const mermaidElements = page.locator('.mermaid, svg[id^="mermaid"]');
  const count = await mermaidElements.count();

  console.log(`\n📊 レンダリングされたMermaid要素: ${count}個`);

  if (count > 0) {
    await mermaidElements.first().screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/mermaid-rendered.png'
    });
    console.log('✅ Mermaid図のスクリーンショット保存');
  }

  await page.screenshot({
    path: '/home/relu/misc/zen-auto-create-article/temp/mermaid-debug-full.png',
    fullPage: true
  });

  await browser.close();
})();
