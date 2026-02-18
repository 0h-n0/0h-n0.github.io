const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage({ viewport: { width: 1280, height: 1024 } });

  const ARTICLE_URL = 'https://0h-n0.github.io/posts/techblog-aws-bedrock-structured-outputs/';

  console.log('🌐 ページを開いています:', ARTICLE_URL);
  await page.goto(ARTICLE_URL);
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(3000); // MathJax読み込み待機

  // MathJaxの読み込み確認
  const mathJaxLoaded = await page.evaluate(() => {
    return typeof window.MathJax !== 'undefined';
  });
  console.log(`📐 MathJax読み込み: ${mathJaxLoaded ? '✅' : '❌'}`);

  // 数式要素の確認
  const mathElements = await page.evaluate(() => {
    return {
      mjxContainers: document.querySelectorAll('mjx-container').length,
      unprocessedInline: (document.body.innerText.match(/\$[^$]+\$/g) || []).length,
      unprocessedDisplay: (document.body.innerText.match(/\$\$[^$]+\$\$/g) || []).length
    };
  });
  console.log('📐 数式要素:', mathElements);

  if (mathElements.mjxContainers > 0) {
    console.log('✅ 数式が正しくレンダリングされています！');
  } else if (mathElements.unprocessedInline > 0 || mathElements.unprocessedDisplay > 0) {
    console.log('❌ 数式が未処理のまま残っています');
  }

  // コードブロックの色確認
  const codeColors = await page.evaluate(() => {
    const codeElement = document.querySelector('.highlight code');
    if (!codeElement) return null;

    const codeStyles = window.getComputedStyle(codeElement);
    const nameElement = document.querySelector('.highlight .n');
    const nameStyles = nameElement ? window.getComputedStyle(nameElement) : null;

    return {
      code: {
        color: codeStyles.color,
        backgroundColor: codeStyles.backgroundColor
      },
      name: nameStyles ? {
        color: nameStyles.color,
        backgroundColor: nameStyles.backgroundColor
      } : null
    };
  });
  console.log('💻 コードの色:', JSON.stringify(codeColors, null, 2));

  if (codeColors && codeColors.name) {
    const nameColor = codeColors.name.color;
    if (nameColor !== 'rgba(0, 0, 0, 0)' && nameColor !== 'transparent') {
      console.log('✅ 変数名の色が正しく設定されています！');
    } else {
      console.log('❌ 変数名の色がまだ透明です');
    }
  }

  // スクリーンショット
  console.log('📸 スクリーンショットを保存しています...');
  await page.screenshot({
    path: '/home/relu/misc/zen-auto-create-article/temp/verify-full-page.png',
    fullPage: true
  });

  const codeBlock = page.locator('.highlight').first();
  if (await codeBlock.count() > 0) {
    await codeBlock.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/verify-code-block.png'
    });
  }

  console.log('✅ スクリーンショット保存完了');
  console.log('   - temp/verify-full-page.png');
  console.log('   - temp/verify-code-block.png');

  await browser.close();
  console.log('🎉 検証完了！');
})();
