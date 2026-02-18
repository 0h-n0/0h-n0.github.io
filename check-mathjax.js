const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  const ARTICLE_URL = 'https://0h-n0.github.io/posts/techblog-aws-bedrock-structured-outputs/';

  await page.goto(ARTICLE_URL);
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(3000);

  // すべてのscriptタグを取得
  const scripts = await page.evaluate(() => {
    const scriptTags = Array.from(document.querySelectorAll('script'));
    return scriptTags.map(s => ({
      src: s.src || '(inline)',
      id: s.id || '',
      content: s.src ? '' : s.textContent.substring(0, 200)
    }));
  });

  console.log('📜 読み込まれているscriptタグ:');
  scripts.forEach((s, i) => {
    console.log(`\n${i + 1}. ID: ${s.id || '(none)'}`);
    console.log(`   SRC: ${s.src}`);
    if (s.content) {
      console.log(`   CONTENT: ${s.content}...`);
    }
  });

  // MathJaxのバージョンと設定を確認
  const mathJaxInfo = await page.evaluate(() => {
    if (typeof window.MathJax === 'undefined') {
      return { loaded: false };
    }

    return {
      loaded: true,
      version: window.MathJax.version,
      config: {
        tex: window.MathJax.config?.tex || window.MathJax.tex || {},
        options: window.MathJax.config?.options || window.MathJax.options || {}
      },
      startup: window.MathJax.startup,
      typesetPromise: typeof window.MathJax.typesetPromise
    };
  });

  console.log('\n📐 MathJax情報:');
  console.log(JSON.stringify(mathJaxInfo, null, 2));

  // 手動でMathJaxを実行してみる
  const typesetResult = await page.evaluate(async () => {
    if (typeof window.MathJax === 'undefined') {
      return 'MathJax not loaded';
    }

    if (typeof window.MathJax.typesetPromise === 'function') {
      try {
        await window.MathJax.typesetPromise();
        return 'Typeset completed';
      } catch (e) {
        return 'Typeset error: ' + e.message;
      }
    } else if (typeof window.MathJax.typeset === 'function') {
      try {
        window.MathJax.typeset();
        return 'Typeset (sync) completed';
      } catch (e) {
        return 'Typeset error: ' + e.message;
      }
    } else {
      return 'No typeset method found';
    }
  });

  console.log('\n🔄 手動typeset結果:', typesetResult);

  // 再度数式をチェック
  await page.waitForTimeout(2000);
  const mathCheck = await page.evaluate(() => {
    return {
      mjxContainers: document.querySelectorAll('mjx-container').length,
      unprocessedInline: (document.body.innerText.match(/\$[^$]+\$/g) || []).length
    };
  });

  console.log('📐 手動typeset後の数式:', mathCheck);

  if (mathCheck.mjxContainers > 0) {
    console.log('✅ 手動typesetで数式がレンダリングされました！');
  } else {
    console.log('❌ 手動typesetでも数式がレンダリングされませんでした');
  }

  await page.screenshot({
    path: '/home/relu/misc/zen-auto-create-article/temp/mathjax-check.png',
    fullPage: true
  });

  await browser.close();
})();
