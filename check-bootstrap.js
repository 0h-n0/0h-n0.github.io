const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });

  const ARTICLE_URL = 'https://0h-n0.github.io/posts/techblog-aws-bedrock-structured-outputs/';

  await page.goto(ARTICLE_URL);
  await page.waitForLoadState('networkidle');

  // すべてのCSSファイルをチェック
  const cssFiles = await page.evaluate(() => {
    const sheets = [];
    for (const sheet of document.styleSheets) {
      sheets.push({
        href: sheet.href || '(inline)',
        disabled: sheet.disabled
      });
    }
    return sheets;
  });

  console.log('📄 読み込まれているCSSファイル:');
  cssFiles.forEach((css, i) => {
    console.log(`${i + 1}. ${css.href}`);
    if (css.disabled) console.log('   ⚠️  disabled!');
  });

  // Bootstrap固有のクラスが機能しているか確認
  const bootstrapCheck = await page.evaluate(() => {
    const coreWrapper = document.getElementById('core-wrapper');
    const panelWrapper = document.getElementById('panel-wrapper');

    if (!coreWrapper || !panelWrapper) {
      return { error: 'Elements not found' };
    }

    const coreStyles = window.getComputedStyle(coreWrapper);
    const panelStyles = window.getComputedStyle(panelWrapper);

    // Bootstrapのgridクラスが適用されているか確認
    return {
      coreWrapper: {
        classes: coreWrapper.className,
        display: coreStyles.display,
        flex: coreStyles.flex,
        width: coreStyles.width,
        float: coreStyles.float
      },
      panelWrapper: {
        classes: panelWrapper.className,
        display: panelStyles.display,
        flex: panelStyles.flex,
        width: panelStyles.width,
        float: panelStyles.float
      },
      // Bootstrapの.rowが存在するか
      hasRow: !!document.querySelector('.row')
    };
  });

  console.log('\n🎯 Bootstrap Grid状態:');
  console.log(JSON.stringify(bootstrapCheck, null, 2));

  // Bootstrap CSSが読み込まれているか直接確認
  const hasBootstrapClasses = await page.evaluate(() => {
    const testDiv = document.createElement('div');
    testDiv.className = 'col-6';
    document.body.appendChild(testDiv);
    const styles = window.getComputedStyle(testDiv);
    const hasFlexGrow = styles.flexGrow !== '0' && styles.flexGrow !== '';
    document.body.removeChild(testDiv);

    return {
      hasFlexGrow,
      flexGrow: styles.flexGrow,
      flex: styles.flex,
      width: styles.width
    };
  });

  console.log('\n🔍 Bootstrap .col-* クラステスト:');
  console.log(JSON.stringify(hasBootstrapClasses, null, 2));

  if (!hasBootstrapClasses.hasFlexGrow) {
    console.log('\n❌ Bootstrap CSSが正しく読み込まれていません！');
  } else {
    console.log('\n✅ Bootstrap CSSは読み込まれています');
  }

  await browser.close();
})();
