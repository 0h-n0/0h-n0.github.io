// 本番サイトの具体的な問題診断
const { test, expect } = require('@playwright/test');

const ARTICLE_URL = 'https://0h-n0.github.io/posts/techblog-aws-bedrock-structured-outputs/';

test.describe('本番サイトの具体的問題確認', () => {

  test('右サイドバーのスクリーンショット', async ({ page }) => {
    await page.goto('https://0h-n0.github.io/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);

    // 右サイドバー（Recent Updates）
    const rightSidebar = page.locator('#panel-wrapper');
    if (await rightSidebar.count() > 0) {
      await rightSidebar.screenshot({
        path: '/home/relu/misc/zen-auto-create-article/temp/issue-right-sidebar.png'
      });
      console.log('✅ 右サイドバーのスクリーンショット保存');

      // Recent Updatesセクション
      const recentUpdates = page.locator('#access-lastmod');
      if (await recentUpdates.count() > 0) {
        await recentUpdates.screenshot({
          path: '/home/relu/misc/zen-auto-create-article/temp/issue-recent-updates.png'
        });
        console.log('✅ Recent Updatesのスクリーンショット保存');
      }

      // Trending Tagsセクション
      const trendingTags = page.locator('#access-tags');
      if (await trendingTags.count() > 0) {
        await trendingTags.screenshot({
          path: '/home/relu/misc/zen-auto-create-article/temp/issue-trending-tags.png'
        });
        console.log('✅ Trending Tagsのスクリーンショット保存');
      }
    } else {
      console.log('⚠️  右サイドバーが見つかりません');
    }
  });

  test('数式表示の確認', async ({ page }) => {
    await page.goto(ARTICLE_URL);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000); // MathJax読み込み待機

    // MathJaxが読み込まれているか確認
    const mathJaxLoaded = await page.evaluate(() => {
      return typeof window.MathJax !== 'undefined';
    });
    console.log(`📐 MathJax読み込み: ${mathJaxLoaded ? '✅' : '❌'}`);

    // すべての数式要素を検索
    const mathElements = await page.evaluate(() => {
      const elements = [];

      // MathJax v3の場合
      const mjxContainers = document.querySelectorAll('mjx-container');
      mjxContainers.forEach((el, i) => {
        elements.push({
          index: i,
          type: 'mjx-container',
          display: el.getAttribute('display') || 'inline',
          visible: el.offsetWidth > 0 && el.offsetHeight > 0,
          innerHTML: el.innerHTML.substring(0, 100)
        });
      });

      // $...$ または $$...$$ の未処理テキスト
      const textContent = document.body.innerText;
      const inlineMath = (textContent.match(/\$[^$]+\$/g) || []).length;
      const displayMath = (textContent.match(/\$\$[^$]+\$\$/g) || []).length;

      return {
        mjxContainers: elements.length,
        unprocessedInline: inlineMath,
        unprocessedDisplay: displayMath,
        details: elements.slice(0, 5) // 最初の5つのみ
      };
    });

    console.log('📐 数式要素の状態:');
    console.log(JSON.stringify(mathElements, null, 2));

    // 数式セクションのスクリーンショット
    const mathSections = page.locator('mjx-container').first();
    if (await mathSections.count() > 0) {
      // 最初の数式の親要素をスクリーンショット
      const mathParent = mathSections.locator('xpath=ancestor::p | ancestor::div').first();
      await mathParent.screenshot({
        path: '/home/relu/misc/zen-auto-create-article/temp/issue-math-formula.png'
      });
      console.log('✅ 数式のスクリーンショット保存');
    }

    // 記事全体をスクロールして数式を探す
    await page.evaluate(() => {
      const mathElements = document.querySelectorAll('mjx-container');
      if (mathElements.length > 0) {
        mathElements[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    });
    await page.waitForTimeout(1000);

    await page.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/issue-math-section-full.png',
      fullPage: false
    });
  });

  test('コードブロックの色確認', async ({ page }) => {
    await page.goto(ARTICLE_URL);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);

    // コードブロックを検索
    const codeBlocks = page.locator('div.highlight, pre.highlight, figure.highlight');
    const count = await codeBlocks.count();
    console.log(`💻 コードブロック数: ${count}`);

    if (count > 0) {
      // 最初のコードブロックの詳細情報を取得
      const codeInfo = await codeBlocks.first().evaluate(el => {
        const styles = window.getComputedStyle(el);
        const codeElement = el.querySelector('code');
        const codeStyles = codeElement ? window.getComputedStyle(codeElement) : null;

        return {
          container: {
            backgroundColor: styles.backgroundColor,
            color: styles.color,
            padding: styles.padding,
            borderRadius: styles.borderRadius
          },
          code: codeStyles ? {
            backgroundColor: codeStyles.backgroundColor,
            color: codeStyles.color,
            fontFamily: codeStyles.fontFamily
          } : null,
          innerHTML: el.innerHTML.substring(0, 200)
        };
      });

      console.log('💻 コードブロックのスタイル:');
      console.log(JSON.stringify(codeInfo, null, 2));

      // 最初のコードブロックのスクリーンショット
      await codeBlocks.first().screenshot({
        path: '/home/relu/misc/zen-auto-create-article/temp/issue-code-block-1.png'
      });

      // 2番目のコードブロック（存在すれば）
      if (count > 1) {
        await codeBlocks.nth(1).screenshot({
          path: '/home/relu/misc/zen-auto-create-article/temp/issue-code-block-2.png'
        });
      }

      // 3番目のコードブロック（存在すれば）
      if (count > 2) {
        await codeBlocks.nth(2).screenshot({
          path: '/home/relu/misc/zen-auto-create-article/temp/issue-code-block-3.png'
        });
      }

      console.log('✅ コードブロックのスクリーンショット保存');
    }

    // シンタックスハイライトの色を確認
    const highlightColors = await page.evaluate(() => {
      const colors = {};
      const highlightClasses = [
        '.highlight .k',  // keyword
        '.highlight .s',  // string
        '.highlight .n',  // name
        '.highlight .c',  // comment
        '.highlight .o',  // operator
        '.highlight .nf', // function name
        '.highlight .nc', // class name
      ];

      highlightClasses.forEach(selector => {
        const el = document.querySelector(selector);
        if (el) {
          const styles = window.getComputedStyle(el);
          colors[selector] = {
            color: styles.color,
            backgroundColor: styles.backgroundColor
          };
        }
      });

      return colors;
    });

    console.log('🎨 シンタックスハイライトの色:');
    console.log(JSON.stringify(highlightColors, null, 2));
  });

  test('記事全体の問題箇所スクリーンショット', async ({ page }) => {
    await page.goto(ARTICLE_URL);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);

    // 記事全体
    await page.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/issue-article-full.png',
      fullPage: true
    });

    // "実装例" セクション（コードが含まれる）
    const implementationSection = page.locator('h2:has-text("実装例"), h3:has-text("実装例")');
    if (await implementationSection.count() > 0) {
      await implementationSection.first().screenshot({
        path: '/home/relu/misc/zen-auto-create-article/temp/issue-implementation-section.png'
      });
    }

    // "数式" を含むセクション
    const mathSection = page.locator('h2:has-text("数式"), h3:has-text("数式"), text=/\\$/');
    if (await mathSection.count() > 0) {
      await mathSection.first().screenshot({
        path: '/home/relu/misc/zen-auto-create-article/temp/issue-math-heading.png'
      });
    }

    console.log('✅ 記事全体のスクリーンショット保存完了');
  });

  test('テーマモード（light/dark）の確認', async ({ page }) => {
    // ライトモード
    await page.goto(ARTICLE_URL);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);

    const themeMode = await page.evaluate(() => {
      return document.documentElement.getAttribute('data-mode');
    });
    console.log(`🎨 現在のテーマモード: ${themeMode}`);

    // コードブロックのスクリーンショット（ライトモード）
    const codeBlock = page.locator('div.highlight, pre.highlight, figure.highlight').first();
    if (await codeBlock.count() > 0) {
      await codeBlock.screenshot({
        path: `/home/relu/misc/zen-auto-create-article/temp/issue-code-${themeMode}-mode.png`
      });
    }

    // ダークモードに切り替え（ボタンが存在する場合）
    const themeSwitcher = page.locator('button#mode-toggle, button.mode-toggle');
    if (await themeSwitcher.count() > 0) {
      await themeSwitcher.click();
      await page.waitForTimeout(500);

      const newThemeMode = await page.evaluate(() => {
        return document.documentElement.getAttribute('data-mode');
      });
      console.log(`🎨 切り替え後のテーマモード: ${newThemeMode}`);

      // コードブロックのスクリーンショット（ダークモード）
      if (await codeBlock.count() > 0) {
        await codeBlock.screenshot({
          path: `/home/relu/misc/zen-auto-create-article/temp/issue-code-${newThemeMode}-mode.png`
        });
      }
    }
  });

  test('CSSファイルの確認', async ({ page }) => {
    await page.goto(ARTICLE_URL);
    await page.waitForLoadState('networkidle');

    // すべてのCSSファイルを取得
    const cssFiles = await page.evaluate(() => {
      const files = [];
      for (const sheet of document.styleSheets) {
        if (sheet.href) {
          files.push({
            href: sheet.href,
            disabled: sheet.disabled
          });
        }
      }
      return files;
    });

    console.log('📄 読み込まれているCSSファイル:');
    cssFiles.forEach((file, i) => {
      console.log(`  ${i + 1}. ${file.href} (disabled: ${file.disabled})`);
    });

    // style.css の内容を確認
    const styleCssContent = await page.evaluate(() => {
      for (const sheet of document.styleSheets) {
        if (sheet.href && sheet.href.includes('style.css')) {
          try {
            const rules = Array.from(sheet.cssRules || []);
            return {
              rulesCount: rules.length,
              hasCodeStyles: rules.some(r => r.selectorText && r.selectorText.includes('.highlight')),
              hasMathStyles: rules.some(r => r.selectorText && r.selectorText.includes('mjx')),
              sampleRules: rules.slice(0, 10).map(r => r.cssText).join('\n')
            };
          } catch (e) {
            return { error: e.message };
          }
        }
      }
      return null;
    });

    console.log('📄 style.css の内容:');
    console.log(JSON.stringify(styleCssContent, null, 2));
  });
});
