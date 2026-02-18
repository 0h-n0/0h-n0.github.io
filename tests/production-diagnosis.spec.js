// 本番サイト（https://0h-n0.github.io/）のスタイル診断
const { test, expect } = require('@playwright/test');

const PRODUCTION_URL = 'https://0h-n0.github.io/';

test.describe('本番サイトのスタイル診断', () => {

  test('ホームページのスクリーンショット撮影', async ({ page }) => {
    await page.goto(PRODUCTION_URL);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000); // CSSの読み込み完了待機

    // 全体スクリーンショット
    await page.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/prod-homepage-full.png',
      fullPage: true
    });

    // ヘッダー部分
    const header = page.locator('#topbar-wrapper');
    if (await header.count() > 0) {
      await header.screenshot({
        path: '/home/relu/misc/zen-auto-create-article/temp/prod-header.png'
      });
    }

    // サイドバー
    const sidebar = page.locator('#sidebar');
    if (await sidebar.count() > 0) {
      await sidebar.screenshot({
        path: '/home/relu/misc/zen-auto-create-article/temp/prod-sidebar.png'
      });
    }

    // 最初の記事カード
    const firstCard = page.locator('#post-list .post-preview').first();
    if (await firstCard.count() > 0) {
      await firstCard.screenshot({
        path: '/home/relu/misc/zen-auto-create-article/temp/prod-post-card.png'
      });
    }

    console.log('✅ ホームページのスクリーンショット保存完了');
  });

  test('CSS読み込み状態の確認', async ({ page }) => {
    await page.goto(PRODUCTION_URL);
    await page.waitForLoadState('networkidle');

    // すべてのCSSファイルを取得
    const stylesheets = await page.evaluate(() => {
      const sheets = [];
      for (const sheet of document.styleSheets) {
        try {
          sheets.push({
            href: sheet.href,
            disabled: sheet.disabled,
            rulesCount: sheet.cssRules ? sheet.cssRules.length : 0,
            media: sheet.media.mediaText
          });
        } catch (e) {
          // CORS制限でアクセスできないシート
          sheets.push({
            href: sheet.href,
            disabled: sheet.disabled,
            rulesCount: 'CORS-blocked',
            media: sheet.media.mediaText
          });
        }
      }
      return sheets;
    });

    console.log('📄 読み込まれたCSS:');
    stylesheets.forEach((sheet, i) => {
      console.log(`  ${i + 1}. ${sheet.href || '(inline)'}`);
      console.log(`     Rules: ${sheet.rulesCount}, Disabled: ${sheet.disabled}`);
    });

    // style.cssが読み込まれているか確認
    const hasStyleCss = stylesheets.some(s => s.href && s.href.includes('style.css'));
    console.log(`🎨 style.css読み込み: ${hasStyleCss ? '✅' : '❌'}`);

    // カスタムCSSルールの確認
    const customStyles = await page.evaluate(() => {
      const testSelectors = [
        '.category-badge',
        '.tag-badge',
        'h1[data-toc-skip] .post-emoji',
        '.related-post-card'
      ];

      const results = {};
      for (const selector of testSelectors) {
        const element = document.querySelector(selector);
        if (element) {
          const styles = window.getComputedStyle(element);
          results[selector] = {
            backgroundColor: styles.backgroundColor,
            padding: styles.padding,
            borderRadius: styles.borderRadius,
            display: styles.display
          };
        } else {
          results[selector] = 'Element not found';
        }
      }
      return results;
    });

    console.log('🎨 カスタムスタイルの適用状況:');
    for (const [selector, styles] of Object.entries(customStyles)) {
      console.log(`  ${selector}:`, JSON.stringify(styles, null, 2));
    }
  });

  test('記事詳細ページのスクリーンショット撮影', async ({ page }) => {
    await page.goto(PRODUCTION_URL);
    await page.waitForLoadState('networkidle');

    // 最初の記事をクリック
    const firstPostLink = page.locator('#post-list .post-preview h1.post-title a').first();
    if (await firstPostLink.count() > 0) {
      const articleUrl = await firstPostLink.getAttribute('href');
      console.log('📌 記事URL:', PRODUCTION_URL.replace(/\/$/, '') + articleUrl);

      await firstPostLink.click();
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(2000); // MathJax/Mermaidの読み込み待機

      // 記事全体
      await page.screenshot({
        path: '/home/relu/misc/zen-auto-create-article/temp/prod-article-full.png',
        fullPage: true
      });

      // タイトル部分
      const title = page.locator('h1[data-toc-skip]');
      if (await title.count() > 0) {
        await title.screenshot({
          path: '/home/relu/misc/zen-auto-create-article/temp/prod-article-title.png'
        });

        // 絵文字が表示されているか
        const hasEmoji = await title.locator('.post-emoji').count() > 0;
        console.log(`📌 タイトル絵文字: ${hasEmoji ? '✅' : '❌'}`);
      }

      // カテゴリ・タグエリア
      const tailWrapper = page.locator('.post-tail-wrapper');
      if (await tailWrapper.count() > 0) {
        await tailWrapper.screenshot({
          path: '/home/relu/misc/zen-auto-create-article/temp/prod-article-tail.png'
        });

        // バッジの確認
        const categoryBadges = await tailWrapper.locator('.category-badge').count();
        const tagBadges = await tailWrapper.locator('.tag-badge').count();
        console.log(`📁 カテゴリバッジ数: ${categoryBadges}`);
        console.log(`🏷️  タグバッジ数: ${tagBadges}`);
      }

      // 関連記事セクション
      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      await page.waitForTimeout(500);

      const relatedPosts = page.locator('#related-posts');
      if (await relatedPosts.count() > 0) {
        await relatedPosts.screenshot({
          path: '/home/relu/misc/zen-auto-create-article/temp/prod-related-posts.png'
        });

        const relatedCards = await relatedPosts.locator('.related-post-card').count();
        console.log(`📚 関連記事カード数: ${relatedCards}`);
      }

      console.log('✅ 記事詳細ページのスクリーンショット保存完了');
    } else {
      console.log('⚠️  記事が見つかりません');
    }
  });

  test('ネットワークリソースの読み込みエラー確認', async ({ page }) => {
    const failedRequests = [];
    const slowRequests = [];

    page.on('response', response => {
      const url = response.url();
      const status = response.status();
      const timing = response.timing();

      // 失敗したリクエスト
      if (status >= 400) {
        failedRequests.push({ url, status });
      }

      // 遅いリクエスト（3秒以上）
      if (timing && timing.responseEnd > 3000) {
        slowRequests.push({ url, duration: Math.round(timing.responseEnd) });
      }
    });

    await page.goto(PRODUCTION_URL);
    await page.waitForLoadState('networkidle');

    if (failedRequests.length > 0) {
      console.log('❌ 読み込み失敗したリソース:');
      failedRequests.forEach(req => {
        console.log(`  ${req.status}: ${req.url}`);
      });
    } else {
      console.log('✅ すべてのリソースが正常に読み込まれました');
    }

    if (slowRequests.length > 0) {
      console.log('⚠️  読み込みが遅いリソース（3秒以上）:');
      slowRequests.forEach(req => {
        console.log(`  ${req.duration}ms: ${req.url}`);
      });
    }
  });

  test('ローカル環境との比較スクリーンショット', async ({ page }) => {
    // 本番環境
    await page.goto(PRODUCTION_URL);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);

    await page.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/compare-prod.png',
      fullPage: false, // ファーストビューのみ
      clip: { x: 0, y: 0, width: 1280, height: 1024 }
    });

    // ローカル環境
    await page.goto('http://localhost:4000');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);

    await page.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/compare-local.png',
      fullPage: false,
      clip: { x: 0, y: 0, width: 1280, height: 1024 }
    });

    console.log('✅ 本番・ローカル比較スクリーンショット保存完了');
    console.log('   本番: temp/compare-prod.png');
    console.log('   ローカル: temp/compare-local.png');
  });

  test('DOM構造の確認', async ({ page }) => {
    await page.goto(PRODUCTION_URL);
    await page.waitForLoadState('networkidle');

    const domInfo = await page.evaluate(() => {
      return {
        hasSidebar: !!document.querySelector('#sidebar'),
        hasTopbar: !!document.querySelector('#topbar-wrapper'),
        hasPostList: !!document.querySelector('#post-list'),
        postCount: document.querySelectorAll('#post-list .post-preview').length,
        hasCategoryBadges: document.querySelectorAll('.category-badge').length,
        hasTagBadges: document.querySelectorAll('.tag-badge').length,
        hasEmojis: document.querySelectorAll('.post-emoji').length,
        bodyClasses: document.body.className,
        themeMode: document.documentElement.getAttribute('data-mode')
      };
    });

    console.log('🔍 DOM構造情報:');
    console.log(JSON.stringify(domInfo, null, 2));

    // 重要要素の存在確認
    expect(domInfo.hasSidebar).toBeTruthy();
    expect(domInfo.hasPostList).toBeTruthy();
    expect(domInfo.postCount).toBeGreaterThan(0);
  });
});
