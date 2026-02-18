const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });

  const ARTICLE_URL = 'https://0h-n0.github.io/posts/techblog-aws-bedrock-structured-outputs/';

  await page.goto(ARTICLE_URL);
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(2000);

  // タグの状態を確認
  const tagInfo = await page.evaluate(() => {
    const tagBadges = document.querySelectorAll('.post-tags .tag-badge');

    return Array.from(tagBadges).map((tag, i) => {
      const styles = window.getComputedStyle(tag);
      const rect = tag.getBoundingClientRect();

      return {
        index: i,
        tagName: tag.tagName,
        text: tag.textContent.trim(),
        href: tag.href || null,
        className: tag.className,
        pointerEvents: styles.pointerEvents,
        cursor: styles.cursor,
        display: styles.display,
        position: styles.position,
        zIndex: styles.zIndex,
        rect: {
          top: Math.round(rect.top),
          left: Math.round(rect.left),
          width: Math.round(rect.width),
          height: Math.round(rect.height)
        }
      };
    });
  });

  console.log('🏷️  タグの状態:');
  console.log(JSON.stringify(tagInfo, null, 2));

  if (tagInfo.length > 0) {
    // 最初のタグをクリックしてみる
    console.log('\n🖱️  最初のタグをクリックしてみます...');
    const firstTag = page.locator('.post-tags .tag-badge').first();

    try {
      await firstTag.click({ timeout: 5000 });
      await page.waitForTimeout(1000);

      const newUrl = page.url();
      console.log(`✅ クリック成功！ 遷移先: ${newUrl}`);
    } catch (e) {
      console.log(`❌ クリック失敗: ${e.message}`);

      // タグのスクリーンショット
      await firstTag.screenshot({
        path: '/home/relu/misc/zen-auto-create-article/temp/tag-unclickable.png'
      });
      console.log('📸 クリックできないタグのスクリーンショット保存');
    }
  } else {
    console.log('❌ タグが見つかりません');
  }

  // タグエリア全体のスクリーンショット
  const tagsArea = page.locator('.post-tags');
  if (await tagsArea.count() > 0) {
    await tagsArea.screenshot({
      path: '/home/relu/misc/zen-auto-create-article/temp/tags-area.png'
    });
    console.log('📸 タグエリアのスクリーンショット保存');
  }

  await browser.close();
})();
