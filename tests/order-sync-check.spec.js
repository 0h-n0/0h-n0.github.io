// 右サイドバーとホームページの記事順序が一致するかチェック
const { test, expect } = require('@playwright/test');

test('右サイドバーとホームページの記事順序が一致', async ({ page }) => {
  await page.goto('http://localhost:4000');
  await page.waitForLoadState('networkidle');

  // 右サイドバーの記事タイトルを取得（最大5件）
  const sidebarTitles = await page.locator('#access-lastmod li.recent-post-item a.post-link .post-title-text')
    .allTextContents();

  console.log('\n━━━ 右サイドバー「最近の更新」━━━');
  sidebarTitles.forEach((title, i) => {
    console.log(`${i + 1}. ${title.trim()}`);
  });

  // ホームページの記事タイトルを取得（最大5件）
  const homeTitles = await page.locator('#post-list .post-preview h1.post-title a')
    .allTextContents();

  console.log('\n━━━ ホームページ記事リスト ━━━');
  homeTitles.slice(0, 5).forEach((title, i) => {
    console.log(`${i + 1}. ${title.trim()}`);
  });

  // 順序が一致するか確認（最大5件を比較）
  const compareCount = Math.min(sidebarTitles.length, 5);

  console.log('\n━━━ 順序一致確認 ━━━');
  let allMatched = true;

  for (let i = 0; i < compareCount; i++) {
    const sidebarTitle = sidebarTitles[i]?.trim();
    const homeTitle = homeTitles[i]?.trim();
    const matched = sidebarTitle === homeTitle;

    console.log(`${i + 1}. ${matched ? '✅' : '❌'} ${matched ? '一致' : '不一致'}`);
    if (!matched) {
      console.log(`   サイドバー: ${sidebarTitle}`);
      console.log(`   ホーム    : ${homeTitle}`);
      allMatched = false;
    }
  }

  if (allMatched) {
    console.log('\n🎉 全ての記事順序が一致しています！');
  } else {
    console.log('\n⚠️  順序が一致していない記事があります');
  }

  // 少なくとも最初の記事は一致すべき
  expect(sidebarTitles[0]?.trim()).toBe(homeTitles[0]?.trim());
});
