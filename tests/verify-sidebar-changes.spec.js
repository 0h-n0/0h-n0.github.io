const { test, expect } = require('@playwright/test');

test.describe('Sidebar Changes Verification', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://127.0.0.1:4000/');
    await page.waitForLoadState('networkidle');
  });

  test('should display social links in sidebar', async ({ page }) => {
    // GitHub link
    const githubLink = page.locator('#sidebar a[aria-label="github"]');
    await expect(githubLink).toBeVisible();
    await expect(githubLink).toHaveAttribute('href', 'https://github.com/0h-n0');

    // Twitter/X link
    const twitterLink = page.locator('#sidebar a[aria-label="twitter"]');
    await expect(twitterLink).toBeVisible();
    await expect(twitterLink).toHaveAttribute('href', 'https://x.com/XrZRTgC1ko96643');

    // Email link
    const emailLink = page.locator('#sidebar a[aria-label="email"]');
    await expect(emailLink).toBeVisible();

    // RSS link
    const rssLink = page.locator('#sidebar a[aria-label="rss"]');
    await expect(rssLink).toBeVisible();
    await expect(rssLink).toHaveAttribute('href', '/feed.xml');

    console.log('✅ All social links are displayed correctly');
  });

  test('should display English navigation labels', async ({ page }) => {
    // Check navigation menu labels
    const homeTab = page.locator('#sidebar .nav-link:has-text("HOME")');
    await expect(homeTab).toBeVisible();

    const categoriesTab = page.locator('#sidebar .nav-link:has-text("CATEGORIES")');
    await expect(categoriesTab).toBeVisible();

    const tagsTab = page.locator('#sidebar .nav-link:has-text("TAGS")');
    await expect(tagsTab).toBeVisible();

    const archivesTab = page.locator('#sidebar .nav-link:has-text("ARCHIVES")');
    await expect(archivesTab).toBeVisible();

    const aboutTab = page.locator('#sidebar .nav-link:has-text("ABOUT")');
    await expect(aboutTab).toBeVisible();

    console.log('✅ All navigation labels are in English');
  });

  test('should use plural tags icon', async ({ page }) => {
    // Click on Tags tab
    await page.click('#sidebar .nav-link:has-text("TAGS")');
    await page.waitForURL('**/tags/');

    // Check for plural icon in sidebar (fas fa-tags)
    const tagsIcon = page.locator('#sidebar .nav-link:has-text("TAGS") i');
    const iconClass = await tagsIcon.getAttribute('class');
    expect(iconClass).toContain('fa-tags');

    console.log('✅ Tags icon is plural (fas fa-tags)');
  });

  test('should have dark mode toggle', async ({ page }) => {
    const darkModeToggle = page.locator('#sidebar button#mode-toggle');
    await expect(darkModeToggle).toBeVisible();

    console.log('✅ Dark mode toggle is visible');
  });
});
