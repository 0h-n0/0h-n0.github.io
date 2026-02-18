const { chromium } = require('@playwright/test');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  await page.goto('http://127.0.0.1:4000/posts/techblog-aws-bedrock-structured-outputs/');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(2000);
  
  // Check if jQuery is loaded
  const jQueryLoaded = await page.evaluate(() => typeof $ !== 'undefined');
  console.log('jQuery loaded:', jQueryLoaded);
  
  // Check if Toc is loaded
  const TocLoaded = await page.evaluate(() => typeof Toc !== 'undefined');
  console.log('Toc loaded:', TocLoaded);
  
  // Check if TOC element exists
  const tocExists = await page.evaluate(() => document.querySelector('#toc') !== null);
  console.log('TOC element exists:', tocExists);
  
  // Check if core-wrapper exists
  const coreWrapperExists = await page.evaluate(() => document.querySelector('#core-wrapper') !== null);
  console.log('core-wrapper exists:', coreWrapperExists);
  
  // Count headings in core-wrapper
  const headingCount = await page.evaluate(() => {
    const wrapper = document.querySelector('#core-wrapper');
    if (!wrapper) return 0;
    return wrapper.querySelectorAll('h2, h3').length;
  });
  console.log('Headings in core-wrapper:', headingCount);
  
  // Check TOC content
  const tocHtml = await page.evaluate(() => document.querySelector('#toc')?.innerHTML || '');
  console.log('TOC HTML:', tocHtml);
  
  // Check console errors
  const errors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') {
      errors.push(msg.text());
    }
  });
  
  // Wait a bit more to catch any errors
  await page.waitForTimeout(1000);
  
  if (errors.length > 0) {
    console.log('\nConsole errors:');
    errors.forEach(err => console.log('  -', err));
  }
  
  await browser.close();
})();
