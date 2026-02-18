const { chromium } = require('@playwright/test');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  await page.goto('http://127.0.0.1:4000/posts/techblog-aws-bedrock-structured-outputs/');
  await page.waitForLoadState('networkidle');
  
  // Get all script tags
  const scripts = await page.evaluate(() => {
    return Array.from(document.querySelectorAll('script[src]')).map(s => s.src);
  });
  
  console.log('Script tags:');
  scripts.forEach(src => {
    if (src.includes('bootstrap-toc') || src.includes('jquery') || src.includes('toc')) {
      console.log('  ✓', src);
    }
  });
  
  // Check for bootstrap-toc specifically
  const hasToc = scripts.some(src => src.includes('bootstrap-toc'));
  console.log('\nbootstrap-toc.js present:', hasToc);
  
  // Get all CSS links
  const links = await page.evaluate(() => {
    return Array.from(document.querySelectorAll('link[rel="stylesheet"]')).map(l => l.href);
  });
  
  console.log('\nCSS links with bootstrap-toc:');
  links.forEach(href => {
    if (href.includes('bootstrap-toc')) {
      console.log('  ✓', href);
    }
  });
  
  await browser.close();
})();
