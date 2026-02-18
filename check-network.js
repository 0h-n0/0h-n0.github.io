const { chromium } = require('@playwright/test');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  const failedRequests = [];
  
  page.on('requestfailed', request => {
    failedRequests.push({
      url: request.url(),
      failure: request.failure().errorText
    });
  });
  
  page.on('response', response => {
    if (!response.ok() && response.url().includes('bootstrap-toc')) {
      console.log('Failed to load bootstrap-toc:', response.status(), response.url());
    }
  });
  
  await page.goto('http://127.0.0.1:4000/posts/techblog-aws-bedrock-structured-outputs/');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(3000);
  
  if (failedRequests.length > 0) {
    console.log('Failed requests:');
    failedRequests.forEach(req => {
      console.log('  ✗', req.url);
      console.log('    Error:', req.failure);
    });
  } else {
    console.log('No failed requests');
  }
  
  // Check what's actually available on window
  const windowToc = await page.evaluate(() => {
    const props = [];
    for (let key in window) {
      if (key.toLowerCase().includes('toc') || key === 'Toc') {
        props.push(key);
      }
    }
    return props;
  });
  
  console.log('\nWindow properties with "toc":', windowToc);
  
  // Check if bootstrap-toc loaded
  const bootstrapTocTest = await page.evaluate(() => {
    // Bootstrap TOC might not expose a global Toc object
    // It might be a jQuery plugin instead
    return {
      jQueryTocPlugin: typeof $.fn.toc !== 'undefined',
      Toc: typeof Toc !== 'undefined'
    };
  });
  
  console.log('Bootstrap TOC check:', bootstrapTocTest);
  
  await browser.close();
})();
