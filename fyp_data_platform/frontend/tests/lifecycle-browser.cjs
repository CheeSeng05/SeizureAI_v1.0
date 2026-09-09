// Optional integration smoke test: requires Playwright and a built dist/.
// All API calls are intercepted; no live app data is modified.
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const assert = require('node:assert/strict');

(async () => {
  const dist = path.resolve(__dirname, '../dist');
  const server = http.createServer((req, res) => {
    const asset = req.url.startsWith('/assets/') ? path.join(dist, path.basename(path.dirname(req.url)), path.basename(req.url)) : path.join(dist, 'index.html');
    res.setHeader('Content-Type', asset.endsWith('.js') ? 'application/javascript' : asset.endsWith('.css') ? 'text/css' : 'text/html');
    res.end(fs.readFileSync(asset));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_CHANNEL ? {channel:process.env.BROWSER_CHANNEL} : {}) });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = []; page.on('pageerror', error => errors.push(error.message));
    let archived = false, deleted = false;
    const row = { id:'demo::r1', dataset_id:'demo', recording_source_id:'r1', video:{filename:'sample.mp4',size_bytes:100}, created_at:'2026-09-09' };
    const run = {id:'run1', recording_id:'r1', dataset_id:'demo', status:'completed', modality:'video', require_labels:false, runtime:{timings:{total_sec:1}}, compute:{mode:'cpu'}, artifacts:[], ground_truth:{}, metrics:{prediction_only:true}, windows:[{start_sec:0,end_sec:1,probability:0.7}], duration_sec:60.96, source_video:{url:'/api/sample.mp4',browser_playable:true,filename:'sample.mp4',pose_url:'/api/pose.json'}};
    const sample = path.resolve(__dirname, '../../../FYPModels/test_data/Stitched_Video.mp4');
    await page.route('**/api/**', async route => {
      const url = new URL(route.request().url()), method = route.request().method();
      let body;
      if (url.pathname === '/api/health') body={counts:{datasets:1,patients:1,recordings:1,files:1,runs:1},compute:{mode:'cpu'},status:'ok'};
      else if (url.pathname === '/api/catalog') body={recordings: !deleted && (url.searchParams.get('archived') === 'true') === archived ? [{...row, archived}] : [], runs:[]};
      else if (url.pathname.startsWith('/api/manage/')) {
        if (method === 'PATCH') archived=route.request().postDataJSON().archived;
        if (method === 'DELETE') { assert.equal(route.request().postDataJSON().confirmation,'demo::r1'); deleted=true; }
        body={id:'demo::r1',runs:1,files:2,bytes:100,archived};
      } else if (url.pathname === '/api/runs/run1') body=run;
      else if (url.pathname === '/api/pose.json') body={version:1,width:1920,height:1080,fps:25,edges:[[0,1]],frames:Array.from({length:1500},(_,i)=>[i,[[500+i,300,0.9],[700+i,500,0.5]]])};
      else if (url.pathname === '/api/sample.mp4') { await route.fulfill({status:200,contentType:'video/mp4',body:fs.readFileSync(sample)}); return; }
      else throw new Error('Unexpected API '+url.pathname);
      await route.fulfill({json:body});
    });
    const base = `http://127.0.0.1:${server.address().port}`;
    page.on('dialog', dialog => dialog.accept());
    await page.goto(base+'/recordings');
    await page.getByRole('button',{name:'Archive',exact:true}).click();
    await page.getByText('No recordings found').waitFor();
    await page.goto(base+'/settings');
    await page.getByRole('button',{name:'Restore',exact:true}).click();
    await page.getByText('No archived recordings.').waitFor();
    await page.goto(base+'/recordings');
    await page.getByRole('button',{name:'Archive',exact:true}).click();
    await page.getByText('No recordings found').waitFor();
    await page.goto(base+'/settings');
    await page.getByRole('button',{name:'Delete permanently…',exact:true}).click();
    const confirm = page.getByRole('region',{name:'Confirm permanent deletion'});
    assert.equal(await confirm.getByRole('button',{name:'Permanently delete',exact:true}).isDisabled(),true);
    await confirm.locator('input').fill('demo::r1');
    await confirm.getByRole('button',{name:'Permanently delete',exact:true}).click();
    await page.getByText('No archived recordings.').waitFor();
    assert.equal(deleted,true);
    if (fs.existsSync(sample)) {
      await page.goto(base+'/runs/run1');
      await page.getByRole('checkbox',{name:'Show skeleton'}).check();
      await page.waitForFunction(()=>document.querySelector('.skeleton-canvas')?.width>0 && document.querySelector('video')?.readyState>=2);
      const checkCanvas=()=>page.evaluate(()=>{
        const c=document.querySelector('.skeleton-canvas'),v=document.querySelector('video');
        const alpha=c.getContext('2d').getImageData(0,0,c.width,c.height).data.some((value,i)=>i%4===3&&value>0);
        return {alpha,height:Math.abs(c.getBoundingClientRect().height-v.getBoundingClientRect().height)};
      });
      assert.equal((await checkCanvas()).alpha,true);
      await page.locator('video').evaluate(v=>{v.currentTime=2;});
      await page.waitForFunction(()=>!document.querySelector('video').seeking);
      assert.equal((await checkCanvas()).alpha,true);
      await page.setViewportSize({width:700,height:1000});
      await page.waitForFunction(()=>Math.abs(document.querySelector('.skeleton-canvas').getBoundingClientRect().height-document.querySelector('video').getBoundingClientRect().height)<1);
      assert.ok((await checkCanvas()).height<1);
      await page.getByRole('button',{name:'Fullscreen with overlay'}).click();
      await page.waitForFunction(()=>document.fullscreenElement?.classList.contains('skeleton-stage'));
      await page.waitForFunction(()=>Math.abs(document.querySelector('.skeleton-canvas').getBoundingClientRect().height-document.querySelector('video').getBoundingClientRect().height)<1);
      assert.equal((await checkCanvas()).alpha,true);
      await page.evaluate(()=>document.exitFullscreen());
      await page.getByRole('checkbox',{name:'Show skeleton'}).uncheck();
      assert.equal(await page.locator('.skeleton-canvas').count(),0);
      console.log('Skeleton: seek, resize, toggle and rendered pixels passed');
    } else console.log('Skeleton playback skipped: optional sample video absent');
    assert.deepEqual(errors,[]);
    console.log('Archive, restore, typed deletion and browser error checks passed');
  } finally { if(browser) await browser.close(); server.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
