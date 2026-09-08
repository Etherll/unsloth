// SPDX-License-Identifier: AGPL-3.0-only
import { chromium } from './node_modules/playwright-core/index.mjs';
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
const target='70775b3bdf147d10e941306c17c25d51d98a4bca';
mkdirSync('native-evidence',{recursive:true});
let browser;
for(let i=0;i<60;i++){try{browser=await chromium.connectOverCDP('http://127.0.0.1:19266');break;}catch{await new Promise(r=>setTimeout(r,1000));}}
assert.ok(browser,'Native WebView2 CDP endpoint must start');
const context=browser.contexts()[0];
let page;
for(let i=0;i<30;i++){page=context.pages()[0];if(page)break;await new Promise(r=>setTimeout(r,1000));}
assert.ok(page,'Native main webview required');
await page.goto('http://localhost:5173/review.html');
await page.waitForFunction(()=>document.documentElement.dataset.harnessReady==='true',{},{timeout:60000});
assert.equal(await page.evaluate(()=>!!window.__TAURI_INTERNALS__),true);
const baseline=await page.evaluate(()=>({dpr:devicePixelRatio,width:innerWidth,outer:outerWidth}));
const results=[];
for(const scale of [100,50,150,200,100]){
 const field=page.getByRole('spinbutton',{name:'Interface scale',exact:true});
 await field.fill(String(scale));await field.press('Tab');
 await page.waitForFunction(({scale,dpr})=>Math.abs(devicePixelRatio-dpr*scale/100)<0.02,{scale,dpr:baseline.dpr});
 const facts=await page.evaluate(()=>({scale:window.settingsHarness.snapshot().scale,dpr:devicePixelRatio,width:innerWidth,outer:outerWidth,stored:localStorage.getItem('unsloth_interface_scale'),chrome:document.documentElement.style.getPropertyValue('--studio-window-chrome-top')}));
 assert.equal(facts.scale,scale);assert.equal(JSON.parse(facts.stored).state.scale,scale);assert.equal(facts.chrome,'34px');
 assert.ok(Math.abs(facts.width*scale/100-baseline.width)<3,'viewport must reflect actual native zoom');
 results.push({requested:scale,...facts});
 await page.screenshot({path:`native-evidence/scale-${scale}.png`});
}
await page.getByRole('spinbutton',{name:'Interface scale',exact:true}).fill('150');await page.getByRole('spinbutton',{name:'Interface scale',exact:true}).press('Tab');
await page.waitForFunction(dpr=>Math.abs(devicePixelRatio-dpr*1.5)<0.02,baseline.dpr);
await page.getByRole('button',{name:'Reset customization',exact:true}).click();
await page.waitForFunction(dpr=>Math.abs(devicePixelRatio-dpr)<0.02,baseline.dpr);
assert.equal(await page.getByRole('spinbutton',{name:'Interface scale',exact:true}).inputValue(),'100');
writeFileSync('native-evidence/proof.json',JSON.stringify({target,mode:'Native Windows WebView2; controlled production controls/effects mount; no backend or actual OS file drag',userAgent:await page.evaluate(()=>navigator.userAgent),baseline,results,reset:'pass'},null,2));
console.log('PASS native WebView2 actual zoom 50/100/150/200 and reset',JSON.stringify(results));
process.exit(0);

