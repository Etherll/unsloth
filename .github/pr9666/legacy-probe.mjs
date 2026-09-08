// SPDX-License-Identifier: AGPL-3.0-only
import {chromium} from './node_modules/playwright-core/index.mjs';
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
const phase=process.env.PR9666_PHASE;
const expectedSource=['existing','rollback'].includes(phase)?'5c44c8c96b7fad0777cb6ea1240e5427f81e6fe4':'70775b3bdf147d10e941306c17c25d51d98a4bca';
let browser;
for(let i=0;i<45;i++){try{browser=await chromium.connectOverCDP('http://127.0.0.1:19266',{timeout:1500});break}catch{await new Promise(r=>setTimeout(r,1000))}}
assert.ok(browser,'Native connection required');
let page;
for(let i=0;i<30;i++){page=browser.contexts()[0].pages()[0];if(page)break;await new Promise(r=>setTimeout(r,1000))}
assert.ok(page,'Native main webview required');
await page.goto('http://localhost:5173/review.html');
const font=page.getByRole('spinbutton',{name:'UI font size',exact:true});
await font.waitFor({timeout:60000});
assert.equal(await page.evaluate(()=>!!window.__TAURI_INTERNALS__),true);
assert.equal(await page.evaluate(()=>window.__TAURI_INTERNALS__.invoke('set_close_to_tray',{enabled:false})),false);
assert.equal(await page.evaluate(()=>window.__TAURI_INTERNALS__.invoke('get_close_to_tray')),false);
const field=page.getByRole('spinbutton',{name:'Interface scale',exact:true});
const baseline=JSON.parse(readFileSync('native-evidence/proof.json','utf8')).baseline.dpr;
if(phase==='existing'){
 assert.equal(await field.count(),0);
 await font.fill('20');await font.press('Tab');assert.equal(await font.inputValue(),'20');
 await page.evaluate(()=>localStorage.setItem('pr9666_retained_control','preserve'));
}else{
 assert.equal(await page.evaluate(()=>localStorage.getItem('pr9666_retained_control')),'preserve');
 assert.equal(await font.inputValue(),phase==='upgrade'?'20':'18');
 if(phase==='upgrade'){
  assert.equal(await field.inputValue(),'100');
  await field.fill('175');await field.press('Tab');
  await font.fill('18');await font.press('Tab');assert.equal(await font.inputValue(),'18');
 }else if(phase==='rollback'){
  assert.equal(await field.count(),0);
  assert.equal(await page.evaluate(()=>JSON.parse(localStorage.getItem('unsloth_interface_scale')).state.scale),175);
 }else{
  assert.equal(await field.inputValue(),'175');
 }
}
const factor=['upgrade','re-upgrade'].includes(phase)?1.75:1;
await page.waitForFunction(({baseline,factor})=>Math.abs(devicePixelRatio-baseline*factor)<.02,{baseline,factor});
const facts=await page.evaluate(()=>({dpr:devicePixelRatio,font:(document.querySelector('input[aria-label="UI font size"]')||{}).value,retained:localStorage.getItem('pr9666_retained_control'),scale:localStorage.getItem('unsloth_interface_scale')}));
await page.screenshot({path:`native-evidence/legacy-${phase}.png`});
writeFileSync(`native-evidence/legacy-${phase}.json`,JSON.stringify({phase,source:expectedSource,target:'70775b3bdf147d10e941306c17c25d51d98a4bca',mode:'actual native source-built executables; same disposable WebView profile; controlled production settings mount; installer execution excluded',facts},null,2));
console.log('PASS retained-profile phase',phase);
process.exit(0);
