// SPDX-License-Identifier: AGPL-3.0-only
import { createServer } from '../../studio/frontend/node_modules/vite/dist/node/index.js';
import { fileURLToPath } from 'node:url';
const frontend=fileURLToPath(new URL('../../studio/frontend/',import.meta.url));
const run=fileURLToPath(new URL('.',import.meta.url));
const server=await createServer({root:frontend,configFile:frontend+'/vite.config.ts',server:{host:'127.0.0.1',port:5173,strictPort:true,proxy:{},fs:{allow:[frontend,run]}},plugins:[{name:'review-harness',transform(code,id){if(id.endsWith('/src/app/provider.tsx'))return code+'\nexport { DesktopChromeVarsEffect as ReviewChromeEffect, AppearanceCustomizationEffect as ReviewAppearanceEffect };';if(id.endsWith('/src/features/settings/tabs/general-tab.tsx'))return code+'\nexport { resetAllPrefs as reviewResetAllPrefs };';},configureServer(s){s.middlewares.use('/review.html',async(req,res)=>{res.setHeader('Content-Type','text/html');res.end(await s.transformIndexHtml('/review.html',`<!doctype html><html><head><title>PR 9666 controls probe</title></head><body><div id="root"></div><script type="module" src="/@fs/${run.replaceAll('\\','/')}settings-harness.tsx"></script></body></html>`));});}}]});
// Remove inherited backend proxies: this isolated component harness never uses a real backend.
server.config.server.proxy={};
await server.listen();console.log('HARNESS_READY http://localhost:5173/review.html');
