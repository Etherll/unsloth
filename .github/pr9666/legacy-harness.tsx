// SPDX-License-Identifier: AGPL-3.0-only
import React from 'react';
import {createRoot} from 'react-dom/client';
import {AppearanceTab} from '@/features/settings/tabs/appearance-tab';
import {TooltipProvider} from '@/components/ui/tooltip';
import {MotionConfig} from 'motion/react';
import '@/index.css';
if(!('__TAURI_INTERNALS__' in window))throw new Error('Actual native bridge required');
createRoot(document.getElementById('root')!).render(<MotionConfig reducedMotion="always"><TooltipProvider><main style={{padding:24}}><p>WINDOWS NATIVE WEBVIEW — old-version settings controls</p><AppearanceTab/></main></TooltipProvider></MotionConfig>);
requestAnimationFrame(()=>{document.documentElement.dataset.harnessReady='true'});
