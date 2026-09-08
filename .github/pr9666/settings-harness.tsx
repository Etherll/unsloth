// SPDX-License-Identifier: AGPL-3.0-only
// Report-only original-head browser harness. Serve via the frontend Vite server
// in a fresh isolated browser context. This mounts production components; it is
// simulated-native evidence and cannot prove native zoom or the app provider.
// Harness endpoint: window.settingsHarness after data-harness-ready="true".
// Suggested controls: getByRole('spinbutton', {name:'Interface scale'}),
// 'UI font size', 'Code font size'; Reset customization is the real tab button.
import React from "react";
import { createRoot } from "react-dom/client";
import "@/index.css";

type BridgeCall = { command: string; args: unknown };
type Transition = { store: string; before: unknown; after: unknown };
const host = window as unknown as Record<string, any>;
const desktop = new URLSearchParams(location.search).get("desktop") !== "0";
const calls: BridgeCall[] = [];
const transitions: Transition[] = [];

async function mount() {
  if (!("__TAURI_INTERNALS__" in window)) throw new Error("Actual Tauri bridge required");
  const [tab, scale, appearance, tooltip, motion] = await Promise.all([
    import("@/features/settings/tabs/appearance-tab"),
    import("@/features/settings/stores/interface-scale-store"),
    import("@/features/settings/stores/appearance-custom-store"),
    import("@/components/ui/tooltip"),
    import("motion/react"),
  ]);
  const { AppearanceTab } = tab;
  const { TooltipProvider } = tooltip;
  const { MotionConfig } = motion;
  const unsubscribeScale = scale.useInterfaceScaleStore.subscribe((next, prev) => {
    transitions.push({ store: "scale", before: prev.scale, after: next.scale });
  });
  const unsubscribeAppearance = appearance.useAppearanceCustomStore.subscribe((next, prev) => {
    transitions.push({ store: "appearance", before: prev.customization, after: next.customization });
  });
  const snapshot = () => ({
    evidenceMode: "native Windows webview; controlled production component mount",
    target: "70775b3bdf147d10e941306c17c25d51d98a4bca",
    scale: scale.useInterfaceScaleStore.getState().scale,
    customization: appearance.useAppearanceCustomStore.getState().customization,
    scaleStorage: localStorage.getItem(scale.INTERFACE_SCALE_STORAGE_KEY),
    appearanceStorage: localStorage.getItem("unsloth_appearance_customization"),
    transitions: [...transitions],
    bridgeCalls: [...calls],
  });
  const rootNode = document.getElementById("root") ?? document.body.appendChild(document.createElement("div"));
  const root = createRoot(rootNode);
  host.settingsHarness = {
    async mountProductionEffects(nativeMac = true, custom = false) {
      const provider = await import("@/app/provider") as any;
      const container = document.body.appendChild(document.createElement("div"));
      const effects = createRoot(container);
      effects.render(React.createElement(React.Fragment, null,
        React.createElement(provider.ReviewAppearanceEffect),
        React.createElement(provider.ReviewChromeEffect, {usesNativeMacTitlebar:nativeMac,usesCustomTitlebar:custom})));
      host.reviewEffects = {unmount(){effects.unmount();container.remove();}};
    },
    snapshot,
    clearTrace() { calls.length = 0; transitions.length = 0; },
    // These prepare real authoritative stores. Actual assertions should mutate
    // inputs and click the production reset button through browser events.
    setScale(value: number) { scale.useInterfaceScaleStore.getState().setScale(value); },
    patchAppearance(value: Record<string, unknown>) { appearance.useAppearanceCustomStore.getState().patch(value); },
    resetStores() {
      appearance.useAppearanceCustomStore.getState().resetAll();
      scale.useInterfaceScaleStore.getState().reset();
    },
    // Explicit seam probe only: AppearanceTab does not install the production
    // provider effect. Calling this must not be described as automatic live zoom.
    applyCurrentScale() { return scale.applyInterfaceScale(scale.useInterfaceScaleStore.getState().scale); },
    unmount() { root.unmount(); unsubscribeScale(); unsubscribeAppearance(); },
  };
  root.render(
    <MotionConfig reducedMotion="always">
      <TooltipProvider>
        <main style={{ maxWidth: 960, margin: "0 auto", padding: 24 }}>
          <p data-evidence-mode={desktop ? "simulated-native" : "browser"}>
            {desktop ? "WINDOWS NATIVE WEBVIEW — controlled settings mount" : "BROWSER CONTROL — actual React controls"}
          </p>
          <AppearanceTab />
        </main>
      </TooltipProvider>
    </MotionConfig>,
  );
  await host.settingsHarness.mountProductionEffects(false, true);
  requestAnimationFrame(() => { document.documentElement.dataset.harnessReady = "true"; });
}
void mount().catch((error) => {
  host.settingsHarnessError = String(error?.stack ?? error);
  document.body.appendChild(document.createElement("pre")).textContent = host.settingsHarnessError;
});

// Parent browser matrix (poll snapshot after each real input action):
// 1. Scale 100 -> fill 150 -> blur: DOM 150, store/storage 150.
// 2. Fill 250 -> Enter: DOM 200, store 200; trace records duplicate commits.
// 3. Already 200 -> fill 250 -> blur: DOM must reconcile to 200.
// 4. Fill 66.6 -> Enter: DOM/store 67; clear -> blur: store 100.
// 5. Clear when already 100: empty DOM with placeholder 100 is valid default.
// 6. Set scale 150 only; production reset button enabled; click -> store 100,
//    customization defaults, button disabled. Repeat after both stores changed.
// 7. UI/code size decimals/out-of-range/clear independently; null is inheritance.
// 8. While scale input has uncommitted draft, externally setScale(175); observe
//    production useEffect reconciliation, then blur. Do not replace handlers.
// 9. New context desktop=0: no Interface scale spinbutton; font rows still mount.
// Nonfinite type=number strings may sanitize to empty in the browser; capture
// DOM validity/value and actual result rather than force impossible React state.

