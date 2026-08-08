#!/usr/bin/env python3
"""Add disposable macOS lifecycle instrumentation to PR 7790 or its base."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path.cwd().resolve()
MAIN = ROOT / "studio/src-tauri/src/main.rs"
CARGO = ROOT / "studio/src-tauri/Cargo.toml"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one sentinel, found {count}")
    return text.replace(old, new, 1)


COMMON = r'''// CI-only lifecycle probe. This block is injected in the disposable staging
// checkout and is never part of the pull request under test.
#[cfg(target_os = "macos")]
fn quit_ci_log(event: &str) {
    use std::io::Write;
    let Ok(path) = std::env::var("UNSLOTH_QUIT_CI_LOG") else {
        return;
    };
    if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{event}");
        let _ = file.flush();
    }
}

#[cfg(target_os = "macos")]
fn quit_ci_state_has(kind: &str) -> bool {
    std::env::var("UNSLOTH_QUIT_CI_STATE")
        .unwrap_or_default()
        .split(',')
        .any(|part| part.trim() == kind)
}

#[cfg(target_os = "macos")]
fn quit_ci_response(kind: &str) -> Option<bool> {
    let spec = std::env::var("UNSLOTH_QUIT_CI_RESPONSE").ok()?;
    let selected = spec.split(',').find_map(|entry| {
        let mut pieces = entry.splitn(2, '=');
        let first = pieces.next()?.trim();
        match pieces.next() {
            Some(value) if first == kind => Some(value.trim()),
            None => Some(first),
            _ => None,
        }
    })?;
    let response = match selected {
        "confirm" => true,
        "cancel" => false,
        _ => return None,
    };
    if let Ok(delay) = std::env::var("UNSLOTH_QUIT_CI_DELAY_MS") {
        if let Ok(delay) = delay.parse::<u64>() {
            std::thread::sleep(std::time::Duration::from_millis(delay));
        }
    }
    quit_ci_log(&format!("response {kind} {response}"));
    Some(response)
}

#[cfg(target_os = "macos")]
fn quit_ci_native_terminate(app: &tauri::AppHandle) {
    let result = app.run_on_main_thread(|| unsafe {
        let nsapp: *mut objc2::runtime::AnyObject =
            objc2::msg_send![objc2::class!(NSApplication), sharedApplication];
        let () =
            objc2::msg_send![nsapp, terminate: std::ptr::null_mut::<objc2::runtime::AnyObject>()];
    });
    if let Err(error) = result {
        quit_ci_log(&format!("native terminate scheduling failed: {error}"));
    }
}

#[cfg(target_os = "macos")]
fn setup_quit_ci_state(app: &tauri::App) {
    if quit_ci_state_has("training") {
        if let Some(state) = app.try_state::<TrainingActivityState>() {
            if let Ok(mut running) = state.lock() {
                *running = true;
            }
        }
    }
    let state = std::env::var("UNSLOTH_QUIT_CI_STATE").unwrap_or_else(|_| "inactive".into());
    quit_ci_log(&format!("ready state={state}"));

    let Some(trigger) = std::env::var("UNSLOTH_QUIT_CI_TRIGGER").ok() else {
        return;
    };
    let handle = app.handle().clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(350));
        quit_ci_log(&format!("trigger {trigger}"));
        match trigger.as_str() {
            "native" => quit_ci_native_terminate(&handle),
            "native-double" => {
                quit_ci_native_terminate(&handle);
                std::thread::sleep(std::time::Duration::from_millis(100));
                quit_ci_native_terminate(&handle);
            }
            "menu" => confirm_then_quit(&handle),
            "programmatic" => handle.exit(42),
            _ => quit_ci_log("unknown trigger"),
        }
    });
}
'''


def instrument(mode: str) -> None:
    main = MAIN.read_text(encoding="utf-8")
    main = replace_once(
        main,
        "fn setup_logging() {",
        COMMON + "\nfn setup_logging() {",
        "common instrumentation",
    )
    setup = """            #[cfg(unix)]
            setup_unix_termination_signals(app)?;
            Ok(())"""
    main = replace_once(
        main,
        setup,
        """            #[cfg(unix)]
            setup_unix_termination_signals(app)?;
            #[cfg(target_os = "macos")]
            setup_quit_ci_state(app);
            Ok(())""",
        "setup hook",
    )

    if mode == "base":
        main = replace_once(
            main,
            '            "menu" => confirm_then_quit(&handle),',
            '            "menu" => quit_ci_log("menu trigger unavailable on base"),',
            "base menu trigger removal",
        )
        cargo = CARGO.read_text(encoding="utf-8")
        if "[target.'cfg(target_os = \"macos\")'.dependencies]" not in cargo:
            cargo = replace_once(
                cargo,
                "[target.'cfg(target_os = \"linux\")'.dependencies]",
                "[target.'cfg(target_os = \"macos\")'.dependencies]\nobjc2 = \"0.6\"\n\n[target.'cfg(target_os = \"linux\")'.dependencies]",
                "base objc2 dependency",
            )
            CARGO.write_text(cargo, encoding="utf-8")
    else:
        main = replace_once(
            main,
            """fn training_is_active(app: &tauri::AppHandle) -> bool {
    let Some(state)""",
            """fn training_is_active(app: &tauri::AppHandle) -> bool {
    #[cfg(target_os = "macos")]
    if quit_ci_state_has("training") {
        return true;
    }
    let Some(state)""",
            "training activity override",
        )
        main = replace_once(
            main,
            """fn install_is_active(app: &tauri::AppHandle) -> bool {
    let Some(state)""",
            """fn install_is_active(app: &tauri::AppHandle) -> bool {
    #[cfg(target_os = "macos")]
    if quit_ci_state_has("install") {
        return true;
    }
    let Some(state)""",
            "install activity override",
        )
        main = replace_once(
            main,
            """    if !training_is_active(app) {
        return true;
    }
    app.dialog()""",
            """    if !training_is_active(app) {
        return true;
    }
    #[cfg(target_os = "macos")]
    {
        quit_ci_log("prompt training");
        if let Some(response) = quit_ci_response("training") {
            return response;
        }
    }
    app.dialog()""",
            "training response hook",
        )
        main = replace_once(
            main,
            """    if !install_is_active(app) {
        return true;
    }
    app.dialog()""",
            """    if !install_is_active(app) {
        return true;
    }
    #[cfg(target_os = "macos")]
    {
        quit_ci_log("prompt install");
        if let Some(response) = quit_ci_response("install") {
            return response;
        }
    }
    app.dialog()""",
            "install response hook",
        )
        main = replace_once(
            main,
            """fn confirm_then_quit(app: &tauri::AppHandle) {
    spawn_quit_confirmation""",
            """fn confirm_then_quit(app: &tauri::AppHandle) {
    #[cfg(target_os = "macos")]
    quit_ci_log("menu confirmation path");
    spawn_quit_confirmation""",
            "menu log hook",
        )
        main = replace_once(
            main,
            """    const NS_TERMINATE_LATER: usize = 2;

    let Some(app)""",
            """    const NS_TERMINATE_LATER: usize = 2;

    quit_ci_log("applicationShouldTerminate entered");
    let Some(app)""",
            "delegate entry log",
        )
        main = replace_once(
            main,
            """    if !install_is_active(app) && !training_is_active(app) {
        return NS_TERMINATE_NOW;
    }""",
            """    if !install_is_active(app) && !training_is_active(app) {
        quit_ci_log("applicationShouldTerminate NOW");
        return NS_TERMINATE_NOW;
    }""",
            "inactive result log",
        )
        main = replace_once(
            main,
            """    if spawn_quit_confirmation(app, |app, proceed| {
        if proceed {""",
            """    if spawn_quit_confirmation(app, |app, proceed| {
        quit_ci_log(&format!("confirmation completed {proceed}"));
        if proceed {""",
            "confirmation completion log",
        )
        main = replace_once(
            main,
            """    }) {
        NS_TERMINATE_LATER
    } else {
        // Another quit path""",
            """    }) {
        quit_ci_log("applicationShouldTerminate LATER");
        NS_TERMINATE_LATER
    } else {
        quit_ci_log("applicationShouldTerminate CANCEL duplicate");
        // Another quit path""",
            "delegate result log",
        )
        main = replace_once(
            main,
            """fn reply_to_termination_request(app: &tauri::AppHandle, proceed: bool) {
    use objc2""",
            """fn reply_to_termination_request(app: &tauri::AppHandle, proceed: bool) {
    quit_ci_log(&format!("replyToApplicationShouldTerminate {proceed}"));
    use objc2""",
            "reply log",
        )

    MAIN.write_text(main, encoding="utf-8")
    print(f"instrumented {mode}: {MAIN}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("base", "target"))
    instrument(parser.parse_args().mode)
