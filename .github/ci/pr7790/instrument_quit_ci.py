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
        quit_ci_log("native terminate main-thread entered");
        let nsapp: *mut objc2::runtime::AnyObject =
            objc2::msg_send![objc2::class!(NSApplication), sharedApplication];
        let () =
            objc2::msg_send![nsapp, terminate: std::ptr::null_mut::<objc2::runtime::AnyObject>()];
        quit_ci_log("native terminate main-thread returned");
    });
    if let Err(error) = result {
        quit_ci_log(&format!("native terminate scheduling failed: {error}"));
    }
}

#[cfg(target_os = "macos")]
fn quit_ci_native_menu(app: &tauri::AppHandle) {
    let result = app.run_on_main_thread(|| unsafe {
        let nsapp: *mut objc2::runtime::AnyObject =
            objc2::msg_send![objc2::class!(NSApplication), sharedApplication];
        if nsapp.is_null() {
            quit_ci_log("native menu missing NSApplication");
            return;
        }
        let main_menu: *mut objc2::runtime::AnyObject = objc2::msg_send![nsapp, mainMenu];
        if main_menu.is_null() {
            quit_ci_log("native menu missing mainMenu");
            return;
        }
        let app_item: *mut objc2::runtime::AnyObject =
            objc2::msg_send![main_menu, itemAtIndex: 0isize];
        if app_item.is_null() {
            quit_ci_log("native menu missing app item");
            return;
        }
        let app_menu: *mut objc2::runtime::AnyObject = objc2::msg_send![app_item, submenu];
        if app_menu.is_null() {
            quit_ci_log("native menu missing app submenu");
            return;
        }
        let count: isize = objc2::msg_send![app_menu, numberOfItems];
        quit_ci_log(&format!("native app menu item count {count}"));
        if count < 1 {
            quit_ci_log("native menu has no items");
            return;
        }
        quit_ci_log("native menu item performClick");
        let () = objc2::msg_send![app_menu, performActionForItemAtIndex: count - 1];
    });
    if let Err(error) = result {
        quit_ci_log(&format!("native menu scheduling failed: {error}"));
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
        let delay = if trigger == "native-menu" { 1_500 } else { 350 };
        std::thread::sleep(std::time::Duration::from_millis(delay));
        quit_ci_log(&format!("trigger {trigger}"));
        match trigger.as_str() {
            "native" => quit_ci_native_terminate(&handle),
            "native-double" => {
                quit_ci_native_terminate(&handle);
                std::thread::sleep(std::time::Duration::from_millis(100));
                quit_ci_native_terminate(&handle);
            }
            "delegate-double" => quit_ci_direct_duplicate(),
            "native-menu" => quit_ci_native_menu(&handle),
            "menu" => request_quit(&handle),
            "programmatic" => handle.exit(42),
            _ => quit_ci_log("unknown trigger"),
        }
    });
}
'''


def instrument(mode: str, *, check_only: bool = False) -> None:
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
            '            "delegate-double" => quit_ci_direct_duplicate(),',
            '            "delegate-double" => quit_ci_log("delegate probe unavailable on base"),',
            "base delegate probe removal",
        )
        main = replace_once(
            main,
            '            "menu" => request_quit(&handle),',
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
            if not check_only:
                CARGO.write_text(cargo, encoding="utf-8")
    else:
        main = replace_once(
            main,
            '''#[cfg(target_os = "macos")]
extern "C-unwind" fn application_should_terminate(''',
            '''#[cfg(target_os = "macos")]
fn quit_ci_direct_duplicate() {
    let selector = objc2::sel!(applicationShouldTerminate:);
    let first = application_should_terminate(std::ptr::null_mut(), selector, std::ptr::null_mut());
    let second = application_should_terminate(std::ptr::null_mut(), selector, std::ptr::null_mut());
    quit_ci_log(&format!("direct duplicate results {first} {second}"));
}

#[cfg(target_os = "macos")]
extern "C-unwind" fn application_should_terminate(''',
            "direct duplicate probe",
        )
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
        for function_name, state_name in (
            ("confirm_quit_during_update", "update"),
            ("confirm_quit_during_shell_update", "shell-update"),
            ("confirm_quit_during_downloads", "downloads"),
        ):
            sentinel = f'''fn {function_name}(app: &tauri::AppHandle) -> bool {{
    use tauri_plugin_dialog::{{DialogExt, MessageDialogButtons, MessageDialogKind}};
'''
            replacement = sentinel + f'''
    #[cfg(target_os = "macos")]
    if quit_ci_state_has("{state_name}") {{
        quit_ci_log("prompt {state_name}");
        if let Some(response) = quit_ci_response("{state_name}") {{
            return response;
        }}
    }}
'''
            main = replace_once(
                main,
                sentinel,
                replacement,
                f"{state_name} response hook",
            )
        main = replace_once(
            main,
            '''fn quit_requires_confirmation(app: &tauri::AppHandle) -> bool {
    let update_active''',
            '''fn quit_requires_confirmation(app: &tauri::AppHandle) -> bool {
    if ["update", "shell-update", "downloads"]
        .iter()
        .any(|kind| quit_ci_state_has(kind))
    {
        return true;
    }
    let update_active''',
            "native state override",
        )
        main = replace_once(
            main,
            """fn request_quit(app: &tauri::AppHandle) {
    spawn_quit_confirmation""",
            """fn request_quit(app: &tauri::AppHandle) {
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
            """    if !quit_requires_confirmation(app) {
        return NS_TERMINATE_NOW;
    }""",
            """    if !quit_requires_confirmation(app) {
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

    if not check_only:
        MAIN.write_text(main, encoding="utf-8")
    print(f"{'checked' if check_only else 'instrumented'} {mode}: {MAIN}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("base", "target"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    instrument(args.mode, check_only=args.check_only)
