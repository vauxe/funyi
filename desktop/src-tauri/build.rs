fn main() {
    configure_macos_linker();
    tauri_build::build();
}

#[cfg(target_os = "macos")]
fn configure_macos_linker() {
    println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET=13.0");
    println!("cargo:rustc-link-arg=-mmacosx-version-min=13.0");

    for path in swift_runtime_library_paths() {
        println!("cargo:rustc-link-search=native={path}");
    }
    if let Some(sdk_path) = command_stdout("xcrun", &["--show-sdk-path"]) {
        println!("cargo:rustc-link-search=framework={sdk_path}/System/Library/Frameworks");
    }
}

#[cfg(not(target_os = "macos"))]
fn configure_macos_linker() {}

#[cfg(target_os = "macos")]
fn swift_runtime_library_paths() -> Vec<String> {
    let Some(target_info) = command_stdout("swift", &["-print-target-info"]) else {
        return Vec::new();
    };

    let mut paths = Vec::new();
    let mut in_runtime_paths = false;
    for line in target_info.lines() {
        if line.contains("\"runtimeLibraryPaths\"") {
            in_runtime_paths = true;
            continue;
        }
        if in_runtime_paths && line.contains(']') {
            break;
        }
        if in_runtime_paths {
            if let Some(path) = json_string_value(line) {
                paths.push(path);
            }
        }
    }
    paths
}

#[cfg(target_os = "macos")]
fn json_string_value(line: &str) -> Option<String> {
    let trimmed = line.trim().trim_end_matches(',');
    if !(trimmed.starts_with('"') && trimmed.ends_with('"')) {
        return None;
    }
    Some(trimmed.trim_matches('"').to_string())
}

#[cfg(target_os = "macos")]
fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(command)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    Some(text.trim().to_string())
}
