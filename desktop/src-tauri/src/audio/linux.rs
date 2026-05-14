use std::io::{ErrorKind, Read};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;

use tauri::AppHandle;

use super::{
    emit_audio_capture_error, emit_audio_frame, make_handle_with_shutdown, AudioSource,
    CaptureHandle, FRAME_BYTES, OUTPUT_CHANNELS, OUTPUT_SAMPLE_RATE,
};

const SOURCE_PREFIX: &str = "pulse:";
const UNAVAILABLE_ID: &str = "linux_monitor_unavailable";
const PULSE_SAMPLE_FORMAT: &str = "s16le";

pub fn list_audio_sources() -> Vec<AudioSource> {
    if !has_command("pactl") || !has_command("parec") {
        return vec![unavailable_source(
            "PipeWire/PulseAudio monitor capture requires pactl and parec on PATH.",
        )];
    }

    match discover_monitor_sources() {
        Ok(sources) if !sources.is_empty() => sources,
        Ok(_) => vec![unavailable_source(
            "No PipeWire/PulseAudio monitor sources were found. Select a playback monitor source, not a microphone input.",
        )],
        Err(error) => vec![unavailable_source(format!(
            "Could not list PipeWire/PulseAudio monitor sources: {error}",
        ))],
    }
}

pub fn start_audio_capture(app: AppHandle, source_id: &str) -> Result<CaptureHandle, String> {
    let source_name = source_id
        .strip_prefix(SOURCE_PREFIX)
        .filter(|source| !source.is_empty())
        .ok_or_else(|| format!("unsupported Linux audio source: {source_id}"))?;

    let mut child = Command::new("parec")
        .arg(format!("--device={source_name}"))
        .arg(format!("--format={PULSE_SAMPLE_FORMAT}"))
        .arg(format!("--rate={OUTPUT_SAMPLE_RATE}"))
        .arg(format!("--channels={OUTPUT_CHANNELS}"))
        .arg("--raw")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("failed to start parec: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "failed to read parec stdout".to_string())?;
    let child = Arc::new(Mutex::new(child));

    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let thread_child = Arc::clone(&child);
    let join = thread::Builder::new()
        .name("funyi-pulse-monitor".to_string())
        .spawn(move || {
            if let Err(error) = capture_child_loop(app.clone(), thread_stop, stdout, thread_child) {
                emit_audio_capture_error(&app, error);
            }
        })
        .map_err(|error| error.to_string())?;

    let shutdown_child = Arc::clone(&child);
    Ok(make_handle_with_shutdown(stop, join, move || {
        if let Ok(mut child) = shutdown_child.lock() {
            let _ = child.kill();
        }
    }))
}

fn discover_monitor_sources() -> Result<Vec<AudioSource>, String> {
    let output = Command::new("pactl")
        .args(["list", "short", "sources"])
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }

    Ok(parse_pactl_short_sources(&String::from_utf8_lossy(
        &output.stdout,
    )))
}

fn parse_pactl_short_sources(output: &str) -> Vec<AudioSource> {
    output
        .lines()
        .filter_map(parse_pactl_short_source_line)
        .collect()
}

fn parse_pactl_short_source_line(line: &str) -> Option<AudioSource> {
    let mut fields = line.split('\t');
    let _index = fields.next()?;
    let raw_name = fields.next()?.trim();
    if raw_name.is_empty() || !raw_name.ends_with(".monitor") {
        return None;
    }

    Some(AudioSource {
        id: format!("{SOURCE_PREFIX}{raw_name}"),
        name: format!("{} (monitor)", raw_name.trim_end_matches(".monitor")),
        kind: "system".to_string(),
        is_available: true,
        detail: format!("Captures playback from PipeWire/PulseAudio monitor source {raw_name}."),
    })
}

fn capture_child_loop(
    app: AppHandle,
    stop: Arc<AtomicBool>,
    mut stdout: ChildStdout,
    child: Arc<Mutex<Child>>,
) -> Result<(), String> {
    let mut seq = 0_u64;
    let mut frame = vec![0_u8; FRAME_BYTES];

    while !stop.load(Ordering::SeqCst) {
        match stdout.read_exact(&mut frame) {
            Ok(()) => {
                emit_audio_frame(&app, seq, &frame).map_err(|error| error.to_string())?;
                seq = seq.saturating_add(1);
            }
            Err(_) if stop.load(Ordering::SeqCst) => break,
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => {
                return child_exit_error(child);
            }
            Err(error) => return Err(format!("failed reading parec audio: {error}")),
        }
    }

    Ok(())
}

fn child_exit_error(child: Arc<Mutex<Child>>) -> Result<(), String> {
    let status = child
        .lock()
        .map_err(|_| "parec state lock failed".to_string())?
        .wait()
        .map_err(|error| format!("failed waiting for parec: {error}"))?;
    Err(format!("parec exited with status {status}"))
}

fn has_command(command: &str) -> bool {
    Command::new(command)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

fn unavailable_source(detail: impl Into<String>) -> AudioSource {
    AudioSource {
        id: UNAVAILABLE_ID.to_string(),
        name: "System audio monitor".to_string(),
        kind: "system".to_string(),
        is_available: false,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::parse_pactl_short_sources;

    #[test]
    fn parses_monitor_sources_and_skips_microphones() {
        let sources = parse_pactl_short_sources(
            "\
42\talsa_input.pci-0000_00_1f.3.analog-stereo\tPipeWire\tfloat32le 2ch 48000Hz\tSUSPENDED\n\
43\talsa_output.pci-0000_00_1f.3.analog-stereo.monitor\tPipeWire\tfloat32le 2ch 48000Hz\tRUNNING\n",
        );

        assert_eq!(sources.len(), 1);
        assert_eq!(
            sources[0].id,
            "pulse:alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
        );
        assert!(sources[0].is_available);
    }
}
