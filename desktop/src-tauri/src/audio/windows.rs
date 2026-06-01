use std::collections::VecDeque;
use std::error::Error;
use std::ptr;
use std::slice;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;

use tauri::AppHandle;
use wasapi::{deinitialize, initialize_mta, Device, DeviceEnumerator, Direction};
use windows::{
    core::{PCSTR, PWSTR},
    Win32::{
        Foundation::{CloseHandle, HANDLE},
        Media::Audio::{
            eCapture, eConsole, eRender, IAudioCaptureClient, IAudioClient, IMMDevice,
            IMMDeviceEnumerator, MMDeviceEnumerator, AUDCLNT_BUFFERFLAGS_SILENT,
            AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
            AUDCLNT_STREAMFLAGS_EVENTCALLBACK, AUDCLNT_STREAMFLAGS_LOOPBACK,
            AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, DEVICE_STATE_ACTIVE, WAVEFORMATEX,
            WAVE_FORMAT_PCM,
        },
        System::{
            Com::{CoCreateInstance, CoTaskMemFree, CLSCTX_ALL},
            Threading::{CreateEventA, WaitForSingleObject},
        },
    },
};

use super::{
    emit_pending_audio_frames, spawn_capture_thread, AudioSource, AudioSourceKind, CaptureHandle,
    FRAME_BYTES, OUTPUT_BITS, OUTPUT_CHANNELS, OUTPUT_SAMPLE_RATE,
};

const SOURCE_ID: &str = "system_default";
const MICROPHONE_SOURCE_PREFIX: &str = "windows_microphone:";

pub fn list_audio_sources() -> Vec<AudioSource> {
    let mut sources = vec![AudioSource {
        id: SOURCE_ID.to_string(),
        name: "Audio".to_string(),
        kind: AudioSourceKind::System,
        is_available: true,
        detail: "Captures audio currently playing on the default Windows playback device with WASAPI loopback."
            .to_string(),
    }];

    match list_microphone_sources() {
        Ok(microphones) => sources.extend(microphones),
        Err(error) => sources.push(unavailable_microphone_source(format!(
            "Windows microphone enumeration failed: {error}"
        ))),
    }
    sources
}

pub fn start_audio_capture(app: AppHandle, source_id: &str) -> Result<CaptureHandle, String> {
    let source = WindowsAudioSource::parse(source_id)?;
    validate_audio_source(&source)?;
    let thread_name = source.thread_name();

    spawn_capture_thread(thread_name, app, move |app, stop| {
        capture_loop(app, stop, source).map_err(|error| error.to_string())
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum WindowsAudioSource {
    SystemLoopback,
    Microphone { device_id: String },
}

impl WindowsAudioSource {
    fn parse(source_id: &str) -> Result<Self, String> {
        if source_id == SOURCE_ID {
            return Ok(Self::SystemLoopback);
        }
        if let Some(device_id) = source_id.strip_prefix(MICROPHONE_SOURCE_PREFIX) {
            if device_id.is_empty() {
                return Err("missing Windows microphone device id".to_string());
            }
            return Ok(Self::Microphone {
                device_id: device_id.to_string(),
            });
        }
        Err(format!("unsupported Windows audio source: {source_id}"))
    }

    fn thread_name(&self) -> &'static str {
        match self {
            Self::SystemLoopback => "funyi-wasapi-loopback",
            Self::Microphone { .. } => "funyi-wasapi-microphone",
        }
    }
}

fn microphone_source_id(device_id: &str) -> String {
    format!("{MICROPHONE_SOURCE_PREFIX}{device_id}")
}

fn validate_audio_source(source: &WindowsAudioSource) -> Result<(), String> {
    let source = source.clone();
    let join = thread::Builder::new()
        .name("funyi-wasapi-validate".to_string())
        .spawn(move || {
            let _com = ComApartment::initialize_mta().map_err(|error| error.to_string())?;
            capture_endpoint(source)
                .map(|_| ())
                .map_err(|error| error.to_string())
        })
        .map_err(|error| error.to_string())?;
    join.join()
        .map_err(|_| "Windows audio source validation thread panicked".to_string())?
}

fn list_microphone_sources() -> Result<Vec<AudioSource>, String> {
    let join = thread::Builder::new()
        .name("funyi-wasapi-enumerate".to_string())
        .spawn(|| list_microphone_sources_on_current_thread().map_err(|error| error.to_string()))
        .map_err(|error| error.to_string())?;
    join.join()
        .map_err(|_| "Windows microphone enumeration thread panicked".to_string())?
}

fn list_microphone_sources_on_current_thread() -> Result<Vec<AudioSource>, Box<dyn Error>> {
    let _com = ComApartment::initialize_mta()?;

    let enumerator = DeviceEnumerator::new()?;
    let default_device_id = enumerator
        .get_default_device(&Direction::Capture)
        .ok()
        .and_then(|device| device.get_id().ok());
    let devices = enumerator.get_device_collection(&Direction::Capture)?;
    let mut sources = Vec::new();

    for device in &devices {
        let device = device?;
        let id = device.get_id()?;
        let is_default = default_device_id.as_deref() == Some(id.as_str());
        let source = microphone_source_from_device(&device, &id, is_default);
        if is_default {
            sources.insert(0, source);
        } else {
            sources.push(source);
        }
    }

    if sources.is_empty() {
        sources.push(unavailable_microphone_source(
            "No active Windows microphone input device was reported by WASAPI.".to_string(),
        ));
    }
    Ok(sources)
}

fn microphone_source_from_device(device: &Device, id: &str, is_default: bool) -> AudioSource {
    let name = device
        .get_friendlyname()
        .unwrap_or_else(|_| "Microphone".to_string());
    AudioSource {
        id: microphone_source_id(id),
        name,
        kind: AudioSourceKind::Microphone,
        is_available: true,
        detail: if is_default {
            "Captures default Windows microphone input with WASAPI."
        } else {
            "Captures Windows microphone input with WASAPI."
        }
        .to_string(),
    }
}

fn unavailable_microphone_source(detail: String) -> AudioSource {
    AudioSource {
        id: "windows_microphone_unavailable".to_string(),
        name: "Microphone".to_string(),
        kind: AudioSourceKind::Microphone,
        is_available: false,
        detail,
    }
}

fn capture_loop(
    app: AppHandle,
    stop: Arc<AtomicBool>,
    source: WindowsAudioSource,
) -> Result<(), Box<dyn Error>> {
    let _com = ComApartment::initialize_mta()?;

    let stream = WasapiCaptureStream::start(source)?;
    // Room for a few 100 ms output frames plus one device buffer so steady-state
    // draining rarely reallocates. The queue is fully drained on every wake, so this
    // is only an initial hint, not a cap.
    let mut sample_queue =
        VecDeque::with_capacity(FRAME_BYTES * 4 + stream.buffer_capacity_bytes());
    let mut seq = 0_u64;

    while !stop.load(Ordering::SeqCst) {
        stream.drain_packets(&mut sample_queue)?;
        // Tolerate a transient emit failure (e.g. a teardown-time race) the way the
        // macOS callback does, rather than treating it as fatal and tearing down a
        // healthy capture; the stop flag still drives shutdown.
        let _ = emit_pending_audio_frames(&app, &mut sample_queue, &mut seq);
        stream.wait_for_packet(100);
    }
    Ok(())
}

struct WasapiCaptureStream {
    audio_client: IAudioClient,
    capture_client: IAudioCaptureClient,
    event: EventHandle,
    block_align: usize,
    buffer_frame_count: u32,
}

impl WasapiCaptureStream {
    fn start(source: WindowsAudioSource) -> Result<Self, Box<dyn Error>> {
        let is_loopback = source == WindowsAudioSource::SystemLoopback;
        let endpoint = capture_endpoint(source)?;
        let audio_client: IAudioClient = unsafe { endpoint.Activate(CLSCTX_ALL, None)? };

        let format = output_wave_format();
        let event = EventHandle::new()?;
        unsafe {
            audio_client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                capture_stream_flags(is_loopback),
                0,
                0,
                &format,
                None,
            )?;
            audio_client.SetEventHandle(event.raw())?;
        }
        let buffer_frame_count = unsafe { audio_client.GetBufferSize()? };
        let capture_client = unsafe { audio_client.GetService::<IAudioCaptureClient>()? };
        unsafe {
            audio_client.Start()?;
        }

        Ok(Self {
            audio_client,
            capture_client,
            event,
            block_align: output_block_align(),
            buffer_frame_count,
        })
    }

    fn buffer_capacity_bytes(&self) -> usize {
        self.block_align * self.buffer_frame_count as usize
    }

    fn drain_packets(&self, sample_queue: &mut VecDeque<u8>) -> Result<(), Box<dyn Error>> {
        while unsafe { self.capture_client.GetNextPacketSize()? } > 0 {
            self.read_packet(sample_queue)?;
        }
        Ok(())
    }

    fn read_packet(&self, sample_queue: &mut VecDeque<u8>) -> Result<(), Box<dyn Error>> {
        let mut data = ptr::null_mut();
        let mut frame_count = 0_u32;
        let mut flags = 0_u32;
        unsafe {
            self.capture_client
                .GetBuffer(&mut data, &mut frame_count, &mut flags, None, None)?;
        }

        let append_result = unsafe {
            append_capture_packet_bytes(sample_queue, data, frame_count, flags, self.block_align)
        };
        let release_result = unsafe { self.capture_client.ReleaseBuffer(frame_count) };
        append_result?;
        release_result?;
        Ok(())
    }

    fn wait_for_packet(&self, timeout_ms: u32) {
        self.event.wait(timeout_ms);
    }
}

impl Drop for WasapiCaptureStream {
    fn drop(&mut self) {
        unsafe {
            let _ = self.audio_client.Stop();
        }
    }
}

fn capture_stream_flags(is_loopback: bool) -> u32 {
    let mut flags = AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
        | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY
        | AUDCLNT_STREAMFLAGS_EVENTCALLBACK;
    if is_loopback {
        flags |= AUDCLNT_STREAMFLAGS_LOOPBACK;
    }
    flags
}

fn output_wave_format() -> WAVEFORMATEX {
    let block_align = output_block_align();
    WAVEFORMATEX {
        wFormatTag: WAVE_FORMAT_PCM as u16,
        nChannels: OUTPUT_CHANNELS as u16,
        nSamplesPerSec: OUTPUT_SAMPLE_RATE as u32,
        nAvgBytesPerSec: (OUTPUT_SAMPLE_RATE * block_align) as u32,
        nBlockAlign: block_align as u16,
        wBitsPerSample: OUTPUT_BITS as u16,
        cbSize: 0,
    }
}

fn output_block_align() -> usize {
    OUTPUT_CHANNELS * OUTPUT_BITS / 8
}

unsafe fn append_capture_packet_bytes(
    sample_queue: &mut VecDeque<u8>,
    data: *const u8,
    frame_count: u32,
    flags: u32,
    block_align: usize,
) -> Result<(), Box<dyn Error>> {
    let byte_count = frame_count as usize * block_align;
    if byte_count == 0 {
        return Ok(());
    }

    if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 {
        sample_queue.extend(std::iter::repeat_n(0_u8, byte_count));
        return Ok(());
    }
    if data.is_null() {
        return Err("WASAPI returned a null non-silent capture packet".into());
    }

    let packet = unsafe { slice::from_raw_parts(data, byte_count) };
    sample_queue.extend(packet);
    Ok(())
}

fn capture_endpoint(source: WindowsAudioSource) -> Result<IMMDevice, Box<dyn Error>> {
    let enumerator: IMMDeviceEnumerator =
        unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)? };
    match source {
        WindowsAudioSource::SystemLoopback => {
            Ok(unsafe { enumerator.GetDefaultAudioEndpoint(eRender, eConsole)? })
        }
        WindowsAudioSource::Microphone { device_id } => {
            capture_endpoint_by_id(&enumerator, &device_id)
        }
    }
}

fn capture_endpoint_by_id(
    enumerator: &IMMDeviceEnumerator,
    device_id: &str,
) -> Result<IMMDevice, Box<dyn Error>> {
    let devices = unsafe { enumerator.EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE)? };
    let count = unsafe { devices.GetCount()? };
    for index in 0..count {
        let device = unsafe { devices.Item(index)? };
        if endpoint_id(&device)? == device_id {
            return Ok(device);
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("Windows microphone source is no longer available: {device_id}"),
    )
    .into())
}

fn endpoint_id(device: &IMMDevice) -> Result<String, Box<dyn Error>> {
    let id = CoTaskMemPwstr(unsafe { device.GetId()? });
    Ok(unsafe { id.0.to_string()? })
}

struct CoTaskMemPwstr(PWSTR);

impl Drop for CoTaskMemPwstr {
    fn drop(&mut self) {
        unsafe {
            CoTaskMemFree(Some(self.0.as_ptr().cast()));
        }
    }
}

struct EventHandle(HANDLE);

impl EventHandle {
    fn new() -> Result<Self, Box<dyn Error>> {
        Ok(Self(unsafe {
            CreateEventA(None, false, false, PCSTR::null())?
        }))
    }

    fn raw(&self) -> HANDLE {
        self.0
    }

    fn wait(&self, timeout_ms: u32) {
        unsafe {
            let _ = WaitForSingleObject(self.0, timeout_ms);
        }
    }
}

impl Drop for EventHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

struct ComApartment;

impl ComApartment {
    fn initialize_mta() -> Result<Self, Box<dyn Error>> {
        initialize_mta().ok()?;
        Ok(Self)
    }
}

impl Drop for ComApartment {
    fn drop(&mut self) {
        deinitialize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_system_loopback_source_id() {
        assert_eq!(
            WindowsAudioSource::parse(SOURCE_ID),
            Ok(WindowsAudioSource::SystemLoopback)
        );
    }

    #[test]
    fn parses_microphone_source_id_as_opaque_wasapi_device_id() {
        let source_id =
            microphone_source_id(r#"{0.0.1.00000000}.{a1b2c3d4-e5f6-4789-a012-3456789abcde}"#);

        assert_eq!(
            WindowsAudioSource::parse(&source_id),
            Ok(WindowsAudioSource::Microphone {
                device_id: r#"{0.0.1.00000000}.{a1b2c3d4-e5f6-4789-a012-3456789abcde}"#.to_string(),
            })
        );
    }

    #[test]
    fn rejects_empty_microphone_device_id() {
        assert_eq!(
            WindowsAudioSource::parse(MICROPHONE_SOURCE_PREFIX),
            Err("missing Windows microphone device id".to_string())
        );
    }

    #[test]
    fn appends_zeroes_for_silent_packet_without_reading_data() {
        let mut pending = VecDeque::from(vec![1_u8, 2]);

        unsafe {
            append_capture_packet_bytes(
                &mut pending,
                ptr::null(),
                2,
                AUDCLNT_BUFFERFLAGS_SILENT.0 as u32,
                2,
            )
            .unwrap();
        }

        assert_eq!(Vec::from(pending), vec![1, 2, 0, 0, 0, 0]);
    }

    #[test]
    fn appends_non_silent_packet_data() {
        let packet = [3_u8, 4, 5, 6];
        let mut pending = VecDeque::new();

        unsafe {
            append_capture_packet_bytes(&mut pending, packet.as_ptr(), 2, 0, 2).unwrap();
        }

        assert_eq!(Vec::from(pending), packet);
    }

    #[test]
    fn rejects_null_non_silent_packet_data() {
        let mut pending = VecDeque::new();

        let error = unsafe {
            append_capture_packet_bytes(&mut pending, ptr::null(), 2, 0, 2)
                .expect_err("null non-silent packet should fail")
        };

        assert_eq!(
            error.to_string(),
            "WASAPI returned a null non-silent capture packet"
        );
    }
}
