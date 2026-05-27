use std::collections::VecDeque;

use super::OUTPUT_SAMPLE_RATE;

pub(crate) struct PcmCaptureBuffer {
    pub(crate) pending: VecDeque<u8>,
    resampler: PcmResampler,
}

impl PcmCaptureBuffer {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            pending: VecDeque::with_capacity(capacity),
            resampler: PcmResampler::new(),
        }
    }
}

struct PcmResampler {
    input_index: u64,
    input_rate: usize,
    next_output_at: f64,
    previous_sample: Option<(f64, f32)>,
}

impl PcmResampler {
    fn new() -> Self {
        Self {
            input_index: 0,
            input_rate: OUTPUT_SAMPLE_RATE,
            next_output_at: 0.0,
            previous_sample: None,
        }
    }

    fn reset(&mut self, input_rate: usize) {
        self.input_index = 0;
        self.input_rate = input_rate.max(1);
        self.next_output_at = 0.0;
        self.previous_sample = None;
    }
}

pub(crate) fn append_pcm_s16le(
    state: &mut PcmCaptureBuffer,
    data: &[u8],
    sample_count: usize,
    input_rate: usize,
) {
    if input_rate == OUTPUT_SAMPLE_RATE && sample_count > 0 && data.len() == sample_count * 2 {
        append_direct_pcm_bytes(state, data, sample_count);
        return;
    }

    let samples = extract_mono_samples(data, sample_count);
    if samples.is_empty() {
        return;
    }

    let input_rate = input_rate.max(1);
    if state.resampler.input_rate != input_rate {
        state.resampler.reset(input_rate);
    }

    if input_rate == OUTPUT_SAMPLE_RATE {
        append_direct_samples(state, samples, sample_count);
        return;
    }

    let step = input_rate as f64 / OUTPUT_SAMPLE_RATE as f64;
    for sample in samples {
        let current_at = state.resampler.input_index as f64;
        while state.resampler.next_output_at <= current_at {
            let output_sample =
                if let Some((previous_at, previous_sample)) = state.resampler.previous_sample {
                    let span = (current_at - previous_at).max(f64::EPSILON);
                    let t = ((state.resampler.next_output_at - previous_at) / span).clamp(0.0, 1.0);
                    previous_sample + (sample - previous_sample) * t as f32
                } else {
                    sample
                };
            push_i16_sample(&mut state.pending, output_sample);
            state.resampler.next_output_at += step;
        }
        state.resampler.previous_sample = Some((current_at, sample));
        state.resampler.input_index = state.resampler.input_index.saturating_add(1);
    }
}

fn append_direct_pcm_bytes(state: &mut PcmCaptureBuffer, data: &[u8], sample_count: usize) {
    state.pending.extend(data.iter().copied());
    advance_direct_samples(state, sample_count);
}

fn append_direct_samples(
    state: &mut PcmCaptureBuffer,
    samples: impl IntoIterator<Item = f32>,
    sample_count: usize,
) {
    for sample in samples {
        push_i16_sample(&mut state.pending, sample);
    }
    advance_direct_samples(state, sample_count);
}

fn advance_direct_samples(state: &mut PcmCaptureBuffer, sample_count: usize) {
    state.resampler.input_index = state
        .resampler
        .input_index
        .saturating_add(sample_count as u64);
    state.resampler.next_output_at = state.resampler.input_index as f64;
    state.resampler.previous_sample = None;
}

fn extract_mono_samples(data: &[u8], sample_count: usize) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }

    if sample_count > 0 {
        let f32_len = sample_count.saturating_mul(4);
        let i16_len = sample_count.saturating_mul(2);
        if data.len() == f32_len {
            return data.chunks_exact(4).map(f32_sample).collect();
        }
        if data.len() == i16_len {
            return data.chunks_exact(2).map(i16_sample).collect();
        }
        if f32_len > 0 && data.len().is_multiple_of(f32_len) {
            let channels = data.len() / f32_len;
            return average_interleaved_f32(data, sample_count, channels);
        }
        if i16_len > 0 && data.len().is_multiple_of(i16_len) {
            let channels = data.len() / i16_len;
            return average_interleaved_i16(data, sample_count, channels);
        }
    }

    data.chunks_exact(2).map(i16_sample).collect()
}

fn average_interleaved_f32(data: &[u8], sample_count: usize, channels: usize) -> Vec<f32> {
    data.chunks_exact(4)
        .map(f32_sample)
        .collect::<Vec<_>>()
        .chunks_exact(channels)
        .take(sample_count)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

fn average_interleaved_i16(data: &[u8], sample_count: usize, channels: usize) -> Vec<f32> {
    data.chunks_exact(2)
        .map(i16_sample)
        .collect::<Vec<_>>()
        .chunks_exact(channels)
        .take(sample_count)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

fn f32_sample(chunk: &[u8]) -> f32 {
    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).clamp(-1.0, 1.0)
}

fn i16_sample(chunk: &[u8]) -> f32 {
    i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0
}

fn push_i16_sample(output: &mut VecDeque<u8>, sample: f32) {
    let scaled = (sample.clamp(-1.0, 1.0) * i16::MAX as f32)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    output.extend(scaled.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resamples_96khz_float_mic_to_16khz_pcm() {
        let mut state = capture_buffer();
        let input = f32_bytes((0..96).map(|index| index as f32 / 96.0));

        append_pcm_s16le(&mut state, &input, 96, 96_000);

        assert_eq!(state.pending.len(), 16 * 2);
    }

    #[test]
    fn preserves_16khz_i16_pcm() {
        let mut state = capture_buffer();
        let mut input = Vec::new();
        for value in [-32768_i16, 0, 32767] {
            input.extend(value.to_le_bytes());
        }

        append_pcm_s16le(&mut state, &input, 3, 16_000);

        assert_eq!(state.pending.into_iter().collect::<Vec<_>>(), input);
    }

    #[test]
    fn averages_interleaved_stereo_float_samples() {
        let mut input = Vec::new();
        for sample in [0.0_f32, 1.0, -1.0, 1.0] {
            input.extend(sample.to_le_bytes());
        }

        assert_eq!(extract_mono_samples(&input, 2), vec![0.5, 0.0]);
    }

    fn capture_buffer() -> PcmCaptureBuffer {
        PcmCaptureBuffer::with_capacity(0)
    }

    fn f32_bytes(samples: impl IntoIterator<Item = f32>) -> Vec<u8> {
        let mut bytes = Vec::new();
        for sample in samples {
            bytes.extend(sample.to_le_bytes());
        }
        bytes
    }
}
