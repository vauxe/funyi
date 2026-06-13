# coding=utf-8
# Lint E721 is suppressed file-wide: the body below is vendored verbatim
# (upstream asserts ``type(raw_prob) == float``). Do not "fix" it here.
# ruff: noqa: E721
#
# Vendored verbatim from FireRedVAD, the upstream reference implementation:
#   repo:   https://github.com/FireRedTeam/FireRedVAD
#   file:   fireredvad/core/stream_vad_postprocessor.py
#   commit: c30ec49e8cc69642b0ee65362eba11b9d11c6e54 (2026-05-06)
#   license: Apache-2.0
#
# Kept byte-for-byte (only this NOTICE block is added) so the streaming VAD
# state machine matches upstream exactly. Do not hand-edit; re-vendor from
# upstream if it changes. The original copyright header follows.
#
# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Wenpeng Li, Kai Huang, Kun Liu)

import enum
from collections import deque
from dataclasses import dataclass


@dataclass
class StreamVadFrameResult:
    frame_idx: int  # 1-based
    is_speech: bool
    raw_prob: float
    smoothed_prob: float
    is_speech_start: bool = False
    is_speech_end: bool = False
    speech_start_frame: int = -1  # 1-based
    speech_end_frame: int = -1    # 1-based


@enum.unique
class VadState(enum.Enum):
    SILENCE = 0
    POSSIBLE_SPEECH = 1
    SPEECH = 2
    POSSIBLE_SILENCE = 3


class StreamVadPostprocessor:
    def __init__(self,
                 smooth_window_size,
                 speech_threshold,
                 pad_start_frame,
                 min_speech_frame,
                 max_speech_frame,
                 min_silence_frame):
        self.smooth_window_size = max(1, smooth_window_size)
        self.speech_threshold = speech_threshold
        self.pad_start_frame = max(self.smooth_window_size, pad_start_frame)
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self.reset()

    def reset(self):
        self.frame_cnt = 0
        # smooth window
        self.smooth_window = deque()
        self.smooth_window_sum = 0.0
        # state transition
        self.state = VadState.SILENCE
        self.speech_cnt = 0
        self.silence_cnt = 0
        self.hit_max_speech = False
        self.last_speech_start_frame = -1
        self.last_speech_end_frame = -1

    def process_one_frame(self, raw_prob):
        assert type(raw_prob) == float
        assert 0.0 <= raw_prob and raw_prob <= 1.0
        self.frame_cnt += 1

        smoothed_prob = self.smooth_prob(raw_prob)

        is_speech = self.apply_threshold(smoothed_prob)

        result = StreamVadFrameResult(
            frame_idx = self.frame_cnt,
            is_speech=is_speech,
            raw_prob=round(raw_prob, 3),
            smoothed_prob=round(smoothed_prob, 3)
        )

        result = self.state_transition(is_speech, result)

        return result

    def smooth_prob(self, prob):
        if self.smooth_window_size <= 1:
            return prob
        self.smooth_window.append(prob)
        self.smooth_window_sum += prob
        if len(self.smooth_window) > self.smooth_window_size:
            left = self.smooth_window.popleft()
            self.smooth_window_sum -= left
        smoothed_prob = self.smooth_window_sum / len(self.smooth_window)
        return smoothed_prob

    def apply_threshold(self, prob):
        return int(prob >= self.speech_threshold)

    def state_transition(self, is_speech, result):
        if self.hit_max_speech:
            result.is_speech_start = True
            result.speech_start_frame = self.frame_cnt
            self.last_speech_start_frame = result.speech_start_frame
            self.hit_max_speech = False

        if self.state == VadState.SILENCE:
            if is_speech:
                self.state = VadState.POSSIBLE_SPEECH
                self.speech_cnt += 1
            else:
                self.silence_cnt += 1
                self.speech_cnt = 0

        elif self.state == VadState.POSSIBLE_SPEECH:
            if is_speech:
                self.speech_cnt += 1
                if self.speech_cnt >= self.min_speech_frame:
                    self.state = VadState.SPEECH
                    result.is_speech_start = True
                    result.speech_start_frame = max(1,
                        self.frame_cnt - self.speech_cnt + 1 - self.pad_start_frame,
                        self.last_speech_end_frame + 1)
                    self.last_speech_start_frame = result.speech_start_frame
                    self.silence_cnt = 0
            else:
                self.state = VadState.SILENCE
                self.silence_cnt = 1
                self.speech_cnt = 0

        elif self.state == VadState.SPEECH:
            self.speech_cnt += 1
            if is_speech:
                self.silence_cnt = 0
                if self.speech_cnt >= self.max_speech_frame:
                    self.hit_max_speech = True
                    self.speech_cnt = 0
                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame
                    self.last_speech_start_frame = -1
                    self.last_speech_end_frame = result.speech_end_frame
            else:
                self.state = VadState.POSSIBLE_SILENCE
                self.silence_cnt += 1

        elif self.state == VadState.POSSIBLE_SILENCE:
            self.speech_cnt += 1
            if is_speech:
                self.state = VadState.SPEECH
                self.silence_cnt = 0
                if self.speech_cnt >= self.max_speech_frame:
                    self.hit_max_speech = True
                    self.speech_cnt = 0
                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame
                    self.last_speech_start_frame = -1
                    self.last_speech_end_frame = result.speech_end_frame

            else:
                self.silence_cnt += 1
                if self.silence_cnt >= self.min_silence_frame:
                    self.state = VadState.SILENCE
                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame
                    self.last_speech_end_frame = result.speech_end_frame
                    self.last_speech_start_frame = -1
                    self.speech_cnt = 0

        return result
