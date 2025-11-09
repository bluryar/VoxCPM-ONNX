from typing import Tuple
import numpy as np
import soundfile as sf

from .constants import SAMPLE_RATE


def resample_audio_linear(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    duration = audio.shape[0] / float(sr_in)
    new_len = int(round(duration * sr_out))
    x_old = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def read_audio_mono(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = resample_audio_linear(audio, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    audio = audio.astype(np.float32, copy=False)
    return audio, sr

__all__ = ["read_audio_mono", "resample_audio_linear"]