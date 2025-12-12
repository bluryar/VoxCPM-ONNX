from typing import Optional, Tuple
import time
import numpy as np
import onnxruntime as ort

from .constants import AUDIO_START_TOKEN, LATENT_DIM
from .audio import read_audio_mono
from .vae import encode_audio_to_patches


def build_inputs(tokenizer,
                 target_text: str,
                 prompt_text: str,
                 prompt_wav_path: Optional[str],
                 vae_enc_sess: ort.InferenceSession,
                 patch_size: int,
                 inference_dtype: np.dtype = np.float32,
                 run_options: Optional[ort.RunOptions] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not prompt_wav_path:
        text = target_text
        t_tok0 = time.perf_counter()
        text_token = np.array(tokenizer(text), dtype=np.int64)
        tok_dt = time.perf_counter() - t_tok0
        print(f"分词耗时: {tok_dt:.3f}s，token数: {text_token.shape[0]}")
        text_token = np.concatenate([text_token, np.array([AUDIO_START_TOKEN], dtype=np.int64)], axis=-1)
        text_length = text_token.shape[0]
        audio_feat = np.zeros((text_length, patch_size, LATENT_DIM), dtype=inference_dtype)
        text_mask = np.ones(text_length, dtype=np.int32)
        audio_mask = np.zeros(text_length, dtype=np.int32)
    else:
        text = (prompt_text or "") + (target_text or "")
        t_tok0 = time.perf_counter()
        text_token = np.array(tokenizer(text), dtype=np.int64)
        tok_dt = time.perf_counter() - t_tok0
        print(f"分词耗时: {tok_dt:.3f}s，token数: {text_token.shape[0]}")
        text_token = np.concatenate([text_token, np.array([AUDIO_START_TOKEN], dtype=np.int64)], axis=-1)
        text_length = text_token.shape[0]
        t_audio0 = time.perf_counter()
        audio, sr = read_audio_mono(prompt_wav_path)
        audio_dt = time.perf_counter() - t_audio0
        print(f"音频读取+重采样耗时: {audio_dt:.3f}s，采样率: {sr}，样本数: {audio.shape[0]}")
        t_vaeenc0 = time.perf_counter()
        patches = encode_audio_to_patches(vae_enc_sess, audio, patch_size, inference_dtype, run_options)
        vae_enc_dt = time.perf_counter() - t_vaeenc0
        print(f"VAE编码耗时: {vae_enc_dt:.3f}s，patch形状: {patches.shape}")
        audio_length = patches.shape[0]
        text_pad = np.zeros(audio_length, dtype=np.int64)
        text_token = np.concatenate([text_token, text_pad])
        audio_pad = np.zeros((text_length, patch_size, LATENT_DIM), dtype=inference_dtype)
        audio_feat = np.concatenate([audio_pad, patches], axis=0)
        text_mask = np.concatenate([np.ones(text_length), np.zeros(audio_length)]).astype(np.int32)
        audio_mask = np.concatenate([np.zeros(text_length), np.ones(audio_length)]).astype(np.int32)

    text_token = np.expand_dims(text_token, 0)
    text_mask = np.expand_dims(text_mask, 0)
    audio_feat = np.expand_dims(audio_feat, 0)
    audio_mask = np.expand_dims(audio_mask, 0)
    audio_feat = audio_feat.astype(inference_dtype, copy=False)
    return text_token, text_mask, audio_feat, audio_mask

__all__ = ["build_inputs"]

def build_inputs_with_patches(tokenizer,
                              target_text: str,
                              prompt_text: str,
                              patches: np.ndarray,
                              patch_size: int,
                              inference_dtype: np.dtype = np.float32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    text = (prompt_text or "") + (target_text or "")
    t_tok0 = time.perf_counter()
    text_token = np.array(tokenizer(text), dtype=np.int64)
    tok_dt = time.perf_counter() - t_tok0
    print(f"分词耗时: {tok_dt:.3f}s，token数: {text_token.shape[0]}")
    text_token = np.concatenate([text_token, np.array([AUDIO_START_TOKEN], dtype=np.int64)], axis=-1)
    text_length = text_token.shape[0]

    audio_length = patches.shape[0]
    text_pad = np.zeros(audio_length, dtype=np.int64)
    text_token = np.concatenate([text_token, text_pad])
    audio_pad = np.zeros((text_length, patch_size, LATENT_DIM), dtype=inference_dtype)
    audio_feat = np.concatenate([audio_pad, patches.astype(inference_dtype)], axis=0)
    text_mask = np.concatenate([np.ones(text_length), np.zeros(audio_length)]).astype(np.int32)
    audio_mask = np.concatenate([np.zeros(text_length), np.ones(audio_length)]).astype(np.int32)

    text_token = np.expand_dims(text_token, 0)
    text_mask = np.expand_dims(text_mask, 0)
    audio_feat = np.expand_dims(audio_feat, 0)
    audio_mask = np.expand_dims(audio_mask, 0)
    return text_token, text_mask, audio_feat, audio_mask

__all__.append("build_inputs_with_patches")
