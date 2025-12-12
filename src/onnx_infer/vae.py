from typing import Optional

import numpy as np
import onnxruntime as ort

from .constants import CHUNK_SIZE, LATENT_DIM


def get_model_input_dtype(session: ort.InferenceSession, input_name: str) -> np.dtype:
    for inp in session.get_inputs():
        if inp.name == input_name:
            onnx_type = inp.type
            s = str(onnx_type)
            if 'tensor_type' in s:
                if 'float16' in s:
                    return np.float16
                elif 'float32' in s:
                    return np.float32
                elif 'float64' in s:
                    return np.float64
                elif 'int32' in s:
                    return np.int32
                elif 'int64' in s:
                    return np.int64
    return np.float32


def encode_audio_to_patches(vae_enc_sess: ort.InferenceSession, audio: np.ndarray, patch_size: int, inference_dtype: np.dtype = np.float32, run_options: Optional[ort.RunOptions] = None) -> np.ndarray:
    patch_len = patch_size * CHUNK_SIZE
    if audio.shape[0] % patch_len != 0:
        pad_len = patch_len - (audio.shape[0] % patch_len)
        audio = np.pad(audio, (0, pad_len), mode="constant")
    vae_input_dtype = get_model_input_dtype(vae_enc_sess, "audio_data")
    inp = audio.reshape(1, 1, -1).astype(vae_input_dtype)
    z = vae_enc_sess.run(None, {"audio_data": inp}, run_options=run_options)[0]
    z = np.squeeze(z, axis=0)
    D = z.shape[0]
    L = z.shape[1]
    assert D == LATENT_DIM, f"Unexpected latent dim {D}"
    if L % patch_size != 0:
        pad_len = patch_size - (L % patch_size)
        z = np.pad(z, ((0, 0), (0, pad_len)), mode="constant")
        L = z.shape[1]
    T = L // patch_size
    patches = z.reshape(D, T, patch_size).transpose(1, 2, 0)
    if patches.shape[0] > 0:
        patches = patches[:-1, ...]
    return patches.astype(inference_dtype)


def decode_audio(vae_dec_sess: ort.InferenceSession, latents: np.ndarray, device_type: str = 'cpu', device_id: int = 0, inference_dtype: np.dtype = np.float32, run_options: Optional[ort.RunOptions] = None) -> np.ndarray:
    if latents.shape[-1] == 0:
        return np.zeros((0,), dtype=inference_dtype)
    
    # latents shape: [B, D, T]
    # We need to make sure latents are properly scaled/formatted if needed
    # In original VoxCPM, VAE decode input is [B, D, T]
    
    vae_dec_in_names = [inp.name for inp in vae_dec_sess.get_inputs()]
    vae_dec_out_names = [out.name for out in vae_dec_sess.get_outputs()]
    vae_input_dtype = get_model_input_dtype(vae_dec_sess, "z")
    
    # Original implementation passes latents directly.
    # Check if we need to add/remove dims. 
    # Current input latents from run_inference is likely [B, T, D] or [B, D, T] ?
    # Let's trace back:
    # run_inference returns [1, T, P, D] -> [1, D, T*P] via some reshape?
    # No, run_inference (infer_loop.py) returns latents as [1, T*P, D] or similar?
    # Let's check infer_loop.py return value.
    # Ah, infer_loop returns `pred_feat_seq` which is [1, T, P, D].
    # But wait, run_inference returns the output of decode_sess?
    # No, run_inference returns `pred_feat` accumulated.
    
    # In app.py:
    # latents = run_inference(...)
    # audio = decode_audio(..., latents, ...)
    
    # infer_loop.py returns `feat_pred` which is rearranged:
    # feat_pred = rearrange(pred_feat_seq, "b t p d -> b d (t p)", ...)
    # So latents is [B, D, L] where L = T*P.
    
    latents_ort = ort.OrtValue.ortvalue_from_numpy(latents.astype(vae_input_dtype), device_type, device_id)
    input_feed_vae_dec = {vae_dec_in_names[0]: latents_ort}
    audio_out = vae_dec_sess.run_with_ort_values(vae_dec_out_names, input_feed_vae_dec)[0]
    audio = audio_out.numpy()
    
    # audio output from VAE decoder is [B, 1, Samples]
    audio = np.squeeze(audio, axis=1) # [B, Samples]
    audio = audio[0] # [Samples]
    
    # Original code does trimming: decode_audio = decode_audio[..., 640:-640]
    # This might be important if the VAE adds padding or has artifacts at boundaries.
    # The trim amount 640 depends on CHUNK_SIZE or similar.
    # Let's add trimming.
    trim_len = CHUNK_SIZE
    if audio.shape[0] > 2 * trim_len:
        audio = audio[trim_len:-trim_len]
        
    return audio.astype(inference_dtype)

__all__ = ["get_model_input_dtype", "encode_audio_to_patches", "decode_audio"]