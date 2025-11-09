from typing import Optional

import numpy as np
import onnxruntime as ort
from tqdm import tqdm


def run_inference(prefill_sess: ort.InferenceSession,
                  decode_sess: ort.InferenceSession,
                  text_token: np.ndarray,
                  text_mask: np.ndarray,
                  audio_feat: np.ndarray,
                  audio_mask: np.ndarray,
                  min_len: int,
                  max_len: int,
                  cfg_value: float,
                  device_type: str = 'cpu',
                  device_id: int = 0,
                  inference_dtype: np.dtype = np.float32,
                  run_options: Optional[ort.RunOptions] = None) -> np.ndarray:
    prefill_in_names = [inp.name for inp in prefill_sess.get_inputs()]
    prefill_out_names = [out.name for out in prefill_sess.get_outputs()]
    decode_in_names = [inp.name for inp in decode_sess.get_inputs()]
    decode_out_names = [out.name for out in decode_sess.get_outputs()]

    text_token_ort = ort.OrtValue.ortvalue_from_numpy(text_token, device_type, device_id)
    text_mask_ort = ort.OrtValue.ortvalue_from_numpy(text_mask, device_type, device_id)
    audio_feat_ort = ort.OrtValue.ortvalue_from_numpy(audio_feat.astype(inference_dtype), device_type, device_id)
    audio_mask_ort = ort.OrtValue.ortvalue_from_numpy(audio_mask, device_type, device_id)

    input_feed_prefill = {
        prefill_in_names[0]: text_token_ort,
        prefill_in_names[1]: text_mask_ort,
        prefill_in_names[2]: audio_feat_ort,
        prefill_in_names[3]: audio_mask_ort,
    }

    outputs = prefill_sess.run_with_ort_values(prefill_out_names, input_feed_prefill)

    (
        dit_hidden_ort,
        base_next_keys_ort,
        base_next_values_ort,
        residual_next_keys_ort,
        residual_next_values_ort,
        prefix_feat_cond_ort,
    ) = outputs

    pred_seq = []
    cfg_scalar = np.array(cfg_value, dtype=inference_dtype)
    cfg_scalar_ort = ort.OrtValue.ortvalue_from_numpy(cfg_scalar, device_type, device_id)

    for _ in tqdm(range(max_len), desc="Decoding", unit="step"):
        noise = np.random.randn(*prefix_feat_cond_ort.shape()).astype(inference_dtype)
        noise_ort = ort.OrtValue.ortvalue_from_numpy(noise, device_type, device_id)

        input_feed_decode = {
            decode_in_names[0]: dit_hidden_ort,
            decode_in_names[1]: base_next_keys_ort,
            decode_in_names[2]: base_next_values_ort,
            decode_in_names[3]: residual_next_keys_ort,
            decode_in_names[4]: residual_next_values_ort,
            decode_in_names[5]: prefix_feat_cond_ort,
            decode_in_names[6]: noise_ort,
            decode_in_names[7]: cfg_scalar_ort,
        }

        dec_out = decode_sess.run_with_ort_values(decode_out_names, input_feed_decode)

        (
            pred_feat_ort,
            new_dit_hidden_ort,
            new_base_next_keys_ort,
            new_base_next_values_ort,
            new_residual_next_keys_ort,
            new_residual_next_values_ort,
            stop_flag_ort,
        ) = dec_out

        pred_feat = pred_feat_ort.numpy()
        pred_seq.append(pred_feat)

        prefix_feat_cond_ort = pred_feat_ort
        dit_hidden_ort = new_dit_hidden_ort
        base_next_keys_ort = new_base_next_keys_ort
        base_next_values_ort = new_base_next_values_ort
        residual_next_keys_ort = new_residual_next_keys_ort
        residual_next_values_ort = new_residual_next_values_ort

        flag = bool(stop_flag_ort.numpy().reshape(-1)[0])
        if len(pred_seq) > min_len and flag:
            break

    if len(pred_seq) == 0:
        return np.zeros((1, 64, 0), dtype=inference_dtype)
    seq = np.concatenate([s[np.newaxis, ...] for s in pred_seq], axis=1)  # [1, T, 2, 64]
    seq = np.transpose(seq, (0, 3, 1, 2))  # [1, 64, T, 2]
    B, D, T, P = seq.shape
    return seq.reshape(B, D, T * P).astype(inference_dtype)

__all__ = ["run_inference"]