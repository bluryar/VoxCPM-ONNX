import io
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import ffmpeg
import numpy as np
import soundfile as sf
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from onnx_infer.constants import SAMPLE_RATE, MAX_THREADS
from onnx_infer.runtime import (
    create_session_options,
    create_run_options,
    configure_providers,
    create_session,
    get_device_info_from_providers,
)
from onnx_infer.tokenize import load_tokenizer
from onnx_infer.inputs import build_inputs, build_inputs_with_patches
from onnx_infer.infer_loop import run_inference
from onnx_infer.vae import decode_audio, encode_audio_to_patches
from onnx_infer.store import init_db, save_ref_features, load_ref_features


class TTSRequest(BaseModel):
    # OpenAI 兼容字段
    model: str = "voxcpm-onnx"  # 可随意传入，仅作兼容
    input: str  # 对应 text
    voice: str = "default"  # 对应 feat_id，支持 "default" 或预置别名
    response_format: str = "mp3"  # 仅兼容，实际固定 16k wav
    speed: float = 1.0  # 占位，暂不支持变速
    # 内部扩展字段
    feat_id: Optional[str] = None  # 直接指定预编码特征，优先级高于 voice
    prompt_audio_path: Optional[str] = None  # 本地路径备用
    prompt_text: Optional[str] = None
    min_len: int = 2
    max_len: int = 2000
    cfg_value: float = 2.0


app = FastAPI(title="VoxCPM ONNX TTS Service")


class TTSState:
    def __init__(self):
        self.initialized = False
        self.tokenizer = None
        self.prefill_sess = None
        self.decode_sess = None
        self.vae_enc_sess = None
        self.vae_dec_sess = None
        self.providers = None
        self.provider_options = None
        self.session_opts = None
        self.run_opts = None
        self.patch_size = 2
        self.device_type = 'cpu'
        self.device_id = 0
        self.models_dir = None
        self.tokenizer_dir = None
        self.sqlite_path: Optional[str] = None

    def init(self, models_dir: str, tokenizer_dir: str, device: str, device_id: int, optimize: bool, max_threads: int, dtype: str, sqlite_path: str):
        if self.initialized:
            return
        self.models_dir = models_dir
        self.tokenizer_dir = tokenizer_dir
        self.sqlite_path = sqlite_path
        self.tokenizer = load_tokenizer(tokenizer_dir)
        self.providers, self.provider_options = configure_providers(device, max_threads, device_id)
        self.session_opts = create_session_options(max_threads, optimize)
        self.run_opts = create_run_options()
        self.prefill_sess = create_session(os.path.join(models_dir, "voxcpm_prefill.onnx"), self.session_opts, self.providers, self.provider_options)
        self.decode_sess = create_session(os.path.join(models_dir, "voxcpm_decode_step.onnx"), self.session_opts, self.providers, self.provider_options)
        self.vae_enc_sess = create_session(os.path.join(models_dir, "audio_vae_encoder.onnx"), self.session_opts, self.providers, self.provider_options)
        self.vae_dec_sess = create_session(os.path.join(models_dir, "audio_vae_decoder.onnx"), self.session_opts, self.providers, self.provider_options)

        try:
            with open(os.path.join(models_dir, "config.json"), "r") as f:
                cfg = json.load(f)
                self.patch_size = int(cfg.get("patch_size", 2))
        except Exception:
            self.patch_size = 2

        self.device_type, self.device_id = get_device_info_from_providers(self.providers, device_id)
        self.initialized = True


state = TTSState()

DEFAULT_MODELS_DIR = os.environ.get("VOX_MODELS_DIR", "/root/code/VoxCPM/onnx_models")
DEFAULT_TOKENIZER_DIR = os.environ.get("VOX_TOKENIZER_DIR", DEFAULT_MODELS_DIR)
DEFAULT_DEVICE = os.environ.get("VOX_DEVICE", "cuda")
DEFAULT_DEVICE_ID = int(os.environ.get("VOX_DEVICE_ID", "0"))
DEFAULT_OPTIMIZE = os.environ.get("VOX_OPTIMIZE", "1").lower() in ("1", "true", "yes")
DEFAULT_DTYPE = os.environ.get("VOX_DTYPE", "fp32")
DEFAULT_SQLITE_PATH = os.environ.get("VOX_SQLITE_PATH", "/tmp/voxcpm_ref.db")


@app.on_event("startup")
def _startup_init():
    try:
        state.init(
            models_dir=DEFAULT_MODELS_DIR,
            tokenizer_dir=DEFAULT_TOKENIZER_DIR,
            device=DEFAULT_DEVICE,
            device_id=DEFAULT_DEVICE_ID,
            optimize=DEFAULT_OPTIMIZE,
            max_threads=MAX_THREADS,
            dtype=DEFAULT_DTYPE,
            sqlite_path=DEFAULT_SQLITE_PATH,
        )
        app.state.init_error = None
    except Exception as e:
        app.state.init_error = str(e)
        state.initialized = False


@app.get("/health")
def health():
    return {
        "status": "ok" if state.initialized else "error",
        "initialized": state.initialized,
        "models_dir": state.models_dir,
        "tokenizer_dir": state.tokenizer_dir,
        "device_type": state.device_type,
        "device_id": state.device_id,
        "sqlite_path": state.sqlite_path,
        "error": getattr(app.state, "init_error", None),
    }


@app.post("/ref_feat")
async def ref_feat(
    feat_id: str = Form(...),
    prompt_audio: UploadFile = File(...),
    prompt_text: Optional[str] = Form(None),
):
    """上传参考音频并编码存储特征到 sqlite"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail=f"Model not initialized: {getattr(app.state, 'init_error', None)}")

    inference_dtype = {
        "fp32": np.float32,
        "fp16": np.float16,
        "bf16": np.float32,
    }.get(DEFAULT_DTYPE, np.float32)

    # 保存上传文件到临时目录
    output_dir = Path(os.environ.get("VOX_OUTPUT_DIR", "/tmp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_audio = output_dir / f"ref_upload_{int(time.time()*1000)}.wav"
    try:
        with tmp_audio.open("wb") as f:
            shutil.copyfileobj(prompt_audio.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded audio: {e}")

    try:
        # 读取并编码
        audio, sr = sf.read(tmp_audio, always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            from onnx_infer.audio import resample_audio_linear
            audio = resample_audio_linear(audio, sr, SAMPLE_RATE)
        patches = encode_audio_to_patches(
            state.vae_enc_sess,
            audio,
            state.patch_size,
            inference_dtype,
            state.run_opts,
        )
        # 存储到 sqlite
        save_ref_features(
            state.sqlite_path,
            feat_id,
            prompt_text or "",
            state.patch_size,
            DEFAULT_DTYPE,
            patches,
        )
        return {"feat_id": feat_id, "patches_shape": patches.shape}
    finally:
        # 清理临时文件
        tmp_audio.unlink(missing_ok=True)


# ------------------ GET /tts ------------------
async def _tts_core(
    model: str,
    input: str,
    voice: Optional[str],
    response_format: str,
    speed: float,
    prompt_text: Optional[str],
    min_len: int,
    max_len: int,
    cfg_value: float,
    prompt_audio: Optional[UploadFile] = None,
):
    """复用 POST 的核心逻辑，仅 input 必填"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail=f"Model not initialized: {getattr(app.state, 'init_error', None)}")

    inference_dtype = {
        "fp32": np.float32,
        "fp16": np.float16,
        "bf16": np.float32,
    }.get(DEFAULT_DTYPE, np.float32)

    target_text = input
    if voice is not None:
        row = load_ref_features(state.sqlite_path, voice)
        if row is None:
            raise HTTPException(status_code=404, detail=f"voice alias '{voice}' not found")
        patches, patch_size, prompt_text_stored, _ = row
        text_token, text_mask, audio_feat, audio_mask = build_inputs_with_patches(
            state.tokenizer,
            target_text=target_text,
            prompt_text=prompt_text_stored,
            patches=patches,
            patch_size=patch_size,
            inference_dtype=inference_dtype,
        )
    else:
        audio_path = None
        if prompt_audio is not None:
            output_dir = Path(os.environ.get("VOX_OUTPUT_DIR", "/tmp"))
            output_dir.mkdir(parents=True, exist_ok=True)
            tmp_audio = output_dir / f"tts_upload_{int(time.time()*1000)}.wav"
            try:
                with tmp_audio.open("wb") as f:
                    shutil.copyfileobj(prompt_audio.file, f)
                audio_path = str(tmp_audio)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to save uploaded audio: {e}")
        text_token, text_mask, audio_feat, audio_mask = build_inputs(
            state.tokenizer,
            target_text=target_text,
            prompt_text=prompt_text or "",
            prompt_wav_path=audio_path,
            vae_enc_sess=state.vae_enc_sess,
            patch_size=state.patch_size,
            inference_dtype=inference_dtype,
            run_options=state.run_opts,
        )
        if prompt_audio is not None and audio_path:
            Path(audio_path).unlink(missing_ok=True)

    latents = run_inference(
        state.prefill_sess,
        state.decode_sess,
        text_token,
        text_mask,
        audio_feat,
        audio_mask,
        min_len=min_len,
        max_len=max_len,
        cfg_value=cfg_value,
        device_type=state.device_type,
        device_id=state.device_id,
        inference_dtype=inference_dtype,
        run_options=state.run_opts,
    )

    audio = decode_audio(
        state.vae_dec_sess,
        latents,
        device_type=state.device_type,
        device_id=state.device_id,
        inference_dtype=inference_dtype,
        run_options=state.run_opts,
    )

    # 写盘 + 转码
    output_dir = Path(os.environ.get("VOX_OUTPUT_DIR", "/tmp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"voxcpm_{int(time.time()*1000)}.wav"
    print(f"写入 {wav_path}")
    sf.write(wav_path, audio, SAMPLE_RATE, format="WAV")

    fmt = response_format.lower()
    if fmt not in {"mp3", "opus", "aac", "flac", "wav", "pcm"}:
        fmt = "mp3"

    out_path = wav_path
    media_type = "audio/wav"

    try:
        if fmt != "wav":
            converted_path = wav_path.with_suffix(f".{fmt}")
            try:
                stream = ffmpeg.input(str(wav_path))
                if fmt == "pcm":
                    stream = ffmpeg.output(stream, str(converted_path), acodec="pcm_s16le", ac=1, ar=str(SAMPLE_RATE))
                else:
                    acodec_map = {"mp3": "libmp3lame", "opus": "libopus", "aac": "aac", "flac": "flac"}
                    stream = ffmpeg.output(stream, str(converted_path), acodec=acodec_map[fmt])
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                out_path = converted_path
                media_map = {"mp3": "audio/mpeg", "opus": "audio/opus", "aac": "audio/aac", "flac": "audio/flac", "pcm": "audio/pcm"}
                media_type = media_map.get(fmt, "audio/mpeg")
            except ffmpeg.Error as e:
                raise HTTPException(status_code=500, detail=f"ffmpeg transcoding failed: {e.stderr.decode()}")

        content = out_path.read_bytes()
        return Response(content=content, media_type=media_type)

    finally:
        keep_audio_files = os.environ.get("VOX_KEEP_AUDIO_FILES", "false").lower() in ("true", "1", "yes", "on")
        if not keep_audio_files:
            wav_path.unlink(missing_ok=True)
            if out_path != wav_path and out_path.exists():
                out_path.unlink(missing_ok=True)
        else:
            print(f"Keeping audio file: {wav_path}")
            if out_path != wav_path and out_path.exists():
                print(f"Keeping audio file: {out_path}")

@app.post("/tts")
async def tts(
    model: str = Form("voxcpm-onnx"),
    input: str = Form(...),  # 对应 text
    voice: Optional[str] = Form(None),  # 对应 feat_id
    response_format: str = Form("mp3"),  # 兼容字段，实际固定 wav
    speed: float = Form(1.0),  # 占位
    prompt_text: Optional[str] = Form(None),
    min_len: int = Form(2),
    max_len: int = Form(2000),
    cfg_value: float = Form(2.0),
    prompt_audio: Optional[UploadFile] = None,  # 纯可选，不传即可省略
):
    """TTS 生成，支持 feat_id、上传音频或音频路径三选一；优先级：feat_id > 上传文件 > 路径"""
    return await _tts_core(
        model=model,
        input=input,
        voice=voice,
        response_format=response_format,
        speed=speed,
        prompt_text=prompt_text,
        min_len=min_len,
        max_len=max_len,
        cfg_value=cfg_value,
        prompt_audio=prompt_audio,
    )

@app.get("/tts")
async def tts_get(
    input: str,
    model: str = "voxcpm-onnx",
    voice: Optional[str] = None,
    response_format: str = "mp3",
    speed: float = 1.0,
    prompt_text: Optional[str] = None,
    min_len: int = 2,
    max_len: int = 2000,
    cfg_value: float = 2.0,
):
    """GET 版本 /tts，仅 input 必填，其余可选"""
    return await _tts_core(
        model=model,
        input=input,
        voice=voice,
        response_format=response_format,
        speed=speed,
        prompt_text=prompt_text,
        min_len=min_len,
        max_len=max_len,
        cfg_value=cfg_value,
        prompt_audio=None,
    )