from typing import Any, Dict, List, Tuple

import onnxruntime as ort


def create_session_options(max_threads: int, optimize: bool) -> ort.SessionOptions:
    opts = ort.SessionOptions()

    def add_opt(k, v):
        try:
            opts.add_session_config_entry(k, str(v))
        except Exception:
            pass

    if optimize:
        opts.inter_op_num_threads = int(max_threads)
        opts.intra_op_num_threads = int(max_threads)
        opts.log_severity_level = 4
        opts.log_verbosity_level = 4
        opts.enable_cpu_mem_arena = True
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        add_opt("session.set_denormal_as_zero", "1")
        add_opt("session.intra_op.allow_spinning", "1")
        add_opt("session.inter_op.allow_spinning", "1")
        add_opt("session.enable_quant_qdq_cleanup", "1")
        add_opt("session.qdq_matmulnbits_accuracy_level", "4")
        add_opt("optimization.enable_gelu_approximation", "1")
        add_opt("session.disable_synchronize_execution_providers", "1")
        add_opt("optimization.minimal_build_optimizations", "")
        add_opt("session.use_device_allocator_for_initializers", "1")
        add_opt("optimization.enable_cast_chain_elimination", "1")
        add_opt("session.graph_optimizations_loop_level", "2")
    return opts


def create_run_options() -> ort.RunOptions:
    run_opts = ort.RunOptions()
    try:
        run_opts.log_severity_level = 4
        run_opts.log_verbosity_level = 4
        run_opts.add_run_config_entry('disable_synchronize_execution_providers', '1')
    except Exception:
        pass
    return run_opts


def configure_providers(device: str, max_threads: int, device_id: int = 0) -> Tuple[List[str], List[Dict[str, Any]]]:
    avail = ort.get_available_providers()
    providers: List[str] = []
    provider_options: List[Dict[str, Any]] = []

    dev_lower = (device or '').lower()

    # TensorRT
    if dev_lower in ('tensorrt', 'trt') and 'TensorrtExecutionProvider' in avail:
        import os
        trt_engine_cache = os.path.join(os.getcwd(), 'trt_engine_cache')
        trt_timing_cache = os.path.join(os.getcwd(), 'trt_timing_cache')
        try:
            os.makedirs(trt_engine_cache, exist_ok=True)
            os.makedirs(trt_timing_cache, exist_ok=True)
        except Exception:
            pass

        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        provider_options = [
            {
                'device_id': device_id,
                'trt_fp16_enable': True,
                'trt_int8_enable': False,
                'trt_layer_norm_fp32_fallback': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': trt_engine_cache,
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': trt_timing_cache,
                'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,
                'trt_force_sequential_engine_build': True,
                'trt_context_memory_sharing_enable': True,
                'trt_sparsity_enable': True,
                'trt_build_heuristics_enable': True,
                'trt_cuda_graph_enable': True,
                'trt_dla_enable': False,
                'trt_dla_core': 0,
            },
            {
                'device_id': device_id,
                'gpu_mem_limit': 24 * 1024 * 1024 * 1024,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'sdpa_kernel': '2',
                'use_tf32': '1',
                'fuse_conv_bias': '0',
                'cudnn_conv_use_max_workspace': '1',
                'cudnn_conv1d_pad_to_nc1d': '1',
                'tunable_op_enable': '0',
                'tunable_op_tuning_enable': '0',
                'tunable_op_max_tuning_duration_ms': 10,
                'do_copy_in_default_stream': '0',
                'enable_cuda_graph': '0',
                'prefer_nhwc': '0',
                'enable_skip_layer_norm_strict_mode': '0',
                'use_ep_level_unified_stream': '0',
            },
            {}
        ]

    # OpenVINO
    elif dev_lower == 'openvino' and 'OpenVINOExecutionProvider' in avail:
        providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        provider_options = [
            {
                'device_type': 'CPU',
                'precision': 'ACCURACY',
                'num_of_threads': (max_threads if max_threads != 0 else 8),
                'num_streams': 1,
                'enable_opencl_throttling': False,
                'enable_qdq_optimizer': False,
                'disable_dynamic_shapes': False,
            },
            {}
        ]

    # CUDA
    elif dev_lower == 'cuda' and 'CUDAExecutionProvider' in avail:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        provider_options = [
            {
                'device_id': device_id,
                'gpu_mem_limit': 24 * 1024 * 1024 * 1024,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'sdpa_kernel': '2',
                'use_tf32': '1',
                'fuse_conv_bias': '0',
                'cudnn_conv_use_max_workspace': '1',
                'cudnn_conv1d_pad_to_nc1d': '1',
                'tunable_op_enable': '0',
                'tunable_op_tuning_enable': '0',
                'tunable_op_max_tuning_duration_ms': 10,
                'do_copy_in_default_stream': '0',
                'enable_cuda_graph': '0',
                'prefer_nhwc': '0',
                'enable_skip_layer_norm_strict_mode': '1',
                'use_ep_level_unified_stream': '1',
            },
            {}
        ]

    # DML
    elif dev_lower == 'dml' and 'DmlExecutionProvider' in avail:
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        provider_options = [
            {
                'device_id': device_id,
                'performance_preference': 'high_performance',
                'device_filter': 'npu',
            },
            {}
        ]

    else:
        providers = ['CPUExecutionProvider']
        provider_options = [{}]

    return providers, provider_options


def create_session(model_path: str,
                   session_opts: ort.SessionOptions,
                   providers: List[str],
                   provider_options: List[Dict[str, Any]]) -> ort.InferenceSession:
    def model_has_trt_incompatible_ops(path: str) -> bool:
        try:
            import onnx
            m = onnx.load(path)
            for n in m.graph.node:
                if n.op_type == 'MatMulNBits':
                    return True
            return False
        except Exception:
            return False

    trt_requested = any(p == 'TensorrtExecutionProvider' for p in providers)
    if trt_requested and model_has_trt_incompatible_ops(model_path):
        fallback_providers: List[str] = []
        fallback_options: List[Dict[str, Any]] = []
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            fallback_providers.append('CUDAExecutionProvider')
            cuda_opt = None
            for i, p in enumerate(providers):
                if p == 'CUDAExecutionProvider':
                    cuda_opt = provider_options[i] if i < len(provider_options) else None
                    break
            fallback_options.append(cuda_opt or {'device_id': 0})
        fallback_providers.append('CPUExecutionProvider')
        fallback_options.append({})
        try:
            return ort.InferenceSession(model_path, sess_options=session_opts, providers=fallback_providers, provider_options=fallback_options)
        except Exception:
            return ort.InferenceSession(model_path, sess_options=session_opts, providers=['CPUExecutionProvider'])

    try:
        return ort.InferenceSession(model_path, sess_options=session_opts, providers=providers, provider_options=provider_options)
    except Exception:
        fallback_providers: List[str] = []
        fallback_options: List[Dict[str, Any]] = []
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            fallback_providers.append('CUDAExecutionProvider')
            cuda_opt = None
            for i, p in enumerate(providers):
                if p == 'CUDAExecutionProvider':
                    cuda_opt = provider_options[i] if i < len(provider_options) else None
                    break
            fallback_options.append(cuda_opt or {'device_id': 0})
        fallback_providers.append('CPUExecutionProvider')
        fallback_options.append({})
        try:
            return ort.InferenceSession(model_path, sess_options=session_opts, providers=fallback_providers, provider_options=fallback_options)
        except Exception:
            try:
                return ort.InferenceSession(model_path, sess_options=session_opts, providers=['CPUExecutionProvider'])
            except Exception:
                return ort.InferenceSession(model_path, sess_options=session_opts, providers=providers)


def get_device_info_from_providers(providers: List[str], device_id: int = 0) -> Tuple[str, int]:
    if not providers:
        return 'cpu', 0

    primary_provider = providers[0].lower()

    if 'cuda' in primary_provider or 'tensorrt' in primary_provider:
        return 'cuda', device_id
    elif 'dml' in primary_provider:
        return 'dml', device_id
    elif 'openvino' in primary_provider:
        return 'cpu', 0
    else:
        return 'cpu', 0

__all__ = [
    "create_session_options",
    "create_run_options",
    "configure_providers",
    "create_session",
    "get_device_info_from_providers",
]