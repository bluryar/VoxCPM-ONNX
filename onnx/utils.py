import onnx
import onnxruntime as ort
import torch
import numpy as np
from typing import Optional, Union, List, Dict, Any, Tuple
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)


# https://github.com/onnx/onnx/pull/6556
MAXIMUM_PROTOBUF = 2147483648  # 2GiB


def strict_check_model(model_or_path: Union[onnx.ModelProto, str, Path]):
    try:
        onnx.checker.check_model(model_or_path, full_check=True)
    except Exception as e:
        if "No Op registered for" in str(e):
            pass
        else:
            raise e


def check_and_save_model(model: onnx.ModelProto, save_path: Optional[Union[str, Path]]):
    # for large models, a path must be provided instead of a ModelProto:
    # https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#checking-a-large-onnx-model-2gb
    if model.ByteSize() < MAXIMUM_PROTOBUF:
        # For the try catch, refer to https://github.com/microsoft/onnxruntime/issues/14768
        strict_check_model(model)
        if save_path:
            # Overwrite.
            save_path = Path(save_path).as_posix()
            external_file_name = os.path.basename(save_path) + "_data"
            # path/to/model.onnx_data
            external_path = os.path.join(os.path.dirname(save_path), external_file_name)

            if save_path.endswith(".onnx") and os.path.isfile(save_path):
                os.remove(save_path)
            if os.path.isfile(external_path):
                # The new model may be below the maximum protobuf size, overwritting a model that was larger. Hence this os.remove.
                os.remove(external_path)

            onnx.save(
                model,
                save_path,
                convert_attribute=True,
            )
    elif save_path is not None:
        # path/to/model.onnx
        save_path = Path(save_path).as_posix()

        external_file_name = os.path.basename(save_path) + "_data"
        # path/to/model.onnx_data
        external_path = os.path.join(os.path.dirname(save_path), external_file_name)

        if save_path.endswith(".onnx") and os.path.isfile(save_path):
            os.remove(save_path)
        if os.path.isfile(external_path):
            os.remove(external_path)

        onnx.save(
            model,
            save_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_file_name,
            convert_attribute=True,
        )

    else:
        logger.info(
            "Merged ONNX model exceeds 2GB, the model will not be checked without `save_path` given."
        )


def compare_torch_onnx_outputs(
    torch_outputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
    onnx_outputs: List[np.ndarray],
    output_names: List[str],
    rtol: float = 1e-3,
    atol: float = 1e-4,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    比较PyTorch模型和ONNX模型的输出，计算容差。
    
    Args:
        torch_outputs: PyTorch模型的输出，可以是单个tensor、tuple或list
        onnx_outputs: ONNX模型的输出，list of numpy arrays
        output_names: 输出名称列表
        rtol: 相对容差
        atol: 绝对容差
        verbose: 是否打印详细信息
    
    Returns:
        包含每个输出比较结果的字典
    """
    # 将torch_outputs转换为list格式
    if isinstance(torch_outputs, torch.Tensor):
        torch_outputs = [torch_outputs]
    elif isinstance(torch_outputs, tuple):
        torch_outputs = list(torch_outputs)
    
    results = {}
    
    if len(torch_outputs) != len(onnx_outputs):
        logger.error(f"输出数量不匹配: PyTorch={len(torch_outputs)}, ONNX={len(onnx_outputs)}")
        return results
    
    for i, (torch_out, onnx_out, name) in enumerate(zip(torch_outputs, onnx_outputs, output_names)):
        # 转换torch tensor到numpy
        torch_np = torch_out.detach().cpu().numpy()
        
        # 检查形状
        shape_match = torch_np.shape == onnx_out.shape
        
        # 计算差异
        if shape_match:
            # 检查数据类型，对布尔类型特殊处理
            if torch_np.dtype == bool or onnx_out.dtype == bool:
                # 对于布尔类型，使用XOR来计算差异
                abs_diff = np.logical_xor(torch_np, onnx_out).astype(np.float32)
                max_abs_diff = np.max(abs_diff)
                mean_abs_diff = np.mean(abs_diff)
                
                # 布尔类型的相对差异没有意义，设为与绝对差异相同
                max_rel_diff = max_abs_diff
                mean_rel_diff = mean_abs_diff
                
                # 检查是否完全匹配（布尔类型应该完全相等）
                is_close = np.array_equal(torch_np, onnx_out)
            else:
                # 数值类型的正常处理
                abs_diff = np.abs(torch_np - onnx_out)
                max_abs_diff = np.max(abs_diff)
                mean_abs_diff = np.mean(abs_diff)
                
                # 相对差异（避免除零）
                rel_diff = abs_diff / (np.abs(torch_np) + 1e-8)
                max_rel_diff = np.max(rel_diff)
                mean_rel_diff = np.mean(rel_diff)
                
                # 检查是否在容差范围内
                is_close = np.allclose(torch_np, onnx_out, rtol=rtol, atol=atol)
            
            results[name] = {
                'shape_match': shape_match,
                'torch_shape': torch_np.shape,
                'onnx_shape': onnx_out.shape,
                'max_abs_diff': float(max_abs_diff),
                'mean_abs_diff': float(mean_abs_diff),
                'max_rel_diff': float(max_rel_diff),
                'mean_rel_diff': float(mean_rel_diff),
                'is_close': is_close,
                'rtol': rtol,
                'atol': atol
            }
        else:
            results[name] = {
                'shape_match': shape_match,
                'torch_shape': torch_np.shape,
                'onnx_shape': onnx_out.shape,
                'is_close': False,
                'rtol': rtol,
                'atol': atol
            }
        
        if verbose:
            logger.info(f"输出 '{name}' 比较结果:")
            logger.info(f"  形状匹配: {results[name]['shape_match']}")
            logger.info(f"  PyTorch形状: {results[name]['torch_shape']}")
            logger.info(f"  ONNX形状: {results[name]['onnx_shape']}")
            if shape_match:
                logger.info(f"  最大绝对差异: {results[name]['max_abs_diff']:.2e}")
                logger.info(f"  平均绝对差异: {results[name]['mean_abs_diff']:.2e}")
                logger.info(f"  最大相对差异: {results[name]['max_rel_diff']:.2e}")
                logger.info(f"  平均相对差异: {results[name]['mean_rel_diff']:.2e}")
                logger.info(f"  在容差范围内: {results[name]['is_close']}")
            logger.info("")
    
    return results


def validate_onnx_model_with_torch(
    torch_model: torch.nn.Module,
    onnx_path: str,
    test_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
    input_names: List[str],
    output_names: List[str],
    rtol: float = 1e-3,
    atol: float = 1e-4,
    num_tests: int = 3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    使用多组测试输入验证ONNX模型与PyTorch模型的一致性。
    
    Args:
        torch_model: PyTorch模型
        onnx_path: ONNX模型路径
        test_inputs: 测试输入，可以是单个tensor、tuple或list
        input_names: 输入名称列表
        output_names: 输出名称列表
        rtol: 相对容差
        atol: 绝对容差
        num_tests: 测试次数
        verbose: 是否打印详细信息
    
    Returns:
        包含所有测试结果的字典
    """
    logger.info(f"开始验证ONNX模型: {onnx_path}")
    logger.info(f"将进行 {num_tests} 次测试")
    
    # 创建ONNX Runtime会话
    try:
        ort_session = ort.InferenceSession(onnx_path)
    except Exception as e:
        logger.error(f"无法加载ONNX模型: {e}")
        return {'error': str(e)}
    
    torch_model.eval()
    
    all_results = []
    
    for test_idx in range(num_tests):
        logger.info(f"执行第 {test_idx + 1} 次测试...")
        
        # 准备输入
        if isinstance(test_inputs, torch.Tensor):
            current_inputs = [test_inputs]
        elif isinstance(test_inputs, (tuple, list)):
            current_inputs = list(test_inputs)
        else:
            logger.error("不支持的输入类型")
            continue
        
        # 如果需要多次测试，为每次测试生成不同的随机输入
        if num_tests > 1 and test_idx > 0:
            # 基于原始输入的形状生成新的随机输入
            current_inputs = []
            for inp in (test_inputs if isinstance(test_inputs, (tuple, list)) else [test_inputs]):
                if inp.dtype == torch.float32:
                    new_inp = torch.randn_like(inp)
                elif inp.dtype == torch.int64:
                    new_inp = torch.randint_like(inp, low=0, high=10000)
                elif inp.dtype == torch.int32:
                    new_inp = torch.ones_like(inp)
                else:
                    new_inp = inp.clone()
                current_inputs.append(new_inp)
        
        try:
            # PyTorch推理
            with torch.no_grad():
                if len(current_inputs) == 1:
                    torch_outputs = torch_model(current_inputs[0])
                else:
                    torch_outputs = torch_model(*current_inputs)
            
            # ONNX推理
            onnx_inputs = {}
            for name, inp in zip(input_names, current_inputs):
                onnx_inputs[name] = inp.detach().cpu().numpy()
            
            onnx_outputs = ort_session.run(output_names, onnx_inputs)
            
            # 比较输出
            test_results = compare_torch_onnx_outputs(
                torch_outputs, onnx_outputs, output_names, rtol, atol, verbose
            )
            
            test_results['test_index'] = test_idx
            all_results.append(test_results)
            
        except Exception as e:
            logger.error(f"第 {test_idx + 1} 次测试失败: {e}")
            all_results.append({'test_index': test_idx, 'error': str(e)})
    
    # 汇总结果
    summary = {
        'total_tests': num_tests,
        'successful_tests': len([r for r in all_results if 'error' not in r]),
        'failed_tests': len([r for r in all_results if 'error' in r]),
        'all_results': all_results
    }
    
    # 计算总体统计
    if summary['successful_tests'] > 0:
        all_close_counts = {}
        for output_name in output_names:
            all_close_counts[output_name] = sum(
                1 for r in all_results 
                if 'error' not in r and output_name in r and r[output_name]['is_close']
            )
        
        summary['output_accuracy'] = {
            name: f"{count}/{summary['successful_tests']}" 
            for name, count in all_close_counts.items()
        }
    
    if verbose:
        logger.info("=" * 50)
        logger.info("验证结果汇总:")
        logger.info(f"总测试次数: {summary['total_tests']}")
        logger.info(f"成功测试次数: {summary['successful_tests']}")
        logger.info(f"失败测试次数: {summary['failed_tests']}")
        if 'output_accuracy' in summary:
            logger.info("各输出准确率:")
            for name, accuracy in summary['output_accuracy'].items():
                logger.info(f"  {name}: {accuracy}")
        logger.info("=" * 50)
    
    return summary
