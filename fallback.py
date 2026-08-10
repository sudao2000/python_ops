import torch
import functools
import inspect
import json
from typing import Callable

FALLBACK_ENABLE = False
origin_ops = {}
LOG_ENABLE = True

def to_cpu(obj):
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        return obj.cpu()
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
    if isinstance(obj, set):
        return {to_cpu(v) for v in obj}
    return obj

def fallback_log(msg: str):
    if LOG_ENABLE:
        # Get the caller's stack frame
        stack = inspect.stack()
        # stack[0] is fallback_log, stack[1] is wrapper, stack[2] is the caller of the wrapper
        # However, due to decorators, we might need to search up the stack
        caller_frame = None
        for frame_info in stack:
             # Skip fallback implementation details
             if frame_info.filename != __file__:
                 caller_frame = frame_info
                 break
        
        caller_info = ""
        if caller_frame:
            caller_info = f" (at {caller_frame.filename}:{caller_frame.lineno})"
            
        print(f"Fallback: {msg}{caller_info}")

def fallback(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        has_cuda = False
        origin_device = None

        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.is_cuda:
                has_cuda = True
                origin_device = arg.device
                break
            if isinstance(arg, (list, tuple, set)):
                for i, subarg in enumerate(arg):
                    if isinstance(subarg, torch.Tensor) and subarg.is_cuda:
                        has_cuda = True
                        origin_device = subarg.device
                        break
        if not has_cuda:
            for v in kwargs.values():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    has_cuda = True
                    origin_device = v.device
                    break

        if not has_cuda:
            return func(*args, **kwargs)
        # Prepare cpu args/kwargs; handle 'out' specially so we can copy back
        cpu_args = [to_cpu(arg) for arg in args]
        cpu_kwargs = {}
        out_map = {}
        for k, v in kwargs.items():
            if k == 'out' and isinstance(v, torch.Tensor) and v.is_cuda:
                cpu_out = v.cpu()
                cpu_kwargs[k] = cpu_out
                out_map[k] = (v, cpu_out)
            else:
                cpu_kwargs[k] = to_cpu(v)

        result = func(*cpu_args, **cpu_kwargs)

        # If we used an 'out' CPU tensor, copy its contents back to original CUDA tensor
        if out_map:
            for k, (orig_cuda, cpu_out) in out_map.items():
                try:
                    orig_cuda.copy_(cpu_out)
                except Exception:
                    # best-effort: if copy_ fails, try to move result back instead
                    pass
            # If function returns the cpu_out, return the original CUDA tensor
            if isinstance(result, torch.Tensor):
                # if result is the cpu_out object, return orig_cuda instead
                for (_, cpu_out) in out_map.values():
                    if result is cpu_out:
                        # return first orig_cuda
                        return list(out_map.values())[0][0]
            # For in-place/out style functions, prefer returning the original CUDA out
            return list(out_map.values())[0][0]

        if isinstance(result, torch.Tensor):
            return result.to(origin_device)
        elif isinstance(result, (list, tuple, set)):
            return type(result)(to_cpu(v).to(origin_device) if isinstance(v, torch.Tensor) else v for v in result)
        else:
            return result
    return wrapper

def _safe_json_value(value):
    if isinstance(value, torch.Tensor):
        return {
            "data_type": str(value.dtype),
            "shape": list(value.shape),
            "stride": list(value.stride()),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return [_safe_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    return repr(value)


def _format_arg(arg):
    if isinstance(arg, torch.Tensor):
        return _safe_json_value(arg)
    return {
        "data_type": type(arg).__name__,
        "value": _safe_json_value(arg),
    }

def _format_args(func, args, kwargs):
    formatted = {}

    try:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        arguments = bound_args.arguments
        for name, value in arguments.items():
            formatted[name] = _format_arg(value)
    except Exception:
        for i, a in enumerate(args):
            key = f"arg{i}"
            formatted[key] = _format_arg(a)
        for k, v in kwargs.items():
            formatted[k] = _format_arg(v)

    return json.dumps(formatted, ensure_ascii=False)

def torch_func_wrapper(origin_func, func_name):
    @fallback
    @functools.wraps(origin_func)
    def wrapper(*args, **kwargs):
        args_info = _format_args(origin_func, args, kwargs)
        fallback_log(f"torch.{func_name} fallback to CPU. Args: {args_info}")
        return origin_func(*args, **kwargs)
    return wrapper

def torch_tensor_func_wrapper(origin_func, func_name):
    @fallback
    @functools.wraps(origin_func)
    def wrapper(self, *args, **kwargs):
        args_info = _format_args(origin_func, (self,) + args, kwargs)
        fallback_log(f"Tensor.{func_name} fallback to CPU. Args: {args_info}")
        return origin_func(self, *args, **kwargs)
    return wrapper

def register_fallback_ops():
    ops_dict = {}
    torch_functions = {
        '__and__': True, '__or__': True, '_assert_async': True, '_has_compatible_shallow_copy_type': True, 
        '_local_scalar_dense': True, '_reshape_alias': True, '_safe_softmax': True, 
        '_scaled_dot_product_attention_math': True, '_softmax': True, '_to_copy': True, '_unsafe_view': True, 
        'add': True, 'add_': True, 'alias': True, 'all': True, 'any': True, 'arange': True, 'argmax': True, 'as_strided': True, 
        'bitwise_and': True, 'bitwise_not': True, 'bitwise_or': True, 'bmm': True, 'cat': True, 'clone': True, 
        'contiguous': True, 'copy_': True, 'cos': True, 'cumsum': True, 'detach': True, 'detach_': True, 'div': True, 'div_': True, 
        'embedding': True, 'empty': True, 'empty_like': True, 'empty_strided': True, 'eq': True, 'expand': True, 
        'exponential_': True, 'fill_': True, 'flatten': True, 'full': True, 'ge': True, 'gt': True, 'index': True, 
        'index_select': True, 'is_nonzero': True, 'isin': True, 'isneginf': True, 'item': True, 'le': True, 
        'lift_fresh': True, 'linear': True, 'lt': True, 'masked_fill': True, 'masked_fill_': True, 'matmul': True, 
        'max': True, 'mean': True, 'min': True, 'mm': True, 'mul': True, 'multinomial': True, 'neg': True, 'new_empty': True, 
        'new_ones': True, 'ones': True, 'pow': True, 'reciprocal': True, 'repeat_interleave': True, 'reshape': True, 
        'resize_': True, 'resolve_conj': True, 'resolve_neg': True, 'result_type': True, 'rsqrt': True, 'rsub': True, 
        'scalar_tensor': True, 'scaled_dot_product_attention': True, 'scatter': True, 'select': True, 
        'set_': True, 'silu': True, 'sin': True, 'slice': True, 'softmax': True, 'sort': True, 'squeeze': True, 'sub': True, 
        'sum': True, 't': True, 'topk': True, 'transpose': True, 'tril': True, 'unsqueeze': True, 'view': True, 
        'view_as': True, 'where': True, '__lt__': True, '__gt__': True, '__le__': True, '__ge__': True, '__eq__': True, '__ne__': True,
        '__div__': True,
        'ceil': True,
        '__truediv__': True,
        '__itruediv__': True,
        '__floordiv__': True,
        '__ifloordiv__': True,
        'abs': True,
        'fill': True,
        'sub_': True,
        'ones_': True,
        '__sub__': True,
        '__add__': True,
        '__mul__': True,
        'mul_': True,
        '__getitem__': True,
        '__setitem__': True,
        'rsub__': True,
        '__rsub__': True,
        'gather': True,
        '__neg__': True,
        '__matmul__': True,
        'zeros': True,
        'not': True,
        '__invert__': True,
        'randn': True,
        'scaled_dot_product_attention': True,
    }
    missing_ops = []
    for func_name, enabled in torch_functions.items():
        if not enabled:
            continue
        found = False
        if hasattr(torch, func_name):
            origin_func = getattr(torch, func_name)
            ops_dict[func_name] = torch_func_wrapper(origin_func, func_name)
            found = True
        if hasattr(torch.Tensor, func_name):
            origin_func = getattr(torch.Tensor, func_name)
            ops_dict[f'Tensor.{func_name}'] = torch_tensor_func_wrapper(origin_func, func_name)
            found = True
        
        # Check in specialized submodules if not found in top-level

        special_modules = [
            (torch.nn.functional, 'nn.functional'),
            (torch._C._nn, '_C._nn'), 
            (torch._C, '_C')
        ]
        for module, module_name in special_modules:
            if hasattr(module, func_name):
                origin_func = getattr(module, func_name)
                # We wrap it as torch.{func_name} for fallback logging purposes
                # or keep module_name in log if preferred
                ops_dict[f'{module_name}.{func_name}'] = torch_func_wrapper(origin_func, f'{module_name}.{func_name}')
                found = True
                break

        if not found:
            missing_ops.append(func_name)
    
    if missing_ops:
        print(f"Warning: The following ops were not found in torch or torch.Tensor: {missing_ops}")
        
    return ops_dict

def disable_fallback():
    global FALLBACK_ENABLE, origin_ops, LOG_ENABLE
    if not FALLBACK_ENABLE:
        print("CUDA fallback not enabled")
        return
    for op_name, original_func in origin_ops.items():
        if op_name.startswith('Tensor.'):
            method_name = op_name.split('.')[1]
            setattr(torch.Tensor, method_name, original_func)
        elif op_name.startswith('nn.functional.'):
             method_name = op_name.split('.')[2]
             setattr(torch.nn.functional, method_name, original_func)
        elif op_name.startswith('_C._nn.'):
             method_name = op_name.split('.')[2]
             setattr(torch._C._nn, method_name, original_func)
        elif op_name.startswith('_C.'):
             method_name = op_name.split('.')[1]
             setattr(torch._C, method_name, original_func)
        elif hasattr(torch, op_name):
            setattr(torch, op_name, original_func)
            
    origin_ops.clear()
    FALLBACK_ENABLE = False
    LOG_ENABLE = False
    print("unregister callback ops finish")

def enable_fallback():
    global FALLBACK_ENABLE, origin_ops, LOG_ENABLE
    if FALLBACK_ENABLE:
        print("fallback already enabled")
        return
    custom_ops = register_fallback_ops()
    for op_name, custom_func in custom_ops.items():
        if op_name.startswith('Tensor.'):
            method_name = op_name.split('.')[1]
            origin_ops[op_name] = getattr(torch.Tensor, method_name)
            setattr(torch.Tensor, method_name, custom_func)
        elif op_name.startswith('nn.functional.'):
            method_name = op_name.split('.')[2]
            origin_ops[op_name] = getattr(torch.nn.functional, method_name)
            setattr(torch.nn.functional, method_name, custom_func)
        elif op_name.startswith('_C._nn.'):
            method_name = op_name.split('.')[2]
            origin_ops[op_name] = getattr(torch._C._nn, method_name)
            setattr(torch._C._nn, method_name, custom_func)
        elif op_name.startswith('_C.'):
            method_name = op_name.split('.')[1]
            origin_ops[op_name] = getattr(torch._C, method_name)
            setattr(torch._C, method_name, custom_func)
        elif hasattr(torch, op_name):
            origin_ops[op_name] = getattr(torch, op_name)
            setattr(torch, op_name, custom_func)
    FALLBACK_ENABLE = True
    print(f"fallback enabled for {custom_ops}")