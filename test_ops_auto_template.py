"""Template for generating auto fallback operator tests.

This file is used by generate_test_ops_auto.py.
"""

HEADER = '''import json
import unittest
import torch
import sys

# import fallback
# fallback.enable_fallback() # 引入fallback机制
DEBUG = True

# Parse command-line arguments
if len(sys.argv) > 1:
    param = sys.argv[1]
    if '=' in param:
        key, value = param.split('=')
        if key == 'm':
            mm = int(value)
            print(f"parameter m is: {mm}")
        else:
            print(f"Unknown key: {key}")
            mm = 13
    else:
        print("Parameter format should be key=value")
        mm = 13
else:
    mm = 13

class TestOpsAuto(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")

    def _resolve_callable(self, func_name: str):
        if func_name.startswith("Tensor."):
            method_name = func_name.split(".", 1)[1]
            return getattr(torch.Tensor, method_name, None)

        if func_name.startswith("torch."):
            path = func_name.split(".")[1:]
            obj = torch
            for p in path:
                obj = getattr(obj, p, None)
                if obj is None:
                    return None
            return obj

        return None

    def _to_dtype(self, value):
        if not isinstance(value, str):
            return None
        if value.startswith("torch."):
            name = value.split(".", 1)[1]
            return getattr(torch, name, None)
        return None

    def _materialize_value(self, value, float_tensor_dtype=None):
        if isinstance(value, dict) and "data_type" in value:
            return self._materialize_arg(value, float_tensor_dtype=float_tensor_dtype)
        if isinstance(value, list):
            return [self._materialize_value(v, float_tensor_dtype=float_tensor_dtype) for v in value]
        if isinstance(value, tuple):
            return tuple(self._materialize_value(v, float_tensor_dtype=float_tensor_dtype) for v in value)
        if isinstance(value, dict):
            return {k: self._materialize_value(v, float_tensor_dtype=float_tensor_dtype) for k, v in value.items()}
        return value

    def _has_float_tensor_spec(self, value):
        if isinstance(value, dict):
            data_type = value.get("data_type")
            if (
                isinstance(data_type, str)
                and data_type.startswith("torch.")
                and "shape" in value
                and data_type in ("torch.bfloat16")
            ):
                return True
            return any(self._has_float_tensor_spec(v) for v in value.values())
        if isinstance(value, list):
            return any(self._has_float_tensor_spec(v) for v in value)
        if isinstance(value, tuple):
            return any(self._has_float_tensor_spec(v) for v in value)
        return False

    def _iter_test_dtypes(self, args_info):
        if self._has_float_tensor_spec(args_info):
            return [torch.float32, torch.bfloat16, torch.float16]
        return [None]

    def _materialize_arg(self, spec, float_tensor_dtype=None):
        if not isinstance(spec, dict) or "data_type" not in spec:
            return self._materialize_value(spec, float_tensor_dtype=float_tensor_dtype)

        data_type = spec.get("data_type")
        # 添加对 ellipsis 类型的处理
        if data_type == "ellipsis":
            return Ellipsis
        
        if isinstance(data_type, str) and data_type.startswith("torch.") and "shape" in spec:
            dtype = self._to_dtype(data_type) or torch.float32
            if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16) and float_tensor_dtype is not None:
                dtype = float_tensor_dtype
            shape = tuple(spec.get("shape", []))
            if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
                if dtype == torch.bool:
                    return torch.randint(0, 2, shape).bool().to(self.device) if len(shape) > 0 else torch.tensor(True).to(self.device)
                return torch.randint(0, 10, shape, dtype=dtype).to(self.device) if len(shape) > 0 else torch.tensor(1, dtype=dtype).to(self.device)
            return torch.rand(*shape, dtype=dtype).to(self.device) if len(shape) > 0 else torch.tensor(1.0, dtype=dtype).to(self.device)

        value = spec.get("value")
        if data_type == "dtype":
            return self._to_dtype(value) or torch.float32
        if data_type == "device":
            return self.device
        if data_type == "memory_format":
            # 处理 memory_format 类型
            if isinstance(value, str) and value.startswith("torch."):
                format_name = value.split(".", 1)[1]
                return getattr(torch, format_name, None)
            return value
        if data_type == "tuple":
            # 处理元组中的特殊字符串
            if isinstance(value, list):
                tuple_values = []
                for v in value:
                    if isinstance(v, str):
                        if v == "Ellipsis":
                            tuple_values.append(Ellipsis)
                        elif v.startswith("slice(") and v.endswith(")"):
                            # 解析 slice 字符串，例如 "slice(-1, None, None)"
                            slice_args = v[6:-1].split(", ")
                            slice_args = [None if arg == "None" else int(arg) for arg in slice_args]
                            tuple_values.append(slice(*slice_args))
                        else:
                            tuple_values.append(v)
                    else:
                        tuple_values.append(self._materialize_value(v, float_tensor_dtype=float_tensor_dtype))
                return tuple(tuple_values)
            return tuple(self._materialize_value(value, float_tensor_dtype=float_tensor_dtype))
        if data_type == "list":
            return [self._materialize_value(v, float_tensor_dtype=float_tensor_dtype) for v in value]
        return self._materialize_value(value, float_tensor_dtype=float_tensor_dtype)

    def _build_call_args(self, func_name, args_info, float_tensor_dtype=None):
        positional = []
        keyword = {}

        for k, v in args_info.items():
            if k == "self":
                # 将self作为第一个位置参数
                positional.append((0, self._materialize_arg(v, float_tensor_dtype=float_tensor_dtype)))
            elif k.startswith("arg") and k[3:].isdigit():
                positional.append((int(k[3:]), self._materialize_arg(v, float_tensor_dtype=float_tensor_dtype)))
            else:
                keyword[k] = self._materialize_arg(v, float_tensor_dtype=float_tensor_dtype)

        positional = [v for _, v in sorted(positional, key=lambda x: x[0])]
        return positional, keyword

    def _invoke(self, func_name, fn, args, kwargs):
        if func_name == "Tensor.__getitem__":
            if not args:
                raise ValueError("Tensor method missing self argument")
            self_tensor, *rest = args
            return self_tensor[...] if len(rest) > 0 else self_tensor
        if func_name == "Tensor.__setitem__":
            if not args:
                raise ValueError("Tensor method missing self argument")
            self_tensor = args[0]
            key = args[1] if len(args) > 1 else kwargs.get("key")
            value = args[2] if len(args) > 2 else kwargs.get("value")
            self_tensor[key] = value
            return self_tensor  # __setitem__ 返回 None，但我们需要返回一个值以供测试
        if func_name.startswith("Tensor."):
            if not args:
                raise ValueError("Tensor method missing self argument")
            self_tensor, *rest = args
            method_name = func_name.split(".", 1)[1]
            method = getattr(self_tensor, method_name)
            return method(*rest, **kwargs)
        return fn(*args, **kwargs)

'''

TEST_METHOD_TEMPLATE = '''    def test_auto_{test_name}(self):
        func_name = {func_name!r}
        args_info = {args_info}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {{func_name}}")
        print(f"Testing {{func_name}} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{{float_tensor_dtype}}: {{exc}}")

        if errors:
            self.fail(f"auto case failed for {{func_name}} -> " + " | ".join(errors))

'''

FOOTER = '''
if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
'''
