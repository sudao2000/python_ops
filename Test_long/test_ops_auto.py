import json
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


    # Source: ops.log line 3
    def test_auto_001_torch_empty(self):
        func_name = 'torch.empty'
        args_info = {'arg0': {'data_type': 'tuple', 'value': [152064, 5120]}, 'device': {'data_type': 'NoneType', 'value': None}, 'dtype': {'data_type': 'NoneType', 'value': None}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 4
    def test_auto_002_torch_empty(self):
        func_name = 'torch.empty'
        args_info = {'arg0': {'data_type': 'tuple', 'value': [5120, 5120]}, 'device': {'data_type': 'NoneType', 'value': None}, 'dtype': {'data_type': 'NoneType', 'value': None}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 5
    def test_auto_003_torch_empty(self):
        func_name = 'torch.empty'
        args_info = {'arg0': {'data_type': 'int', 'value': 5120}, 'device': {'data_type': 'NoneType', 'value': None}, 'dtype': {'data_type': 'NoneType', 'value': None}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 6
    def test_auto_004_torch_empty(self):
        func_name = 'torch.empty'
        args_info = {'arg0': {'data_type': 'tuple', 'value': [1024, 5120]}, 'device': {'data_type': 'NoneType', 'value': None}, 'dtype': {'data_type': 'NoneType', 'value': None}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 7
    def test_auto_005_torch_empty(self):
        func_name = 'torch.empty'
        args_info = {'arg0': {'data_type': 'int', 'value': 1024}, 'device': {'data_type': 'NoneType', 'value': None}, 'dtype': {'data_type': 'NoneType', 'value': None}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 11
    def test_auto_006_torch_empty(self):
        func_name = 'torch.empty'
        args_info = {'arg0': {'data_type': 'tuple', 'value': [13824, 5120]}, 'device': {'data_type': 'NoneType', 'value': None}, 'dtype': {'data_type': 'NoneType', 'value': None}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 13
    def test_auto_007_torch_empty(self):
        func_name = 'torch.empty'
        args_info = {'arg0': {'data_type': 'tuple', 'value': [5120, 13824]}, 'device': {'data_type': 'NoneType', 'value': None}, 'dtype': {'data_type': 'NoneType', 'value': None}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 14
    def test_auto_008_torch_ones(self):
        func_name = 'torch.ones'
        args_info = {'arg0': {'data_type': 'int', 'value': 5120}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 17
    def test_auto_009_torch_arange(self):
        func_name = 'torch.arange'
        args_info = {'arg0': {'data_type': 'int', 'value': 0}, 'arg1': {'data_type': 'int', 'value': 128}, 'arg2': {'data_type': 'int', 'value': 2}, 'dtype': {'data_type': 'dtype', 'value': 'torch.int64'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 18
    def test_auto_010_tensor_truediv(self):
        func_name = 'Tensor.__truediv__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [64], 'stride': [1]}, 'arg1': {'data_type': 'int', 'value': 128}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 19
    def test_auto_011_tensor_div(self):
        func_name = 'Tensor.div'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [64], 'stride': [1]}, 'arg1': {'data_type': 'int', 'value': 128}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 20
    def test_auto_012_torch_pow(self):
        func_name = 'torch.pow'
        args_info = {'arg0': {'data_type': 'float', 'value': 1000000.0}, 'arg1': {'data_type': 'torch.float32', 'shape': [64], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 21
    def test_auto_013_tensor_reciprocal(self):
        func_name = 'Tensor.reciprocal'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [64], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 22
    def test_auto_014_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [64], 'stride': [1]}, 'arg1': {'data_type': 'float', 'value': 1.0}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 23
    def test_auto_015_tensor_clone(self):
        func_name = 'Tensor.clone'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [64], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 26
    def test_auto_016_tensor_detach(self):
        func_name = 'Tensor.detach'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [152064, 5120], 'stride': [5120, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 27
    def test_auto_017_tensor_detach(self):
        func_name = 'Tensor.detach'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [5120, 5120], 'stride': [5120, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 28
    def test_auto_018_tensor_detach(self):
        func_name = 'Tensor.detach'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [5120], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 29
    def test_auto_019_tensor_detach(self):
        func_name = 'Tensor.detach'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1024, 5120], 'stride': [5120, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 30
    def test_auto_020_tensor_detach(self):
        func_name = 'Tensor.detach'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1024], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 34
    def test_auto_021_tensor_detach(self):
        func_name = 'Tensor.detach'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [13824, 5120], 'stride': [5120, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 36
    def test_auto_022_tensor_detach(self):
        func_name = 'Tensor.detach'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [5120, 13824], 'stride': [13824, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 56
    def test_auto_023_tensor_view(self):
        func_name = 'Tensor.view'
        args_info = {'arg0': {'data_type': 'torch.uint8', 'shape': [1557135360], 'stride': [1]}, 'dtype': {'data_type': 'dtype', 'value': 'torch.bfloat16'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 57
    def test_auto_024_tensor_reshape(self):
        func_name = 'Tensor.reshape'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [778567680], 'stride': [1]}, 'arg1': {'data_type': 'list', 'value': [152064, 5120]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 59
    def test_auto_025_tensor_view(self):
        func_name = 'Tensor.view'
        args_info = {'arg0': {'data_type': 'torch.uint8', 'shape': [141557760], 'stride': [1]}, 'dtype': {'data_type': 'dtype', 'value': 'torch.bfloat16'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 60
    def test_auto_026_tensor_view(self):
        func_name = 'Tensor.view'
        args_info = {'arg0': {'data_type': 'torch.uint8', 'shape': [10240], 'stride': [1]}, 'dtype': {'data_type': 'dtype', 'value': 'torch.bfloat16'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 62
    def test_auto_027_tensor_reshape(self):
        func_name = 'Tensor.reshape'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [70778880], 'stride': [1]}, 'arg1': {'data_type': 'list', 'value': [5120, 13824]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 63
    def test_auto_028_tensor_reshape(self):
        func_name = 'Tensor.reshape'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [5120], 'stride': [1]}, 'arg1': {'data_type': 'list', 'value': [5120]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 64
    def test_auto_029_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [152064, 5120], 'stride': [5120, 1]}, 'key': {'data_type': 'ellipsis', 'value': 'Ellipsis'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 65
    def test_auto_030_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [5120], 'stride': [1]}, 'key': {'data_type': 'ellipsis', 'value': 'Ellipsis'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 66
    def test_auto_031_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [5120, 13824], 'stride': [13824, 1]}, 'key': {'data_type': 'ellipsis', 'value': 'Ellipsis'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 73
    def test_auto_032_tensor_reshape(self):
        func_name = 'Tensor.reshape'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [70778880], 'stride': [1]}, 'arg1': {'data_type': 'list', 'value': [13824, 5120]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 76
    def test_auto_033_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [13824, 5120], 'stride': [5120, 1]}, 'key': {'data_type': 'ellipsis', 'value': 'Ellipsis'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 77
    def test_auto_034_tensor_view(self):
        func_name = 'Tensor.view'
        args_info = {'arg0': {'data_type': 'torch.uint8', 'shape': [10485760], 'stride': [1]}, 'dtype': {'data_type': 'dtype', 'value': 'torch.bfloat16'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 79
    def test_auto_035_tensor_view(self):
        func_name = 'Tensor.view'
        args_info = {'arg0': {'data_type': 'torch.uint8', 'shape': [52428800], 'stride': [1]}, 'dtype': {'data_type': 'dtype', 'value': 'torch.bfloat16'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 82
    def test_auto_036_tensor_reshape(self):
        func_name = 'Tensor.reshape'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1024], 'stride': [1]}, 'arg1': {'data_type': 'list', 'value': [1024]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 83
    def test_auto_037_tensor_reshape(self):
        func_name = 'Tensor.reshape'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [5242880], 'stride': [1]}, 'arg1': {'data_type': 'list', 'value': [1024, 5120]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 84
    def test_auto_038_tensor_reshape(self):
        func_name = 'Tensor.reshape'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [26214400], 'stride': [1]}, 'arg1': {'data_type': 'list', 'value': [5120, 5120]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 85
    def test_auto_039_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [5120, 5120], 'stride': [5120, 1]}, 'key': {'data_type': 'ellipsis', 'value': 'Ellipsis'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 88
    def test_auto_040_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [1024, 5120], 'stride': [5120, 1]}, 'key': {'data_type': 'ellipsis', 'value': 'Ellipsis'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 89
    def test_auto_041_tensor_view(self):
        func_name = 'Tensor.view'
        args_info = {'arg0': {'data_type': 'torch.uint8', 'shape': [2048], 'stride': [1]}, 'dtype': {'data_type': 'dtype', 'value': 'torch.bfloat16'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 92
    def test_auto_042_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [1024], 'stride': [1]}, 'key': {'data_type': 'ellipsis', 'value': 'Ellipsis'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 122
    def test_auto_043_torch_empty_like(self):
        func_name = 'torch.empty_like'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [64], 'stride': [1]}, 'device': {'data_type': 'str', 'value': 'cpu'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 129
    def test_auto_044_tensor_copy(self):
        func_name = 'Tensor.copy_'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [64], 'stride': [1]}, 'arg1': {'data_type': 'torch.float32', 'shape': [64], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 131
    def test_auto_045_torch_has_compatible_shallow_copy_type(self):
        func_name = 'torch._has_compatible_shallow_copy_type'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [152064, 5120], 'stride': [5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [152064, 5120], 'stride': [5120, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 132
    def test_auto_046_torch_has_compatible_shallow_copy_type(self):
        func_name = 'torch._has_compatible_shallow_copy_type'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [5120, 5120], 'stride': [5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [5120, 5120], 'stride': [5120, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 133
    def test_auto_047_torch_has_compatible_shallow_copy_type(self):
        func_name = 'torch._has_compatible_shallow_copy_type'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [5120], 'stride': [1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [5120], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 134
    def test_auto_048_torch_has_compatible_shallow_copy_type(self):
        func_name = 'torch._has_compatible_shallow_copy_type'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1024, 5120], 'stride': [5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1024, 5120], 'stride': [5120, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 135
    def test_auto_049_torch_has_compatible_shallow_copy_type(self):
        func_name = 'torch._has_compatible_shallow_copy_type'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1024], 'stride': [1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1024], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 139
    def test_auto_050_torch_has_compatible_shallow_copy_type(self):
        func_name = 'torch._has_compatible_shallow_copy_type'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [13824, 5120], 'stride': [5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [13824, 5120], 'stride': [5120, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 141
    def test_auto_051_torch_has_compatible_shallow_copy_type(self):
        func_name = 'torch._has_compatible_shallow_copy_type'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [5120, 13824], 'stride': [13824, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [5120, 13824], 'stride': [13824, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 146
    def test_auto_052_torch_isin(self):
        func_name = 'torch.isin'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [2], 'stride': [1]}, 'arg1': {'data_type': 'torch.int64', 'shape': [], 'stride': []}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 147
    def test_auto_053_tensor_any(self):
        func_name = 'Tensor.any'
        args_info = {'arg0': {'data_type': 'torch.bool', 'shape': [2], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 148
    def test_auto_054_tensor_lt(self):
        func_name = 'Tensor.__lt__'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [2], 'stride': [1]}, 'arg1': {'data_type': 'int', 'value': 0}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 150
    def test_auto_055_tensor_cumsum(self):
        func_name = 'Tensor.cumsum'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1, 13], 'stride': [13, 1]}, 'arg1': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 151
    def test_auto_056_tensor_sub(self):
        func_name = 'Tensor.__sub__'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1, 13], 'stride': [13, 1]}, 'arg1': {'data_type': 'int', 'value': 1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 152
    def test_auto_057_tensor_eq(self):
        func_name = 'Tensor.__eq__'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1, 13], 'stride': [13, 1]}, 'arg1': {'data_type': 'int', 'value': 0}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 153
    def test_auto_058_tensor_masked_fill(self):
        func_name = 'Tensor.masked_fill'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1, 13], 'stride': [13, 1]}, 'arg1': {'data_type': 'torch.bool', 'shape': [1, 13], 'stride': [13, 1]}, 'arg2': {'data_type': 'int', 'value': 0}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 154
    def test_auto_059_torch_ones(self):
        func_name = 'torch.ones'
        args_info = {'arg0': {'data_type': 'int', 'value': 1}, 'dtype': {'data_type': 'dtype', 'value': 'torch.int64'}, 'device': {'data_type': 'device', 'value': "device(type='cuda', index=0)"}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 155
    def test_auto_060_torch_ones(self):
        func_name = 'torch.ones'
        args_info = {'arg0': {'data_type': 'int', 'value': 13}, 'dtype': {'data_type': 'dtype', 'value': 'torch.int64'}, 'device': {'data_type': 'device', 'value': "device(type='cuda', index=0)"}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 156
    def test_auto_061_tensor_cumsum(self):
        func_name = 'Tensor.cumsum'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [13], 'stride': [1]}, 'arg1': {'data_type': 'int', 'value': 0}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 157
    def test_auto_062_tensor_sub(self):
        func_name = 'Tensor.__sub__'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [13], 'stride': [1]}, 'arg1': {'data_type': 'int', 'value': 1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 158
    def test_auto_063_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.int64', 'shape': [1, 13], 'stride': [13, 1]}, 'key': {'data_type': 'tuple', 'value': ['slice(None, None, None)', 'slice(-13, None, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 159
    def test_auto_064_tensor_clone(self):
        func_name = 'Tensor.clone'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1, 13], 'stride': [13, 1]}, 'memory_format': {'data_type': 'memory_format', 'value': 'torch.contiguous_format'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 160
    def test_auto_065_torch_nn_functional_embedding(self):
        func_name = 'torch.nn.functional.embedding'
        args_info = {'input': {'data_type': 'torch.int64', 'shape': [1, 13], 'stride': [13, 1]}, 'weight': {'data_type': 'torch.bfloat16', 'shape': [152064, 5120], 'stride': [5120, 1]}, 'padding_idx': {'data_type': 'NoneType', 'value': None}, 'max_norm': {'data_type': 'NoneType', 'value': None}, 'norm_type': {'data_type': 'float', 'value': 2.0}, 'scale_grad_by_freq': {'data_type': 'bool', 'value': False}, 'sparse': {'data_type': 'bool', 'value': False}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 161
    def test_auto_066_torch_embedding(self):
        func_name = 'torch.embedding'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [152064, 5120], 'stride': [5120, 1]}, 'arg1': {'data_type': 'torch.int64', 'shape': [1, 13], 'stride': [13, 1]}, 'arg2': {'data_type': 'int', 'value': -1}, 'arg3': {'data_type': 'bool', 'value': False}, 'arg4': {'data_type': 'bool', 'value': False}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 162
    def test_auto_067_tensor_all(self):
        func_name = 'Tensor.all'
        args_info = {'arg0': {'data_type': 'torch.bool', 'shape': [1, 13], 'stride': [13, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 163
    def test_auto_068_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.float32', 'shape': [64], 'stride': [1]}, 'key': {'data_type': 'tuple', 'value': [None, 'slice(None, None, None)', None]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 164
    def test_auto_069_tensor_expand(self):
        func_name = 'Tensor.expand'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 64, 1], 'stride': [64, 1, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'int', 'value': -1}, 'arg3': {'data_type': 'int', 'value': 1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 165
    def test_auto_070_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.int64', 'shape': [1, 13], 'stride': [13, 1]}, 'key': {'data_type': 'tuple', 'value': ['slice(None, None, None)', None, 'slice(None, None, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 166
    def test_auto_071_tensor_matmul(self):
        func_name = 'Tensor.__matmul__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 64, 1], 'stride': [64, 1, 1]}, 'arg1': {'data_type': 'torch.float32', 'shape': [1, 1, 13], 'stride': [13, 13, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 167
    def test_auto_072_tensor_transpose(self):
        func_name = 'Tensor.transpose'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 64, 13], 'stride': [832, 13, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'int', 'value': 2}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 168
    def test_auto_073_torch_cat(self):
        func_name = 'torch.cat'
        args_info = {'arg0': {'data_type': 'tuple', 'value': [{'data_type': 'torch.float32', 'shape': [1, 13, 64], 'stride': [832, 1, 13]}, {'data_type': 'torch.float32', 'shape': [1, 13, 64], 'stride': [832, 1, 13]}]}, 'dim': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 169
    def test_auto_074_tensor_cos(self):
        func_name = 'Tensor.cos'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 13, 128], 'stride': [1664, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 170
    def test_auto_075_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 13, 128], 'stride': [1664, 128, 1]}, 'arg1': {'data_type': 'float', 'value': 1.0}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 171
    def test_auto_076_tensor_sin(self):
        func_name = 'Tensor.sin'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 13, 128], 'stride': [1664, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 173
    def test_auto_077_tensor_pow(self):
        func_name = 'Tensor.pow'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 13, 5120], 'stride': [66560, 5120, 1]}, 'arg1': {'data_type': 'int', 'value': 2}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 174
    def test_auto_078_tensor_mean(self):
        func_name = 'Tensor.mean'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 13, 5120], 'stride': [66560, 5120, 1]}, 'arg1': {'data_type': 'int', 'value': -1}, 'keepdim': {'data_type': 'bool', 'value': True}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 175
    def test_auto_079_tensor_add(self):
        func_name = 'Tensor.__add__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 13, 1], 'stride': [13, 1, 1]}, 'arg1': {'data_type': 'float', 'value': 1e-06}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 176
    def test_auto_080_torch_rsqrt(self):
        func_name = 'torch.rsqrt'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 13, 1], 'stride': [13, 1, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 177
    def test_auto_081_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 13, 5120], 'stride': [66560, 5120, 1]}, 'arg1': {'data_type': 'torch.float32', 'shape': [1, 13, 1], 'stride': [13, 1, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 178
    def test_auto_082_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [5120], 'stride': [1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 5120], 'stride': [66560, 5120, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 179
    def test_auto_083_torch_nn_functional_linear(self):
        func_name = 'torch.nn.functional.linear'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 5120], 'stride': [66560, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [5120, 5120], 'stride': [5120, 1]}, 'arg2': {'data_type': 'torch.bfloat16', 'shape': [5120], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 180
    def test_auto_084_tensor_view(self):
        func_name = 'Tensor.view'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 5120], 'stride': [66560, 5120, 1]}, 'arg1': {'data_type': 'tuple', 'value': [1, 13, -1, 128]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 181
    def test_auto_085_tensor_transpose(self):
        func_name = 'Tensor.transpose'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 40, 128], 'stride': [66560, 5120, 128, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'int', 'value': 2}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 182
    def test_auto_086_torch_nn_functional_linear(self):
        func_name = 'torch.nn.functional.linear'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 5120], 'stride': [66560, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1024, 5120], 'stride': [5120, 1]}, 'arg2': {'data_type': 'torch.bfloat16', 'shape': [1024], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 183
    def test_auto_087_tensor_view(self):
        func_name = 'Tensor.view'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 1024], 'stride': [13312, 1024, 1]}, 'arg1': {'data_type': 'tuple', 'value': [1, 13, -1, 128]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 184
    def test_auto_088_tensor_transpose(self):
        func_name = 'Tensor.transpose'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 8, 128], 'stride': [13312, 1024, 128, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'int', 'value': 2}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 188
    def test_auto_089_tensor_unsqueeze(self):
        func_name = 'Tensor.unsqueeze'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 128], 'stride': [1664, 128, 1]}, 'arg1': {'data_type': 'int', 'value': 1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 190
    def test_auto_090_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 13, 128], 'stride': [66560, 128, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 13, 128], 'stride': [1664, 1664, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 191
    def test_auto_091_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 13, 128], 'stride': [66560, 128, 5120, 1]}, 'key': {'data_type': 'tuple', 'value': ['Ellipsis', 'slice(None, 64, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 192
    def test_auto_092_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 13, 128], 'stride': [66560, 128, 5120, 1]}, 'key': {'data_type': 'tuple', 'value': ['Ellipsis', 'slice(64, None, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 193
    def test_auto_093_tensor_neg(self):
        func_name = 'Tensor.__neg__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 13, 64], 'stride': [33280, 64, 2560, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 194
    def test_auto_094_torch_cat(self):
        func_name = 'torch.cat'
        args_info = {'arg0': {'data_type': 'tuple', 'value': [{'data_type': 'torch.bfloat16', 'shape': [1, 40, 13, 64], 'stride': [33280, 64, 2560, 1]}, {'data_type': 'torch.bfloat16', 'shape': [1, 40, 13, 64], 'stride': [33280, 64, 2560, 1]}]}, 'dim': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 195
    def test_auto_095_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 13, 128], 'stride': [66560, 1664, 128, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 13, 128], 'stride': [1664, 1664, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 196
    def test_auto_096_tensor_add(self):
        func_name = 'Tensor.__add__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 13, 128], 'stride': [66560, 128, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 13, 128], 'stride': [66560, 1664, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 197
    def test_auto_097_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 13, 128], 'stride': [13312, 128, 1024, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 13, 128], 'stride': [1664, 1664, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 198
    def test_auto_098_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 13, 128], 'stride': [13312, 128, 1024, 1]}, 'key': {'data_type': 'tuple', 'value': ['Ellipsis', 'slice(None, 64, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 199
    def test_auto_099_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 13, 128], 'stride': [13312, 128, 1024, 1]}, 'key': {'data_type': 'tuple', 'value': ['Ellipsis', 'slice(64, None, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 200
    def test_auto_100_tensor_neg(self):
        func_name = 'Tensor.__neg__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 13, 64], 'stride': [6656, 64, 512, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 201
    def test_auto_101_torch_cat(self):
        func_name = 'torch.cat'
        args_info = {'arg0': {'data_type': 'tuple', 'value': [{'data_type': 'torch.bfloat16', 'shape': [1, 8, 13, 64], 'stride': [6656, 64, 512, 1]}, {'data_type': 'torch.bfloat16', 'shape': [1, 8, 13, 64], 'stride': [6656, 64, 512, 1]}]}, 'dim': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 202
    def test_auto_102_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 13, 128], 'stride': [13312, 1664, 128, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 13, 128], 'stride': [1664, 1664, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 203
    def test_auto_103_tensor_add(self):
        func_name = 'Tensor.__add__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 13, 128], 'stride': [13312, 128, 1024, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 13, 128], 'stride': [13312, 1664, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 204
    def test_auto_104_torch_cat(self):
        func_name = 'torch.cat'
        args_info = {'arg0': {'data_type': 'list', 'value': [{'data_type': 'torch.bfloat16', 'shape': [0], 'stride': [1]}, {'data_type': 'torch.bfloat16', 'shape': [1, 8, 13, 128], 'stride': [13312, 128, 1024, 1]}]}, 'dim': {'data_type': 'int', 'value': -2}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 206
    def test_auto_105_torch_nn_functional_scaled_dot_product_attention(self):
        func_name = 'torch.nn.functional.scaled_dot_product_attention'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 13, 128], 'stride': [66560, 128, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 13, 128], 'stride': [13312, 1664, 128, 1]}, 'arg2': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 13, 128], 'stride': [13312, 1664, 128, 1]}, 'attn_mask': {'data_type': 'NoneType', 'value': None}, 'dropout_p': {'data_type': 'float', 'value': 0.0}, 'scale': {'data_type': 'float', 'value': 0.08838834764831845}, 'is_causal': {'data_type': 'bool', 'value': True}, 'enable_gqa': {'data_type': 'bool', 'value': True}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 207
    def test_auto_106_tensor_transpose(self):
        func_name = 'Tensor.transpose'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 13, 128], 'stride': [66560, 1664, 128, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'int', 'value': 2}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 208
    def test_auto_107_tensor_contiguous(self):
        func_name = 'Tensor.contiguous'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 40, 128], 'stride': [66560, 128, 1664, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 209
    def test_auto_108_tensor_reshape(self):
        func_name = 'Tensor.reshape'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 40, 128], 'stride': [66560, 5120, 128, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'int', 'value': 13}, 'arg3': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 210
    def test_auto_109_tensor_contiguous(self):
        func_name = 'Tensor.contiguous'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 5120], 'stride': [66560, 5120, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 211
    def test_auto_110_torch_nn_functional_linear(self):
        func_name = 'torch.nn.functional.linear'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 5120], 'stride': [66560, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [5120, 5120], 'stride': [5120, 1]}, 'arg2': {'data_type': 'NoneType', 'value': None}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 212
    def test_auto_111_tensor_add(self):
        func_name = 'Tensor.__add__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 5120], 'stride': [66560, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 5120], 'stride': [66560, 5120, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 219
    def test_auto_112_torch_nn_functional_linear(self):
        func_name = 'torch.nn.functional.linear'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 5120], 'stride': [66560, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [13824, 5120], 'stride': [5120, 1]}, 'arg2': {'data_type': 'NoneType', 'value': None}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 220
    def test_auto_113_torch_nn_functional_silu(self):
        func_name = 'torch.nn.functional.silu'
        args_info = {'input': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 13824], 'stride': [179712, 13824, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 222
    def test_auto_114_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 13824], 'stride': [179712, 13824, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 13824], 'stride': [179712, 13824, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 223
    def test_auto_115_torch_nn_functional_linear(self):
        func_name = 'torch.nn.functional.linear'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 13824], 'stride': [179712, 13824, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [5120, 13824], 'stride': [13824, 1]}, 'arg2': {'data_type': 'NoneType', 'value': None}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 231
    def test_auto_116_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [1, 13, 5120], 'stride': [66560, 5120, 1]}, 'key': {'data_type': 'tuple', 'value': ['slice(None, None, None)', 'slice(-1, None, None)', 'slice(None, None, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 232
    def test_auto_117_torch_nn_functional_linear(self):
        func_name = 'torch.nn.functional.linear'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 5120], 'stride': [66560, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [152064, 5120], 'stride': [5120, 1]}, 'arg2': {'data_type': 'NoneType', 'value': None}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 233
    def test_auto_118_torch_arange(self):
        func_name = 'torch.arange'
        args_info = {'arg0': {'data_type': 'int', 'value': 1}, 'dtype': {'data_type': 'dtype', 'value': 'torch.int64'}, 'device': {'data_type': 'device', 'value': "device(type='cuda', index=0)"}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 234
    def test_auto_119_tensor_view(self):
        func_name = 'Tensor.view'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1], 'stride': [1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 235
    def test_auto_120_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.int64', 'shape': [1, 13], 'stride': [13, 1]}, 'key': {'data_type': 'tuple', 'value': ['Ellipsis', 'slice(-1, None, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 236
    def test_auto_121_tensor_add(self):
        func_name = 'Tensor.__add__'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1, 1], 'stride': [1, 1]}, 'arg1': {'data_type': 'torch.int64', 'shape': [1, 1], 'stride': [13, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 237
    def test_auto_122_tensor_add(self):
        func_name = 'Tensor.__add__'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1, 1], 'stride': [1, 1]}, 'arg1': {'data_type': 'int', 'value': 1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 238
    def test_auto_123_torch_cat(self):
        func_name = 'torch.cat'
        args_info = {'arg0': {'data_type': 'list', 'value': [{'data_type': 'torch.int64', 'shape': [1, 13], 'stride': [13, 1]}, {'data_type': 'torch.int64', 'shape': [1, 1], 'stride': [1, 1]}]}, 'dim': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 239
    def test_auto_124_tensor_new_ones(self):
        func_name = 'Tensor.new_ones'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1, 13], 'stride': [13, 1]}, 'arg1': {'data_type': 'tuple', 'value': [1, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 242
    def test_auto_125_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.int64', 'shape': [13], 'stride': [1]}, 'key': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 243
    def test_auto_126_tensor_add(self):
        func_name = 'Tensor.__add__'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1], 'stride': [1]}, 'arg1': {'data_type': 'torch.int64', 'shape': [], 'stride': []}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 244
    def test_auto_127_tensor_add(self):
        func_name = 'Tensor.__add__'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1], 'stride': [1]}, 'arg1': {'data_type': 'int', 'value': 1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 245
    def test_auto_128_torch_cat(self):
        func_name = 'torch.cat'
        args_info = {'arg0': {'data_type': 'tuple', 'value': [{'data_type': 'torch.int64', 'shape': [13], 'stride': [1]}, {'data_type': 'torch.int64', 'shape': [1], 'stride': [1]}]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 246
    def test_auto_129_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 152064], 'stride': [152064, 152064, 1]}, 'key': {'data_type': 'tuple', 'value': ['slice(None, None, None)', -1, 'slice(None, None, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 247
    def test_auto_130_torch_gather(self):
        func_name = 'torch.gather'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 152064], 'stride': [152064, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'torch.int64', 'shape': [1, 13], 'stride': [13, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 248
    def test_auto_131_tensor_lt(self):
        func_name = 'Tensor.__lt__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 13], 'stride': [13, 1]}, 'arg1': {'data_type': 'int', 'value': 0}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 249
    def test_auto_132_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 13], 'stride': [13, 1]}, 'arg1': {'data_type': 'float', 'value': 1.05}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 250
    def test_auto_133_tensor_truediv(self):
        func_name = 'Tensor.__truediv__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 13], 'stride': [13, 1]}, 'arg1': {'data_type': 'float', 'value': 1.05}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 251
    def test_auto_134_torch_where(self):
        func_name = 'torch.where'
        args_info = {'arg0': {'data_type': 'torch.bool', 'shape': [1, 13], 'stride': [13, 1]}, 'arg1': {'data_type': 'torch.float32', 'shape': [1, 13], 'stride': [13, 1]}, 'arg2': {'data_type': 'torch.float32', 'shape': [1, 13], 'stride': [13, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 252
    def test_auto_135_tensor_scatter(self):
        func_name = 'Tensor.scatter'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 152064], 'stride': [152064, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'torch.int64', 'shape': [1, 13], 'stride': [13, 1]}, 'arg3': {'data_type': 'torch.float32', 'shape': [1, 13], 'stride': [13, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 253
    def test_auto_136_tensor_truediv(self):
        func_name = 'Tensor.__truediv__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 152064], 'stride': [152064, 1]}, 'arg1': {'data_type': 'float', 'value': 0.7}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 254
    def test_auto_137_torch_topk(self):
        func_name = 'torch.topk'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 152064], 'stride': [152064, 1]}, 'arg1': {'data_type': 'int', 'value': 20}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 255
    def test_auto_138_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.float32', 'shape': [1, 20], 'stride': [20, 1]}, 'key': {'data_type': 'tuple', 'value': ['Ellipsis', -1, None]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 256
    def test_auto_139_tensor_lt(self):
        func_name = 'Tensor.__lt__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 152064], 'stride': [152064, 1]}, 'arg1': {'data_type': 'torch.float32', 'shape': [1, 1], 'stride': [20, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 258
    def test_auto_140_torch_sort(self):
        func_name = 'torch.sort'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 152064], 'stride': [152064, 1]}, 'descending': {'data_type': 'bool', 'value': False}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 259
    def test_auto_141_tensor_softmax(self):
        func_name = 'Tensor.softmax'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 152064], 'stride': [152064, 1]}, 'dim': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 260
    def test_auto_142_tensor_cumsum(self):
        func_name = 'Tensor.cumsum'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 152064], 'stride': [152064, 1]}, 'dim': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 261
    def test_auto_143_tensor_le(self):
        func_name = 'Tensor.__le__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 152064], 'stride': [152064, 1]}, 'arg1': {'data_type': 'float', 'value': 0.19999999999999996}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 262
    def test_auto_144_tensor_setitem(self):
        func_name = 'Tensor.__setitem__'
        args_info = {'self': {'data_type': 'torch.bool', 'shape': [1, 152064], 'stride': [152064, 1]}, 'key': {'data_type': 'tuple', 'value': ['Ellipsis', 'slice(-1, None, None)']}, 'value': {'data_type': 'int', 'value': 0}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 263
    def test_auto_145_tensor_scatter(self):
        func_name = 'Tensor.scatter'
        args_info = {'arg0': {'data_type': 'torch.bool', 'shape': [1, 152064], 'stride': [152064, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'torch.int64', 'shape': [1, 152064], 'stride': [152064, 1]}, 'arg3': {'data_type': 'torch.bool', 'shape': [1, 152064], 'stride': [152064, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 265
    def test_auto_146_torch_nn_functional_softmax(self):
        func_name = 'torch.nn.functional.softmax'
        args_info = {'input': {'data_type': 'torch.float32', 'shape': [1, 152064], 'stride': [152064, 1]}, 'dim': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 266
    def test_auto_147_tensor_softmax(self):
        func_name = 'Tensor.softmax'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 152064], 'stride': [152064, 1]}, 'arg1': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 267
    def test_auto_148_torch_multinomial(self):
        func_name = 'torch.multinomial'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 152064], 'stride': [152064, 1]}, 'num_samples': {'data_type': 'int', 'value': 1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 268
    def test_auto_149_tensor_squeeze(self):
        func_name = 'Tensor.squeeze'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1, 1], 'stride': [1, 1]}, 'arg1': {'data_type': 'int', 'value': 1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 269
    def test_auto_150_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1], 'stride': [1]}, 'arg1': {'data_type': 'torch.int64', 'shape': [1], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 270
    def test_auto_151_tensor_rsub(self):
        func_name = 'Tensor.__rsub__'
        args_info = {'self': {'data_type': 'torch.int64', 'shape': [1], 'stride': [1]}, 'other': {'data_type': 'int', 'value': 1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 271
    def test_auto_152_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [], 'stride': []}, 'arg1': {'data_type': 'torch.int64', 'shape': [1], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 272
    def test_auto_153_tensor_add(self):
        func_name = 'Tensor.__add__'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1], 'stride': [1]}, 'arg1': {'data_type': 'torch.int64', 'shape': [1], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 273
    def test_auto_154_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.int64', 'shape': [1], 'stride': [1]}, 'key': {'data_type': 'tuple', 'value': ['slice(None, None, None)', None]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 275
    def test_auto_155_torch_full(self):
        func_name = 'torch.full'
        args_info = {'arg0': {'data_type': 'tuple', 'value': [1]}, 'arg1': {'data_type': 'bool', 'value': False}, 'device': {'data_type': 'device', 'value': "device(type='cuda', index=0)"}, 'dtype': {'data_type': 'dtype', 'value': 'torch.bool'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 277
    def test_auto_156_tensor_or(self):
        func_name = 'Tensor.__or__'
        args_info = {'arg0': {'data_type': 'torch.bool', 'shape': [1], 'stride': [1]}, 'arg1': {'data_type': 'torch.bool', 'shape': [1], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 278
    def test_auto_157_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.int64', 'shape': [1, 14], 'stride': [14, 1]}, 'key': {'data_type': 'tuple', 'value': ['slice(None, None, None)', -1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 279
    def test_auto_158_torch_isin(self):
        func_name = 'torch.isin'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1], 'stride': [14]}, 'arg1': {'data_type': 'torch.int64', 'shape': [2], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 281
    def test_auto_159_tensor_invert(self):
        func_name = 'Tensor.__invert__'
        args_info = {'arg0': {'data_type': 'torch.bool', 'shape': [1], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 282
    def test_auto_160_tensor_and(self):
        func_name = 'Tensor.__and__'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1], 'stride': [1]}, 'arg1': {'data_type': 'torch.bool', 'shape': [1], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 283
    def test_auto_161_tensor_max(self):
        func_name = 'Tensor.max'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 284
    def test_auto_162_tensor_eq(self):
        func_name = 'Tensor.__eq__'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [], 'stride': []}, 'arg1': {'data_type': 'int', 'value': 0}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 285
    def test_auto_163_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.int64', 'shape': [1, 14], 'stride': [14, 1]}, 'key': {'data_type': 'tuple', 'value': ['slice(None, None, None)', 'slice(-1, None, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 286
    def test_auto_164_tensor_clone(self):
        func_name = 'Tensor.clone'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1, 1], 'stride': [14, 1]}, 'memory_format': {'data_type': 'memory_format', 'value': 'torch.contiguous_format'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 287
    def test_auto_165_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.int64', 'shape': [1, 14], 'stride': [14, 1]}, 'key': {'data_type': 'tuple', 'value': ['Ellipsis', 'slice(-1, None, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 289
    def test_auto_166_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.int64', 'shape': [14], 'stride': [1]}, 'key': {'data_type': 'tuple', 'value': ['Ellipsis', 'slice(-1, None, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 290
    def test_auto_167_tensor_clone(self):
        func_name = 'Tensor.clone'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1], 'stride': [1]}, 'memory_format': {'data_type': 'memory_format', 'value': 'torch.contiguous_format'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 291
    def test_auto_168_torch_nn_functional_embedding(self):
        func_name = 'torch.nn.functional.embedding'
        args_info = {'input': {'data_type': 'torch.int64', 'shape': [1, 1], 'stride': [1, 1]}, 'weight': {'data_type': 'torch.bfloat16', 'shape': [152064, 5120], 'stride': [5120, 1]}, 'padding_idx': {'data_type': 'NoneType', 'value': None}, 'max_norm': {'data_type': 'NoneType', 'value': None}, 'norm_type': {'data_type': 'float', 'value': 2.0}, 'scale_grad_by_freq': {'data_type': 'bool', 'value': False}, 'sparse': {'data_type': 'bool', 'value': False}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 292
    def test_auto_169_torch_embedding(self):
        func_name = 'torch.embedding'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [152064, 5120], 'stride': [5120, 1]}, 'arg1': {'data_type': 'torch.int64', 'shape': [1, 1], 'stride': [1, 1]}, 'arg2': {'data_type': 'int', 'value': -1}, 'arg3': {'data_type': 'bool', 'value': False}, 'arg4': {'data_type': 'bool', 'value': False}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 293
    def test_auto_170_tensor_all(self):
        func_name = 'Tensor.all'
        args_info = {'arg0': {'data_type': 'torch.bool', 'shape': [1, 14], 'stride': [14, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 296
    def test_auto_171_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.int64', 'shape': [1, 1], 'stride': [1, 1]}, 'key': {'data_type': 'tuple', 'value': ['slice(None, None, None)', None, 'slice(None, None, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 297
    def test_auto_172_tensor_matmul(self):
        func_name = 'Tensor.__matmul__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 64, 1], 'stride': [64, 1, 1]}, 'arg1': {'data_type': 'torch.float32', 'shape': [1, 1, 1], 'stride': [1, 1, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 298
    def test_auto_173_tensor_transpose(self):
        func_name = 'Tensor.transpose'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 64, 1], 'stride': [64, 1, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'int', 'value': 2}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 299
    def test_auto_174_torch_cat(self):
        func_name = 'torch.cat'
        args_info = {'arg0': {'data_type': 'tuple', 'value': [{'data_type': 'torch.float32', 'shape': [1, 1, 64], 'stride': [64, 1, 1]}, {'data_type': 'torch.float32', 'shape': [1, 1, 64], 'stride': [64, 1, 1]}]}, 'dim': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 300
    def test_auto_175_tensor_cos(self):
        func_name = 'Tensor.cos'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 1, 128], 'stride': [128, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 301
    def test_auto_176_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 1, 128], 'stride': [128, 128, 1]}, 'arg1': {'data_type': 'float', 'value': 1.0}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 302
    def test_auto_177_tensor_sin(self):
        func_name = 'Tensor.sin'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 1, 128], 'stride': [128, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 304
    def test_auto_178_tensor_pow(self):
        func_name = 'Tensor.pow'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 1, 5120], 'stride': [5120, 5120, 1]}, 'arg1': {'data_type': 'int', 'value': 2}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 305
    def test_auto_179_tensor_mean(self):
        func_name = 'Tensor.mean'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 1, 5120], 'stride': [5120, 5120, 1]}, 'arg1': {'data_type': 'int', 'value': -1}, 'keepdim': {'data_type': 'bool', 'value': True}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 306
    def test_auto_180_tensor_add(self):
        func_name = 'Tensor.__add__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 1, 1], 'stride': [1, 1, 1]}, 'arg1': {'data_type': 'float', 'value': 1e-06}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 307
    def test_auto_181_torch_rsqrt(self):
        func_name = 'torch.rsqrt'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 1, 1], 'stride': [1, 1, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 308
    def test_auto_182_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 1, 5120], 'stride': [5120, 5120, 1]}, 'arg1': {'data_type': 'torch.float32', 'shape': [1, 1, 1], 'stride': [1, 1, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 309
    def test_auto_183_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [5120], 'stride': [1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 5120], 'stride': [5120, 5120, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 310
    def test_auto_184_torch_nn_functional_linear(self):
        func_name = 'torch.nn.functional.linear'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 5120], 'stride': [5120, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [5120, 5120], 'stride': [5120, 1]}, 'arg2': {'data_type': 'torch.bfloat16', 'shape': [5120], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 311
    def test_auto_185_tensor_view(self):
        func_name = 'Tensor.view'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 5120], 'stride': [5120, 5120, 1]}, 'arg1': {'data_type': 'tuple', 'value': [1, 1, -1, 128]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 312
    def test_auto_186_tensor_transpose(self):
        func_name = 'Tensor.transpose'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 40, 128], 'stride': [5120, 5120, 128, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'int', 'value': 2}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 313
    def test_auto_187_torch_nn_functional_linear(self):
        func_name = 'torch.nn.functional.linear'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 5120], 'stride': [5120, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1024, 5120], 'stride': [5120, 1]}, 'arg2': {'data_type': 'torch.bfloat16', 'shape': [1024], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 314
    def test_auto_188_tensor_view(self):
        func_name = 'Tensor.view'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 1024], 'stride': [1024, 1024, 1]}, 'arg1': {'data_type': 'tuple', 'value': [1, 1, -1, 128]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 315
    def test_auto_189_tensor_transpose(self):
        func_name = 'Tensor.transpose'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 8, 128], 'stride': [1024, 1024, 128, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'int', 'value': 2}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 319
    def test_auto_190_tensor_unsqueeze(self):
        func_name = 'Tensor.unsqueeze'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 128], 'stride': [128, 128, 1]}, 'arg1': {'data_type': 'int', 'value': 1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 321
    def test_auto_191_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 1, 128], 'stride': [5120, 128, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 1, 128], 'stride': [128, 128, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 322
    def test_auto_192_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 1, 128], 'stride': [5120, 128, 5120, 1]}, 'key': {'data_type': 'tuple', 'value': ['Ellipsis', 'slice(None, 64, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 323
    def test_auto_193_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 1, 128], 'stride': [5120, 128, 5120, 1]}, 'key': {'data_type': 'tuple', 'value': ['Ellipsis', 'slice(64, None, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 324
    def test_auto_194_tensor_neg(self):
        func_name = 'Tensor.__neg__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 1, 64], 'stride': [2560, 64, 2560, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 325
    def test_auto_195_torch_cat(self):
        func_name = 'torch.cat'
        args_info = {'arg0': {'data_type': 'tuple', 'value': [{'data_type': 'torch.bfloat16', 'shape': [1, 40, 1, 64], 'stride': [2560, 64, 64, 1]}, {'data_type': 'torch.bfloat16', 'shape': [1, 40, 1, 64], 'stride': [2560, 64, 2560, 1]}]}, 'dim': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 326
    def test_auto_196_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 1, 128], 'stride': [5120, 128, 128, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 1, 128], 'stride': [128, 128, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 327
    def test_auto_197_tensor_add(self):
        func_name = 'Tensor.__add__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 1, 128], 'stride': [5120, 128, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 1, 128], 'stride': [5120, 128, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 328
    def test_auto_198_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 1, 128], 'stride': [1024, 128, 1024, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 1, 128], 'stride': [128, 128, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 329
    def test_auto_199_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 1, 128], 'stride': [1024, 128, 1024, 1]}, 'key': {'data_type': 'tuple', 'value': ['Ellipsis', 'slice(None, 64, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 330
    def test_auto_200_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 1, 128], 'stride': [1024, 128, 1024, 1]}, 'key': {'data_type': 'tuple', 'value': ['Ellipsis', 'slice(64, None, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 331
    def test_auto_201_tensor_neg(self):
        func_name = 'Tensor.__neg__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 1, 64], 'stride': [512, 64, 512, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 332
    def test_auto_202_torch_cat(self):
        func_name = 'torch.cat'
        args_info = {'arg0': {'data_type': 'tuple', 'value': [{'data_type': 'torch.bfloat16', 'shape': [1, 8, 1, 64], 'stride': [512, 64, 64, 1]}, {'data_type': 'torch.bfloat16', 'shape': [1, 8, 1, 64], 'stride': [512, 64, 512, 1]}]}, 'dim': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 333
    def test_auto_203_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 1, 128], 'stride': [1024, 128, 128, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 1, 128], 'stride': [128, 128, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 334
    def test_auto_204_tensor_add(self):
        func_name = 'Tensor.__add__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 1, 128], 'stride': [1024, 128, 1024, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 1, 128], 'stride': [1024, 128, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 335
    def test_auto_205_torch_cat(self):
        func_name = 'torch.cat'
        args_info = {'arg0': {'data_type': 'list', 'value': [{'data_type': 'torch.bfloat16', 'shape': [1, 8, 13, 128], 'stride': [13312, 1664, 128, 1]}, {'data_type': 'torch.bfloat16', 'shape': [1, 8, 1, 128], 'stride': [1024, 128, 128, 1]}]}, 'dim': {'data_type': 'int', 'value': -2}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 336
    def test_auto_206_torch_cat(self):
        func_name = 'torch.cat'
        args_info = {'arg0': {'data_type': 'list', 'value': [{'data_type': 'torch.bfloat16', 'shape': [1, 8, 13, 128], 'stride': [13312, 1664, 128, 1]}, {'data_type': 'torch.bfloat16', 'shape': [1, 8, 1, 128], 'stride': [1024, 128, 1024, 1]}]}, 'dim': {'data_type': 'int', 'value': -2}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 337
    def test_auto_207_torch_nn_functional_scaled_dot_product_attention(self):
        func_name = 'torch.nn.functional.scaled_dot_product_attention'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 1, 128], 'stride': [5120, 128, 128, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 14, 128], 'stride': [14336, 1792, 128, 1]}, 'arg2': {'data_type': 'torch.bfloat16', 'shape': [1, 8, 14, 128], 'stride': [14336, 1792, 128, 1]}, 'attn_mask': {'data_type': 'NoneType', 'value': None}, 'dropout_p': {'data_type': 'float', 'value': 0.0}, 'scale': {'data_type': 'float', 'value': 0.08838834764831845}, 'is_causal': {'data_type': 'bool', 'value': False}, 'enable_gqa': {'data_type': 'bool', 'value': True}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 338
    def test_auto_208_tensor_transpose(self):
        func_name = 'Tensor.transpose'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 40, 1, 128], 'stride': [5120, 128, 128, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'int', 'value': 2}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 339
    def test_auto_209_tensor_contiguous(self):
        func_name = 'Tensor.contiguous'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 40, 128], 'stride': [5120, 128, 128, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 340
    def test_auto_210_tensor_reshape(self):
        func_name = 'Tensor.reshape'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 40, 128], 'stride': [5120, 128, 128, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'int', 'value': 1}, 'arg3': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 341
    def test_auto_211_tensor_contiguous(self):
        func_name = 'Tensor.contiguous'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 5120], 'stride': [5120, 5120, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 342
    def test_auto_212_torch_nn_functional_linear(self):
        func_name = 'torch.nn.functional.linear'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 5120], 'stride': [5120, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [5120, 5120], 'stride': [5120, 1]}, 'arg2': {'data_type': 'NoneType', 'value': None}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 343
    def test_auto_213_tensor_add(self):
        func_name = 'Tensor.__add__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 5120], 'stride': [5120, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 5120], 'stride': [5120, 5120, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 350
    def test_auto_214_torch_nn_functional_linear(self):
        func_name = 'torch.nn.functional.linear'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 5120], 'stride': [5120, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [13824, 5120], 'stride': [5120, 1]}, 'arg2': {'data_type': 'NoneType', 'value': None}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 351
    def test_auto_215_torch_nn_functional_silu(self):
        func_name = 'torch.nn.functional.silu'
        args_info = {'input': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 13824], 'stride': [13824, 13824, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 353
    def test_auto_216_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 13824], 'stride': [13824, 13824, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 13824], 'stride': [13824, 13824, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 354
    def test_auto_217_torch_nn_functional_linear(self):
        func_name = 'torch.nn.functional.linear'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 13824], 'stride': [13824, 13824, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [5120, 13824], 'stride': [13824, 1]}, 'arg2': {'data_type': 'NoneType', 'value': None}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 362
    def test_auto_218_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 5120], 'stride': [5120, 5120, 1]}, 'key': {'data_type': 'tuple', 'value': ['slice(None, None, None)', 'slice(-1, None, None)', 'slice(None, None, None)']}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 363
    def test_auto_219_torch_nn_functional_linear(self):
        func_name = 'torch.nn.functional.linear'
        args_info = {'arg0': {'data_type': 'torch.bfloat16', 'shape': [1, 1, 5120], 'stride': [5120, 5120, 1]}, 'arg1': {'data_type': 'torch.bfloat16', 'shape': [152064, 5120], 'stride': [5120, 1]}, 'arg2': {'data_type': 'NoneType', 'value': None}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 367
    def test_auto_220_tensor_add(self):
        func_name = 'Tensor.__add__'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1, 1], 'stride': [1, 1]}, 'arg1': {'data_type': 'torch.int64', 'shape': [1, 1], 'stride': [14, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 369
    def test_auto_221_torch_cat(self):
        func_name = 'torch.cat'
        args_info = {'arg0': {'data_type': 'list', 'value': [{'data_type': 'torch.int64', 'shape': [1, 14], 'stride': [14, 1]}, {'data_type': 'torch.int64', 'shape': [1, 1], 'stride': [1, 1]}]}, 'dim': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 370
    def test_auto_222_tensor_new_ones(self):
        func_name = 'Tensor.new_ones'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1, 14], 'stride': [14, 1]}, 'arg1': {'data_type': 'tuple', 'value': [1, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 373
    def test_auto_223_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.int64', 'shape': [14], 'stride': [1]}, 'key': {'data_type': 'int', 'value': -1}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 376
    def test_auto_224_torch_cat(self):
        func_name = 'torch.cat'
        args_info = {'arg0': {'data_type': 'tuple', 'value': [{'data_type': 'torch.int64', 'shape': [14], 'stride': [1]}, {'data_type': 'torch.int64', 'shape': [1], 'stride': [1]}]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 378
    def test_auto_225_torch_gather(self):
        func_name = 'torch.gather'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 152064], 'stride': [152064, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'torch.int64', 'shape': [1, 14], 'stride': [14, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 379
    def test_auto_226_tensor_lt(self):
        func_name = 'Tensor.__lt__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 14], 'stride': [14, 1]}, 'arg1': {'data_type': 'int', 'value': 0}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 380
    def test_auto_227_tensor_mul(self):
        func_name = 'Tensor.__mul__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 14], 'stride': [14, 1]}, 'arg1': {'data_type': 'float', 'value': 1.05}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 381
    def test_auto_228_tensor_truediv(self):
        func_name = 'Tensor.__truediv__'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 14], 'stride': [14, 1]}, 'arg1': {'data_type': 'float', 'value': 1.05}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 382
    def test_auto_229_torch_where(self):
        func_name = 'torch.where'
        args_info = {'arg0': {'data_type': 'torch.bool', 'shape': [1, 14], 'stride': [14, 1]}, 'arg1': {'data_type': 'torch.float32', 'shape': [1, 14], 'stride': [14, 1]}, 'arg2': {'data_type': 'torch.float32', 'shape': [1, 14], 'stride': [14, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 383
    def test_auto_230_tensor_scatter(self):
        func_name = 'Tensor.scatter'
        args_info = {'arg0': {'data_type': 'torch.float32', 'shape': [1, 152064], 'stride': [152064, 1]}, 'arg1': {'data_type': 'int', 'value': 1}, 'arg2': {'data_type': 'torch.int64', 'shape': [1, 14], 'stride': [14, 1]}, 'arg3': {'data_type': 'torch.float32', 'shape': [1, 14], 'stride': [14, 1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 407
    def test_auto_231_torch_full(self):
        func_name = 'torch.full'
        args_info = {'arg0': {'data_type': 'tuple', 'value': [1]}, 'arg1': {'data_type': 'bool', 'value': True}, 'device': {'data_type': 'device', 'value': "device(type='cuda', index=0)"}, 'dtype': {'data_type': 'dtype', 'value': 'torch.bool'}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 409
    def test_auto_232_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.int64', 'shape': [1, 15], 'stride': [15, 1]}, 'key': {'data_type': 'tuple', 'value': ['slice(None, None, None)', -1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 410
    def test_auto_233_torch_isin(self):
        func_name = 'torch.isin'
        args_info = {'arg0': {'data_type': 'torch.int64', 'shape': [1], 'stride': [15]}, 'arg1': {'data_type': 'torch.int64', 'shape': [2], 'stride': [1]}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))


    # Source: ops.log line 416
    def test_auto_234_tensor_getitem(self):
        func_name = 'Tensor.__getitem__'
        args_info = {'self': {'data_type': 'torch.int64', 'shape': [1, 15], 'stride': [15, 1]}, 'key': {'data_type': 'int', 'value': 0}}
        fn = self._resolve_callable(func_name)
        self.assertIsNotNone(fn, f"callable not found: {func_name}")
        print(f"Testing {func_name} args_info:", args_info) if DEBUG else None

        errors = []
        for float_tensor_dtype in self._iter_test_dtypes(args_info):
            with self.subTest(float_tensor_dtype=str(float_tensor_dtype) if float_tensor_dtype is not None else "original"):
                try:
                    args, kwargs = self._build_call_args(func_name, args_info, float_tensor_dtype=float_tensor_dtype)
                    result = self._invoke(func_name, fn, args, kwargs)
                    self.assertIsNotNone(result)
                except Exception as exc:
                    errors.append(f"{float_tensor_dtype}: {exc}")

        if errors:
            self.fail(f"auto case failed for {func_name} -> " + " | ".join(errors))



if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
