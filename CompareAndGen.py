"""
Compare two test_ops_auto.py files generated with different input lengths,
detect which integers change with sequence length, and produce a parameterized
test_ops_auto_res.py using `mm` as the variable.

Usage:
    python generate_param_test.py Test1/test_ops_auto.py Test2/test_ops_auto.py test_ops_auto_res.py
"""

import ast
import re
import sys
from pathlib import Path

# Model-specific constants that should never be parameterized
# These are derived from Qwen2.5-7B config.json
KNOWN_DIMS = {
    3584,       # hidden_size
    18944,      # intermediate_size
    152064,     # vocab_size
    28,         # num_attention_heads
    4,          # num_key_value_heads
    128,        # head_dim = 3584 / 28
    64,         # half head_dim
    512,        # 4 * 128 (kv_heads * head_dim)
    256,
    1,          # batch size / new token
    2,
    3,
    0,          # broadcast stride
    -1,         # common index
    -2,         # common index
    7,          # num_kv_groups = 28 / 4
}


def _collect_integers(obj):
    """Recursively collect all integers from a nested structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _collect_integers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _collect_integers(v)
    elif isinstance(obj, int):
        yield obj


def _collect_shapes(obj):
    """Recursively collect shape lists from args_info."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _collect_shapes(v)
    elif isinstance(obj, list):
        if all(isinstance(x, int) for x in obj) and len(obj) >= 2:
            yield obj
        else:
            for v in obj:
                yield from _collect_shapes(v)
    elif isinstance(obj, tuple):
        for v in obj:
            yield from _collect_shapes(v)


def detect_seq_len(args_dicts):
    """Detect the original sequence length from a list of parsed args_info dicts.

    Uses shape-position heuristics:
    - Only looks at 'shape' lists (not stride, which is noisy)
    - Skips batch dimension (index 0)
    - Excludes known model dimensions
    - Bonus for (candidate+1) also appearing (KV cache pattern)
    """
    candidates = {}
    for d in args_dicts:
        for shape in _collect_shapes(d):
            for i, dim in enumerate(shape):
                if i == 0:
                    continue
                if dim in KNOWN_DIMS:
                    continue
                if dim < 2 or dim > 1000:
                    continue
                candidates[dim] = candidates.get(dim, 0) + 1

    if not candidates:
        raise ValueError("Could not detect sequence length from args_info")

    # Score: frequency + bonus if (candidate+1) also exists (KV cache confirmation)
    best = None
    best_score = 0
    for val, freq in sorted(candidates.items(), key=lambda x: -x[1]):
        score = freq
        if (val + 1) in candidates:
            score += candidates[val + 1] * 2
        if score > best_score:
            best_score = score
            best = val

    print(f"  candidates: {dict(sorted(candidates.items(), key=lambda x: -x[1])[:8])}")
    return best


def param_expr(v1, v2, sl1, sl2):
    """Derive the mm expression for two corresponding values that differ.

    v1 is from file with seq_len=sl1, v2 from file with seq_len=sl2.

    Returns an expression string like "mm", "mm + 1", "mm * 3584",
    "(mm + 1) * 128", etc. Returns None if the value should stay literal.
    """
    if not isinstance(v1, int) or not isinstance(v2, int):
        return None
    if v1 == v2:
        return None  # same in both, keep literal
    if v1 in KNOWN_DIMS:
        return None  # model constant

    # --- Direct equality ---
    if v1 == sl1 and v2 == sl2:
        return "mm"

    # --- Offset: v1 = sl1 + k, v2 = sl2 + k ---
    k = v1 - sl1
    if v2 - sl2 == k:
        if k == 1:
            return "mm + 1"
        if k == 2:
            return "mm + 2"
        if k == -1:
            return "mm - 1"
        if k == -2:
            return "mm - 2"
        if k > 0:
            return f"mm + {k}"
        if k < 0:
            return f"mm - {abs(k)}"

    # --- Multiple of seq_len: v1 = k * sl1, v2 = k * sl2 ---
    if sl1 > 0 and sl2 > 0:
        if v1 % sl1 == 0 and v2 % sl2 == 0:
            k1 = v1 // sl1
            k2 = v2 // sl2
            if k1 == k2:
                if k1 == 1:
                    return "mm"
                return f"mm * {k1}"

    # --- Multiple of (seq_len + offset) ---
    for offset in [1, 2, -1, -2]:
        b1, b2 = sl1 + offset, sl2 + offset
        if b1 > 0 and b2 > 0 and v1 % b1 == 0 and v2 % b2 == 0:
            k1, k2 = v1 // b1, v2 // b2
            if k1 == k2:
                if offset > 0:
                    base = "mm + 1" if offset == 1 else f"mm + {offset}"
                else:
                    base = "mm - 1" if offset == -1 else f"mm - {abs(offset)}"
                if k1 == 1:
                    return base
                return f"({base}) * {k1}"

    # --- Multiple of seq_len^2 (attention score strides) ---
    sq1, sq2 = sl1 * sl1, sl2 * sl2
    if sq1 > 0 and sq2 > 0 and v1 % sq1 == 0 and v2 % sq2 == 0:
        k1, k2 = v1 // sq1, v2 // sq2
        if k1 == k2:
            if k1 == 1:
                return "mm * mm"
            return f"mm * mm * {k1}"

    # --- Failed ---
    print(f"  WARNING: cannot parameterize {v1}↔{v2} (sl1={sl1}, sl2={sl2})")
    return None


def deep_param(obj1, obj2, sl1, sl2):
    """Walk two parallel structures. Returns a version of obj1 where
    integers that differ from obj2 are replaced with mm expression strings.
    """
    if type(obj1) != type(obj2):
        return obj1

    if isinstance(obj1, dict):
        result = {}
        for k in obj1:
            if k in obj2:
                result[k] = deep_param(obj1[k], obj2[k], sl1, sl2)
            else:
                result[k] = obj1[k]
        return result

    elif isinstance(obj1, list):
        return [deep_param(v1, v2, sl1, sl2) if i < len(obj2) else v1
                for i, (v1, v2) in enumerate(zip(obj1, obj2))]

    elif isinstance(obj1, tuple):
        return tuple(deep_param(v1, v2, sl1, sl2) if i < len(obj2) else v1
                     for i, (v1, v2) in enumerate(zip(obj1, obj2)))

    elif isinstance(obj1, int) and isinstance(obj2, int):
        expr = param_expr(obj1, obj2, sl1, sl2)
        if expr is not None:
            return expr
        return obj1

    else:
        return obj1


# ── Formatting: turn param dict (with embedded mm-expression strings) into Python source ──

def is_expr_str(v):
    """Check if a string is an mm expression (not a regular string like 'torch.float32')."""
    return 'mm' in v


def format_value(v):
    """Format a value as Python source code.
    Dict keys use repr(). Values that are mm-expression strings are emitted
    without quotes; everything else uses repr().
    """
    if isinstance(v, str) and is_expr_str(v):
        return v  # bare expression, no quotes

    if isinstance(v, dict):
        items = []
        for k, val in v.items():
            items.append(f"{k!r}: {format_value(val)}")
        return "{" + ", ".join(items) + "}"

    if isinstance(v, list):
        items = [format_value(x) for x in v]
        return "[" + ", ".join(items) + "]"

    if isinstance(v, tuple):
        items = [format_value(x) for x in v]
        return "(" + ", ".join(items) + ")"

    if isinstance(v, str):
        return repr(v)

    if isinstance(v, bool):
        return repr(v)

    if v is None:
        return "None"

    if isinstance(v, (int, float)):
        return repr(v)

    return repr(v)


def process(test1_path, test2_path, output_path):
    # ── Read both files ──
    with open(test1_path, "r", encoding="utf-8") as f:
        lines1 = f.readlines()
    with open(test2_path, "r", encoding="utf-8") as f:
        lines2 = f.readlines()

    # Pattern to match an args_info assignment line
    args_re = re.compile(r"^(\s*)args_info\s*=\s*(.+)$")

    # ── First pass: parse all args_info to detect seq_lens ──
    dicts1, dicts2 = [], []
    for line in lines1:
        m = args_re.match(line)
        if m:
            try:
                dicts1.append(ast.literal_eval(m.group(2)))
            except Exception:
                pass
    for line in lines2:
        m = args_re.match(line)
        if m:
            try:
                dicts2.append(ast.literal_eval(m.group(2)))
            except Exception:
                pass

    sl1 = detect_seq_len(dicts1)
    sl2 = detect_seq_len(dicts2)
    print(f"Detected: sl1={sl1}, sl2={sl2}")

    # ── Second pass: process line-by-line ──
    out_lines = []
    param_count = 0
    unparam_count = 0

    for i, (l1, l2) in enumerate(zip(lines1, lines2)):
        m1 = args_re.match(l1)
        m2 = args_re.match(l2)

        if m1 and m2:
            indent = m1.group(1)
            try:
                d1 = ast.literal_eval(m1.group(2))
                d2 = ast.literal_eval(m2.group(2))
            except Exception:
                out_lines.append(l1.rstrip("\n"))
                continue

            result = deep_param(d1, d2, sl1, sl2)

            # Count how many values were parameterized
            def count_expr(v):
                if isinstance(v, str) and is_expr_str(v):
                    return 1
                if isinstance(v, dict):
                    return sum(count_expr(x) for x in v.values())
                if isinstance(v, (list, tuple)):
                    return sum(count_expr(x) for x in v)
                return 0

            n = count_expr(result)
            if n > 0:
                param_count += n
            else:
                unparam_count += 1

            out_lines.append(f"{indent}args_info = {format_value(result)}")
        else:
            out_lines.append(l1.rstrip("\n"))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    print(f"Generated {output_path}")
    print(f"  {param_count} values parameterized with mm")
    print(f"  {unparam_count} args_info left as literals (no change between sl1/sl2)")

    # Validate: try to parse the output
    print("  Validating syntax...")
    with open(output_path, "r", encoding="utf-8") as f:
        source = f.read()
    try:
        compile(source, output_path, "exec")
        print("  Syntax OK")
    except SyntaxError as e:
        print(f"  SYNTAX ERROR: {e}")

    # ── Extract test method names and generate .sh ──
    test_re = re.compile(r"^\s*def (test_auto_\w+)\(self\):")
    test_names = []
    for line in out_lines:
        m = test_re.match(line)
        if m:
            test_names.append(m.group(1))

    # Write the .sh file
    out_dir = Path(output_path).parent
    sh_path = out_dir / "run_all_tests.sh"
    with open(sh_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Auto-generated: {len(test_names)} test cases\n\n")
        for tname in test_names:
            f.write(f"pytest -v -s test_ops_auto_res.py::TestOpsAuto::{tname}\n")

    print(f"Generated {sh_path} with {len(test_names)} test commands")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    process(sys.argv[1], sys.argv[2], sys.argv[3])
