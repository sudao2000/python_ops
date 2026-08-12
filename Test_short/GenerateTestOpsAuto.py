import json
import re
import sys
from pathlib import Path

Debug = False

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from test_ops_auto_template import HEADER, TEST_METHOD_TEMPLATE, FOOTER

LOG_PATH = Path("ops.log")
OUTPUT_PATH = Path("test_ops_auto.py")

LINE_RE = re.compile(r"^Fallback: (?P<func>.+?) fallback to CPU\. Args: (?P<args>\{.*\}) \(at ")


def sanitize_test_name(func_name: str, index: int) -> str:
    s = func_name.lower().replace(".", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return f"{index:03d}_{s}"


def clean_json_text(text: str) -> str:
    text = text.replace("-Infinity", '"-Infinity"')
    text = text.replace("Infinity", '"Infinity"')
    text = text.replace("NaN", '"NaN"')
    return text


def _make_hashable(obj):
    """Convert nested dicts/lists to a hashable representation."""
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, list):
        return tuple(_make_hashable(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(_make_hashable(item) for item in obj)
    else:
        return obj


def parse_cases(max_cases_per_op: int = 2, max_total: int = 1000):
    cases = []
    per_op_count = {}
    seen_signatures = {}  # Track (func_name, args_info) -> first line_no
    duplicate_mapping = {}  # Track duplicate line_no -> original line_no

    with LOG_PATH.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            m = LINE_RE.search(line.strip())
            if not m:
                print(f"Skipping line {line_no}: {line}")
                continue

            func_name = m.group("func")
            args_text = clean_json_text(m.group("args"))
            try:
                args_info = json.loads(args_text)
            except Exception:
                continue

            # Create signature from func_name and entire args_info (including values)
            args_hashable = _make_hashable(args_info)
            signature = (func_name, args_hashable)
            
            # Check if we've already seen this func_name + args combination
            if signature in seen_signatures:
                duplicate_mapping[line_no] = seen_signatures[signature]
                continue
            
            seen_signatures[signature] = line_no
            cnt = per_op_count.get(func_name, 0)
            # if cnt >= max_cases_per_op:
            #     import pdb; pdb.set_trace()
            #     continue

            per_op_count[func_name] = cnt + 1
            cases.append((func_name, args_info, line_no))

            if len(cases) >= max_total:
                break

    return cases, duplicate_mapping


def generate_file():
    cases, duplicate_mapping = parse_cases()

    lines = [HEADER]
    for i, (func_name, args_info, line_no) in enumerate(cases, start=1):
        test_name = sanitize_test_name(func_name, i)
        args_literal = repr(args_info)
        lines.append(f"    # Source: {LOG_PATH.name} line {line_no}")
        lines.append(
            TEST_METHOD_TEMPLATE.format(
                test_name=test_name,
                func_name=func_name,
                args_info=args_literal,
            )
        )

    # Add comments for duplicated lines
    if duplicate_mapping:
        if Debug:
            lines.append("\n# Duplicated cases (skipped):")
            for dup_line, orig_line in sorted(duplicate_mapping.items()):
                lines.append(f"# Line {dup_line} -> same as line {orig_line}")

    lines.append(FOOTER)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Generated {OUTPUT_PATH} with {len(cases)} cases ({len(duplicate_mapping)} duplicates skipped)")



if __name__ == "__main__":
    generate_file()
