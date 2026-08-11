"""
AutoPipeline — 一键完成算子抓取 → 对比 → 参数化 → 清理

Usage:
    python AutoPipeline.py          # Windows
    python3 AutoPipeline.py         # Linux
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

TEMP_DIRS = ["Test_short", "Test_long"]


def step(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run(cmd, **kwargs):
    """Run a command, print output in real time and return CompletedProcess."""
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    return subprocess.run(cmd, cwd=str(SCRIPT_DIR), **kwargs)


def main():
    # ── Step 0: Clean old temp dirs ──
    step("Step 0: Clean old temp files")
    for d in TEMP_DIRS:
        if Path(d).exists():
            shutil.rmtree(d)
            print(f"  Removed {d}/")

    # ── Step 1: Run MainTest.py with short text → Test_short/ ──
    step("Step 1: Capture ops with SHORT text (0)")
    Path("Test_short").mkdir(exist_ok=True)
    with open("Test_short/ops.log", "w", encoding="utf-8") as f:
        subprocess.run(
            [sys.executable, "MainTest.py", "0"],
            stdout=f, stderr=subprocess.STDOUT,
            cwd=str(SCRIPT_DIR), timeout=300
        )
    print("  → Test_short/ops.log")

    # ── Step 2: Run MainTest.py with LONG text → Test_long/ ──
    step("Step 2: Capture ops with LONG text (1)")
    Path("Test_long").mkdir(exist_ok=True)
    with open("Test_long/ops.log", "w", encoding="utf-8") as f:
        subprocess.run(
            [sys.executable, "MainTest.py", "1"],
            stdout=f, stderr=subprocess.STDOUT,
            cwd=str(SCRIPT_DIR), timeout=300
        )
    print("  → Test_long/ops.log")

    # ── Step 3: Generate test_ops_auto.py from each ops.log ──
    step("Step 3: Generate test_ops_auto.py from ops.log")

    # Copy GenerateTestOpsAuto.py + template into each temp dir and run
    for d in TEMP_DIRS:
        target = Path(d)
        shutil.copy(SCRIPT_DIR / "GenerateTestOpsAuto.py", target / "GenerateTestOpsAuto.py")
        shutil.copy(SCRIPT_DIR / "test_ops_auto_template.py", target / "test_ops_auto_template.py")
        subprocess.run(
            [sys.executable, "GenerateTestOpsAuto.py"],
            cwd=str(target), timeout=30
        )
        print(f"  → {d}/test_ops_auto.py")

    # ── Step 4: Compare and generate parameterized result ──
    step("Step 4: CompareAndGen → test_ops_auto_res.py")
    subprocess.run(
        [sys.executable, "CompareAndGen.py",
         "Test_short/test_ops_auto.py",
         "Test_long/test_ops_auto.py",
         "test_ops_auto_res.py"],
        timeout=30
    )

    # ── Step 5: Clean up temp files ──
    step("Step 5: Clean up temporary files")
    for d in TEMP_DIRS:
        if Path(d).exists():
            shutil.rmtree(d)
            print(f"  Removed {d}/")

    print(f"\n{'=' * 60}")
    print(f"  Done!")
    print(f"  Final output:")
    print(f"    test_ops_auto_res.py  — parameterized tests")
    print(f"    run_all_tests.sh      — individual pytest commands")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
