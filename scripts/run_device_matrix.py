#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from RotaryPosEmb import impl


CASES = [
    ("f32_small", 1, 3, 2, 20, "float32", "small", "random"),
    ("f32_pos0", 1, 8, 1, 16, "float32", "zero", "random"),
    ("f16_pos0", 1, 8, 1, 16, "float16", "zero", "random"),
    ("f32_non_aligned", 2, 16, 3, 24, "float32", "small", "random"),
    ("f16_non_aligned", 2, 16, 3, 24, "float16", "small", "random"),
    ("f32_tail34", 1, 2, 17, 34, "float32", "small", "random"),
    ("f16_tail34", 1, 2, 17, 34, "float16", "small", "random"),
    ("f16_tail62", 2, 7, 3, 62, "float16", "small", "random"),
    ("f32_large_pos", 1, 17, 8, 64, "float32", "high", "random"),
    ("f16_large_pos", 1, 17, 8, 64, "float16", "high", "random"),
    ("f16_tail96", 2, 8, 17, 96, "float16", "high", "random"),
    ("f32_dim1024", 1, 2, 1, 1024, "float32", "small", "random"),
    ("f16_dim1024", 1, 2, 1, 1024, "float16", "small", "random"),
    ("f32_p8192", 1, 1, 1, 64, "float32", "fixed_8192", "random"),
    ("f32_p32768", 1, 1, 1, 64, "float32", "fixed_32768", "random"),
    ("f32_p65535", 1, 1, 1, 64, "float32", "fixed_65535", "random"),
    ("f32_p131072", 1, 1, 1, 64, "float32", "fixed_131072", "random"),
    ("f16_p8192", 1, 1, 1, 64, "float16", "fixed_8192", "random"),
    ("f16_p131072", 1, 1, 1, 64, "float16", "fixed_131072", "random"),
]


def diagnostic_cases():
    positions = [0, 1, 8192, 32768, 65535, 131072]
    cases = []
    for dtype in ("float32", "float16"):
        prefix = "f32" if dtype == "float32" else "f16"
        for position in positions:
            cases.append((
                f"diag_{prefix}_seq_p{position}", 1, 1, 1, 64,
                dtype, f"fixed_{position}", "sequence"))
        for index in (0, 1, 2, 3, 8, 16, 31):
            cases.append((
                f"diag_{prefix}_onehot_i{index}", 1, 1, 1, 64,
                dtype, "fixed_131072", f"onehot_{index}"))
    return cases


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_input(rng, b, s, h, dim, dtype, position_mode, x_mode):
    if x_mode == "random":
        x = rng.uniform(-10.0, 10.0, size=(b, s, h, dim))
    elif x_mode.startswith("onehot_"):
        x = np.zeros((b, s, h, dim), dtype=np.float32)
        index = int(x_mode[len("onehot_"):])
        if index >= dim:
            raise ValueError(f"onehot index {index} is outside dim={dim}")
        x[..., index] = 1.0
    else:
        x = np.arange(b * s * h * dim, dtype=np.float32).reshape(b, s, h, dim)
        x = (x % 97.0) - 48.0
    x = x.astype(np.float16 if dtype == "float16" else np.float32)
    if position_mode == "zero":
        positions = np.zeros((b, s), dtype=np.int32)
    elif position_mode.startswith("fixed_"):
        value = int(position_mode.split("_", 1)[1])
        positions = np.full((b, s), value, dtype=np.int32)
    elif position_mode.startswith("fixed_"):
        position = int(position_mode[len("fixed_"):])
        positions = np.full((b, s), position, dtype=np.int32)
    elif position_mode == "high":
        positions = rng.integers(0, 131073, size=(b, s), dtype=np.int32)
    else:
        positions = rng.integers(0, 50, size=(b, s), dtype=np.int32)
    return x, positions


def verify(output, golden, dtype):
    if dtype == "float16":
        atol, rtol, max_error = 1.95e-3, 1.95e-3, 1e-1
    else:
        atol, rtol, max_error = 9.77e-4, 1.53e-5, 1e-2
    out_f = output.astype(np.float32)
    golden_f = golden.astype(np.float32)
    close = np.isclose(out_f, golden_f, rtol=rtol, atol=atol, equal_nan=True)
    mismatch = np.flatnonzero(~close)
    diff = np.abs(out_f - golden_f)
    ratio = float(mismatch.size) / max(1, golden.size)
    max_diff = float(np.nanmax(diff)) if diff.size else 0.0
    dim_error_count = np.sum(~close, axis=(0, 1, 2))
    dim_max_error = np.max(diff, axis=(0, 1, 2))
    return {
        "passed": mismatch.size / max(1, golden.size) <= 0.01 and max_diff <= max_error,
        "mismatch_count": int(mismatch.size),
        "mismatch_ratio": ratio,
        "max_abs_error": max_diff,
        "first_indices": mismatch[:12].tolist(),
        "mod64": np.bincount(mismatch % 64, minlength=64).tolist() if mismatch.size else [0] * 64,
        "dim_error_count": dim_error_count.astype(np.int64).tolist(),
        "dim_max_error": dim_max_error.astype(np.float32).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("eval_runs"))
    parser.add_argument("--kernel", type=Path, default=Path("kernel.asc"))
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--diagnostic", action="store_true",
                        help="run fixed-position and one-hot diagnostic cases")
    args = parser.parse_args()

    exe = args.exe.resolve()
    workdir = args.workdir.resolve()
    output_root = args.output_root if args.output_root.is_absolute() else workdir / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    kernel_path = args.kernel if args.kernel.is_absolute() else workdir / args.kernel
    case_catalog = diagnostic_cases() if args.diagnostic else CASES
    selected = [case for case in case_catalog if not args.only or case[0] in args.only]
    results = []
    for case_id, b, s, h, dim, dtype, position_mode, x_mode in selected:
        case_dir = output_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(20260927 + sum(ord(c) for c in case_id))
        x, positions = make_input(rng, b, s, h, dim, dtype, position_mode, x_mode)
        golden = impl(x, positions, theta=10000.0)
        input_x = case_dir / "x.bin"
        input_pos = case_dir / "positions.bin"
        output_y = case_dir / "y.bin"
        golden_path = case_dir / "golden.bin"
        x.tofile(input_x)
        positions.tofile(input_pos)
        golden.tofile(golden_path)
        command = [
            str(exe), "--b", str(b), "--s", str(s), "--h", str(h), "--dim", str(dim),
            "--dtype", dtype, "--theta", "10000.0", "--input_x", str(input_x),
            "--input_pos", str(input_pos), "--output_y", str(output_y),
        ]
        proc = subprocess.run(command, cwd=str(workdir), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        result = {"case": case_id, "shape": [b, s, h, dim], "dtype": dtype,
                  "returncode": proc.returncode, "command": command,
                  "kernel_sha256": sha256(kernel_path),
                  "binary_sha256": sha256(exe), "output": proc.stdout}
        if proc.returncode == 0 and output_y.exists():
            output = np.fromfile(output_y, dtype=golden.dtype)
            if output.size == golden.size:
                result.update(verify(output.reshape(golden.shape), golden, dtype))
            else:
                result.update({"passed": False, "size_mismatch": [int(output.size), int(golden.size)]})
        else:
            result["passed"] = False
        results.append(result)
        status = "PASS" if result.get("passed") else "FAIL"
        print(f"{status:4s} {case_id:20s} rc={proc.returncode} "
              f"mismatch={result.get('mismatch_ratio', 1.0) * 100:.4f}% "
              f"max={result.get('max_abs_error', float('nan'))}")
        if result.get("first_indices"):
            print(f"     first_indices={result['first_indices']}")
        if args.diagnostic and result.get("dim_error_count"):
            bad_dims = [
                (index, count, result["dim_max_error"][index])
                for index, count in enumerate(result["dim_error_count"])
                if count
            ]
            if bad_dims:
                print(f"     bad_dims={bad_dims}")
    report = output_root / "report.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Report: {report}")
    return 0 if all(item.get("passed", False) for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
