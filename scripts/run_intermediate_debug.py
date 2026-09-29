#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

import numpy as np


INDICES = np.array([0, 1, 2, 3, 8, 16], dtype=np.int32)


def expected(position: int, theta: float, dim: int):
    index_f = INDICES.astype(np.float32)
    exponent = np.float32(-2.0 / np.float32(dim)) * index_f
    theta_f = np.float32(theta)
    log_theta = np.float32(np.log(theta_f))
    exp_freq = np.exp(np.float32(log_theta * exponent)).astype(np.float32)
    positive_power = np.power(theta_f, -exponent).astype(np.float32)
    reciprocal_freq = np.divide(np.float32(1.0), positive_power).astype(np.float32)
    position_f = np.float32(position)
    exp_angle = np.multiply(position_f, exp_freq).astype(np.float32)
    reciprocal_angle = np.multiply(position_f, reciprocal_freq).astype(np.float32)
    return (
        (exp_freq, exp_angle, np.cos(exp_angle), np.sin(exp_angle)),
        (reciprocal_freq, reciprocal_angle,
         np.cos(reciprocal_angle), np.sin(reciprocal_angle)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--position", type=int, default=131072)
    parser.add_argument("--theta", type=float, default=10000.0)
    args = parser.parse_args()

    run_dir = args.workdir / "debug_intermediate"
    run_dir.mkdir(parents=True, exist_ok=True)
    x_path = run_dir / "x.bin"
    positions_path = run_dir / "positions.bin"
    output_path = run_dir / "y.bin"
    np.zeros((1, 1, 1, 64), dtype=np.float32).tofile(x_path)
    np.array([args.position], dtype=np.int32).tofile(positions_path)
    command = [
        str(args.exe.resolve()), "--b", "1", "--s", "1", "--h", "1", "--dim", "64",
        "--dtype", "float32", "--theta", str(args.theta), "--input_x", str(x_path),
        "--input_pos", str(positions_path), "--output_y", str(output_path),
    ]
    proc = subprocess.run(command, cwd=str(args.workdir), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, end="")
    if proc.returncode != 0:
        return proc.returncode
    actual = np.fromfile(output_path, dtype=np.float32)
    if actual.size < 24:
        print(f"diagnostic output too small: {actual.size} floats")
        return 1
    actual = actual[:24].reshape(4, 6)
    expected_variants = expected(args.position, args.theta, 64)
    labels = ("freq", "angle", "cos", "sin")
    print(f"position={args.position} indices={INDICES.tolist()}")
    for name, values in zip(labels, actual):
        print(f"actual_{name}=" + np.array2string(values, precision=9, floatmode="unique"))
    for variant, values_set in zip(("exp", "positive_power_reciprocal"), expected_variants):
        print(f"expected_{variant}:")
        for row, (name, values) in enumerate(zip(labels, values_set)):
            diff = np.abs(actual[row] - values)
            print(f"  {name}=" + np.array2string(values, precision=9, floatmode="unique") +
                  f" max_abs_diff={float(diff.max()):.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
