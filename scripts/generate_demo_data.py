#!/usr/bin/env python3
"""Generate a small synthetic bearing-fault demo dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def impulse_train(length: int, fs: float, pulse_hz: float, strength: float, width: int = 8) -> np.ndarray:
    signal = np.zeros(length, dtype=float)
    spacing = max(1, int(fs / pulse_hz))
    pulse = np.hanning(width) * strength
    for start in range(0, length - width, spacing):
        signal[start : start + width] += pulse
    return signal


def build_signal(label: str, fs: float, duration_s: float) -> np.ndarray:
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    shaft = 30.0
    signal = 0.2 * np.sin(2 * np.pi * shaft * t)
    signal += 0.05 * np.sin(2 * np.pi * 2 * shaft * t)
    signal += 0.02 * np.random.default_rng(42).normal(size=n)

    fault_map = {
        "healthy": None,
        "inner": 162.0,
        "outer": 108.0,
        "ball": 72.0,
    }
    pulse_hz = fault_map[label]
    if pulse_hz is not None:
        signal += impulse_train(n, fs, pulse_hz, strength=0.9)
    return signal


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "demo_data"
    fs = 12000.0
    duration_s = 2.0
    for label in ["healthy", "inner", "outer", "ball"]:
        label_dir = root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        signal = build_signal(label, fs, duration_s)
        pd.DataFrame({"signal": signal}).to_csv(label_dir / f"{label}_demo.csv", index=False)
    print(f"Demo data written to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
