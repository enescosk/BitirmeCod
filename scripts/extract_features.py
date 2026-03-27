#!/usr/bin/env python3
"""Bearing fault feature extraction pipeline.

This script builds ML-ready time-domain and frequency-domain features from
vibration-like 1D signals stored in common file formats.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat, wavfile
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".npy", ".npz", ".mat", ".wav"}


@dataclass
class SignalRecord:
    path: Path
    label: str
    signal: np.ndarray


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = ["input_dir", "output_dir", "sampling_rate_hz", "segment_length"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    config.setdefault("segment_overlap", 0.5)
    config.setdefault("label_strategy", "parent_dir")
    config.setdefault("signal_column", None)
    config.setdefault("harmonics", 3)
    config.setdefault("fault_band_half_width_hz", 5.0)
    config.setdefault(
        "band_energy_hz",
        [[0.0, 500.0], [500.0, 1500.0], [1500.0, 3000.0]],
    )
    config.setdefault("max_plots_per_label", 1)
    return config


def normalize_signal(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim > 1:
        if 1 in array.shape:
            array = array.reshape(-1)
        else:
            # Prefer the first numeric channel when there are multiple columns.
            array = array[:, 0]
    return array.astype(float, copy=False)


def first_numeric_series(frame: pd.DataFrame, requested_column: str | None) -> np.ndarray:
    if requested_column and requested_column in frame.columns:
        return normalize_signal(frame[requested_column].to_numpy())
    numeric = frame.select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError("No numeric signal column found")
    return normalize_signal(numeric.iloc[:, 0].to_numpy())


def load_signal(path: Path, signal_column: str | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return first_numeric_series(pd.read_csv(path), signal_column)
    if suffix == ".txt":
        try:
            return first_numeric_series(pd.read_csv(path, sep=None, engine="python"), signal_column)
        except Exception:
            return normalize_signal(np.loadtxt(path))
    if suffix == ".npy":
        return normalize_signal(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        keys = list(archive.keys())
        if not keys:
            raise ValueError("NPZ archive is empty")
        return normalize_signal(archive[keys[0]])
    if suffix == ".mat":
        data = loadmat(path)
        preferred = [
            "signal",
            "vibration",
            "data",
            "X",
            "x",
            "DE_time",
            "FE_time",
        ]
        for key in preferred:
            if key in data:
                return normalize_signal(data[key])
        for key, value in data.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value)
            if np.issubdtype(arr.dtype, np.number):
                return normalize_signal(arr)
        raise ValueError("No numeric MAT array found")
    if suffix == ".wav":
        _, signal = wavfile.read(path)
        return normalize_signal(signal)
    raise ValueError(f"Unsupported file type: {suffix}")


def infer_label(path: Path, input_root: Path, strategy: str) -> str:
    if strategy == "filename":
        stem = path.stem.lower()
        for candidate in ["healthy", "normal", "inner", "outer", "ball", "cage", "fault"]:
            if candidate in stem:
                return candidate
        return path.stem
    if strategy == "relative_dir":
        rel = path.relative_to(input_root)
        parts = rel.parts[:-1]
        return "__".join(parts) if parts else "unlabeled"
    parent = path.parent.name
    return parent if parent else "unlabeled"


def iter_signal_files(input_root: Path) -> Iterable[Path]:
    if input_root.is_file() and input_root.suffix.lower() in SUPPORTED_EXTENSIONS:
        yield input_root
        return
    for path in sorted(input_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_records(config: dict) -> list[SignalRecord]:
    input_root = Path(config["input_dir"]).expanduser().resolve()
    signal_column = config.get("signal_column")
    strategy = config.get("label_strategy", "parent_dir")
    records: list[SignalRecord] = []
    for path in iter_signal_files(input_root):
        try:
            signal = load_signal(path, signal_column)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue
        if signal.size == 0:
            print(f"Skipping {path}: empty signal")
            continue
        label = infer_label(path, input_root, strategy)
        records.append(SignalRecord(path=path, label=label, signal=signal))
    return records


def segment_signal(signal: np.ndarray, segment_length: int, segment_overlap: float) -> list[tuple[int, np.ndarray]]:
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    step = max(1, int(segment_length * (1.0 - segment_overlap)))
    if signal.size < segment_length:
        return []
    segments = []
    for start in range(0, signal.size - segment_length + 1, step):
        window = signal[start : start + segment_length]
        segments.append((start, window))
    return segments


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def shannon_entropy(signal: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def time_domain_features(signal: np.ndarray) -> dict[str, float]:
    mean_abs = float(np.mean(np.abs(signal)))
    peak_abs = float(np.max(np.abs(signal)))
    signal_rms = rms(signal)
    sqrt_abs_mean = float(np.mean(np.sqrt(np.abs(signal)))) if np.any(signal) else 0.0
    clearance_denom = sqrt_abs_mean ** 2 if sqrt_abs_mean else 0.0
    features = {
        "td_mean": float(np.mean(signal)),
        "td_std": float(np.std(signal)),
        "td_variance": float(np.var(signal)),
        "td_rms": signal_rms,
        "td_peak_abs": peak_abs,
        "td_peak_to_peak": float(np.ptp(signal)),
        "td_skewness": float(skew(signal, bias=False)),
        "td_kurtosis": float(kurtosis(signal, fisher=False, bias=False)),
        "td_crest_factor": peak_abs / signal_rms if signal_rms else 0.0,
        "td_shape_factor": signal_rms / mean_abs if mean_abs else 0.0,
        "td_impulse_factor": peak_abs / mean_abs if mean_abs else 0.0,
        "td_clearance_factor": peak_abs / clearance_denom if clearance_denom else 0.0,
        "td_energy": float(np.sum(np.square(signal))),
        "td_entropy": shannon_entropy(signal),
    }
    return features


def spectral_moments(freqs: np.ndarray, amplitudes: np.ndarray) -> tuple[float, float]:
    weight_sum = float(np.sum(amplitudes))
    if weight_sum == 0.0:
        return 0.0, 0.0
    centroid = float(np.sum(freqs * amplitudes) / weight_sum)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * amplitudes) / weight_sum))
    return centroid, bandwidth


def band_energy(freqs: np.ndarray, power: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def dominant_frequency(freqs: np.ndarray, amplitudes: np.ndarray) -> float:
    if amplitudes.size <= 1:
        return 0.0
    index = int(np.argmax(amplitudes[1:]) + 1)
    return float(freqs[index])


def amplitude_near(freqs: np.ndarray, amplitudes: np.ndarray, target_hz: float, half_width_hz: float) -> float:
    mask = (freqs >= max(0.0, target_hz - half_width_hz)) & (freqs <= target_hz + half_width_hz)
    if not np.any(mask):
        return 0.0
    return float(np.max(amplitudes[mask]))


def compute_fault_frequencies(config: dict) -> dict[str, float]:
    if "fault_frequencies_hz" in config and config["fault_frequencies_hz"]:
        return {key.upper(): float(value) for key, value in config["fault_frequencies_hz"].items()}

    geometry = config.get("bearing_geometry")
    shaft_frequency_hz = config.get("shaft_frequency_hz")
    if shaft_frequency_hz is None and "rpm" in config:
        shaft_frequency_hz = float(config["rpm"]) / 60.0
    if not geometry or shaft_frequency_hz is None:
        return {}

    num_balls = float(geometry["num_balls"])
    ball_diameter = float(geometry["ball_diameter"])
    pitch_diameter = float(geometry["pitch_diameter"])
    contact_angle_deg = float(geometry.get("contact_angle_deg", 0.0))
    cos_theta = math.cos(math.radians(contact_angle_deg))
    ratio = (ball_diameter / pitch_diameter) * cos_theta
    fr = float(shaft_frequency_hz)
    return {
        "FTF": 0.5 * fr * (1.0 - ratio),
        "BPFO": 0.5 * num_balls * fr * (1.0 - ratio),
        "BPFI": 0.5 * num_balls * fr * (1.0 + ratio),
        "BSF": 0.5 * fr * (pitch_diameter / ball_diameter) * (1.0 - ratio**2),
    }


def frequency_domain_features(signal: np.ndarray, sampling_rate_hz: float, config: dict) -> dict[str, float]:
    signal = signal - np.mean(signal)
    window = np.hanning(signal.size)
    fft = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sampling_rate_hz)
    amplitudes = np.abs(fft) / signal.size
    power = np.square(amplitudes)

    analytic = hilbert(signal)
    envelope = np.abs(analytic) - np.mean(np.abs(analytic))
    envelope_fft = np.fft.rfft(envelope * window)
    envelope_amplitudes = np.abs(envelope_fft) / signal.size

    centroid, bandwidth = spectral_moments(freqs, amplitudes)
    features = {
        "fd_dominant_freq_hz": dominant_frequency(freqs, amplitudes),
        "fd_spectral_centroid_hz": centroid,
        "fd_spectral_bandwidth_hz": bandwidth,
        "fd_total_band_energy": float(np.sum(power)),
        "fd_envelope_dominant_freq_hz": dominant_frequency(freqs, envelope_amplitudes),
    }

    for low_hz, high_hz in config.get("band_energy_hz", []):
        key = f"fd_band_energy_{int(low_hz)}_{int(high_hz)}_hz"
        features[key] = band_energy(freqs, power, float(low_hz), float(high_hz))

    fault_freqs = compute_fault_frequencies(config)
    half_width_hz = float(config.get("fault_band_half_width_hz", 5.0))
    harmonics = int(config.get("harmonics", 3))
    for fault_name, base_hz in fault_freqs.items():
        features[f"fd_{fault_name.lower()}_hz"] = float(base_hz)
        for harmonic in range(1, harmonics + 1):
            target_hz = base_hz * harmonic
            features[f"fd_{fault_name.lower()}_h{harmonic}_fft_amp"] = amplitude_near(
                freqs, amplitudes, target_hz, half_width_hz
            )
            features[f"fd_{fault_name.lower()}_h{harmonic}_envelope_amp"] = amplitude_near(
                freqs, envelope_amplitudes, target_hz, half_width_hz
            )

    return features


def save_plot(
    output_dir: Path,
    label: str,
    file_stem: str,
    segment_index: int,
    signal: np.ndarray,
    sampling_rate_hz: float,
    config: dict,
) -> None:
    signal_centered = signal - np.mean(signal)
    time_axis = np.arange(signal.size) / sampling_rate_hz
    fft = np.fft.rfft(signal_centered * np.hanning(signal.size))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sampling_rate_hz)
    amplitudes = np.abs(fft) / signal.size
    envelope = np.abs(hilbert(signal_centered)) - np.mean(np.abs(hilbert(signal_centered)))
    envelope_fft = np.fft.rfft(envelope * np.hanning(signal.size))
    envelope_amplitudes = np.abs(envelope_fft) / signal.size

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(time_axis, signal, linewidth=1.0)
    axes[0].set_title(f"Time Signal - {label}")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(freqs, amplitudes, linewidth=1.0)
    axes[1].set_title("FFT Spectrum")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Amplitude")

    axes[2].plot(freqs, envelope_amplitudes, linewidth=1.0)
    axes[2].set_title("Envelope Spectrum")
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Amplitude")

    for fault_name, fault_hz in compute_fault_frequencies(config).items():
        for ax in axes[1:]:
            ax.axvline(fault_hz, color="tab:red", linestyle="--", alpha=0.5)
            ax.text(fault_hz, ax.get_ylim()[1] * 0.9, fault_name, rotation=90, fontsize=8)

    fig.tight_layout()
    plot_dir = output_dir / "plots" / label
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"{file_stem}_segment_{segment_index}.png", dpi=150)
    plt.close(fig)


def build_feature_tables(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = load_records(config)
    if not records:
        raise ValueError(
            "No supported signal files found. Put dataset files under input_dir or update the config."
        )

    sampling_rate_hz = float(config["sampling_rate_hz"])
    segment_length = int(config["segment_length"])
    segment_overlap = float(config.get("segment_overlap", 0.5))
    output_rows = []
    plots_per_label: dict[str, int] = {}
    output_dir = Path(config["output_dir"]).expanduser().resolve()

    for record in records:
        segments = segment_signal(record.signal, segment_length, segment_overlap)
        for segment_index, (start, segment) in enumerate(segments):
            metadata = {
                "label": record.label,
                "source_file": str(record.path),
                "segment_index": segment_index,
                "start_sample": start,
                "end_sample": start + segment_length,
            }
            td = time_domain_features(segment)
            fd = frequency_domain_features(segment, sampling_rate_hz, config)
            row = metadata | td | fd
            output_rows.append(row)

            if plots_per_label.get(record.label, 0) < int(config.get("max_plots_per_label", 1)):
                save_plot(
                    output_dir=output_dir,
                    label=record.label,
                    file_stem=record.path.stem,
                    segment_index=segment_index,
                    signal=segment,
                    sampling_rate_hz=sampling_rate_hz,
                    config=config,
                )
                plots_per_label[record.label] = plots_per_label.get(record.label, 0) + 1

    combined = pd.DataFrame(output_rows)
    metadata_cols = ["label", "source_file", "segment_index", "start_sample", "end_sample"]
    time_cols = metadata_cols + [column for column in combined.columns if column.startswith("td_")]
    freq_cols = metadata_cols + [column for column in combined.columns if column.startswith("fd_")]
    return combined, combined[time_cols], combined[freq_cols]


def save_outputs(config: dict, combined: pd.DataFrame, time_df: pd.DataFrame, freq_df: pd.DataFrame) -> None:
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined.to_csv(output_dir / "combined_features.csv", index=False)
    time_df.to_csv(output_dir / "time_domain_features.csv", index=False)
    freq_df.to_csv(output_dir / "frequency_domain_features.csv", index=False)

    summary = {
        "config_path": config.get("_config_path"),
        "input_dir": config["input_dir"],
        "output_dir": str(output_dir),
        "sampling_rate_hz": config["sampling_rate_hz"],
        "segment_length": config["segment_length"],
        "segment_overlap": config.get("segment_overlap", 0.5),
        "fault_frequencies_hz": compute_fault_frequencies(config),
        "num_rows": int(len(combined)),
        "labels": sorted(combined["label"].dropna().astype(str).unique().tolist()),
        "num_time_features": int(len([c for c in combined.columns if c.startswith("td_")])),
        "num_frequency_features": int(len([c for c in combined.columns if c.startswith("fd_")])),
        "has_missing_values": bool(combined.isna().any().any()),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract bearing-fault features from 1D signal files.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a JSON config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config["_config_path"] = str(args.config.resolve())
    combined, time_df, freq_df = build_feature_tables(config)
    save_outputs(config, combined, time_df, freq_df)
    print(f"Saved {len(combined)} segments to {config['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
