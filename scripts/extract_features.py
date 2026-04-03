#!/usr/bin/env python3
"""Bearing fault feature extraction pipeline.

This script builds ML-ready time-domain and frequency-domain features from
vibration-like 1D signals stored in common file formats.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat, wavfile
from scipy.signal import find_peaks, hilbert, stft, welch
from scipy.stats import kurtosis, skew


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".npy", ".npz", ".mat", ".wav"}
DEFAULT_MAT_PRIORITIES = ["de_time", "fe_time", "ba_time", "signal", "vibration", "data", "x"]


@dataclass
class SignalRecord:
    path: Path
    label: str
    signal: np.ndarray
    source_metadata: dict[str, object]


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
    config.setdefault("band_energy_hz", [[0.0, 500.0], [500.0, 2000.0], [2000.0, None]])
    config.setdefault("max_plots_per_label", 1)
    config.setdefault("save_stft_artifacts", True)
    config.setdefault("max_stft_artifacts_per_file", 1)
    config.setdefault("stft_nperseg", 256)
    config.setdefault("stft_noverlap", 128)
    config.setdefault("welch_nperseg", 1024)
    config.setdefault("infer_sampling_rate", False)
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


def preferred_mat_priorities(path: Path) -> list[str]:
    path_text = str(path).lower()
    if "fan" in path_text:
        return ["fe_time", "de_time", "ba_time", "signal", "vibration", "data", "x"]
    if "drive" in path_text or "48k" in path_text:
        return DEFAULT_MAT_PRIORITIES.copy()

    file_id = extract_numeric_file_id(path)
    fan_end_ids = (
        set(range(270, 303))
        | {305, 306, 307}
        | set(range(309, 314))
        | {315, 316, 317, 318}
    )
    if file_id in fan_end_ids:
        return ["fe_time", "de_time", "ba_time", "signal", "vibration", "data", "x"]
    return DEFAULT_MAT_PRIORITIES.copy()


def pick_mat_signal_key(data: dict[str, object], priorities: list[str] | None = None) -> str | None:
    priorities = priorities or DEFAULT_MAT_PRIORITIES
    visible_keys = [key for key in data if not key.startswith("__")]
    for token in priorities:
        for key in visible_keys:
            if token in key.lower():
                array = normalize_signal(np.asarray(data[key]))
                if array.size > 1000:
                    return key
    for key in visible_keys:
        array = np.asarray(data[key])
        if np.issubdtype(array.dtype, np.number):
            signal = normalize_signal(array)
            if signal.size > 1000:
                return key
    return None


def load_signal(path: Path, signal_column: str | None) -> tuple[np.ndarray, dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return first_numeric_series(pd.read_csv(path), signal_column), {}
    if suffix == ".txt":
        try:
            return first_numeric_series(pd.read_csv(path, sep=None, engine="python"), signal_column), {}
        except Exception:
            return normalize_signal(np.loadtxt(path)), {}
    if suffix == ".npy":
        return normalize_signal(np.load(path, allow_pickle=False)), {}
    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        keys = list(archive.keys())
        if not keys:
            raise ValueError("NPZ archive is empty")
        return normalize_signal(archive[keys[0]]), {"signal_key": keys[0]}
    if suffix == ".mat":
        data = loadmat(path)
        mat_priorities = preferred_mat_priorities(path)
        if signal_column and signal_column in data:
            signal_key = signal_column
        else:
            signal_key = pick_mat_signal_key(data, priorities=mat_priorities)
        if signal_key is None:
            raise ValueError("No numeric MAT array found")
        rpm_key = next((key for key in data if key.endswith("RPM")), None)
        metadata: dict[str, object] = {"signal_key": signal_key, "mat_priorities": mat_priorities}
        if rpm_key is not None:
            metadata["rpm"] = float(np.asarray(data[rpm_key]).reshape(-1)[0])
        return normalize_signal(data[signal_key]), metadata
    if suffix == ".wav":
        _, signal = wavfile.read(path)
        return normalize_signal(signal), {}
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


def infer_bearing_type(path: Path, signal_key: str | None) -> str:
    text = f"{path} {signal_key or ''}".lower()
    if "de_time" in text or "drive" in text:
        return "DRIVE_END"
    if "fe_time" in text or "fan" in text:
        return "FAN_END"
    return "UNKNOWN"


def extract_numeric_file_id(path: Path) -> int | None:
    return int(path.stem) if path.stem.isdigit() else None


def parse_named_cwru_stem(stem: str, path: Path, signal_key: str | None) -> dict[str, object] | None:
    bearing_type = infer_bearing_type(path, signal_key)
    normal_match = re.fullmatch(r"Normal_(\d+)", stem, flags=re.IGNORECASE)
    if normal_match:
        return {
            "output_fault_status": "NO_FAULT",
            "output_bearing_type": "NONE",
            "output_fault_location": "NONE",
            "output_fault_diameter_in": 0.0,
            "output_motor_load_hp": int(normal_match.group(1)),
            "output_outer_race_position": "NONE",
        }

    fault_match = re.fullmatch(r"(IR|OR|B)(\d{3})(?:@(\d+))?_(\d+)", stem, flags=re.IGNORECASE)
    if not fault_match:
        return None

    location_code = fault_match.group(1).upper()
    diameter_code = fault_match.group(2)
    outer_position = fault_match.group(3)
    load_hp = int(fault_match.group(4))
    location_map = {"IR": "INNER_RACE", "OR": "OUTER_RACE", "B": "BALL"}
    return {
        "output_fault_status": "FAULT",
        "output_bearing_type": bearing_type,
        "output_fault_location": location_map[location_code],
        "output_fault_diameter_in": float(f"0.{diameter_code}"),
        "output_motor_load_hp": load_hp,
        "output_outer_race_position": f"@{outer_position}:00" if outer_position else "NONE",
    }


def parse_numeric_cwru_id(stem: str, path: Path, signal_key: str | None) -> dict[str, object] | None:
    if not stem.isdigit():
        return None
    file_id = int(stem)
    bearing_type = infer_bearing_type(path, signal_key)
    blocks = [
        (97, 100, "NO_FAULT", "NONE", 0.0, "NONE"),
        (105, 108, "FAULT", "INNER_RACE", 0.007, "NONE"),
        (118, 121, "FAULT", "BALL", 0.007, "NONE"),
        (130, 133, "FAULT", "OUTER_RACE", 0.007, "@6:00"),
        (169, 172, "FAULT", "INNER_RACE", 0.014, "NONE"),
        (185, 188, "FAULT", "BALL", 0.014, "NONE"),
        (197, 200, "FAULT", "OUTER_RACE", 0.014, "@6:00"),
        (209, 212, "FAULT", "INNER_RACE", 0.021, "NONE"),
        (222, 225, "FAULT", "BALL", 0.021, "NONE"),
        (234, 237, "FAULT", "OUTER_RACE", 0.021, "@6:00"),
        (246, 249, "FAULT", "INNER_RACE", 0.028, "NONE"),
        (3005, 3008, "FAULT", "BALL", 0.028, "NONE"),
    ]
    for start, end, fault_status, fault_location, diameter, outer_position in blocks:
        if start <= file_id <= end:
            return {
                "output_fault_status": fault_status,
                "output_bearing_type": "NONE" if fault_status == "NO_FAULT" else bearing_type,
                "output_fault_location": fault_location,
                "output_fault_diameter_in": diameter,
                "output_motor_load_hp": file_id - start,
                "output_outer_race_position": outer_position,
            }
    return None


def parse_extended_cwru_id(path: Path, signal_key: str | None) -> dict[str, object] | None:
    file_id = extract_numeric_file_id(path)
    if file_id is None:
        return None

    bearing_type = infer_bearing_type(path, signal_key)
    mappings: list[tuple[set[int], str, str, float, str, dict[int, int] | None]] = [
        (set(range(97, 101)), "NO_FAULT", "NONE", 0.0, "NONE", {97: 0, 98: 1, 99: 2, 100: 3}),
        (set(range(105, 109)), "FAULT", "INNER_RACE", 0.007, "NONE", {105: 0, 106: 1, 107: 2, 108: 3}),
        (set(range(109, 113)), "FAULT", "INNER_RACE", 0.007, "NONE", {109: 0, 110: 1, 111: 2, 112: 3}),
        (set(range(118, 122)), "FAULT", "BALL", 0.007, "NONE", {118: 0, 119: 1, 120: 2, 121: 3}),
        (set(range(122, 126)), "FAULT", "BALL", 0.007, "NONE", {122: 0, 123: 1, 124: 2, 125: 3}),
        (set(range(130, 134)), "FAULT", "OUTER_RACE", 0.007, "@6:00", {130: 0, 131: 1, 132: 2, 133: 3}),
        (set(range(135, 139)), "FAULT", "OUTER_RACE", 0.007, "@6:00", {135: 0, 136: 1, 137: 2, 138: 3}),
        (set(range(148, 152)), "FAULT", "OUTER_RACE", 0.007, "@3:00", {148: 0, 149: 1, 150: 2, 151: 3}),
        (set(range(161, 165)), "FAULT", "OUTER_RACE", 0.007, "@12:00", {161: 0, 162: 1, 163: 2, 164: 3}),
        (set(range(169, 173)), "FAULT", "INNER_RACE", 0.014, "NONE", {169: 0, 170: 1, 171: 2, 172: 3}),
        (set(range(174, 178)), "FAULT", "INNER_RACE", 0.014, "NONE", {174: 0, 175: 1, 176: 2, 177: 3}),
        (set(range(185, 189)), "FAULT", "BALL", 0.014, "NONE", {185: 0, 186: 1, 187: 2, 188: 3}),
        (set(range(189, 193)), "FAULT", "BALL", 0.014, "NONE", {189: 0, 190: 1, 191: 2, 192: 3}),
        (set(range(197, 201)), "FAULT", "OUTER_RACE", 0.014, "@6:00", {197: 0, 198: 1, 199: 2, 200: 3}),
        (set(range(201, 205)), "FAULT", "OUTER_RACE", 0.014, "@6:00", {201: 0, 202: 1, 203: 2, 204: 3}),
        ({209, 210, 211, 212}, "FAULT", "INNER_RACE", 0.021, "NONE", {209: 0, 210: 1, 211: 2, 212: 3}),
        ({213, 214, 215, 217}, "FAULT", "INNER_RACE", 0.021, "NONE", {213: 0, 214: 1, 215: 2, 217: 3}),
        ({222, 223, 224, 225}, "FAULT", "BALL", 0.021, "NONE", {222: 0, 223: 1, 224: 2, 225: 3}),
        ({226, 227, 228, 229}, "FAULT", "BALL", 0.021, "NONE", {226: 0, 227: 1, 228: 2, 229: 3}),
        ({234, 235, 236, 237}, "FAULT", "OUTER_RACE", 0.021, "@6:00", {234: 0, 235: 1, 236: 2, 237: 3}),
        ({238, 239, 240, 241}, "FAULT", "OUTER_RACE", 0.021, "@6:00", {238: 0, 239: 1, 240: 2, 241: 3}),
        ({246, 247, 248, 249}, "FAULT", "INNER_RACE", 0.028, "NONE", {246: 0, 247: 1, 248: 2, 249: 3}),
        ({250, 251, 252, 253}, "FAULT", "OUTER_RACE", 0.021, "@3:00", {250: 0, 251: 1, 252: 2, 253: 3}),
        ({262, 263, 264, 265}, "FAULT", "OUTER_RACE", 0.021, "@12:00", {262: 0, 263: 1, 264: 2, 265: 3}),
        ({270, 271, 272, 273}, "FAULT", "INNER_RACE", 0.021, "NONE", {270: 0, 271: 1, 272: 2, 273: 3}),
        ({274, 275, 276, 277}, "FAULT", "INNER_RACE", 0.014, "NONE", {274: 0, 275: 1, 276: 2, 277: 3}),
        ({278, 279, 280, 281}, "FAULT", "INNER_RACE", 0.007, "NONE", {278: 0, 279: 1, 280: 2, 281: 3}),
        ({282, 283, 284, 285}, "FAULT", "BALL", 0.007, "NONE", {282: 0, 283: 1, 284: 2, 285: 3}),
        ({286, 287, 288, 289}, "FAULT", "BALL", 0.014, "NONE", {286: 0, 287: 1, 288: 2, 289: 3}),
        ({290, 291, 292, 293}, "FAULT", "BALL", 0.021, "NONE", {290: 0, 291: 1, 292: 2, 293: 3}),
        ({294, 295, 296, 297}, "FAULT", "OUTER_RACE", 0.007, "@6:00", {294: 0, 295: 1, 296: 2, 297: 3}),
        ({298, 299, 300, 301}, "FAULT", "OUTER_RACE", 0.007, "@3:00", {298: 0, 299: 1, 300: 2, 301: 3}),
        ({302, 305, 306, 307}, "FAULT", "OUTER_RACE", 0.007, "@12:00", {302: 0, 305: 1, 306: 2, 307: 3}),
        ({309, 310, 311, 312}, "FAULT", "OUTER_RACE", 0.014, "@3:00", {310: 0, 309: 1, 311: 2, 312: 3}),
        ({313}, "FAULT", "OUTER_RACE", 0.014, "@6:00", {313: 0}),
        ({315}, "FAULT", "OUTER_RACE", 0.021, "@6:00", {315: 0}),
        ({316, 317, 318}, "FAULT", "OUTER_RACE", 0.021, "@3:00", {316: 1, 317: 2, 318: 3}),
        ({3005, 3006, 3007, 3008}, "FAULT", "BALL", 0.028, "NONE", {3005: 0, 3006: 1, 3007: 2, 3008: 3}),
    ]

    for file_ids, fault_status, fault_location, diameter, outer_position, load_map in mappings:
        if file_id in file_ids:
            return {
                "output_fault_status": fault_status,
                "output_bearing_type": "NONE" if fault_status == "NO_FAULT" else bearing_type,
                "output_fault_location": fault_location,
                "output_fault_diameter_in": diameter,
                "output_motor_load_hp": load_map[file_id] if load_map else np.nan,
                "output_outer_race_position": outer_position,
            }
    return None


def infer_outputs(path: Path, label: str, signal_key: str | None) -> dict[str, object]:
    stem = path.stem
    outputs = (
        parse_named_cwru_stem(stem, path, signal_key)
        or parse_extended_cwru_id(path, signal_key)
        or parse_numeric_cwru_id(stem, path, signal_key)
    )
    if outputs is not None:
        return outputs

    label_lower = label.lower()
    if label_lower in {"healthy", "normal"}:
        return {
            "output_fault_status": "NO_FAULT",
            "output_bearing_type": "NONE",
            "output_fault_location": "NONE",
            "output_fault_diameter_in": 0.0,
            "output_motor_load_hp": np.nan,
            "output_outer_race_position": "NONE",
        }

    location_map = {
        "inner": "INNER_RACE",
        "outer": "OUTER_RACE",
        "ball": "BALL",
        "cage": "CAGE",
    }
    return {
        "output_fault_status": "FAULT" if label_lower in location_map else "UNKNOWN",
        "output_bearing_type": infer_bearing_type(path, signal_key),
        "output_fault_location": location_map.get(label_lower, "UNKNOWN"),
        "output_fault_diameter_in": np.nan,
        "output_motor_load_hp": np.nan,
        "output_outer_race_position": "NONE",
    }


def infer_rpm(source_metadata: dict[str, object], outputs: dict[str, object]) -> float:
    rpm = source_metadata.get("rpm")
    if rpm is not None and not np.isnan(rpm):
        return float(rpm)
    load_hp = outputs.get("output_motor_load_hp")
    approximate_rpm_by_load = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}
    if load_hp in approximate_rpm_by_load:
        return approximate_rpm_by_load[int(load_hp)]
    return np.nan


def infer_sampling_rate(path: Path, default_sampling_rate_hz: float) -> float:
    path_text = str(path).lower()
    if "48k" in path_text or "48000" in path_text:
        return 48000.0
    file_id = extract_numeric_file_id(path)
    if file_id is None:
        return float(default_sampling_rate_hz)
    forty_eight_k_ids = (
        set(range(109, 113))
        | set(range(122, 126))
        | set(range(135, 139))
        | set(range(148, 152))
        | set(range(161, 165))
        | set(range(174, 178))
        | set(range(189, 193))
        | set(range(201, 205))
        | {213, 214, 215, 217}
        | {226, 227, 228, 229}
        | {238, 239, 240, 241}
        | {250, 251, 252, 253}
        | {262, 263, 264, 265}
    )
    if file_id in forty_eight_k_ids:
        return 48000.0
    return float(default_sampling_rate_hz)


def derive_class_label(outputs: dict[str, object], fallback_label: str) -> str:
    if outputs.get("output_fault_status") == "NO_FAULT":
        return "healthy"
    location = outputs.get("output_fault_location")
    mapping = {
        "INNER_RACE": "inner_race",
        "OUTER_RACE": "outer_race",
        "BALL": "ball",
        "CAGE": "cage",
    }
    return mapping.get(location, fallback_label)


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
            signal, source_metadata = load_signal(path, signal_column)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue
        if signal.size == 0:
            print(f"Skipping {path}: empty signal")
            continue
        label = infer_label(path, input_root, strategy)
        source_metadata["outputs"] = infer_outputs(path, label, source_metadata.get("signal_key"))
        records.append(SignalRecord(path=path, label=label, signal=signal, source_metadata=source_metadata))
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


def psd_band_energy(freqs: np.ndarray, density: np.ndarray, low_hz: float, high_hz: float | None) -> float:
    if high_hz is None:
        mask = freqs >= low_hz
    else:
        mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(density[mask], freqs[mask]))


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


def top_spectrum_peaks(freqs: np.ndarray, amplitudes: np.ndarray, top_n: int = 5) -> dict[str, float]:
    if amplitudes.size <= 1:
        return {f"fd_peak{i}_freq_hz": 0.0 for i in range(1, top_n + 1)} | {
            f"fd_peak{i}_amp": 0.0 for i in range(1, top_n + 1)
        }

    peak_indices, _ = find_peaks(amplitudes[1:])
    peak_indices = peak_indices + 1
    if peak_indices.size == 0:
        ranked = np.argsort(amplitudes[1:])[::-1][:top_n] + 1
    else:
        ranked = peak_indices[np.argsort(amplitudes[peak_indices])[::-1][:top_n]]

    if ranked.size < top_n:
        existing = set(int(index) for index in ranked.tolist())
        for index in (np.argsort(amplitudes[1:])[::-1] + 1).tolist():
            if int(index) not in existing:
                ranked = np.append(ranked, index)
                existing.add(int(index))
            if ranked.size >= top_n:
                break

    ranked = ranked[:top_n]
    features: dict[str, float] = {}
    for peak_number, index in enumerate(ranked, start=1):
        features[f"fd_peak{peak_number}_freq_hz"] = float(freqs[int(index)])
        features[f"fd_peak{peak_number}_amp"] = float(amplitudes[int(index)])
    return features


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
    welch_freqs, welch_density = welch(
        signal,
        fs=sampling_rate_hz,
        nperseg=min(int(config.get("welch_nperseg", 1024)), signal.size),
    )

    analytic = hilbert(signal)
    envelope = np.abs(analytic) - np.mean(np.abs(analytic))
    envelope_fft = np.fft.rfft(envelope * window)
    envelope_amplitudes = np.abs(envelope_fft) / signal.size

    centroid, bandwidth = spectral_moments(freqs, amplitudes)
    features = {
        "fd_dominant_freq_hz": dominant_frequency(freqs, amplitudes),
        "fd_spectral_centroid_hz": centroid,
        "fd_spectral_bandwidth_hz": bandwidth,
        "fd_total_spectral_energy": float(np.sum(power)),
        "fd_envelope_dominant_freq_hz": dominant_frequency(freqs, envelope_amplitudes),
    }

    for low_hz, high_hz in config.get("band_energy_hz", []):
        low_value = float(low_hz)
        high_value = None if high_hz is None else float(high_hz)
        if high_value is None:
            key = f"fd_band_energy_{int(low_value)}_plus_hz"
        else:
            key = f"fd_band_energy_{int(low_value)}_{int(high_value)}_hz"
        features[key] = psd_band_energy(welch_freqs, welch_density, low_value, high_value)

    features.update(top_spectrum_peaks(freqs, amplitudes, top_n=5))

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


def stft_summary_features(signal: np.ndarray, sampling_rate_hz: float, config: dict) -> tuple[dict[str, float], pd.DataFrame]:
    stft_nperseg = min(int(config.get("stft_nperseg", 256)), signal.size)
    stft_noverlap = min(int(config.get("stft_noverlap", 128)), max(0, stft_nperseg - 1))
    freqs, times, zxx = stft(
        signal,
        fs=sampling_rate_hz,
        nperseg=stft_nperseg,
        noverlap=stft_noverlap,
        boundary=None,
    )
    magnitude = np.abs(zxx)
    power = magnitude**2
    energy_over_time = np.sum(power, axis=0)
    max_freq_indices = np.argmax(magnitude, axis=0) if magnitude.size else np.array([], dtype=int)
    max_freq_over_time = freqs[max_freq_indices] if max_freq_indices.size else np.array([], dtype=float)

    if times.size > 1:
        energy_slope = float(np.polyfit(times, energy_over_time, 1)[0])
        max_freq_slope = float(np.polyfit(times, max_freq_over_time, 1)[0])
    else:
        energy_slope = 0.0
        max_freq_slope = 0.0

    features = {
        "tf_stft_energy_mean": float(np.mean(energy_over_time)) if energy_over_time.size else 0.0,
        "tf_stft_energy_std": float(np.std(energy_over_time)) if energy_over_time.size else 0.0,
        "tf_stft_energy_max": float(np.max(energy_over_time)) if energy_over_time.size else 0.0,
        "tf_stft_energy_min": float(np.min(energy_over_time)) if energy_over_time.size else 0.0,
        "tf_stft_energy_slope": energy_slope,
        "tf_stft_max_freq_mean_hz": float(np.mean(max_freq_over_time)) if max_freq_over_time.size else 0.0,
        "tf_stft_max_freq_std_hz": float(np.std(max_freq_over_time)) if max_freq_over_time.size else 0.0,
        "tf_stft_max_freq_max_hz": float(np.max(max_freq_over_time)) if max_freq_over_time.size else 0.0,
        "tf_stft_max_freq_min_hz": float(np.min(max_freq_over_time)) if max_freq_over_time.size else 0.0,
        "tf_stft_max_freq_slope_hz_per_s": max_freq_slope,
    }
    series_df = pd.DataFrame(
        {
            "time_sec": times,
            "stft_energy": energy_over_time,
            "stft_max_freq_hz": max_freq_over_time,
        }
    )
    return features, series_df


def save_stft_artifacts(
    output_dir: Path,
    label: str,
    file_stem: str,
    segment_index: int,
    signal: np.ndarray,
    sampling_rate_hz: float,
    config: dict,
    series_df: pd.DataFrame,
) -> tuple[str, str]:
    stft_dir = output_dir / "stft"
    plot_dir = stft_dir / "plots" / label
    series_dir = stft_dir / "series" / label
    plot_dir.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{file_stem}_segment_{segment_index}"
    series_path = series_dir / f"{stem}_stft_series.csv"
    series_df.to_csv(series_path, index=False)

    stft_nperseg = min(int(config.get("stft_nperseg", 256)), signal.size)
    stft_noverlap = min(int(config.get("stft_noverlap", 128)), max(0, stft_nperseg - 1))
    freqs, times, zxx = stft(
        signal,
        fs=sampling_rate_hz,
        nperseg=stft_nperseg,
        noverlap=stft_noverlap,
        boundary=None,
    )
    magnitude = np.abs(zxx)

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, freqs, magnitude, shading="gouraud")
    plt.title(f"STFT Magnitude - {label}")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plot_path = plot_dir / f"{stem}_stft.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(plot_path), str(series_path)


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

    default_sampling_rate_hz = float(config["sampling_rate_hz"])
    segment_length = int(config["segment_length"])
    segment_overlap = float(config.get("segment_overlap", 0.5))
    output_rows = []
    plots_per_label: dict[str, int] = {}
    stft_artifacts_per_file: dict[str, int] = {}
    output_dir = Path(config["output_dir"]).expanduser().resolve()

    for record in records:
        outputs = record.source_metadata.get("outputs", {})
        class_label = derive_class_label(outputs, record.label)
        sampling_rate_hz = (
            infer_sampling_rate(record.path, default_sampling_rate_hz)
            if config.get("infer_sampling_rate", False)
            else default_sampling_rate_hz
        )
        segments = segment_signal(record.signal, segment_length, segment_overlap)
        for segment_index, (start, segment) in enumerate(segments):
            metadata = {
                "label": class_label,
                "source_file": str(record.path),
                "source_basename": record.path.name,
                "segment_index": segment_index,
                "start_sample": start,
                "end_sample": start + segment_length,
                "signal_key": record.source_metadata.get("signal_key", ""),
                "sampling_rate_hz": sampling_rate_hz,
                **outputs,
            }
            metadata["rpm"] = infer_rpm(record.source_metadata, outputs)
            td = time_domain_features(segment)
            fd = frequency_domain_features(segment, sampling_rate_hz, config)
            tf, stft_series_df = stft_summary_features(segment, sampling_rate_hz, config)
            row = metadata | td | fd | tf

            row["tf_stft_plot_path"] = "NOT_SAVED"
            row["tf_stft_series_path"] = "NOT_SAVED"
            if config.get("save_stft_artifacts", True):
                saved_count = stft_artifacts_per_file.get(str(record.path), 0)
                if saved_count < int(config.get("max_stft_artifacts_per_file", 1)):
                    plot_path, series_path = save_stft_artifacts(
                        output_dir=output_dir,
                        label=class_label,
                        file_stem=record.path.stem,
                        segment_index=segment_index,
                        signal=segment,
                        sampling_rate_hz=sampling_rate_hz,
                        config=config,
                        series_df=stft_series_df,
                    )
                    row["tf_stft_plot_path"] = plot_path
                    row["tf_stft_series_path"] = series_path
                    stft_artifacts_per_file[str(record.path)] = saved_count + 1

            output_rows.append(row)

            if plots_per_label.get(class_label, 0) < int(config.get("max_plots_per_label", 1)):
                save_plot(
                    output_dir=output_dir,
                    label=class_label,
                    file_stem=record.path.stem,
                    segment_index=segment_index,
                    signal=segment,
                    sampling_rate_hz=sampling_rate_hz,
                    config=config,
                )
                plots_per_label[class_label] = plots_per_label.get(class_label, 0) + 1

    combined = pd.DataFrame(output_rows)
    metadata_cols = [
        "label",
        "source_file",
        "source_basename",
        "segment_index",
        "start_sample",
        "end_sample",
        "signal_key",
        "sampling_rate_hz",
        "rpm",
        "output_fault_status",
        "output_bearing_type",
        "output_fault_location",
        "output_fault_diameter_in",
        "output_motor_load_hp",
        "output_outer_race_position",
        "tf_stft_plot_path",
        "tf_stft_series_path",
    ]
    time_cols = metadata_cols + [column for column in combined.columns if column.startswith("td_")]
    freq_cols = metadata_cols + [column for column in combined.columns if column.startswith("fd_")]
    tf_cols = metadata_cols + [column for column in combined.columns if column.startswith("tf_")]
    metadata_df = combined[metadata_cols]
    return metadata_df, combined, combined[time_cols], combined[freq_cols], combined[tf_cols]


def save_outputs(
    config: dict,
    metadata_df: pd.DataFrame,
    combined: pd.DataFrame,
    time_df: pd.DataFrame,
    freq_df: pd.DataFrame,
    tf_df: pd.DataFrame,
) -> None:
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df.to_csv(output_dir / "metadata_outputs.csv", index=False)
    combined.to_csv(output_dir / "combined_features.csv", index=False)
    combined.to_csv(output_dir / "combined_features_with_outputs.csv", index=False)
    time_df.to_csv(output_dir / "time_domain_features.csv", index=False)
    freq_df.to_csv(output_dir / "frequency_domain_features.csv", index=False)
    tf_df.to_csv(output_dir / "time_frequency_features.csv", index=False)

    summary = {
        "config_path": config.get("_config_path"),
        "input_dir": config["input_dir"],
        "output_dir": str(output_dir),
        "sampling_rate_hz": config["sampling_rate_hz"],
        "infer_sampling_rate": bool(config.get("infer_sampling_rate", False)),
        "segment_length": config["segment_length"],
        "segment_overlap": config.get("segment_overlap", 0.5),
        "fault_frequencies_hz": compute_fault_frequencies(config),
        "num_rows": int(len(combined)),
        "labels": sorted(combined["label"].dropna().astype(str).unique().tolist()),
        "num_time_features": int(len([c for c in combined.columns if c.startswith("td_")])),
        "num_frequency_features": int(len([c for c in combined.columns if c.startswith("fd_")])),
        "num_time_frequency_features": int(len([c for c in combined.columns if c.startswith("tf_")])),
        "has_missing_values": bool(combined.isna().any().any()),
        "sampling_rate_counts_hz": combined["sampling_rate_hz"].value_counts(dropna=False).to_dict(),
        "output_fault_status_counts": combined["output_fault_status"].value_counts(dropna=False).to_dict(),
        "output_fault_location_counts": combined["output_fault_location"].value_counts(dropna=False).to_dict(),
        "output_bearing_type_counts": combined["output_bearing_type"].value_counts(dropna=False).to_dict(),
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
    metadata_df, combined, time_df, freq_df, tf_df = build_feature_tables(config)
    save_outputs(config, metadata_df, combined, time_df, freq_df, tf_df)
    print(f"Saved {len(combined)} segments to {config['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
