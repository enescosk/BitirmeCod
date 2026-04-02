# Bearing Fault Feature Extraction

This repository now includes a working feature-extraction pipeline for bearing or spindle vibration datasets. The goal is to turn raw signals into ML-ready feature tables and quick meeting visuals.

## Included files

- `scripts/extract_features.py`: main CLI that extracts time-domain and frequency-domain features.
- `scripts/generate_demo_data.py`: creates a small synthetic demo dataset for verification.
- `configs/example_config.json`: template for your real dataset.
- `configs/demo_config.json`: ready-to-run config for the synthetic demo data.
- `docs/physical_interpretation.md`: short notes for the meeting.

## Extracted features

Time-domain:

- mean
- std
- variance
- RMS
- peak absolute value
- peak-to-peak
- skewness
- kurtosis
- crest factor
- shape factor
- impulse factor
- clearance factor
- energy
- entropy

Frequency-domain:

- dominant frequency
- top 5 spectrum peak frequencies and amplitudes
- spectral centroid
- spectral bandwidth
- total spectral energy
- envelope-spectrum dominant frequency
- configurable band energies (`0-500 Hz`, `500-2000 Hz`, `2000+ Hz`)
- amplitudes near `BPFI`, `BPFO`, `BSF`, `FTF` and their harmonics in both FFT and envelope spectrum

Time-frequency:

- STFT energy statistics over time
- STFT max-frequency statistics over time
- optional saved STFT plots and time-series CSV artifacts

## Supported input formats

- `.csv`
- `.txt`
- `.npy`
- `.npz`
- `.mat`
- `.wav`

## Expected dataset layout

The default label strategy is `parent_dir`, so this structure works well:

```text
dataset/
  healthy/
    sample1.csv
  inner/
    sample2.csv
  outer/
    sample3.csv
  ball/
    sample4.csv
```

Each file should contain a 1D numeric signal. For CSV files, set `signal_column` in the config if needed.

## Quick start

Generate demo data:

```bash
python3 scripts/generate_demo_data.py
```

Run the pipeline on the demo data:

```bash
python3 scripts/extract_features.py --config configs/demo_config.json
```

Run the pipeline on your real dataset:

1. Copy `configs/example_config.json`.
2. Update `input_dir`, `output_dir`, `sampling_rate_hz`, segment settings, and fault-frequency values.
3. Run:

```bash
python3 scripts/extract_features.py --config path/to/your_config.json
```

## Outputs

The script writes:

- `metadata_outputs.csv`
- `combined_features.csv`
- `combined_features_with_outputs.csv`
- `time_domain_features.csv`
- `frequency_domain_features.csv`
- `time_frequency_features.csv`
- `run_summary.json`
- `plots/<label>/*.png`
- `stft/plots/<label>/*.png`
- `stft/series/<label>/*.csv`

## Notes

- If the dataset documentation already gives `BPFI/BPFO/BSF/FTF`, use those values directly.
- If not, use the formulas in `docs/physical_interpretation.md` with bearing geometry and shaft speed.
- Time-frequency features were intentionally left as a second step so the current deliverable stays focused on the meeting requirement.
