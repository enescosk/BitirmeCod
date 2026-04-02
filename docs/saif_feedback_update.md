# Saif Feedback Update

This update addresses the feedback on the feature database and the missing output labels.

## Added to the database

- top 5 frequency peaks with both frequency and amplitude
- band energies for:
  - `0-500 Hz`
  - `500-2000 Hz`
  - `2000+ Hz`
- STFT-based time-frequency features:
  - energy over time summaries
  - max frequency over time summaries
- saved STFT images
- saved STFT time-series CSV files

## Added output labels for each row

- `output_fault_status`
  - `FAULT`
  - `NO_FAULT`
- `output_bearing_type`
  - `DRIVE_END`
  - `FAN_END`
  - `NONE`
- `output_fault_location`
  - `INNER_RACE`
  - `OUTER_RACE`
  - `BALL`
  - `NONE`
- `output_fault_diameter_in`
  - `0.007`
  - `0.014`
  - `0.021`
  - `0.028`
  - `0.0` for healthy data
- `output_motor_load_hp`
- `output_outer_race_position`

## Database used for the updated run

- healthy:
  - `97.mat`
- inner race:
  - `105.mat`
- outer race:
  - `130.mat`
- ball fault:
  - `118.mat`, `119.mat`, `120.mat`, `121.mat`
  - `185.mat`, `186.mat`, `187.mat`, `188.mat`
  - `222.mat`, `223.mat`, `224.mat`, `225.mat`
  - `3005.mat`, `3006.mat`, `3007.mat`, `3008.mat`

## Main output files

- `outputs/cwru_database_run/metadata_outputs.csv`
- `outputs/cwru_database_run/time_domain_features.csv`
- `outputs/cwru_database_run/frequency_domain_features.csv`
- `outputs/cwru_database_run/time_frequency_features.csv`
- `outputs/cwru_database_run/combined_features_with_outputs.csv`
- `outputs/cwru_database_run/run_summary.json`
