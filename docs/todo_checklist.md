# Bearing Fault Project Checklist

## Done

- [x] Chose the initial public dataset: CWRU Bearing Data Center
- [x] Read the dataset structure enough to identify file types, classes, RPM, and sampling setup
- [x] Selected one representative file per condition
- [x] Downloaded the selected files
- [x] Mapped the selected files to classes
  - [x] `97.mat` -> healthy
  - [x] `105.mat` -> inner race fault
  - [x] `118.mat` -> ball fault
  - [x] `130.mat` -> outer race fault
- [x] Created a working feature-extraction pipeline
- [x] Extracted time-domain features
  - [x] mean
  - [x] std
  - [x] variance
  - [x] RMS
  - [x] peak / peak-to-peak
  - [x] skewness
  - [x] kurtosis
  - [x] crest factor
  - [x] shape / impulse / clearance factor
  - [x] energy / entropy
- [x] Extracted frequency-domain features
  - [x] FFT-based dominant frequency
  - [x] spectral centroid / bandwidth
  - [x] band energies
  - [x] envelope spectrum
  - [x] harmonic amplitudes near `BPFI`, `BPFO`, `BSF`, `FTF`
- [x] Generated one example plot per class
- [x] Produced ML-ready CSV outputs
  - [x] combined feature table
  - [x] time-domain feature table
  - [x] frequency-domain feature table
- [x] Validated the run
  - [x] all classes processed
  - [x] no missing values
  - [x] output files generated successfully
- [x] Wrote a short CWRU dataset summary
- [x] Wrote short physical interpretation notes for characteristic frequencies

## In Progress / Next

- [ ] Turn the feature list into a clean report/slides section
- [ ] Share the output CSV files and example plots with teammates
- [ ] Prepare a short English summary for TA/professor
- [ ] Add time-frequency features
  - [ ] STFT-based features or spectrogram outputs
  - [ ] CWT-based features or scalogram outputs
- [ ] Expand beyond one file per class if you want a stronger first ML dataset
- [ ] Start the ML stage after the team agrees on the final feature set

## Nice to Have

- [ ] Add more CWRU operating conditions (`_1`, `_2`, `_3`)
- [ ] Add more fault sizes (`014`, `021`, `028`)
- [ ] Compare drive-end vs fan-end signals
- [ ] Add feature-importance analysis after the first ML baseline
- [ ] Compare CWRU with a harder dataset later for generalization

## Files Ready To Share

- `/Users/ec/Desktop/automation/hidayet/BitirmeCod/outputs/cwru_selected_run/combined_features.csv`
- `/Users/ec/Desktop/automation/hidayet/BitirmeCod/outputs/cwru_selected_run/time_domain_features.csv`
- `/Users/ec/Desktop/automation/hidayet/BitirmeCod/outputs/cwru_selected_run/frequency_domain_features.csv`
- `/Users/ec/Desktop/automation/hidayet/BitirmeCod/outputs/cwru_selected_run/plots/healthy/97_segment_0.png`
- `/Users/ec/Desktop/automation/hidayet/BitirmeCod/outputs/cwru_selected_run/plots/inner/105_segment_0.png`
- `/Users/ec/Desktop/automation/hidayet/BitirmeCod/outputs/cwru_selected_run/plots/ball/118_segment_0.png`
- `/Users/ec/Desktop/automation/hidayet/BitirmeCod/outputs/cwru_selected_run/plots/outer/130_segment_0.png`
- `/Users/ec/Desktop/automation/hidayet/BitirmeCod/docs/cwru_summary.md`
- `/Users/ec/Desktop/automation/hidayet/BitirmeCod/docs/physical_interpretation.md`
