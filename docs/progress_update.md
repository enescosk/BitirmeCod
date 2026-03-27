# Short Progress Update

We selected the CWRU dataset as the first benchmark dataset and extracted one representative sample for each main condition: healthy, inner race fault, ball fault, and outer race fault. The selected files are `97.mat`, `105.mat`, `118.mat`, and `130.mat`.

We built and ran a working feature-extraction pipeline on these files. The pipeline currently extracts time-domain features and frequency-domain features, including FFT-based indicators, envelope-based indicators, and amplitudes around characteristic bearing frequencies (`BPFI`, `BPFO`, `BSF`, `FTF`).

The run completed successfully and produced ML-ready outputs:

- a combined feature table
- a time-domain feature table
- a frequency-domain feature table
- one example plot per condition

We also prepared a short summary of the CWRU dataset and a short physical interpretation note for the characteristic fault frequencies. The main remaining task before starting ML is to package the feature list cleanly for reporting and, if needed, extend the pipeline with time-frequency features such as STFT and CWT.
