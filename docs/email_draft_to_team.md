# Draft Message To Team

Hi all,

I finished the first CWRU feature-extraction step.

Dataset used:

- `97.mat` -> healthy
- `105.mat` -> inner race fault
- `118.mat` -> ball fault
- `130.mat` -> outer race fault

What is done:

- CWRU dataset summary prepared
- feature-extraction pipeline implemented
- time-domain and frequency-domain features extracted
- example plots generated for each class
- ML-ready CSV files produced

Main output files:

- `combined_features.csv`
- `time_domain_features.csv`
- `frequency_domain_features.csv`

I also prepared short notes for the physical meaning of `BPFI`, `BPFO`, `BSF`, and `FTF`.

What is still left:

- package the feature list for report/slides
- optionally add time-frequency features (`STFT`, `CWT`)
- expand to more CWRU samples if we want a larger first dataset

If you want, I can also share the plots and the short CWRU summary file.

Best,
