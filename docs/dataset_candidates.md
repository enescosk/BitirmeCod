# Dataset Candidates For Next ML Stage

The paper attachment was not included in this workspace, so this shortlist uses three standard public bearing datasets that fit our upcoming ML goals well.

## 1. CWRU Bearing Data Center

Why keep it:

- already integrated into our current feature-extraction pipeline
- very common benchmark for fault classification
- fast to use for feature-selection and neural-network baselines

Main tradeoff:

- faults are artificially introduced in a lab setup, so generalization to real industrial conditions is limited

Source:

- [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/welcome)

## 2. Paderborn University Bearing DataCenter

Why shortlist it:

- includes vibration and motor-current measurements
- covers multiple operating conditions
- often considered more challenging and realistic than CWRU
- useful for testing cross-dataset generalization after building a CWRU baseline

Main tradeoff:

- dataset structure is more complex than CWRU, so preprocessing effort is higher

Source:

- [Paderborn KAt Bearing DataCenter](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter)

## 3. XJTU-SY Bearing Datasets

Why shortlist it:

- includes full run-to-failure bearing life data
- directly useful for future prognostics / RUL work
- supports both diagnosis and degradation-tracking studies

Main tradeoff:

- this dataset is more suitable for prognostics than for simple first-step fault classification

Source:

- [XJTU-SY Bearing Datasets](https://biaowang.tech/xjtu-sy-bearing-datasets/)

## Recommendation For Monday

If the next step is mainly feature selection + classification:

- start with `CWRU`
- keep `Paderborn` as the second serious benchmark

If the team also wants to keep the door open for RUL/prognostics soon:

- shortlist `XJTU-SY` as the third candidate

So the clean recommendation is:

1. `CWRU`
2. `Paderborn`
3. `XJTU-SY`
