# CWRU Dataset Short Summary

## What this dataset is

The Case Western Reserve University Bearing Data Center dataset is a public bearing-fault benchmark collected on a motor-driven test rig. It is widely used in fault-diagnosis papers because it includes healthy and faulty bearing vibration signals with known operating conditions.

## Experimental setup

- Test rig includes a motor, torque transducer, and dynamometer.
- Faults were seeded on bearings using EDM, so the defects are artificial and controlled.
- Common fault classes are healthy, inner race fault, outer race fault, and ball fault.
- Data is provided in MATLAB `.mat` format.
- Files typically contain drive-end (`DE`), fan-end (`FE`), and sometimes base (`BA`) accelerometer signals plus RPM.
- Common sampling rates are `12 kHz` and `48 kHz`; the selected run here uses the `12 kHz drive-end` data.

## Selected files for this run

- `97.mat` -> healthy
- `105.mat` -> inner race fault
- `118.mat` -> ball fault
- `130.mat` -> outer race fault

These were chosen as one representative sample per class under roughly the same low-load operating condition, which is enough for the current feature-extraction task.

## Why researchers like CWRU

- Easy to access and well known.
- Clear labels.
- Good for benchmarking feature extraction and classification pipelines.
- Many papers use it, so comparison is easier.

## Main shortcomings

- Faults are artificial, not naturally developed.
- Lab conditions are much cleaner than real industrial spindle environments.
- Signals are often easier than real-world data, so models may look better than they really are.
- It is a bearing-rig dataset, not a full CNC spindle-health dataset.
- Generalization from CWRU to real industry setups can be weak.

## Notes for your report/meeting

- Mention that CWRU is suitable for the first feature-extraction stage and proof-of-concept ML work.
- Also mention that later stages should use harder datasets or more realistic spindle data for stronger validation.
