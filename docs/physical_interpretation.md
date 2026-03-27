# Physical Interpretation Notes

- `BPFI` (`Ball Pass Frequency Inner race`): inner race fault. Rolling elements hit the inner-race defect periodically, so its harmonics should appear in the spectrum or envelope spectrum.
- `BPFO` (`Ball Pass Frequency Outer race`): outer race fault. Impulses repeat when the rolling elements pass the damaged point on the outer race.
- `BSF` (`Ball Spin Frequency`): rolling element or ball fault. Peaks near this frequency suggest damage on the ball itself.
- `FTF` (`Fundamental Train Frequency`): cage fault. This is usually the lowest characteristic bearing frequency.

## Standard bearing formulas

When the dataset documentation does not provide the fault frequencies directly, they can be estimated from bearing geometry and shaft speed:

- `FTF = 0.5 * fr * (1 - (d / D) * cos(theta))`
- `BPFO = 0.5 * N * fr * (1 - (d / D) * cos(theta))`
- `BPFI = 0.5 * N * fr * (1 + (d / D) * cos(theta))`
- `BSF = 0.5 * fr * (D / d) * (1 - ((d / D) * cos(theta))^2)`

Where:

- `fr`: shaft rotational frequency in Hz
- `N`: number of rolling elements
- `d`: ball diameter
- `D`: pitch diameter
- `theta`: contact angle

## What to show in the meeting

- One clean feature table for ML.
- Separate `time-domain` and `frequency-domain` CSV files.
- A few example plots for healthy vs inner/outer/ball faults.
- A short explanation of which characteristic frequency points to which physical fault.
