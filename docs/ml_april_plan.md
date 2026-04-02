# ML Plan for April

## Goal

Build a first supervised ML pipeline on top of the extracted CWRU feature database, then use the same pipeline structure later with the larger database once feature extraction is finalized.

## Planned outputs

- baseline classification for `FAULT` vs `NO_FAULT`
- baseline classification for `fault location` (`INNER_RACE`, `OUTER_RACE`, `BALL`)
- optional regression/classification experiments for `fault diameter`
- feature-selection results to identify which time, frequency, and time-frequency features are most useful

## Planned workflow

1. Clean the feature database and verify the output labels
2. Split the data into train/validation/test sets
3. Start with simple baselines:
   - Logistic Regression
   - Random Forest
   - Support Vector Machine
4. Compare model performance using accuracy, precision, recall, F1, and confusion matrices
5. Run feature selection:
   - correlation filtering
   - feature importance from tree-based models
   - recursive feature elimination if needed
6. Compare models with:
   - time-domain only
   - frequency-domain only
   - time + frequency
   - time + frequency + STFT features
7. Document which feature groups are most informative and decide the next model direction

## Notes

- The first ML stage can begin immediately with the current database.
- Once the database grows, the same ML pipeline can be rerun without changing the overall workflow.
- If class imbalance becomes an issue, class weighting or resampling can be added later.
