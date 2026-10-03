# PRIVATE Inference Readiness

PRIVATE inference was intentionally not executed.

## Input placement

Place the 87 PRIVATE Parquet light curves in a directory such as:

`private_pack/private/`

The files must be named exactly:

`STAR_0000.parquet` through `STAR_0086.parquet`

No PRIVATE labels or truth files are needed or allowed.

## Frozen configuration and model

The runner uses [configs/final_config.json](../configs/final_config.json).
The frozen decision source is the statistical candidate score with threshold
`0.19`. The trained model artifact is generated as
`outputs_private/model.json`; it is not used for the final decision because
the statistical baseline won TRAIN OOF model comparison.

## Command for tomorrow

```powershell
py run_final_pipeline.py `
  --input-dir private_pack\private `
  --output-dir outputs_private `
  --train-dir train_pack\train `
  --labels train_pack\train_labels.csv `
  --dev-dir dev_pack\dev `
  --dev-labels dev_pack\dev_labels.csv `
  --config configs\final_config.json `
  --expected-count 87
```

This command performs no PRIVATE retraining or tuning. It reuses the frozen
TRAIN-selected configuration and validates the contiguous STAR ID sequence.

## Output and validation

The generated file will be:

`outputs_private/submission.csv`

with exact columns:

`star_id,prediction,confidence,period,depth_ppm,duration_hours`

Run the strict validator after generation:

```powershell
@'
from submission.validate_submission import validate
validate('outputs_private/submission.csv',
         [f'STAR_{i:04d}' for i in range(87)])
print('VALID')
'@ | py -
```

The validator checks row count, IDs, duplicates, predictions, confidence
range, and positive/negative characterization fields.
