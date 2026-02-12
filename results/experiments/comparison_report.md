# Experiment Comparison Report

Generated: 2026-02-12 18:23:25

---

## Performance Comparison

| Experiment        |   Batch Size |     LR |   L1 Lambda |   Epochs |   PSNR (dB) |   SSIM |   L1 Error |
|:------------------|-------------:|-------:|------------:|---------:|------------:|-------:|-----------:|
| exp_001_baseline  |           16 | 0.0002 |         100 |       50 |       24.12 |   0.8  |      0.085 |
| exp_002_lambda50  |           16 | 0.0002 |          50 |       50 |       22.14 |   0.78 |      0.095 |
| exp_003_lambda150 |           16 | 0.0002 |         150 |       50 |       25.89 |   0.85 |      0.072 |
| exp_004_lr_decay  |           16 | 0.0002 |         100 |       50 |       25.12 |   0.84 |      0.078 |


## Best Model

- **Best Experiment:** exp_003_lambda150
- **PSNR:** 25.89 dB

## Experiment Details

### exp_001_baseline

- **Timestamp:** 2026-02-12T18:23:25.400195
- **Notes:** Baseline pix2pix model


### exp_002_lambda50

- **Timestamp:** 2026-02-12T18:23:25.400407
- **Notes:** Reduced L1 weight


### exp_003_lambda150

- **Timestamp:** 2026-02-12T18:23:25.400539
- **Notes:** Increased L1 weight


### exp_004_lr_decay

- **Timestamp:** 2026-02-12T18:23:25.400659
- **Notes:** StepLR scheduler (gamma=0.5, step=30)

