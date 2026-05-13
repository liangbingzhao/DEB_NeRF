# Experiments

All experiment results for Walnut PCCT SAX-NeRF project.

## Directory Structure

```
experiments/
├── README.md                  # This file
├── exp_log.md                 # Experiment log (chronological record)
├── walnut_total_50/           # Line 2: Total energy, 50 sparse views
│   ├── train_<jobid>.log      # SLURM training log
│   ├── eval_<jobid>.log       # SLURM eval log
│   ├── checkpoints/           # Symlink to SAX-NeRF/logs/...
│   ├── visualizations/        # Generated plots and images
│   └── results.md             # Summary of metrics
└── walnut_10kev_50/           # Line 1: 10 keV energy, 50 sparse views
    ├── ...                    # Same structure
    └── results.md
```

## Data Summary

| Dataset | Image Shape | Image Range | Proj Shape | Proj Max | Source |
|---------|-------------|-------------|------------|----------|--------|
| chest_50 (baseline) | 128³ | [0, 1] | 50×256×256 | 0.067 | Synthetic |
| walnut_total_50 | 128³ | [0, 1] | 50×256×256 | 0.019 | Reconstruction/Total → TIGRE fwd proj |
| walnut_10kev_50 | 128³ | [0, 1] | 50×256×256 | 0.002 | VirtualMonoImg/10Kev → TIGRE fwd proj |

## Notes
- All data generated via Plan B: GT volume → TIGRE forward projection (self-consistent)
- Current resolution: 128³ (may revisit with 256³/512³ if results are poor)
- Geometry: ideal params matching chest baseline (DSD=1500, DSO=1000, dDetector=1.0, dVoxel=1.0)
