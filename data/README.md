# Data Directory

## Structure
```
data/
└── keypoints/
    ├── hello/
    │   ├── 0.npy    ← sequence of 30 frames × 63 keypoints
    │   ├── 1.npy
    │   └── ...
    ├── help/
    └── ...
```

## How to collect data
Run: `python src/collect_data.py`

## Note
`.npy` files are excluded from GitHub via `.gitignore`.  
Share datasets via Google Drive or HuggingFace Datasets.
