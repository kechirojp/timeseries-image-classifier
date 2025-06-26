# Sample Image Placeholder

This directory contains sample images for the dataset.
For actual training, replace these placeholder images with real images.

## Image Format Requirements:
- Format: PNG or JPG
- Dimensions: Consistent across dataset (e.g., 224x224, 512x512)
- Naming: Sequential numbers (e.g., 0.png, 1.png, 2.png, ...)
- Content: Images corresponding to the time series data

## Directory Structure:
```
dataset_X_15m_winsize40/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── class_2/
└── test/
    ├── class_0/
    ├── class_1/
    └── class_2/
```

Each class directory should contain images for that specific classification class:
- class_0: Class 0 images (label value: 0)
- class_1: Class 1 images (label value: 1)
- class_2: Class 2 images (label value: 2)

**Important**: Directory names `class_0`, `class_1`, `class_2` correspond directly to label values 0, 1, 2 respectively.
