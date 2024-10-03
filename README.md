## Description
ALAD implementation with tensorflow 2.

Key changes:
- Doesn't support GPU usage
- Doesn't support images datasets
- Spectral normalization removed, because it was used only on images datasets
- Supports anomaly detection on tabular data with binary labels