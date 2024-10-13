## Description
This implementation is based on the original ALAD (Adversarially Learned Anomaly Detection) [[1]](#1) architecture. The code has been adapted from TensorFlow 1 to TensorFlow 2.

Key changes:
- Doesn't support GPU usage.
- Doesn't support images datasets, only tabular.
- Spectral normalization removed, because it was used only on images datasets.
- Supports anomaly detection on tabular data with binary labels.
- Added CIC-IoT-2023 [[2]](#2) support

## Setup
- Install packages in venv
```commandline
pip install -r requirements.txt
```

## Run
```commandline
python main.py ALAD [-h] [--epochs [EPOCHS]] [--batch_size [BATCH_SIZE]] [--seed [SEED]] [--early_stopping [EARLY_STOPPING]] [--fm_degree [FM_DEGREE]] [--enable_dzz] {cic_iot_2023,kdd99}
```
You can read more about arguments in [main.py](./main.py)

## References
<a id="1">[1]</a>
Zenati, H., Romain, M., Foo, C.S., Lecouat, B. and Chandrasekhar, V., 2018, November. Adversarially learned anomaly detection. In 2018 IEEE International conference on data mining (ICDM) (pp. 727-736). IEEE.
https://arxiv.org/abs/1812.02288

<a id="2">[2]</a>
Neto, E.C.P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R. and Ghorbani, A.A., 2023. CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment. Sensors, 23(13), p.5941.
https://www.mdpi.com/1424-8220/23/13/5941
