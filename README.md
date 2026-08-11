# CRF-Net+ for INFRA-3DRC

CRF-Net+ for INFRA-3DRC adapts the CRF-Net camera-radar fusion framework for automotive object detection using the INFRA-3DRC dataset.

## Main Contributions

- INFRA-3DRC dataset integration into CRF-Net
- Camera-radar fusion training across INFRA-3DRC 25 scenes
- INFRA-3DRC-specific data generator and configuration
- Automated training and evaluation reporting
- Per-class AP, precision, recall, and confusion matrices
- Prediction and ground-truth exports
- Training and evaluation visualisations

## Dataset Configuration

Set the INFRA-3DRC dataset location in:

```text
configs/infra3drc.cfg
```

For example:

```ini
[DATA]
data_set = infra3drc
data_path = /path/to/INFRA-3DRC
```

The calibration-corrected INFRA-3DRC dataset used is not included in this repository. The original INFRA-3DRC dataset is available from the [official INFRA-3DRC dataset website](https://fraunhoferivi.github.io/INFRA-3DRC-Dataset/)

## Training

```bash
python train_crfnet.py --config configs/infra3drc.cfg
```

## Evaluation

Evaluation and automated reporting are implemented through:

```text
evaluate_crfnet.py
utils/eval_test.py
utils/reporting.py
```

Generated outputs include performance metrics, per-class AP, confusion matrices, training curves, predictions, and ground-truth data.

## Author

**Olivier Rukundo, Ph.D.**  
University of Limerick, Ireland

## Attribution

This repository is based on the original CRF-Net camera-radar fusion framework. Original CRF-Net authors/contributors retain authorship of the original implementation.

The INFRA-3DRC integration, dataset-specific modifications, and additional functionalities were developed by [Dr Olivier Rukundo](https://www.ul.ie/scieng/schools-and-departments/department-electronic-and-computer-engineering/dr-olivier-rukundo).

INFRA-3DRC is provided by Fraunhofer IVI.

## License

This repository contains code derived from CRF-Net. Please refer to the included license and the original CRF-Net licensing terms.

## Results

CRF-Net+ was trained and evaluated on all 25 INFRA-3DRC scenes.

| mAP | Precision | Recall |
|---:|---:|---:|
| 0.9675 | 0.9956 | 0.5336 |

## Evaluation Demo

The video below shows sequential object-detection outputs generated during evaluation.

https://github.com/user-attachments/assets/e030da2b-54d3-44cd-b99d-498ecc0c8781


