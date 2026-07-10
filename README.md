# mining_sites_detector

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/agbleze/mining_sites_detector/.github%2Fworkflows%2Fci-cd.yml)
![GitHub Tag](https://img.shields.io/github/v/tag/agbleze/mining_sites_detector)
![GitHub Release](https://img.shields.io/github/v/release/agbleze/mining_sites_detector)
![GitHub License](https://img.shields.io/github/license/agbleze/mining_sites_detector)

![Static Badge](https://img.shields.io/badge/Earth-Observation-brightgreen?style=for-the-badge&labelColor=rgba(143%2C%2068%2C%2042%2C%200.8)&color=rgba(10%2C%20131%2C%20214%2C%200.8))
![Static Badge](https://img.shields.io/badge/Sentinel-2-brightgreen?style=for-the-badge&labelColor=rgba(58%2C%2051%2C%2039%2C%200.8)&color=rgba(39%2C%20245%2C%20235%2C%200.8))


## Project Description

mining_sites_detector package provides complete pipeline for training deep learning models on Sentinel‑2 imagery to detect mining sites. The package handles all necessary preprocessing, data loading, model architecture design, training and evaluation.

A Convolutional Neural Network (CNN) designed from scratch, ingests all 13 bands from sentinel 2.


## Installation

```bash
$ pip install mining_sites_detector
```

## Usage

#### Linear probing

### Dataset required

#### Satellite image dataset

A minimum of two directories containing satelite images (tif) for training and validation dataset is required.

#### Labels for dataset

The labels for dataset are expected to be in a csv file with first column being satelite image name and second column being the groundtruth label. Hence, two csv files for training and validation set in the format described are required

The image name in the csv file and the dataset directory are used to create dataset path to read image for processing.


### Train model using package from CLI

An example command to train the model is provided as follows:

```bash

miner --train_img_dir "YOUR_TRAIN_IMAGE_DIRECTORY_PATH" \
    --val_img_dir "YOUR_VALIDATION_IMAGE_DIRECTORY_PATH" \
    --train_target_file "YOUR_TRAIN_TARGET_FILEPATH.csv" \
    --val_target_file "YOUR_VALIDATION_TARGET_FILEPATH" \
    --num_epochs 50 \
    --save_train_results_as "training_results.json"

```

## Future work
- Model architecturing that leverages multimodal data fusion from heterogenous data sources for mineral prospectivity


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`mining_sites_detector` was created by agbleze. It is licensed under the terms of the MIT license.

## Credits

`mining_sites_detector` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
