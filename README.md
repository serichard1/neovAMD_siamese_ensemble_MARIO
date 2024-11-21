[MICCAI 2024](docs/miccai2024-logo.png)

<<<<<<< HEAD
# Siamese-Ensemble-MARIOchall
Multi-modal Siamese Ensemble for Neovascular AMD classification and prediction from Optical Coherence Tomography
=======
# [🕹️ 🍄 Monitoring Age-related Macular Degeneration Progression In Optical Coherence Tomography (MARIO) - MICCAI Challenge 2024](https://youvenz.github.io/MARIO_challenge.github.io/)

PyTorch implementation of **Multi-modal Siamese Ensemble for Neovascular AMD classification and prediction from Optical Coherence Tomography**

[Method](docs/architecture_siamese_ensemble.eps)

## Getting started

### Set Up the Environment
Create a Conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate mario_siamese
```

### Dataset

The provided ground truth consists of pairwise comparisons between B-scans from the same patient across visits, indicating relative progression as the classes: reduced (0), stable (1), worsened (2), or other (3). There are 68 patients in the dataset, for a total of 14,496 pairwise comparisons. An additional external evaluation set is provided, which includes 34 patients, totaling 7,010 comparisons. For Task 2, a subset of the same patient cohort is used, but only the first B-scan is provided, and the third ("other") class is excluded. There are 61 patients, for a total of 8,082 comparisons. External evaluation for Task 2 includes 29 patients and 3,822 comparisons. 

**Prior to training**
We split the training dataset into training, validation, and test sets, at the patient level to prevent data leakage. For Task 1, splits include 50 patients (10,865 comparisons) for training, 9 for validation (1,844 comparisons), and 9 for testing (1,787 comparisons). For Task 2, splits involve 42 patients (5,337 comparisons) for training, 9 for validation (1,461 comparisons), and 10 for testing (1,284 comparisons).
Note : 
- string attributes (ie. "SEX", "side_eye") are mapped to bool 0/1
- all numerical values are normalized

### Example usage

1. Siamese training:

train each of the 3 encoders siamese, 
example command
```bash
python3 training.py --img_size 384 384 --data_path ./data/ --output_dir convnext_training --model convnext_tiny
```

1. Inference + evaluation:

train each of the 3 encoders siamese, 
example command
```bash
python3 training.py --img_size 384 384 --data_path ./data/ --output_dir convnext_training --model convnext_tiny --eval ../convnext_training/ckpt_bestsofar_2024-07-20_convnext_tiny.pth
```

## Training Parameters

```bash
# Training Hyperparameters
--img_size                [int, int]  Input size for the images (default: depends on model architecture).
--learning_rate           float       Initial value of the learning rate (default: 5e-5).
--model                   str         Model architecture to use (default: "convnext_tiny").
                                      Choices: ["convnext_tiny", "efficientnet_v2_s", "inception_resnet_v2", "siamese_ensemble"].
--weight_decay            float       Regularization strength to prevent overfitting (default: 0.05).
--batch_size_per_gpu      int         Batch size per GPU (default: 16).
--n_epochs                int         Number of training epochs (default: 100).
--patience                int         Early stopping patience: epochs to wait without improvement (default: 5).
--dropout                 float       Dropout rate for classification head (default: 0.0).

--mean                    list        Mean values for OCT image normalization (default: [0.1880, 0.1880, 0.1880]).
--std                     list        Standard deviation for OCT image normalization (default: [0.2244, 0.2244, 0.2244]).

# Training Environment
--use_fp16                bool        Use half-precision for faster training and lower memory usage (default: True).
--num_workers             int         Number of data loading workers per GPU (default: 4).
--data_path               str         Path to the dataset (default: "data").
--output_dir              str         Directory to save logs and checkpoints (default: "./output").
--seed                    int         Random seed for reproducibility (default: 3407).

# Logging and Evaluation
--log_freq                int         Frequency of logging during training (default: 10).
--eval                    str         Path to a pretrained model for evaluation only (default: None).
```
>>>>>>> 8f74f8d (refactored)
