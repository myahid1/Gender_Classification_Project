# Gender Classification Image Project

## Project Overview

This project explores binary image classification using a labelled image dataset organised into `men` and `women` folders. Rather than treating the task as a definitive gender detection system, this project uses the dataset as a computer vision case study to practise building, training, and evaluating image classification models.

The main objective was to compare three modelling approaches:

1. A baseline CNN trained from scratch
2. EfficientNet-B0 using transfer learning with a frozen feature extractor
3. Fine-tuned EfficientNet-B0 with selected final layers unfrozen

The project covers exploratory data analysis, image preprocessing, data augmentation, PyTorch DataLoaders, CNN training, transfer learning, fine-tuning, and model evaluation.

## Dataset

The dataset was organised into training and testing folders, with separate subfolders for each class.

```text
Data/
├── train/
│   ├── men/
│   └── women/
└── test/
    ├── men/
    └── women/
```

The training set contained 110 images per class, while the test set contained 40 images per class.

The dataset itself is not included in this repository because of file size and licensing considerations.

```markdown
The trained model weights file (`finetuned_efficientnet_b0.pth`) is not included in this repository because model weight files can be large. The model can be retrained by running the notebook.

## Project Workflow

1. Exploratory Data Analysis

   * Checked class balance
   * Displayed sample images
   * Inspected image dimensions
   * Identified image orientation issues

2. Image Preprocessing

   * Corrected EXIF orientation metadata
   * Converted images to RGB format
   * Resized images to 224 x 224 pixels
   * Applied random horizontal flipping for training data augmentation
   * Converted images into PyTorch tensors
   * Normalised images using ImageNet mean and standard deviation values

3. Model Training

   * Trained a baseline CNN from scratch
   * Trained EfficientNet-B0 with a frozen pretrained feature extractor
   * Fine-tuned EfficientNet-B0 by unfreezing selected final layers

4. Model Evaluation

   * Compared models using accuracy, precision, recall, F1-score, and confusion matrices

## Models and Results

| Model                      | Test Accuracy | Macro F1 |
| -------------------------- | ------------: | -------: |
| Baseline CNN               |        81.25% |     0.81 |
| EfficientNet-B0 Frozen     |        82.50% |     0.82 |
| Fine-tuned EfficientNet-B0 |        85.00% |     0.85 |

The fine-tuned EfficientNet-B0 model achieved the best performance, with a test accuracy of 85.00%.

## Key Findings

The baseline CNN performed reasonably well despite being trained from scratch on a small dataset. However, because it had to learn visual features directly from the available training images, its performance was likely limited by the dataset size and variation in image quality, pose, lighting, and background.

EfficientNet-B0 with a frozen pretrained feature extractor performed slightly better. This suggests that pretrained visual features were useful for the task, as EfficientNet had already learned general image patterns from large-scale pretraining.

The fine-tuned EfficientNet-B0 model achieved the best result. By unfreezing selected final layers, the model was able to adapt some of its higher-level visual features to the dataset while retaining most of its pretrained knowledge.

## Final Model Performance

The fine-tuned EfficientNet-B0 confusion matrix showed:

| Actual Class | Correct Predictions | Misclassified |
| ------------ | ------------------: | ------------: |
| Men          |             36 / 40 |             4 |
| Women        |             32 / 40 |             8 |

The final model showed higher recall for images labelled as `men` than for images labelled as `women`, meaning it correctly identified a larger proportion of the men-labelled test images.

## Project Files

```text
Gender_Classification_Project/
├── notebooks/
│   └── 01_gender_classification.ipynb
├── charts/
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── baseline_training_loss.png
│   ├── baseline_training_accuracy.png
│   ├── baseline_confusion_matrix.png
│   ├── efficientnet_confusion_matrix.png
│   └── finetuned_efficientnet_confusion_matrix.png
├── results/
│   └── model_metrics_comparison.csv
├── final_report.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Technologies Used

* Python
* PyTorch
* Torchvision
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Pillow

## How to Run This Project

1. Clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/Gender_Classification_Project.git
cd Gender_Classification_Project
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Add the dataset into the `Data/` folder using the expected folder structure:

```text
Data/
├── train/
│   ├── men/
│   └── women/
└── test/
    ├── men/
    └── women/
```

5. Open and run the notebook:

```text
notebooks/01_gender_classification.ipynb
```

## Limitations

* The dataset size was small.
* Image quality, pose, lighting, and background varied across examples.
* Some images required orientation correction.
* Labels were based only on dataset folder names.
* The model should not be treated as a reliable or universal measure of gender identity.
* Model performance may not generalise well to broader real-world images.