# Gender Classification Image Project

## 1. Project Overview

This project explores binary image classification using a labelled image dataset organised into men and women folders. Rather than treating the task as a definitive gender detection system, the project uses the dataset as a computer vision case study to practise building, training, and evaluating image classification models.

The main objective was to compare three modelling approaches: a baseline CNN trained from scratch, an EfficientNet-B0 model using transfer learning with a frozen feature extractor, and a fine-tuned EfficientNet-B0 model with selected final layers unfrozen. This allowed the project to examine how model complexity and transfer learning affect performance on a small image dataset.

The project covers exploratory data analysis, image preprocessing, data augmentation, PyTorch DataLoaders, CNN training, transfer learning, fine-tuning, and model evaluation using accuracy, precision, recall, F1-score, and confusion matrices.

## 2. Dataset Overview

The dataset was organised into training and testing folders, with separate subfolders for `men` and `women`.

The training set was balanced, with 110 images for each class. The test set also contained 40 images for each class.

The class distribution chart is shown in `charts/class_distribution.png`.

## 3. Exploratory Data Analysis

Initial EDA showed that the dataset was balanced across both classes. Sample images showed variation in pose, lighting, background, image orientation, and cropping.

Sample training images are shown in `charts/sample_images.png`.

Some images initially appeared rotated due to EXIF orientation metadata. This was corrected using `ImageOps.exif_transpose()`.

The images also had different widths and heights, so resizing was required before model training.

## 4. Preprocessing

The preprocessing pipeline included:

- Correcting EXIF image orientation
- Converting images to RGB format
- Resizing all images to 224 x 224 pixels
- Applying random horizontal flipping to the training set
- Converting images into PyTorch tensors
- Normalising images using ImageNet mean and standard deviation values

DataLoaders were then used to batch and shuffle the training data.

## 5. Models Trained

Three models were trained and compared:

1. Baseline CNN trained from scratch
2. EfficientNet-B0 with frozen pretrained feature extractor
3. Fine-tuned EfficientNet-B0 with the last feature layers unfrozen

## 6. Results

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| Baseline CNN | 81.25% | 0.81 |
| EfficientNet-B0 Frozen | 82.50% | 0.82 |
| Fine-tuned EfficientNet-B0 | 85.00% | 0.85 |

The fine-tuned EfficientNet-B0 model achieved the best performance, with a test accuracy of 85.00%. The results show a gradual improvement from the baseline CNN to the fine-tuned EfficientNet-B0 model. This suggests that pretrained visual features were useful for the task, while limited fine-tuning allowed the model to adapt more closely to the dataset. However, the improvement from 81.25% to 85.00% is moderate, which may reflect the small dataset size and variation in image quality.

## 7. Final Model Performance

The fine-tuned EfficientNet-B0 confusion matrix showed:

| Actual Class | Correct Predictions | Misclassified |
|---|---:|---:|
| Men | 36 / 40 | 4 |
| Women | 32 / 40 | 8 |

The final model showed higher recall for images labelled as `men` than for images labelled as `women`, meaning it correctly identified a larger proportion of the men-labelled test images.

## 8. Key Findings

The baseline CNN performed reasonably well despite being trained from scratch on a small dataset, achieving 81.25% test accuracy. However, because it had to learn all visual features directly from the available training images, its performance was likely limited by the small dataset size and variation in image quality, pose, lighting, and background.

EfficientNet-B0 with a frozen pretrained feature extractor performed slightly better, achieving 82.50% test accuracy. This improvement is likely because EfficientNet had already learned general image features such as edges, textures, shapes, and higher-level visual patterns from large-scale pretraining. However, since the feature extractor was frozen, only the final classifier layer was trained on this dataset. This limited how much the model could adapt to the specific characteristics of the images used in this project.

The fine-tuned EfficientNet-B0 model achieved the best performance, with 85.00% test accuracy. By unfreezing the final feature layers, the model was able to adjust some of its higher-level visual representations to better fit the dataset, while still retaining most of the useful general features learned during pretraining. This likely explains why fine-tuning outperformed both the baseline CNN and the frozen EfficientNet model.

The final fine-tuned EfficientNet-B0 confusion matrix is saved in `charts/finetuned_efficientnet_confusion_matrix.png`.

Overall, the results suggest that transfer learning was useful for this task, and that limited fine-tuning helped improve model performance. However, the dataset size was still small, so the model may not generalise well to broader real-world images. The model should also be interpreted carefully, as the labels are based on dataset folder categories and should not be treated as a reliable or universal measure of gender identity.

## 9. Limitations

- Small dataset size
- Image quality, lighting, and pose varied across examples
- Some images required orientation correction
- Labels were based only on dataset folder names
- Model performance may not generalise well to real-world images

## 10. Conclusion

This project demonstrated a complete computer vision workflow using PyTorch. A simple CNN was first trained as a baseline, followed by transfer learning and fine-tuning using EfficientNet-B0. Fine-tuning achieved the best performance, improving test accuracy to 85.00%.

The project helped demonstrate image preprocessing, CNN modelling, transfer learning, evaluation using classification metrics, and responsible interpretation of model outputs.