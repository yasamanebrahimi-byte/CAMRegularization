# Research Question Definition

## Complete Question

Dropout regularization consists of dropping selected neurons (in the hidden layers and input layer) during various stages of training a neural network. This has the effect of forcing more of the network to be active, and can improve accuracy, reduce overfitting, and so on. For CNNs, a somewhat analogous technique is cutout regularization, where selected parts of the image are blocked out, forcing the model to learn based on parts of the image that might otherwise be ignored. Typically, dropouts and cutouts are selected at random. However, a recent paper considered a clever use of heatmaps (based on Class Activation Mappings, or CAM) to determine which areas of an image a CNN is focusing most attention on, and then applied cutouts to those “hot” areas. This is a more intelligent and focused approach, as compared to random cutouts, and it should be more effective in the sense of reducing training time and potentially improving the results. For a project, we would want to implement such a heatmap technique, and conduct extensive experiments to determine how models trained based on “heatmap cutouts” compare to analogous models trained on random cutouts and models trained with no cutouts. These experiments could involve any type of dataset (there are many good malware datasets that would be appropriate, for example).

## Summary

- Dropout regularization randomly drops neurons to prevent overtraining
- CNN equivalent is cutout regularization
  - CNNs are best for images
- New technique: instead of cutout being random, we can use heatmaps
  - Find which areas of an image the CNN is focusing on, and cut those out

# Literature Review

Reference paper: https://www.sciencedirect.com/science/article/pii/S2214212625001000

## Datasets

- MalImg
- Big2015
- VX-Zoo

## Background

- Malware visualization converts malware binaries into images so CNNs can classify them
- This approach achieves high accuracy, but has poor replicability
- CNN classifiers are also black-box classifiers, so low explicability
- A saliency map is a visualization that shows which parts of an input most influenced a model’s prediction
- Saliency maps can highlight image regions that influence prediction, but their interpretation is subjective and not standardized

## Methodology

1. Datasets
   - The authors use three datasets to ensure robustness and generalization
     - MalImg and Big2015 as standard benchmark datasets
     - VX-Zoo for generalizing to newer malware
2. CNN Models
   - The authors re-implemented 6 CNNs from cited papers
   - Missing or unclear hyperparameters in original papers are inferred or tuned empirically
     - Custom CNN
     - VGG16
     - ResNet50
     - IMCFN (modified VGG16)
     - DenseNet121
     - EfficientNet-B0
3. Model Interpretation
   - Explainability is studied using Class Activation Maps (CAMs)
     - GradCAM: standard gradient-weighted saliency maps
     - HiResCAM: higher-resolution variant that preserves feature importance
   - Sample-level heatmaps are generated and aggregated into cumulative heatmaps per malware family
   - Analyzed whether different CNNs focus on different image regions when classifying the same family or not
4. Comparisons
   - To compare explainability across models, the authors use:
     - Structural Similarity Index (SSIM) to measure similarity between cumulative heatmaps
     - Cumulative-SSIM (new metric) to compare how consistently different CNNs explain the same malware family
   - This allows ranking CNNs not just by accuracy, but by stability and consistency of explanations
5. Masking Strategy (Performance)
   - Explainability is actively exploited via a masking technique
   - The heatmaps of two CNNs are merged via logical OR
   - Regions deemed **unimportant** (below a threshold) are masked out
   - The classifier is forced to focus on relevant regions only
6. ViT as a Validation Classifier
   - The masked datasets are used to train a ViT to avoid CNNs explaining CNNs
   - Improved ViT performance shows that explainability insights are transferable

# Project Outline

## Datasets

- CIFAR-100 for classification, many classes
- CIFAR-10 as a sanity check dataset

## Methodology

1. Datasets
   - Could use multiple datasets for robustness and generalization
2. CNN Models
   - Five or six standard CNN models for comparability, including pretrained models
     - ResNet-18
     - WideResNet-28-10
     - DenseNet-121
     - EfficientNet-B0
     - ConvNeXt-Tiny
   - Models are trained using identical hyperparameters across all conditions in order to isolate the effect of the masking
   - A warm-up phase with no masking to set the saliency estimates
3. Model Interpretation
   - Use CAM and HiResCAM during training to guide the masking decisions
4. Masking Strategies
   - No masking and random masking (control)
   - Denoising with CAMs
   - Regions with low saliency are masked, similar to referenced paper
   - Regularization with CAMs (similar to dropout)
   - Regions with high saliency are masked so model focuses on ignored features
   - All masking strategies:
     - Mask the same fraction of image area
     - Use identical mask shapes and fill values
     - Are evaluated across multiple masking strengths (e.g., 10%, 20%)
5. Metrics
   - Models could be compared using
     - Accuracy
     - Learning curves and convergence speed
     - Steps or epochs required to reach a target performance
   - Robustness can be tested using
     - Multiple random seeds
     - Mean plus/minus standard deviation
6. Research Questions
   - A direct empirical comparison between:
     - Saliency-guided masking of unimportant regions
     - Saliency-guided masking of important regions
     - Random masking
     - No masking
   - Evidence that CAMs can be used as training signals
   - Insights for when the CAM-guided regularization could improve generalization

# Running Optuna Mask Tuning

Use `tune_optuna.py` to optimize masking hyperparameters with:

- objective = best validation accuracy (`best_val_acc`)
- pruning of weak trials (Hyperband)
- multi-fidelity training (short budget first, then promote promising trials)

Supported masking modes:

- `all`
- `random`
- `cam_high`
- `cam_low`

Automatic number of trials by masking mode:

- `all` -> 100
- `cam_high` or `cam_low` -> 64
- `random` -> 32

Example (single masking mode):

```bash
python tune_optuna.py --dataset cifar100 --model resnet18 --masking_type cam_high
```

Example (all parser arguments):

```bash
python tune_optuna.py --dataset cifar100 --model resnet18 --runs_root ./runs --n_jobs 1 --masking_type all --min_resource_epochs 15 --max_resource_epochs 100 --reduction_factor 2
```
