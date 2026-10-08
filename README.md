# Selective Classification and Learning to Defer

Project for the **Symbolic and Evolutionary Artificial Intelligence** course in the **MSc in Artificial Intelligence and Data Engineering, University of Pisa**.

**Authors:** Tommaso Falaschi and Rossana Antonella Sacco.

This repository investigates **reliable image classification under uncertainty** through two approaches: **SelectiveNet on CIFAR-10** and **Learning to Defer on CIFAR-10H**. Both learn when the model should make a prediction, while Learning to Defer also accounts for the performance of an external expert.

## Project objective

A classifier can improve the accuracy of the predictions it retains by abstaining on selected inputs. When an expert is available, the model can instead learn to delegate decisions to that expert and optimize the accuracy of the combined system.

The project implements and evaluates both settings:

| Experiment | Decision mechanism | Main evaluation objective |
| --- | --- | --- |
| **SelectiveNet** | Predict a class or abstain using a learned selection function | Reduce error on accepted samples while meeting a target coverage |
| **Learning to Defer** | Predict a class or defer to a simulated human expert | Improve the accuracy of the combined human-AI system |

**Coverage** is the fraction of inputs handled directly by the model. **Selective accuracy** measures accuracy only on accepted or non-deferred inputs, and **selective risk** is their error rate. **System accuracy** in Learning to Defer includes both model predictions on non-deferred inputs and human predictions on deferred inputs.

## Repository contents

| Resource | Contents |
| --- | --- |
| [selectivenet-cifar10](selectivenet-cifar10) | SelectiveNet notebook, saved metrics, and figures |
| [learning2defer-cifar10h](learning2defer-cifar10h) | Learning to Defer notebook, human annotation distributions, checkpoints, metrics, and figures |
| [SelectiveNet notebook](selectivenet-cifar10/selectivenet_cifar10.ipynb) | Training, threshold calibration, evaluation, and qualitative analysis |
| [Learning to Defer notebook](learning2defer-cifar10h/learning2defer_cifar10H.ipynb) | Backbone pretraining, deferral learning, human-AI evaluation, and example inspection |
| [SelectiveNet release](https://github.com/LeBonWskii/SEAI-Project/releases/tag/selectivenet-cifar10) | Pretrained weights for the primary and additional SelectiveNet runs |

## Datasets and experimental setup

Both experiments use **32 × 32 RGB images from the 10 CIFAR-10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

- **SelectiveNet:** the 50,000 CIFAR-10 training images are split into 45,000 training and 5,000 validation samples, stratified by class. Evaluation uses the standard 10,000-image test set.
- **Learning to Defer:** ResNet-18 is pretrained using the CIFAR-10 training set, with a 45,000/5,000 training/validation split. The deferral experiment uses the 10,000 CIFAR-10 test images and their **CIFAR-10H human label distributions**, split into 7,000 training, 1,000 validation, and 2,000 test samples.
- **Human expert simulation:** one human prediction is sampled per image from its CIFAR-10H annotation distribution using a seeded random generator. These sampled predictions are kept fixed within the experiment.
- The primary experiments use **seed 42**.

The two experiments use different test sets and evaluate different outcomes. Their reported accuracies describe their respective protocols and should be interpreted accordingly.

## 1. SelectiveNet on CIFAR-10

### Implementation

The notebook implements an end-to-end selective classifier in **Keras with the TensorFlow backend**. A shared convolutional feature extractor feeds three components:

- A **prediction head** producing probabilities for the 10 image classes.
- A **selection head** producing an acceptance score.
- An **auxiliary classification head** supporting representation learning during training.

Training combines classification loss on selected samples, a penalty for falling below the desired coverage, and an auxiliary classification loss. The primary configuration uses:

| Parameter | Value |
| --- | ---: |
| Target coverage | 0.80 |
| Selective loss weight (`ALPHA`) | 0.50 |
| Coverage penalty coefficient | 32 |
| Batch size | 128 |
| Training epochs | 300 |
| Initial learning rate | 0.1 |

Optimization uses SGD with momentum and Nesterov acceleration, a learning rate halved every 25 epochs, and checkpointing based on validation loss. Input processing includes normalization, random horizontal flips, translations, and rotations.

### Evaluation and analysis

The operating threshold is calibrated from validation selection scores to target 80% coverage. The notebook evaluates both this calibrated threshold and a fixed threshold, and includes an additional run monitored at a fixed threshold of 0.80 alongside the primary run monitored at 0.50.

Analysis includes risk-coverage and coverage-accuracy curves, **AURC** and **E-AURC**, selection-score distributions, and visual inspection of accepted, rejected, misclassified, and near-threshold examples. Individual decision cards show the image, predicted probabilities, selection score, threshold, and acceptance decision.

### Saved results

The primary experiment's [saved summary](selectivenet-cifar10/results/selectivenet_cifar10_cov80_summary.json) reports:

| Operating point | Test coverage | Selective accuracy | Selective risk | Accepted images |
| --- | ---: | ---: | ---: | ---: |
| Validation-calibrated threshold (approximately 0.977789) | 79.35% | 98.94% | 1.06% | 7,935 |
| Fixed threshold of 0.50 | 83.11% | 98.56% | 1.44% | 8,311 |

The auxiliary head's accuracy across the full test set is **92.54%**. Selective accuracy is computed only on accepted images.

The additional run's saved notebook output reports **79.37% coverage** and **99.09% selective accuracy** at its validation-calibrated threshold.

## 2. Learning to Defer on CIFAR-10H

### Implementation

The notebook implements a **Realizable Surrogate** learning-to-defer pipeline in **PyTorch**:

1. Pretrain a **ResNet-18** classifier on CIFAR-10 using cross-entropy and AdamW, selecting the backbone with the best validation accuracy.
2. Adapt ResNet-18 to small images using a 3 × 3 first convolution with stride 1 and no initial max-pooling layer.
3. Extend the final layer from **10 to 11 outputs**, adding a deferral option.
4. Transfer the pretrained parameters and, in the saved configuration, freeze the backbone while training the final layer.
5. Combine a deferral-aware surrogate loss with a classification loss, accounting for whether the simulated human prediction is correct.
6. Select the loss trade-off parameter `alpha` and the rejection threshold using validation system accuracy.

The configured training stages use **120 pretraining epochs** and **80 deferral epochs**, with learning rates of `1e-3` and `3e-3`, respectively. The candidate alpha values are `[0.0, 0.1, 0.3, 0.5, 0.9, 1.0]`.

At inference time, the model compares the difference between the deferral probability and its highest class probability with the selected threshold.

### Saved results

The [saved summary](learning2defer-cifar10h/results/summary_metrics.json) reports the following results on the 2,000-image test split:

| Metric | Value |
| --- | ---: |
| Classifier accuracy on all test inputs | 91.00% |
| Simulated human accuracy on all test inputs | 95.50% |
| Model coverage | 75.95% |
| Classifier accuracy on non-deferred inputs | 98.03% |
| Simulated human accuracy on deferred inputs | 90.23% |
| **Combined system accuracy** | **96.15%** |
| Inputs handled by the classifier | 1,519 |
| Inputs deferred to the simulated human | 481 |

The selected configuration uses **alpha = 1.0**, a rejection threshold of approximately **-0.221435**, and a frozen backbone.

In this saved run, the combined system improves accuracy by **5.15 percentage points** over the classifier alone and **0.65 percentage points** over the simulated human alone.

The notebook also plots system accuracy as coverage varies, compares classifier and human accuracy on their operational subsets, and visualizes examples where classification or deferral succeeds or fails.

## Running the notebooks

The notebooks are configured for **Google Colab**, with default paths under `/content`. Open either notebook and execute its cells in order; the experiments can be run independently.

Main dependencies:

| Experiment | Main libraries |
| --- | --- |
| SelectiveNet | TensorFlow, Keras, NumPy, Matplotlib, scikit-learn |
| Learning to Defer | PyTorch, torchvision, NumPy, Matplotlib, scikit-learn, requests, Pillow, tqdm |

The SelectiveNet architecture visualization also requires the Graphviz/pydot tooling used by `plot_model`. No dependency file with pinned versions is included.

### SelectiveNet

- Keep `RUN_MODE = "demo"` and `FORCE_TRAIN = False` to load pretrained weights and run evaluation. Missing local weights are downloaded from the GitHub release.
- Set `RUN_MODE = "train"` to allow training when weights are unavailable. Set `FORCE_TRAIN = True` as well to retrain even when pretrained weights are available.
- `USE_DRIVE_FOR_TRAINING = True` saves training artifacts to Google Drive in Colab.

### Learning to Defer

- Keep `FORCE_TRAIN = False` to load local checkpoints or download them from the repository.
- Set `FORCE_TRAIN = True` to retrain the backbone and deferral model.
- The notebook downloads CIFAR-10 and the included `cifar10h-probs.npy` annotation distributions as needed, and uses CUDA when available.

For local execution, adjust the configured directories and disable Colab-specific Drive integration where applicable. A GPU is useful for training; checkpoint-based evaluation avoids repeating the training stages.

## Saved artifacts

| Artifact | Location |
| --- | --- |
| SelectiveNet primary metrics | [Summary JSON](selectivenet-cifar10/results/selectivenet_cifar10_cov80_summary.json) |
| SelectiveNet plots | [Figures](selectivenet-cifar10/figures) |
| SelectiveNet pretrained weights | [GitHub release](https://github.com/LeBonWskii/SEAI-Project/releases/tag/selectivenet-cifar10) |
| ResNet-18 and deferral checkpoints | [Checkpoints](learning2defer-cifar10h/checkpoints) |
| Learning to Defer summary | [Summary JSON](learning2defer-cifar10h/results/summary_metrics.json) |
| Learning to Defer metrics and coverage sweep | [Final metrics JSON](learning2defer-cifar10h/results/final_metrics.json) |
| Backbone training history | [History JSON](learning2defer-cifar10h/results/backbone_pretraining_history.json) |
| Learning to Defer plots | [Figures](learning2defer-cifar10h/figures) |

The results above are taken from saved experiment artifacts and notebook outputs.

## Acknowledgments

The notebooks identify the following original implementations as their starting points:

- [SelectiveNet original implementation](https://github.com/anonygit32/SelectiveNet/tree/master).
- [Human-AI Deferral original implementation](https://github.com/clinicalml/human_ai_deferral).
