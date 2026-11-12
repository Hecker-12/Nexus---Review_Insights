# Evaluation Upgrade: Statistical Rigor & Detailed Slices

This document records the empirical results of the Phase 1 evaluation upgrade on the 10,000 held-out test reviews from `data/processed/test_preds.parquet`. All bootstrap confidence intervals reflect 2,000 resamples with random seed 42.

## 1. Model Performance with 95% Bootstrap Confidence Intervals

| Model | Metric | Value | 95% CI Lower | 95% CI Upper |
|:---|:---:|:---:|:---:|:---:|
| VADER default | accuracy | 0.7189 | 0.7096 | 0.7274 |
| VADER default | macro_f1 | 0.7041 | 0.6948 | 0.7131 |
| VADER tuned (0.7) | accuracy | 0.7707 | 0.7625 | 0.7789 |
| VADER tuned (0.7) | macro_f1 | 0.7706 | 0.7623 | 0.7787 |
| TF-IDF+LogReg | accuracy | 0.9017 | 0.8962 | 0.9072 |
| TF-IDF+LogReg | macro_f1 | 0.9017 | 0.8962 | 0.9072 |
| DistilBERT zero-shot | accuracy | 0.8980 | 0.8922 | 0.9036 |
| DistilBERT zero-shot | macro_f1 | 0.8980 | 0.8922 | 0.9036 |
| DistilBERT fine-tuned | accuracy | 0.9468 | 0.9426 | 0.9512 |
| DistilBERT fine-tuned | macro_f1 | 0.9468 | 0.9426 | 0.9512 |
| Ensemble (TF-IDF + FT) | accuracy | 0.9474 | 0.9432 | 0.9515 |
| Ensemble (TF-IDF + FT) | macro_f1 | 0.9474 | 0.9432 | 0.9515 |
| Fine-tuned minus TF-IDF | accuracy_diff | 0.0451 | 0.0396 | 0.0506 |
| Zero-shot minus TF-IDF | accuracy_diff | -0.0037 | -0.0102 | 0.0028 |
| Fine-tuned minus Zero-shot | accuracy_diff | 0.0488 | 0.0433 | 0.0541 |

## 2. McNemar Exact Discordant Tests

Exact two-sided binomial test on discordant pairs ($p=0.5$):

| Comparison | Model 1 Right, Model 2 Wrong (b) | Model 1 Wrong, Model 2 Right (c) | Total Discordant (b + c) | Two-Sided p-value |
|:---|:---:|:---:|:---:|:---:|
| Fine-tuned vs TF-IDF | 652 | 201 | 853 | 2.86298727e-56 |
| Zero-shot vs TF-IDF | 556 | 593 | 1149 | 2.88211301e-01 |
| Fine-tuned vs Zero-shot | 659 | 171 | 830 | 2.86589608e-68 |
| Ensemble vs Fine-tuned | 40 | 34 | 74 | 5.61380744e-01 |

## 3. Confidence Calibration & Expected Calibration Error (ECE)

Calculated across 10 equal-width bins on $[0.50, 1.00]$. Overall ECE values:

- **TF-IDF+LogReg**: ECE = **0.0943**
- **DistilBERT zero-shot**: ECE = **0.0849**
- **DistilBERT fine-tuned**: ECE = **0.0295**

ece_running_total is the cumulative sum of bin contributions; its last row per model is the model's ECE.

| Model | Bin Index | Bin Range | Count | Mean Confidence | Empirical Accuracy | Cumulative ECE (ece_running_total) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| TF-IDF+LogReg | 0 | [0.50, 0.55] | 568 | 0.5261 | 0.5616 | 0.002016 |
| TF-IDF+LogReg | 1 | [0.55, 0.60] | 551 | 0.5753 | 0.6116 | 0.004015 |
| TF-IDF+LogReg | 2 | [0.60, 0.65] | 615 | 0.6258 | 0.7772 | 0.013331 |
| TF-IDF+LogReg | 3 | [0.65, 0.70] | 676 | 0.6765 | 0.8210 | 0.023101 |
| TF-IDF+LogReg | 4 | [0.70, 0.75] | 810 | 0.7249 | 0.8877 | 0.036283 |
| TF-IDF+LogReg | 5 | [0.75, 0.80] | 921 | 0.7748 | 0.9316 | 0.050723 |
| TF-IDF+LogReg | 6 | [0.80, 0.85] | 1131 | 0.8262 | 0.9549 | 0.065282 |
| TF-IDF+LogReg | 7 | [0.85, 0.90] | 1393 | 0.8760 | 0.9813 | 0.079956 |
| TF-IDF+LogReg | 8 | [0.90, 0.95] | 1750 | 0.9258 | 0.9857 | 0.090437 |
| TF-IDF+LogReg | 9 | [0.95, 1.00] | 1585 | 0.9721 | 0.9962 | 0.094261 |
| DistilBERT zero-shot | 0 | [0.50, 0.55] | 39 | 0.5235 | 0.6923 | 0.000658 |
| DistilBERT zero-shot | 1 | [0.55, 0.60] | 46 | 0.5768 | 0.4348 | 0.001312 |
| DistilBERT zero-shot | 2 | [0.60, 0.65] | 55 | 0.6257 | 0.5455 | 0.001753 |
| DistilBERT zero-shot | 3 | [0.65, 0.70] | 55 | 0.6788 | 0.5273 | 0.002586 |
| DistilBERT zero-shot | 4 | [0.70, 0.75] | 68 | 0.7245 | 0.4412 | 0.004513 |
| DistilBERT zero-shot | 5 | [0.75, 0.80] | 47 | 0.7772 | 0.5957 | 0.005366 |
| DistilBERT zero-shot | 6 | [0.80, 0.85] | 84 | 0.8283 | 0.5714 | 0.007523 |
| DistilBERT zero-shot | 7 | [0.85, 0.90] | 119 | 0.8779 | 0.5798 | 0.011070 |
| DistilBERT zero-shot | 8 | [0.90, 0.95] | 229 | 0.9289 | 0.5895 | 0.018843 |
| DistilBERT zero-shot | 9 | [0.95, 1.00] | 9258 | 0.9964 | 0.9250 | 0.084886 |
| DistilBERT fine-tuned | 0 | [0.50, 0.55] | 54 | 0.5257 | 0.4815 | 0.000239 |
| DistilBERT fine-tuned | 1 | [0.55, 0.60] | 56 | 0.5758 | 0.6250 | 0.000514 |
| DistilBERT fine-tuned | 2 | [0.60, 0.65] | 62 | 0.6216 | 0.5806 | 0.000768 |
| DistilBERT fine-tuned | 3 | [0.65, 0.70] | 66 | 0.6774 | 0.5303 | 0.001739 |
| DistilBERT fine-tuned | 4 | [0.70, 0.75] | 62 | 0.7244 | 0.6290 | 0.002331 |
| DistilBERT fine-tuned | 5 | [0.75, 0.80] | 81 | 0.7750 | 0.5309 | 0.004308 |
| DistilBERT fine-tuned | 6 | [0.80, 0.85] | 102 | 0.8276 | 0.6961 | 0.005649 |
| DistilBERT fine-tuned | 7 | [0.85, 0.90] | 156 | 0.8798 | 0.7436 | 0.007774 |
| DistilBERT fine-tuned | 8 | [0.90, 0.95] | 306 | 0.9295 | 0.7843 | 0.012215 |
| DistilBERT fine-tuned | 9 | [0.95, 1.00] | 9055 | 0.9939 | 0.9748 | 0.029481 |

## 4. Slice Analysis

| Slice Category | Slice Name | Sample Size (n) | Small n (< 200) | TF-IDF Accuracy | Zero-Shot Accuracy | Fine-Tuned Accuracy |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Review Length Quartiles | Q1 (15 to 43 words) | 2513 | No | 0.9180 | 0.9069 | 0.9471 |
| Review Length Quartiles | Q2 (44 to 72 words) | 2527 | No | 0.8995 | 0.8900 | 0.9446 |
| Review Length Quartiles | Q3 (73 to 110 words) | 2482 | No | 0.9053 | 0.9061 | 0.9472 |
| Review Length Quartiles | Q4 (111 to 201 words) | 2478 | No | 0.8838 | 0.8890 | 0.9483 |
| Fine-Tuned Confidence Bands | < 0.70 | 238 | No | 0.5336 | 0.4958 | 0.5546 |
| Fine-Tuned Confidence Bands | 0.70 to 0.90 | 401 | No | 0.6284 | 0.5910 | 0.6708 |
| Fine-Tuned Confidence Bands | > 0.90 | 9361 | No | 0.9228 | 0.9214 | 0.9686 |
| Product Themes (>= 100 test reviews) | Unclustered (Topic -1) | 4557 | No | 0.8962 | 0.8995 | 0.9434 |
| Product Themes (>= 100 test reviews) | Books (Topic 0) | 1875 | No | 0.9051 | 0.9045 | 0.9504 |
| Product Themes (>= 100 test reviews) | Movies and DVDs (Topic 1) | 1054 | No | 0.9099 | 0.9108 | 0.9554 |
| Product Themes (>= 100 test reviews) | Music CDs and albums (Topic 2) | 632 | No | 0.9082 | 0.8924 | 0.9462 |
| Product Themes (>= 100 test reviews) | Coffee makers (Topic 4) | 246 | No | 0.9065 | 0.9024 | 0.9512 |
| Product Themes (>= 100 test reviews) | Toys and gifts (Topic 3) | 225 | No | 0.8933 | 0.9111 | 0.9422 |
| Product Themes (>= 100 test reviews) | Supplements, food and personal care (Topic 5) | 187 | Yes | 0.9091 | 0.8930 | 0.9679 |
| Product Themes (>= 100 test reviews) | Video games (Topic 6) | 155 | Yes | 0.9097 | 0.9032 | 0.9484 |
| Product Themes (>= 100 test reviews) | Batteries and chargers (Topic 7) | 147 | Yes | 0.8980 | 0.8639 | 0.9456 |
| Product Themes (>= 100 test reviews) | Headphones and phones (Topic 8) | 107 | Yes | 0.9626 | 0.9533 | 0.9813 |

### Fine-Tuned Accuracy by Confidence Band

| Confidence Band | Review Count (n) | Fine-Tuned Accuracy |
|:---|:---:|:---:|
| < 0.70 | 238 | 0.5546 |
| 0.70 to 0.90 | 401 | 0.6708 |
| > 0.90 | 9361 | 0.9686 |

Excluding the 639 reviews with fine-tuned confidence below 0.90, accuracy on the remaining 9,361 reviews is 0.9686. The band edges were fixed in advance; this is descriptive and not a tuned threshold.


## 5. Methodological Limits

- **Sampling variability**: Confidence intervals reflect test set sampling variability only and do not account for data collection bias or annotation shifts.
- **Confidence calibration**: Calibration is evaluated against binary correctness using the maximum predicted class probability.
- **Sample size in slices**: Several slices have sample size below 200 reviews (`small_n = True`), increasing variance in slice-level estimates.
- **Untuned ensemble**: The ensemble model combines TF-IDF and fine-tuned transformer probabilities using fixed equal weights (0.5 / 0.5) without parameter tuning.
- **Single test set**: All evaluations are conducted on a single held-out 10,000-review test split.
