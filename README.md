# Take_away_task

Analytics for Business and Society [HST - 105]: ISL labs and a replication of a research paper.

Team Members: Aashit Kumar Singh and Dhruv Singh Bisht

**The main report is online_shopper_intention_prediction/report_summary.docx** (the same text is in report_summary.md).

## The two parts
1. **ISL_labs:** the lab notebooks from "An Introduction to Statistical Learning with Python" (Ch02 to Ch13), each run and annotated with "My notes" cells.
2. **online_shopper_intention_prediction:** our replication and extension of Sakar et al. (2019), "Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks".

## Setup
pip install -r requirements.txt

## Project structure

```
takeaway-task/
├── README.md
├── requirements.txt
├── .gitignore
│
├── ISL_labs/                                           # ISL lab notebooks with our notes
│   ├── Ch02_lab.ipynb                                  # Statistical learning (Python basics)
│   ├── Ch03_linreg_lab.ipynb                           # Linear regression
│   ├── Ch04_classification_lab.ipynb                   # Logistic regression, LDA, QDA, KNN
│   ├── Ch05_resample_lab.ipynb                         # Cross-validation, bootstrap
│   ├── Ch06_varselect_lab.ipynb                        # Subset selection, ridge, lasso, PCR
│   ├── Ch07_nonlin_lab.ipynb                           # Polynomials, splines, GAMs
│   ├── Ch08_baggboost_lab.ipynb                        # Trees, random forests, boosting, BART
│   ├── Ch09_svm_lab.ipynb                              # Support vector machines
│   ├── Ch10_deeplearning_lab.ipynb                     # Neural networks, CNN, RNN, LSTM
│   ├── Ch11_survival_lab.ipynb                         # Survival analysis
│   ├── Ch12_unsupervised_lab.ipynb                     # PCA, K-means, hierarchical clustering
│   ├── Ch13_multiple_lab.ipynb                         # Multiple testing
│   ├── Auto.csv, Auto.data                             # Data used in Ch02 and later labs
│   ├── imagenet_class_index.json                       # Class labels for the pretrained CNN (Ch10)
│   ├── book_images/                                    # Photos labelled by ResNet50 (Ch10)
│   └── figures/                                        # Plots saved by the Ch02 graphics section
│
└── online_shopper_intention_prediction/                # Replication of Sakar et al. (2019)
    ├── README.md                                       # Guide to this folder
    ├── report_summary.docx                             # MAIN REPORT (Word)
    ├── report_summary.md                               # Same report as text
    ├── report_figures/                                 # Figures used in the report
    ├── 01_data_exploration.ipynb                       # Data checks and plots
    ├── 02_preprocessing.ipynb                          # Encoding, split, scaling, oversampling
    ├── 03_baselines_tree_rf_svm.ipynb                  # Decision tree, random forest, SVM
    ├── 04_mlp_purchase_intention.ipynb                 # Module 1: MLP
    ├── 05_feature_ranking.ipynb                        # Correlation, mutual information, mRMR
    ├── 06_mlp_feature_selection.ipynb                  # MLP on selected attributes, final comparison
    ├── 07_synthetic_clickstream.ipynb                  # Synthetic data for Module 2
    ├── 08_lstm_abandonment.ipynb                       # Module 2: LSTM and comparison models
    ├── 10_extensions.ipynb                             # Our extensions (extra models, SMOTE, cost threshold)
    ├── utils.py                                        # Helper functions used by the notebooks
    ├── results/                                        # Saved results of every run (CSV files)
    ├── data/                                           # UCI dataset and the synthetic clickstream data
    ├── 01 data - about.pdf                             # Dataset description (UCI)
    └── 02 paper 2019.pdf                               # The research paper
```

## Notes
- Ch10 skips the CIFAR100 CNN section because the dataset download was too slow.
- The Ch10 datasets (MNIST, IMDB) download automatically when the notebook runs. They are not stored in this folder.
- The paper's clickstream data is private, so the LSTM part uses synthetic data (with our professor's approval). Only the method is replicated there.
- Module 1 results are averaged over 10 random splits. The paper uses 100.
- There is no notebook 09: the final model comparison is at the end of notebook 06.