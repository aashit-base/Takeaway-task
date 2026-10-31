# Take_away_task

Analytics for Business and Society: ISL labs + paper replication.

## Setup
pip install -r requirements.txt

## Project structure

```
takeaway-task/
├── README.md
├── requirements.txt
├── .gitignore
│
├── ISL_labs/                              ISL lab notebooks with my notes
│   ├── Ch02_lab.ipynb                     Statistical learning (Python basics)
│   ├── Ch03_linreg_lab.ipynb              Linear regression
│   ├── Ch04_classification_lab.ipynb      Logistic regression, LDA, QDA, KNN
│   ├── Ch05_resample_lab.ipynb            Cross-validation, bootstrap
│   ├── Ch06_varselect_lab.ipynb           Subset selection, ridge, lasso, PCR
│   ├── Ch07_nonlin_lab.ipynb              Polynomials, splines, GAMs
│   ├── Ch08_baggboost_lab.ipynb           Trees, random forests, boosting, BART
│   ├── Ch09_svm_lab.ipynb                 Support vector machines
│   ├── Ch10_deeplearning_lab.ipynb        Neural networks, CNN, RNN, LSTM
│   ├── Ch11_survival_lab.ipynb            Survival analysis
│   ├── Ch12_unsupervised_lab.ipynb        PCA, K-means, hierarchical clustering
│   ├── Ch13_multiple_lab.ipynb            Multiple testing
│   ├── Auto.csv                           Dataset used in Ch02 and later labs
│   ├── Auto.data                          Same data, whitespace-delimited
│   ├── imagenet_class_index.json          Class labels for the pretrained CNN (Ch10)
│   ├── book_images/                       Photos labelled by ResNet50 (Ch10)
│   └── figures/                           Plots saved by the Ch02 graphics section
│
└── online_shopper_intention_prediction/   Replication of Sakar et al. (2019)
    ├── README.md
    ├── 01 data - about.pdf                Dataset description (UCI)
    ├── 02 paper 2019.pdf                  The research paper
    ├── data/
    │   └── online_shoppers_intention.csv  UCI Online Shoppers dataset
    ├── 01_data_exploration.ipynb          Data checks and plots
    ├── 02_mlp_purchase_intention.ipynb    Module 1: MLP, SVM, Random Forest
    ├── 03_lstm_abandonment.ipynb          Module 2: LSTM (simulated data)
    ├── 04_extensions.ipynb                Extra methods and analysis
    └── report_summary.md                  Results and comparison with the paper
```

## About the two parts
- **ISL_labs:** lab code from "An Introduction to Statistical Learning with Python" (github.com/intro-stat-learning/ISLP_labs). Each notebook was run and annotated with "My notes" cells.
- **online_shopper_intention_prediction:** replication of "Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks" (Sakar et al., 2019).

## Notes
- Ch10 skips the CIFAR100 CNN section because the dataset download was too slow.
- The Ch10 datasets (MNIST, IMDB) download automatically when the notebook runs. They are not stored in this repo.
- The LSTM in the paper replication uses simulated data, because the paper's clickstream data is not available.