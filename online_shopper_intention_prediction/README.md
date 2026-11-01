The main report is report_summary.docx (the text is also in report_summary.md).

# Online shoppers' purchasing intention: replication and extension

By Aashit Kumar Singh and Dhruv Singh Bisht, for the course Analytics for Business and Society.

We replicate and extend Sakar, Polat, Katircioglu and Kastro (2019), "Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks".

## What is in this folder

| File or folder | What it is for |
|---|---|
| 01_data_exploration.ipynb | Checks the data and reproduces Tables 1 and 2 |
| 02_preprocessing.ipynb | Explains the preparation steps and shows why oversampling comes after the split | 
| 03_baselines_tree_rf_svm.ipynb | Decision tree, random forest and SVM (Tables 3, 5, 6, 8) | 
| 04_mlp_purchase_intention.ipynb | MLP with 10, 20 and 40 hidden neurons (Tables 4, 7) | 
| 05_feature_ranking.ipynb | Ranks attributes with correlation, mutual information and mRMR (Table 10) | 
| 06_mlp_feature_selection.ipynb | MLP on the top attributes, and the comparison of the best models (Tables 11, 9) | 
| 07_synthetic_clickstream.ipynb | Generates the SYNTHETIC page-view data | 
| 08_lstm_abandonment.ipynb | LSTM and three simpler models on the synthetic data (Figure 2, Algorithm 1) | 
| 10_extensions.ipynb | Extra models, SMOTE, and threshold and cost | 
| utils.py | Shared settings and helper functions | 
| results/ | Saved metrics for each repeat, and the feature rankings |
| report_figures/ | Report figures |
| data/ | The real dataset, and the SYNTHETIC data from notebook 07 | 
| 01 data - about.pdf | Description of the dataset | 
| 02 paper 2019.pdf | The paper we replicate | 
| report_summary.docx | The main report | 
| report_summary.md | The text of the report, used to build the Word file | 
| README.md | This guide | 

There is no notebook 09, because that comparison is at the end of notebook 06.

## How to run

Install the packages from the repository root with pip install -r requirements.txt. Then open a terminal in this folder and run the notebooks in number order, using Restart and Run All. Notebook 06 needs the results of 03, 04 and 05, and notebook 08 needs the data from 07. On a GPU, notebooks 04, 06 and 08 took about 1.5, 7 and 9 minutes. Notebook 03 took about 6 minutes, and the others about a minute or less.

N_REPEATS sets the number of random splits. We used 10, and the paper uses 100. Raising it to 100 makes notebooks 03, 04, 06 and 10 about ten times slower. Notebook 08 uses N_SEEDS instead, which we set to 5.

QUICK_TEST is in notebooks 04, 06 and 08. Set it to True for a tiny check that the code runs, with 3 repeats, a small data subset and 2 epochs. It writes separate results/quick_ files, and for notebook 06 you must run notebook 04 in quick mode first. Its numbers mean nothing, so set it back to False for a real run.