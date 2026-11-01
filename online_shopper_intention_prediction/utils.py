"""Shared settings and helper functions for the Sakar et al. (2019) replication."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

DATA_PATH = "data/online_shoppers_intention.csv"
RESULT_DIR = "results"
SEED = 42  # seed for single runs; repeated runs use seeds 0, 1, 2, ...

NUM_COLS = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
CAT_COLS = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
ALL_COLS = NUM_COLS + CAT_COLS + ["Weekend"]  # the 17 original attributes

METRICS = ["accuracy", "tpr", "tnr", "f1"]
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- Plot style ----------
def set_style():
    """Same look for every plot in every notebook."""
    sns.set_theme(style="whitegrid", palette=COLOURS)
    plt.rcParams.update({
        "figure.figsize": (9, 5),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "grid.color": "#e1e0d9",
        "grid.alpha": 0.8,
    })


# ---------- Data ----------
def load_data(path=DATA_PATH):
    """Read the CSV and turn Revenue and Weekend into 0/1."""
    df = pd.read_csv(path)
    df["Revenue"] = df["Revenue"].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)
    return df


def make_inputs(df, cols=None):
    """Return inputs X (categoricals one-hot coded) and target y. cols picks the attributes."""
    if cols is None:
        cols = ALL_COLS
    cats = [c for c in cols if c in CAT_COLS]
    X = pd.get_dummies(df[list(cols)], columns=cats, dtype=float).astype(float)
    return X, df["Revenue"]


def split_data(X, y, seed, oversample=False):
    """70/30 stratified split, scale with train statistics, oversample the train part only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed)

    # Fit the scaler on train only, so the test set does not influence training.
    cols = [c for c in NUM_COLS if c in X.columns]
    if cols:
        scaler = StandardScaler().fit(X_train[cols])
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[cols] = scaler.transform(X_train[cols])
        X_test[cols] = scaler.transform(X_test[cols])

    # Duplicate minority rows after the split, so no copy can reach the test set.
    if oversample:
        sampler = RandomOverSampler(random_state=seed)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test


# ---------- Metrics and tests ----------
def get_metrics(y_true, y_pred):
    """Accuracy, true positive rate, true negative rate and F1 (positive class = purchase)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "tpr": tp / (tp + fn) if tp + fn > 0 else 0.0,
        "tnr": tn / (tn + fp) if tn + fp > 0 else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def stars(p):
    """Paper's significance marks: ** p < 0.01, * p < 0.05, N.S. otherwise."""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "N.S."


def paired_test(a, b):
    """Paired t-test on two lists of per-repeat values. Returns p-value and stars."""
    p = ttest_rel(a, b).pvalue
    if np.isnan(p):  # both models gave identical values on every repeat
        p = 1.0
    return p, stars(p)


def add_balanced(table):
    """Add accuracy and F1 as they would be on a 50/50 test set, computed from TPR and TNR."""
    out = table.copy()
    fpr = 1 - out["tnr"]
    out["accuracy_5050"] = (out["tpr"] + out["tnr"]) / 2
    out["f1_5050"] = 2 * out["tpr"] / (1 + out["tpr"] + fpr)
    return out


# ---------- Results and tables ----------
def save_results(rows, name):
    """Save per-repeat rows to results/<name>.csv and return them as a DataFrame."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"{RESULT_DIR}/{name}.csv", index=False)
    return res


def compare_table(res, paper, balanced=False):
    """Mean of our metrics next to the paper's values. One row per model in `paper`."""
    means = res.groupby("model")[METRICS].mean()
    means = add_balanced(means)
    rows = []
    for model, values in paper.items():
        ours = means.loc[model]
        row = {
            "Model": model,
            "Accuracy %": ours["accuracy"] * 100, "Accuracy % (paper)": values[0],
            "TPR": ours["tpr"], "TPR (paper)": values[1],
            "TNR": ours["tnr"], "TNR (paper)": values[2],
            "F1": ours["f1"], "F1 (paper)": values[3],
        }
        if balanced:
            row["Accuracy % (50/50 test)"] = ours["accuracy_5050"] * 100
            row["F1 (50/50 test)"] = ours["f1_5050"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def show_table(table, caption, best=None, digits=2):
    """Styled table: caption, rounded numbers, best value in each `best` column highlighted."""
    style = table.style.format(precision=digits).set_caption(caption)
    if best:
        style = style.highlight_max(subset=best, props="background-color: #2a6f4f; color: white; font-weight: bold;")
    style = style.set_table_styles([{
        "selector": "caption",
        "props": [("font-weight", "bold"), ("font-size", "120%"), ("text-align", "left")],
    }])
    return style


# ---------- MLP (paper Section 2.2.1) ----------
def build_mlp(n_inputs, hidden):
    """One hidden layer of sigmoid units and one sigmoid output."""
    return nn.Sequential(
        nn.Linear(n_inputs, hidden), nn.Sigmoid(),
        nn.Linear(hidden, 1), nn.Sigmoid(),
    )


def to_tensor(values):
    """DataFrame or array to a float tensor on the GPU (if there is one)."""
    return torch.tensor(np.asarray(values), dtype=torch.float32, device=DEVICE)


def train_mlp(X, y, hidden, seed=0, max_epochs=300, grad_stop=0.2):
    """Full-batch RPROP on the summed squared error. Returns the model and the epochs used."""
    torch.manual_seed(seed)
    X_t = to_tensor(X)
    y_t = to_tensor(y)
    model = build_mlp(X_t.shape[1], hidden).to(DEVICE)
    optimizer = torch.optim.Rprop(model.parameters())

    epochs = 0
    for epochs in range(1, max_epochs + 1):
        optimizer.zero_grad()
        output = model(X_t).squeeze(1)
        # Summed (not averaged) error, so the 0.2 stopping rule means what it does in the paper.
        loss = 0.5 * ((y_t - output) ** 2).sum()
        loss.backward()

        # Paper's stopping rule: every partial derivative is below the threshold.
        largest = max(p.grad.abs().max().item() for p in model.parameters())
        if largest < grad_stop:
            break
        optimizer.step()
    return model, epochs


def score_mlp(model, X):
    """Sigmoid output of the network, as a numpy array."""
    with torch.no_grad():
        return model(to_tensor(X)).squeeze(1).cpu().numpy()


def predict_mlp(model, X, threshold=0.5):
    """Class labels using the paper's 0.5 threshold."""
    return (score_mlp(model, X) > threshold).astype(int)
