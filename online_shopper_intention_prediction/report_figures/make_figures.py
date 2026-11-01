"""Draw the report figures from saved results. Nothing is trained here.

Run from the online_shopper_intention_prediction folder:
    python report_figures/make_figures.py
"""
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import pandas as pd
import seaborn as sns

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
sns.set_theme(style="whitegrid")
plt.rcParams.update({"axes.titleweight": "bold", "grid.color": "#e1e0d9", "font.size": 11})


def add_5050(table):
    """Accuracy and F1 as they would be on a 50/50 test set, from TPR and TNR."""
    table = table.copy()
    table["acc"] = (table["tpr"] + table["tnr"]) / 2 * 100
    table["f1_5050"] = 2 * table["tpr"] / (1 + table["tpr"] + 1 - table["tnr"])
    return table


def model_comparison():
    """Our oversampled results next to the paper's."""
    base = pd.read_csv("results/baselines.csv")
    base = base[base["setting"] == "oversampled"]
    mlp = pd.read_csv("results/mlp.csv")
    mlp = mlp[(mlp["setting"] == "oversampled") & (mlp["model"] == "MLP 10")]
    feat = pd.read_csv("results/mlp_features.csv")
    feat = feat[feat["model"] == "mRMR"]
    best_k = add_5050(feat.groupby("setting")[["tpr", "tnr"]].mean())["acc"].idxmax()
    feat = feat[feat["setting"] == best_k]

    rows = []
    for name, part in [("C4.5", base[base["model"] == "C4.5"]), ("Random\nforest", base[base["model"] == "Random forest"]),
                       ("SVM\nlinear", base[base["model"] == "SVM linear"]), ("SVM\nRBF", base[base["model"] == "SVM RBF"]),
                       ("MLP\n10", mlp), ("MLP\nmRMR", feat)]:
        rows.append({"model": name, **add_5050(part[["tpr", "tnr"]].mean().to_frame().T).iloc[0].to_dict()})
    ours = pd.DataFrame(rows)
    paper_acc = [82.34, 82.29, 84.26, 84.88, 87.24, 87.94]
    paper_f1 = [0.82, 0.81, 0.82, 0.82, 0.86, 0.87]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = range(len(ours))
    width = 0.38
    for ax, values, paper, label in [(axes[0], ours["acc"], paper_acc, "Accuracy %"), (axes[1], ours["f1_5050"], paper_f1, "F1")]:
        ax.bar([i - width / 2 for i in x], values, width, color=COLOURS[0], label="Ours")
        ax.bar([i + width / 2 for i in x], paper, width, color=COLOURS[1], label="Paper")
        ax.set_xticks(list(x), ours["model"], fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(f"{label} on the 50/50 basis")
        ax.set_ylim(0, max(max(values), max(paper)) * 1.25)
        ax.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    fig.savefig("report_figures/model_comparison.png", dpi=170)
    plt.close(fig)


def ranking_heatmap():
    """Rank of each attribute under each filter."""
    ranks = pd.read_csv("results/rankings.csv")
    table = pd.DataFrame({
        "Correlation": {name: rank for rank, name in zip(ranks["rank"], ranks["correlation"])},
        "Mutual information": {name: rank for rank, name in zip(ranks["rank"], ranks["mutual_information"])},
        "mRMR": {name: rank for rank, name in zip(ranks["rank"], ranks["mrmr"])},
    }).loc[ranks["mrmr"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    sns.heatmap(table, annot=True, fmt="d", annot_kws={"size": 8},
                cmap=sns.light_palette("#104281", reverse=True, as_cmap=True),
                cbar_kws={"label": "Rank (1 = best)"}, ax=ax)
    ax.set_title("Feature rank under each filter")
    ax.set_xlabel("Filter")
    ax.set_ylabel("Attribute (sorted by mRMR rank)")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig.savefig("report_figures/ranking_heatmap.png", dpi=170)
    plt.close(fig)


def lstm_accuracy():
    """Accuracy against prediction horizon on the synthetic data."""
    res = pd.read_csv("results/lstm.csv")
    res["h"] = res["setting"].str[2:].astype(int)
    acc = res.groupby(["model", "h"])["accuracy"].mean().unstack(0) * 100
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for colour, marker, name in zip(COLOURS, ["o", "s", "^", "D"], ["LSTM", "MLP", "ELM", "RBF"]):
        ax.plot(acc.index, acc[name], marker=marker, color=colour, linewidth=2, markersize=7, label=name)
    ax.axhline(50, color="#898781", linestyle=":", label="Guessing (50%)")
    ax.set_title("Accuracy against prediction horizon (SYNTHETIC data)")
    ax.set_xlabel("Prediction horizon (number of pages)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    plt.tight_layout()
    fig.savefig("report_figures/lstm_accuracy.png", dpi=170)
    plt.close(fig)


def threshold_cost():
    """The cost curve is not stored in results/, so we copy the plot saved in notebook 10."""
    nb = nbformat.read("10_extensions.ipynb", as_version=4)
    cell = next(c for c in nb.cells if c.cell_type == "code" and "Total cost against threshold" in c.source)
    image = next(o["data"]["image/png"] for o in cell.outputs if "image/png" in o.get("data", {}))
    with open("report_figures/threshold_cost.png", "wb") as f:
        f.write(base64.b64decode(image))


model_comparison()
ranking_heatmap()
lstm_accuracy()
threshold_cost()
print("Saved 4 figures in report_figures/")
