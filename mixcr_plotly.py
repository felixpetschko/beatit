import pandas as pd
import numpy as np
from scipy.stats import entropy
from pathlib import Path
import plotly.graph_objects as go

DATA_DIR = Path("data")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


# === Alpha Diversity Calculation ===
def calculate_full_alpha_metrics(read_counts):
    read_counts = read_counts[read_counts > 0]
    total_reads = read_counts.sum()
    proportions = read_counts / total_reads
    S_obs = len(read_counts)

    richness = S_obs

    freq = read_counts.value_counts()
    f1 = freq.get(1, 0)
    f2 = freq.get(2, 0)

    chao1 = S_obs + (f1**2) / (2 * f2) if f2 > 0 else np.nan
    jack1 = S_obs + f1
    jack2 = S_obs + 2 * f1 - f2 if f2 > 0 else np.nan

    rare_cutoff = 10
    rare = read_counts[read_counts <= rare_cutoff]
    abundant = read_counts[read_counts > rare_cutoff]
    S_rare = len(rare)
    N_rare = rare.sum()
    C_ace = 1 - (f1 / N_rare) if N_rare > 0 else np.nan
    gamma_sq = np.var(rare, ddof=1) / np.mean(rare) ** 2 if len(rare) > 1 else 0
    ace = len(abundant) + (S_rare / C_ace) + (f1 / C_ace) * gamma_sq if C_ace > 0 else np.nan

    shannon_H = entropy(proportions, base=np.e)
    simpson = np.sum(proportions ** 2)
    inv_simpson = 1 / simpson if simpson > 0 else np.nan
    gini_simpson = 1 - simpson
    hill_q1 = np.exp(shannon_H)
    hill_q2 = inv_simpson
    berger_parker = 1 / proportions.max()
    renyi_q2 = np.log(np.sum(proportions**2)) / (1 - 2)

    sorted_p = np.sort(proportions)[::-1]
    tail_index = sum([(1 - (i / len(sorted_p))) ** 2 * p for i, p in enumerate(sorted_p)])

    ef = hill_q1 / richness if richness > 0 else np.nan
    rle = np.log(hill_q1) / np.log(richness) if richness > 1 else np.nan
    pielou = shannon_H / np.log(richness) if richness > 1 else np.nan

    return {
        "Observed Richness": richness,
        "Chao1": chao1,
        "1st-order Jackknife": jack1,
        "2nd-order Jackknife": jack2,
        "ACE": ace,
        "Shannon Entropy": shannon_H,
        "Simpson Index": simpson,
        "Inverse Simpson": inv_simpson,
        "Gini-Simpson": gini_simpson,
        "Hill Number (q=1)": hill_q1,
        "Hill Number (q=2)": hill_q2,
        "Berger-Parker": berger_parker,
        "Rényi Entropy (q=2)": renyi_q2,
        "Tail Index": tail_index,
        "Evenness EF": ef,
        "Evenness RLE": rle,
        "Pielou Index": pielou,
    }


def get_alpha_metric_from_file(file_path, sample_name, metric_name):
    df = pd.read_csv(file_path, sep="\t")
    read_counts = df["readCount"]
    metrics = calculate_full_alpha_metrics(read_counts)
    return sample_name, metrics.get(metric_name, np.nan)


# === Define samples ===
samples = [
    (DATA_DIR / "A1_clones_IGH.tsv", "A1_IgM+_IGH"),
    (DATA_DIR / "A1_clones_IGK.tsv", "A1_IgM+_IGK"),
    (DATA_DIR / "A1_clones_IGL.tsv", "A1_IgM+_IGL"),
    (DATA_DIR / "A2_clones_IGH.tsv", "A2_IgM-_IGH"),
    (DATA_DIR / "A2_clones_IGK.tsv", "A2_IgM-_IGK"),
    (DATA_DIR / "A2_clones_IGL.tsv", "A2_IgM-_IGL"),
    (DATA_DIR / "A3_clones_IGH.tsv", "A3_IgM+_IGH"),
    (DATA_DIR / "A3_clones_IGK.tsv", "A3_IgM+_IGK"),
    (DATA_DIR / "A3_clones_IGL.tsv", "A3_IgM+_IGL"),
    (DATA_DIR / "A4_clones_IGH.tsv", "A4_IgM-_IGH"),
    (DATA_DIR / "A4_clones_IGK.tsv", "A4_IgM-_IGK"),
    (DATA_DIR / "A4_clones_IGL.tsv", "A4_IgM-_IGL"),
    (DATA_DIR / "A5_clones_IGH.tsv", "A5_IgM+_IGH"),
    (DATA_DIR / "A5_clones_IGK.tsv", "A5_IgM+_IGK"),
    (DATA_DIR / "A5_clones_IGL.tsv", "A5_IgM+_IGL"),
    (DATA_DIR / "A6_clones_IGH.tsv", "A6_IgM-_IGH"),
    (DATA_DIR / "A6_clones_IGK.tsv", "A6_IgM-_IGK"),
    (DATA_DIR / "A6_clones_IGL.tsv", "A6_IgM-_IGL"),
    (DATA_DIR / "A7_clones_IGH.tsv", "A7_IgM+_IGH"),
    (DATA_DIR / "A7_clones_IGK.tsv", "A7_IgM+_IGK"),
    (DATA_DIR / "A7_clones_IGL.tsv", "A7_IgM+_IGL"),
    (DATA_DIR / "A8_clones_IGH.tsv", "A8_IgM-_IGH"),
    (DATA_DIR / "A8_clones_IGK.tsv", "A8_IgM-_IGK"),
    (DATA_DIR / "A8_clones_IGL.tsv", "A8_IgM-_IGL"),
    (DATA_DIR / "A9_clones_IGH.tsv", "A9_IgM+_IGH"),
    (DATA_DIR / "A9_clones_IGK.tsv", "A9_IgM+_IGK"),
    (DATA_DIR / "A9_clones_IGL.tsv", "A9_IgM+_IGL"),
    (DATA_DIR / "A10_clones_IGH.tsv", "A10_IgM-_IGH"),
    (DATA_DIR / "A10_clones_IGK.tsv", "A10_IgM-_IGK"),
    (DATA_DIR / "A10_clones_IGL.tsv", "A10_IgM-_IGL"),
    (DATA_DIR / "A11_clones_IGH.tsv", "A11_IgM+_IGH"),
    (DATA_DIR / "A11_clones_IGK.tsv", "A11_IgM+_IGK"),
    (DATA_DIR / "A11_clones_IGL.tsv", "A11_IgM+_IGL"),
    (DATA_DIR / "A12_clones_IGH.tsv", "A12_IgM-_IGH"),
    (DATA_DIR / "A12_clones_IGK.tsv", "A12_IgM-_IGK"),
    (DATA_DIR / "A12_clones_IGL.tsv", "A12_IgM-_IGL"),
    (DATA_DIR / "C1_clones_IGH.tsv", "C1_IgM+_IGH"),
    (DATA_DIR / "C1_clones_IGK.tsv", "C1_IgM+_IGK"),
    (DATA_DIR / "C1_clones_IGL.tsv", "C1_IgM+_IGL"),
    (DATA_DIR / "C2_clones_IGH.tsv", "C2_IgM-_IGH"),
    (DATA_DIR / "C2_clones_IGK.tsv", "C2_IgM-_IGK"),
    (DATA_DIR / "C2_clones_IGL.tsv", "C2_IgM-_IGL"),
    (DATA_DIR / "C5_clones_IGH.tsv", "C5_IgM+_IGH"),
    (DATA_DIR / "C5_clones_IGK.tsv", "C5_IgM+_IGK"),
    (DATA_DIR / "C5_clones_IGL.tsv", "C5_IgM+_IGL"),
    (DATA_DIR / "C6_clones_IGH.tsv", "C6_IgM-_IGH"),
    (DATA_DIR / "C6_clones_IGK.tsv", "C6_IgM-_IGK"),
    (DATA_DIR / "C6_clones_IGL.tsv", "C6_IgM-_IGL"),
    (DATA_DIR / "C7_clones_IGH.tsv", "C7_IgM+_IGH"),
    (DATA_DIR / "C7_clones_IGK.tsv", "C7_IgM+_IGK"),
    (DATA_DIR / "C7_clones_IGL.tsv", "C7_IgM+_IGL"),
    (DATA_DIR / "C8_clones_IGH.tsv", "C8_IgM-_IGH"),
    (DATA_DIR / "C8_clones_IGK.tsv", "C8_IgM-_IGK"),
    (DATA_DIR / "C8_clones_IGL.tsv", "C8_IgM-_IGL"),
    (DATA_DIR / "C9_clones_IGH.tsv", "C9_IgM+_IGH"),
    (DATA_DIR / "C9_clones_IGK.tsv", "C9_IgM+_IGK"),
    (DATA_DIR / "C9_clones_IGL.tsv", "C9_IgM+_IGL"),
    (DATA_DIR / "C10_clones_IGH.tsv", "C10_IgM-_IGH"),
    (DATA_DIR / "C10_clones_IGK.tsv", "C10_IgM-_IGK"),
    (DATA_DIR / "C10_clones_IGL.tsv", "C10_IgM-_IGL"),
    (DATA_DIR / "C11_clones_IGH.tsv", "C11_IgM+_IGH"),
    (DATA_DIR / "C11_clones_IGK.tsv", "C11_IgM+_IGK"),
    (DATA_DIR / "C11_clones_IGL.tsv", "C11_IgM+_IGL"),
    (DATA_DIR / "C12_clones_IGH.tsv", "C12_IgM-_IGH"),
    (DATA_DIR / "C12_clones_IGK.tsv", "C12_IgM-_IGK"),
    (DATA_DIR / "C12_clones_IGL.tsv", "C12_IgM-_IGL"),
    (DATA_DIR / "D1_clones_IGH.tsv", "D1_IgM-_IGH"),
    (DATA_DIR / "D1_clones_IGL.tsv", "D1_IgM-_IGL"),
    (DATA_DIR / "D1_clones_IGK.tsv", "D1_IgM-_IGK"),
    (DATA_DIR / "D2_clones_IGH.tsv", "D2_IgM-_IGH"),
    (DATA_DIR / "D2_clones_IGK.tsv", "D2_IgM-_IGK"),
    (DATA_DIR / "D3_clones_IGH.tsv", "D3_IgM-_IGH"),
    (DATA_DIR / "D3_clones_IGL.tsv", "D3_IgM-_IGL"),
    (DATA_DIR / "D3_clones_IGK.tsv", "D3_IgM-_IGK"),
    (DATA_DIR / "D4_clones_IGH.tsv", "D4_IgM+_IGH"),
    (DATA_DIR / "D4_clones_IGL.tsv", "D4_IgM+_IGL"),
    (DATA_DIR / "D4_clones_IGK.tsv", "D4_IgM+_IGK"),
    (DATA_DIR / "D5_clones_IGH.tsv", "D5_mixed_IGH"),
    (DATA_DIR / "D5_clones_IGL.tsv", "D5_mixed_IGL"),
    (DATA_DIR / "D5_clones_IGK.tsv", "D5_mixed_IGK"),
    (DATA_DIR / "D6_clones_IGH.tsv", "D6_IgM-_IGH"),
    (DATA_DIR / "D6_clones_IGK.tsv", "D6_IgM-_IGK"),
    (DATA_DIR / "D6_clones_IGL.tsv", "D6_IgM-_IGL"),
    (DATA_DIR / "D7_clones_IGH.tsv", "D7_IgM-_IGH"),
    (DATA_DIR / "D7_clones_IGK.tsv", "D7_IgM-_IGK"),
    (DATA_DIR / "D7_clones_IGL.tsv", "D7_IgM-_IGL"),
    (DATA_DIR / "D8_clones_IGH.tsv", "D8_IgM+_IGH"),
    (DATA_DIR / "D8_clones_IGK.tsv", "D8_IgM+_IGK"),
    (DATA_DIR / "D8_clones_IGL.tsv", "D8_IgM+_IGL"),
    (DATA_DIR / "D9_clones_IGH.tsv", "D9_IgM+_IGH"),
    (DATA_DIR / "D9_clones_IGK.tsv", "D9_IgM+_IGK"),
    (DATA_DIR / "D9_clones_IGL.tsv", "D9_IgM+_IGL"),
    (DATA_DIR / "F1_clones_IGH.tsv", "F1_IgM+_IGH"),
    (DATA_DIR / "F1_clones_IGK.tsv", "F1_IgM+_IGK"),
    (DATA_DIR / "F1_clones_IGL.tsv", "F1_IgM+_IGL"),
    (DATA_DIR / "F2_clones_IGH.tsv", "F2_IgM-_IGH"),
    (DATA_DIR / "F2_clones_IGK.tsv", "F2_IgM-_IGK"),
    (DATA_DIR / "F2_clones_IGL.tsv", "F2_IgM-_IGL"),
    (DATA_DIR / "F3_clones_IGH.tsv", "F3_mixed_IGH"),
    (DATA_DIR / "F3_clones_IGK.tsv", "F3_mixed_IGK"),
    (DATA_DIR / "F3_clones_IGL.tsv", "F3_mixed_IGL"),
    (DATA_DIR / "F4_clones_IGH.tsv", "F4_IgM-_IGH"),
    (DATA_DIR / "F4_clones_IGK.tsv", "F4_IgM-_IGK"),
    (DATA_DIR / "F4_clones_IGL.tsv", "F4_IgM-_IGL"),
    (DATA_DIR / "F5_clones_IGH.tsv", "F5_IgM-_IGH"),
    (DATA_DIR / "F5_clones_IGK.tsv", "F5_IgM-_IGK"),
    (DATA_DIR / "F5_clones_IGL.tsv", "F5_IgM-_IGL"),
    (DATA_DIR / "F6_clones_IGH.tsv", "F6_IgM-_IGH"),
    (DATA_DIR / "F6_clones_IGK.tsv", "F6_IgM-_IGK"),
    (DATA_DIR / "F7_clones_IGH.tsv", "F7_IgM+_IGH"),
    (DATA_DIR / "F7_clones_IGK.tsv", "F7_IgM+_IGK"),
    (DATA_DIR / "F7_clones_IGL.tsv", "F7_IgM+_IGL"),
    (DATA_DIR / "F8_clones_IGH.tsv", "F8_IgM+_IGH"),
    (DATA_DIR / "F8_clones_IGK.tsv", "F8_IgM+_IGK"),
    (DATA_DIR / "F9_clones_IGH.tsv", "F9_IgM+_IGH"),
    (DATA_DIR / "F9_clones_IGK.tsv", "F9_IgM+_IGK"),
    (DATA_DIR / "F9_clones_IGL.tsv", "F9_IgM+_IGL"),
    (DATA_DIR / "F10_clones_IGH.tsv", "F10_mixed_IGH"),
    (DATA_DIR / "F10_clones_IGK.tsv", "F10_mixed_IGK"),
    (DATA_DIR / "F10_clones_IGL.tsv", "F10_mixed_IGL"),
    (DATA_DIR / "F11_clones_IGH.tsv", "F11_mixed_IGH"),
    (DATA_DIR / "F11_clones_IGK.tsv", "F11_mixed_IGK"),
    (DATA_DIR / "F11_clones_IGL.tsv", "F11_mixed_IGL"),
]

metric_keys = [
    "Observed Richness",
    "Chao1",
    "1st-order Jackknife",
    "2nd-order Jackknife",
    "ACE",
    "Hill Number (q=1)",
    "Berger-Parker",
    "Rényi Entropy (q=2)",
    "Inverse Simpson",
    "Gini-Simpson",
    "Shannon Entropy",
    "Tail Index",
    "Evenness EF",
    "Evenness RLE",
    "Pielou Index",
]

metric_to_plot = metric_keys[14]


def make_alpha_dataframe():
    group_labels = {
        "A": "Eµ Premalignant",
        "C": "EµTet2KO Premalignant",
        "D": "Eµ Malignant",
        "F": "EµTet2KO Malignant",
    }

    rows = []
    for file_path, sample_name in samples:
        _, metric_val = get_alpha_metric_from_file(file_path, sample_name, metric_to_plot)
        group = sample_name[0]
        chain = sample_name.split("_")[-1]
        rows.append(
            {
                "sample": sample_name,
                "group": group,
                "group_label": group_labels.get(group, group),
                "chain": chain,
                "metric": metric_val,
            }
        )

    df = pd.DataFrame(rows)
    df["igm_status"] = df["sample"].str.extract(r"_(IgM\+|IgM-|mixed)_")
    df["group_chain"] = df["group_label"] + " " + df["chain"]
    return df


def add_grouping_traces(fig, df, column, order=None):
    trace_indices = []
    categories = order or sorted(df[column].dropna().unique())
    for category in categories:
        subset = df[df[column] == category]
        if subset.empty:
            continue
        fig.add_trace(
            go.Box(
                y=subset["metric"],
                name=str(category),
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                marker=dict(size=6, line=dict(width=1, color="black")),
                line=dict(width=1),
                showlegend=False,
            )
        )
        trace_indices.append(len(fig.data) - 1)
    return trace_indices


df_alpha = make_alpha_dataframe()

group_label_order = [
    "Eµ Premalignant",
    "EµTet2KO Premalignant",
    "Eµ Malignant",
    "EµTet2KO Malignant",
]

igm_order = ["IgM+", "IgM-", "mixed"]
chain_order = ["IGH", "IGK", "IGL"]

fig = go.Figure()
trace_groups = {}
trace_groups["Group"] = add_grouping_traces(fig, df_alpha, "group_label", group_label_order)
trace_groups["IgM status"] = add_grouping_traces(fig, df_alpha, "igm_status", igm_order)
trace_groups["Chain"] = add_grouping_traces(fig, df_alpha, "chain", chain_order)
trace_groups["Group + Chain"] = add_grouping_traces(fig, df_alpha, "group_chain")

buttons = []
all_traces = len(fig.data)
for name, indices in trace_groups.items():
    visibility = [False] * all_traces
    for idx in indices:
        visibility[idx] = True
    buttons.append(
        dict(
            label=name,
            method="update",
            args=[
                {"visible": visibility},
                {
                    "title": f"{metric_to_plot} grouped by {name}",
                    "xaxis": {"title": name},
                    "yaxis": {"title": metric_to_plot},
                },
            ],
        )
    )

default_group = "Group"
default_visibility = [False] * all_traces
for idx in trace_groups[default_group]:
    default_visibility[idx] = True
for i, trace in enumerate(fig.data):
    trace.visible = default_visibility[i]

fig.update_layout(
    title=f"{metric_to_plot} grouped by {default_group}",
    xaxis_title=default_group,
    yaxis_title=metric_to_plot,
    boxmode="group",
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
        )
    ],
    margin=dict(l=60, r=200, t=60, b=60),
)

output_path = FIGURES_DIR / "alpha_diversity_interactive.html"
fig.write_html(output_path)
print(f"Wrote {output_path}")
