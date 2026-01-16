import pandas as pd
import numpy as np
from scipy.stats import entropy
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def make_alpha_dataframe(metric_name):
    group_labels = {
        "A": "Eµ Premalignant",
        "C": "EµTet2KO Premalignant",
        "D": "Eµ Malignant",
        "F": "EµTet2KO Malignant",
    }

    rows = []
    for file_path, sample_name in samples:
        _, metric_val = get_alpha_metric_from_file(file_path, sample_name, metric_name)
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
    return df


def build_metric_dataframe():
    frames = []
    for metric_name in metric_keys:
        df_metric = make_alpha_dataframe(metric_name)
        df_metric["metric_name"] = metric_name
        frames.append(df_metric)
    return pd.concat(frames, ignore_index=True)


def filter_malignant_igh(df):
    return df[
        (df["group"].isin(["D", "F"])) &
        (df["chain"] == "IGH") &
        (df["igm_status"].isin(["IgM+", "IgM-"]))
    ].copy()


def get_panel_data(df_metric):
    igm_pos = df_metric[df_metric["igm_status"] == "IgM+"]
    igm_neg = df_metric[df_metric["igm_status"] == "IgM-"]
    eu = df_metric[df_metric["group"] == "D"]
    eutet2 = df_metric[df_metric["group"] == "F"]
    return igm_pos, igm_neg, eu, eutet2


def get_panel_ylim(values):
    max_val = values.max()
    if pd.isna(max_val):
        return [0, 1]
    return [0, float(max_val) * 1.05]


def get_shared_ylim(panels):
    combined = pd.concat([p["metric"] for p in panels], ignore_index=True)
    return get_panel_ylim(combined)


df_all = build_metric_dataframe()
df_default = filter_malignant_igh(df_all[df_all["metric_name"] == metric_to_plot])
igm_pos, igm_neg, eu, eutet2 = get_panel_data(df_default)
igm_colors = {"IgM+": "#1f77b4", "IgM-": "#ff7f0e"}
shared_ylim = get_shared_ylim([igm_pos, igm_neg, eu, eutet2])
igm_pos_samples = igm_pos["sample"].tolist()
igm_neg_samples = igm_neg["sample"].tolist()
eu_samples = eu["sample"].tolist()
eutet2_samples = eutet2["sample"].tolist()

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "IgM+ Malignant IGH Samples",
        "IgM- Malignant IGH Samples",
        "Eµ Malignant IGH Samples",
        "EµTet2KO Malignant IGH Samples",
    ),
)

fig.add_trace(
    go.Bar(
        x=igm_pos_samples,
        y=igm_pos["metric"],
        name="IgM+",
        marker_color=igm_colors["IgM+"],
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=igm_neg_samples,
        y=igm_neg["metric"],
        name="IgM-",
        marker_color=igm_colors["IgM-"],
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Bar(
        x=eu_samples,
        y=eu["metric"],
        name="Eµ",
        marker_color=eu["igm_status"].map(igm_colors),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=eutet2_samples,
        y=eutet2["metric"],
        name="EµTet2KO",
        marker_color=eutet2["igm_status"].map(igm_colors),
    ),
    row=2,
    col=2,
)

fig.update_layout(
    title=f"{metric_to_plot} (Malignant IGH)",
    showlegend=False,
    height=700,
    yaxis=dict(range=shared_ylim, autorange=False),
    yaxis2=dict(range=shared_ylim, autorange=False),
    yaxis3=dict(range=shared_ylim, autorange=False),
    yaxis4=dict(range=shared_ylim, autorange=False),
    margin=dict(l=60, r=40, t=80, b=60),
)
fig.update_xaxes(tickangle=45)
fig.update_xaxes(categoryorder="array", categoryarray=igm_pos_samples, row=1, col=1)
fig.update_xaxes(categoryorder="array", categoryarray=igm_neg_samples, row=1, col=2)
fig.update_xaxes(categoryorder="array", categoryarray=eu_samples, row=2, col=1)
fig.update_xaxes(categoryorder="array", categoryarray=eutet2_samples, row=2, col=2)

buttons = []
for metric_name in metric_keys:
    df_metric = filter_malignant_igh(df_all[df_all["metric_name"] == metric_name])
    igm_pos, igm_neg, eu, eutet2 = get_panel_data(df_metric)
    shared_ylim = get_shared_ylim([igm_pos, igm_neg, eu, eutet2])
    buttons.append(
        dict(
            label=metric_name,
            method="update",
            args=[
                {
                    "y": [
                        igm_pos["metric"],
                        igm_neg["metric"],
                        eu["metric"],
                        eutet2["metric"],
                    ],
                    "marker": [
                        {"color": igm_colors["IgM+"]},
                        {"color": igm_colors["IgM-"]},
                        {"color": eu["igm_status"].map(igm_colors)},
                        {"color": eutet2["igm_status"].map(igm_colors)},
                    ],
                },
                {
                    "title": f"{metric_name} (Malignant IGH)",
                    "yaxis.range": shared_ylim,
                    "yaxis.autorange": False,
                    "yaxis2.range": shared_ylim,
                    "yaxis2.autorange": False,
                    "yaxis3.range": shared_ylim,
                    "yaxis3.autorange": False,
                    "yaxis4.range": shared_ylim,
                    "yaxis4.autorange": False,
                },
            ],
        )
    )

fig.update_layout(
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
        )
    ]
)

output_path = FIGURES_DIR / "alpha_diversity_interactive.html"
fig.write_html(output_path)
print(f"Wrote {output_path}")
