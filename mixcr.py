import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.lines import Line2D
from pathlib import Path

# === Plot Settings ===
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})


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

    chao1 = S_obs + (f1**2) / (2*f2) if f2 > 0 else np.nan
    jack1 = S_obs + f1
    jack2 = S_obs + 2*f1 - f2 if f2 > 0 else np.nan

    rare_cutoff = 10
    rare = read_counts[read_counts <= rare_cutoff]
    abundant = read_counts[read_counts > rare_cutoff]
    S_rare = len(rare)
    N_rare = rare.sum()
    C_ace = 1 - (f1 / N_rare) if N_rare > 0 else np.nan
    gamma_sq = np.var(rare, ddof=1) / np.mean(rare)**2 if len(rare) > 1 else 0
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
    tail_index = sum([(1 - (i / len(sorted_p)))**2 * p for i, p in enumerate(sorted_p)])

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
        "Pielou Index": pielou
    }


# === Load metric per sample ===
def get_alpha_metric_from_file(file_path, sample_name, metric_name):
    df = pd.read_csv(file_path, sep="\t")
    read_counts = df["readCount"]
    metrics = calculate_full_alpha_metrics(read_counts)
    return sample_name, metrics.get(metric_name, np.nan)


# === Filtering and Grouping ===
def filter_alpha_diversity(samples, metric_name, start_text=None, end_text=None):
    sample_names = []
    diversity_values = []

    for file_path, sample_name in samples:
        if start_text and not sample_name.startswith(start_text):
            continue
        if end_text and not sample_name.endswith(end_text):
            continue

        sample_name, value = get_alpha_metric_from_file(file_path, sample_name, metric_name)
        sample_names.append(sample_name)
        diversity_values.append(value)

    return sample_names, diversity_values


# === Plotting ===
def plot_alpha(ax, sample_names, values, ylabel):
    ax.bar(sample_names, values)
    ax.set_xticklabels(sample_names, rotation=45, ha="right")
    ax.set_ylabel(ylabel)


# === Define samples (kurz gehalten) ===
DATA_DIR = Path("data")
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
  # (DATA_DIR / "D2_clones_IGL.tsv", "D2_IgM-_IGL"),
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
  # (DATA_DIR / "F6_clones_IGL.tsv", "F6_IgM-_IGL"),
  (DATA_DIR / "F7_clones_IGH.tsv", "F7_IgM+_IGH"),
  (DATA_DIR / "F7_clones_IGK.tsv", "F7_IgM+_IGK"),
  (DATA_DIR / "F7_clones_IGL.tsv", "F7_IgM+_IGL"),
  (DATA_DIR / "F8_clones_IGH.tsv", "F8_IgM+_IGH"),
  (DATA_DIR / "F8_clones_IGK.tsv", "F8_IgM+_IGK"),
  # (DATA_DIR / "F8_clones_IGL.tsv", "F8_IgM+_IGL"),
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
    "Hill Number (q=1)",  # corresponds to qD in table
    "Berger-Parker",
    "Rényi Entropy (q=2)",
    "Inverse Simpson",
    "Gini-Simpson",
    "Shannon Entropy",
    "Tail Index",
    "Evenness EF",
    "Evenness RLE",
    "Pielou Index"
]


# === Wähle gewünschte Metrik ===
metric_to_plot = metric_keys[14] # "Evenness RLE" #"Chao1" # "Evenness RLE"  # z.B. "Shannon Entropy", "ACE", "Pielou Index", ...
print(f"Metric: {metric_to_plot}")

# === Daten pro Gruppe und Kette ===
A_IGH_labels, A_IGH_values = filter_alpha_diversity(samples, metric_to_plot, start_text="A", end_text="_IGH")
C_IGH_labels, C_IGH_values = filter_alpha_diversity(samples, metric_to_plot, start_text="C", end_text="_IGH")

A_IGK_labels, A_IGK_values = filter_alpha_diversity(samples, metric_to_plot, start_text="A", end_text="_IGK")
C_IGK_labels, C_IGK_values = filter_alpha_diversity(samples, metric_to_plot, start_text="C", end_text="_IGK")

A_IGL_labels, A_IGL_values = filter_alpha_diversity(samples, metric_to_plot, start_text="A", end_text="_IGL")
C_IGL_labels, C_IGL_values = filter_alpha_diversity(samples, metric_to_plot, start_text="C", end_text="_IGL")


# === Plot ===
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(30, 20), sharey=True)

plot_alpha(ax1, A_IGH_labels, A_IGH_values, metric_to_plot)
plot_alpha(ax2, C_IGH_labels, C_IGH_values, metric_to_plot)
plot_alpha(ax3, A_IGK_labels, A_IGK_values, metric_to_plot)
plot_alpha(ax4, C_IGK_labels, C_IGK_values, metric_to_plot)
plot_alpha(ax5, A_IGL_labels, A_IGL_values, metric_to_plot)
plot_alpha(ax6, C_IGL_labels, C_IGL_values, metric_to_plot)

# === Titles ===
ax1.set_title("Eµ Premalignant IGH Samples")
ax2.set_title("EµTet2KO Premalignant IGH Samples")
ax3.set_title("Eµ Premalignant IGK Samples")
ax4.set_title("EµTet2KO Premalignant IGK Samples")
ax5.set_title("Eµ Premalignant IGL Samples")
ax6.set_title("EµTet2KO Premalignant IGL Samples")

plt.tight_layout()
plt.savefig("alpha_diversity_premalignant.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Daten pro Gruppe und Kette ===
D_IGH_labels, D_IGH_values = filter_alpha_diversity(samples, metric_to_plot, start_text="D", end_text="_IGH")
F_IGH_labels, F_IGH_values = filter_alpha_diversity(samples, metric_to_plot, start_text="F", end_text="_IGH")

D_IGK_labels, D_IGK_values = filter_alpha_diversity(samples, metric_to_plot, start_text="D", end_text="_IGK")
F_IGK_labels, F_IGK_values = filter_alpha_diversity(samples, metric_to_plot, start_text="F", end_text="_IGK")

D_IGL_labels, D_IGL_values = filter_alpha_diversity(samples, metric_to_plot, start_text="D", end_text="_IGL")
F_IGL_labels, F_IGL_values = filter_alpha_diversity(samples, metric_to_plot, start_text="F", end_text="_IGL")


# === Plot ===
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(30, 20), sharey=True)

plot_alpha(ax1, D_IGH_labels, D_IGH_values, metric_to_plot)
plot_alpha(ax2, F_IGH_labels, F_IGH_values, metric_to_plot)
plot_alpha(ax3, D_IGK_labels, D_IGK_values, metric_to_plot)
plot_alpha(ax4, F_IGK_labels, F_IGK_values, metric_to_plot)
plot_alpha(ax5, D_IGL_labels, D_IGL_values, metric_to_plot)
plot_alpha(ax6, F_IGL_labels, F_IGL_values, metric_to_plot)

# === Titles ===
ax1.set_title("Eµ Malignant IGH Samples")
ax2.set_title("EµTet2KO Malignant IGH Samples")
ax3.set_title("Eµ Malignant IGK Samples")
ax4.set_title("EµTet2KO Malignant IGK Samples")
ax5.set_title("Eµ Malignant IGL Samples")
ax6.set_title("EµTet2KO Malignant IGL Samples")

plt.tight_layout()
plt.savefig("alpha_diversity_malignant.png", dpi=300, bbox_inches="tight")
plt.close(fig)
