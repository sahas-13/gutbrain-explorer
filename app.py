import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="GutBrain Explorer",
    page_icon="🧬",
    layout="wide"
)

# ── Title ──────────────────────────────────────────────────────────
st.title("🧬 GutBrain Explorer")
st.markdown("### Exploring Gut Microbiome Differences Across Diet Groups")
st.markdown("---")

# ── Load data ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('MicrobiomeWithMetadata.csv')
    metadata_cols = ['Diet', 'Source', 'Donor', 'CollectionMet', 'Sex']
    otu_cols = [col for col in df.columns if col.startswith('OTU')]
    otu_table = df[otu_cols]
    top50 = otu_table.mean().sort_values(ascending=False).head(50).index.tolist()
    otu_top50 = otu_table[top50]
    return df, otu_top50

df, otu_top50 = load_data()

# ── Sidebar ────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")
diet_options = sorted(df['Diet'].unique().tolist())
diet_a = st.sidebar.selectbox("Select Diet Group A", diet_options, index=0)
diet_b = st.sidebar.selectbox("Select Diet Group B", diet_options, index=1)
top_n = st.sidebar.slider("Number of top OTUs to show", 5, 50, 20)

st.sidebar.markdown("---")
st.sidebar.markdown("**About this app**")
st.sidebar.markdown("Built by Sahasrakshi S | MTech Biotechnology, VIT")
st.sidebar.markdown("Data: Human Microbiome Project (Turnbaugh et al.)")

# ── Filter data ────────────────────────────────────────────────────
if diet_a == diet_b:
    st.warning("⚠️ Please select two different diet groups!")
    st.stop()

mask = df['Diet'].isin([diet_a, diet_b])
otu_filtered = otu_top50[mask].reset_index(drop=True)
diet_labels = df['Diet'][mask].reset_index(drop=True)

# ── Summary metrics ────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Samples", df.shape[0])
col2.metric(f"Diet {diet_a} Samples", (df['Diet'] == diet_a).sum())
col3.metric(f"Diet {diet_b} Samples", (df['Diet'] == diet_b).sum())
col4.metric("OTUs Analysed", top_n)

st.markdown("---")

# ── Tab layout ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Heatmap",
    "🌋 Volcano Plot",
    "📊 Top OTUs",
    "🌿 Diversity"
])

# ── Tab 1: Heatmap ─────────────────────────────────────────────────
with tab1:
    st.subheader(f"Microbiome Heatmap: Diet {diet_a} vs Diet {diet_b}")
    sort_idx = diet_labels.sort_values().index
    otu_sorted = otu_filtered.loc[sort_idx].iloc[:, :top_n]

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.heatmap(otu_sorted.T, cmap='YlOrRd',
                xticklabels=False, ax=ax,
                cbar_kws={'label': 'Relative Abundance'})
    n_a = (diet_labels == diet_a).sum()
    ax.axvline(x=n_a, color='blue', linewidth=2, linestyle='--')
    ax.set_title(f'OTU Abundance Heatmap (left=Diet {diet_a}, right=Diet {diet_b})')
    ax.set_xlabel('Samples')
    ax.set_ylabel('OTU')
    plt.tight_layout()
    st.pyplot(fig)

# ── Tab 2: Volcano Plot ────────────────────────────────────────────
with tab2:
    st.subheader(f"Volcano Plot: Diet {diet_a} vs Diet {diet_b}")

    grp_a = otu_filtered[diet_labels == diet_a]
    grp_b = otu_filtered[diet_labels == diet_b]

    results = []
    for otu in otu_filtered.columns[:top_n]:
        stat, pval = stats.mannwhitneyu(grp_a[otu], grp_b[otu], alternative='two-sided')
        results.append({
            'OTU': otu,
            'mean_a': grp_a[otu].mean(),
            'mean_b': grp_b[otu].mean(),
            'pvalue': pval
        })

    results_df = pd.DataFrame(results)
    results_df['log2_fc'] = np.log2(
        results_df['mean_b'] / (results_df['mean_a'] + 1e-12))
    results_df['neg_log10_p'] = -np.log10(results_df['pvalue'] + 1e-12)
    results_df['significant'] = results_df['pvalue'] < 0.05

    fig, ax = plt.subplots(figsize=(10, 6))
    ns = results_df[~results_df['significant']]
    sig = results_df[results_df['significant']]
    ax.scatter(ns['log2_fc'], ns['neg_log10_p'], color='grey', alpha=0.6, s=60, label='Not significant')
    ax.scatter(sig['log2_fc'], sig['neg_log10_p'], color='red', alpha=0.8, s=80, label='Significant (p<0.05)')
    for _, row in results_df.head(10).iterrows():
        ax.annotate(row['OTU'], xy=(row['log2_fc'], row['neg_log10_p']), fontsize=7)
    ax.axhline(y=-np.log10(0.05), color='blue', linestyle='--', label='p=0.05')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10 P-value')
    ax.set_title('Volcano Plot')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(f"**Significant OTUs:** {results_df['significant'].sum()} out of {top_n}")
    st.dataframe(results_df.sort_values('pvalue').head(10)[['OTU','mean_a','mean_b','pvalue','log2_fc']])

# ── Tab 3: Top OTUs Bar Plot ───────────────────────────────────────
with tab3:
    st.subheader(f"Top {top_n} Differentially Abundant OTUs")
    top_results = results_df.sort_values('pvalue').head(10)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(top_results))
    width = 0.35
    ax.bar([i - width/2 for i in x], top_results['mean_a'],
           width, label=f'Diet {diet_a}', color='steelblue', edgecolor='black')
    ax.bar([i + width/2 for i in x], top_results['mean_b'],
           width, label=f'Diet {diet_b}', color='coral', edgecolor='black')
    ax.set_xticks(list(x))
    ax.set_xticklabels(top_results['OTU'], rotation=45, ha='right')
    ax.set_xlabel('OTU')
    ax.set_ylabel('Mean Relative Abundance')
    ax.set_title('Top 10 Differentially Abundant OTUs')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

# ── Tab 4: Diversity ───────────────────────────────────────────────
with tab4:
    st.subheader(f"Shannon Diversity: Diet {diet_a} vs Diet {diet_b}")

    def shannon_diversity(row):
        row = row[row > 0]
        return -np.sum(row * np.log(row))

    df_div = df[df['Diet'].isin([diet_a, diet_b])].copy()
    otu_div = otu_top50[df['Diet'].isin([diet_a, diet_b])]
    df_div['shannon'] = otu_div.apply(shannon_diversity, axis=1).values

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_div, x='Diet', y='shannon',
                hue='Diet', palette=['steelblue', 'coral'],
                width=0.5, legend=False, ax=ax)
    sns.stripplot(data=df_div, x='Diet', y='shannon',
                  color='black', alpha=0.3, size=3, ax=ax)

    d_a = df_div[df_div['Diet'] == diet_a]['shannon']
    d_b = df_div[df_div['Diet'] == diet_b]['shannon']
    _, pval = stats.mannwhitneyu(d_a, d_b)

    ax.set_title(f'Shannon Diversity (p = {pval:.4f})')
    ax.set_xlabel('Diet Group')
    ax.set_ylabel('Shannon Diversity Index')
    plt.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.metric(f"Diet {diet_a} Mean Diversity", f"{d_a.mean():.4f}")
    col2.metric(f"Diet {diet_b} Mean Diversity", f"{d_b.mean():.4f}")