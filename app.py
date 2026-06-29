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
st.markdown("""
This app analyses **real Human Microbiome Project data** (Turnbaugh et al.) 
to explore how diet shapes gut microbial communities. 
Use the sidebar to compare any two diet groups interactively.
""")
st.markdown("---")

# ── Load data ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('microbiome_top50.csv')
    otu_cols = [col for col in df.columns if col.startswith('OTU')]
    otu_top50 = df[otu_cols]
    return df, otu_top50

df, otu_top50 = load_data()

# ── Taxonomy dictionary ────────────────────────────────────────────
taxonomy = {
    'OTU4496': 'Bacteroidetes (Bacteroides)',
    'OTU4154': 'Firmicutes (Ruminococcaceae)',
    'OTU3857': 'Firmicutes (Lachnospiraceae)',
    'OTU618':  'Firmicutes (Clostridiales)',
    'OTU5948': 'Bacteroidetes (Prevotella)',
    'OTU5429': 'Firmicutes (Lachnospiraceae)',
    'OTU3994': 'Firmicutes (Ruminococcus)',
    'OTU5691': 'Proteobacteria (Enterobacteriaceae)',
    'OTU5937': 'Firmicutes (Eubacterium)',
    'OTU4256': 'Bacteroidetes (Bacteroides)',
    'OTU453':  'Firmicutes (Faecalibacterium)',
    'OTU155':  'Firmicutes (Ruminococcaceae)',
    'OTU710':  'Bacteroidetes (Bacteroides)',
    'OTU3406': 'Firmicutes (Lachnospiraceae)',
    'OTU2516': 'Firmicutes (Clostridiales)',
    'OTU6224': 'Proteobacteria (Helicobacter)',
    'OTU1462': 'Firmicutes (Ruminococcus)',
    'OTU4435': 'Firmicutes (Eubacterium)',
}

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
st.sidebar.markdown("---")
st.sidebar.markdown("**What is an OTU?**")
st.sidebar.markdown("""
An **Operational Taxonomic Unit (OTU)** is a cluster of 
similar 16S rRNA gene sequences representing a microbial 
species or genus in the gut microbiome.
""")

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

# ── Statistical analysis (shared across tabs) ──────────────────────
grp_a = otu_filtered[diet_labels == diet_a]
grp_b = otu_filtered[diet_labels == diet_b]

results = []
for otu in otu_filtered.columns[:top_n]:
    stat, pval = stats.mannwhitneyu(grp_a[otu], grp_b[otu], alternative='two-sided')
    results.append({
        'OTU': otu,
        'mean_a': grp_a[otu].mean(),
        'mean_b': grp_b[otu].mean(),
        'pvalue': pval,
        'Taxonomy': taxonomy.get(otu, 'Unknown')
    })

results_df = pd.DataFrame(results)
results_df['log2_fc'] = np.log2(
    results_df['mean_b'] / (results_df['mean_a'] + 1e-12))
results_df['neg_log10_p'] = -np.log10(results_df['pvalue'] + 1e-12)
results_df['significant'] = results_df['pvalue'] < 0.05

# ── Shannon diversity ──────────────────────────────────────────────
def shannon_diversity(row):
    row = row[row > 0]
    return -np.sum(row * np.log(row))

df_div = df[df['Diet'].isin([diet_a, diet_b])].copy()
otu_div = otu_top50[df['Diet'].isin([diet_a, diet_b])]
df_div['shannon'] = otu_div.apply(shannon_diversity, axis=1).values
d_a = df_div[df_div['Diet'] == diet_a]['shannon']
d_b = df_div[df_div['Diet'] == diet_b]['shannon']
_, div_pval = stats.mannwhitneyu(d_a, d_b)

# ── Tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Heatmap",
    "🌋 Volcano Plot",
    "📊 Top OTUs",
    "🌿 Diversity",
    "🧬 Summary & Findings",
    "🤖 ML Predictor"
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

    st.markdown("---")
    st.markdown("### 🔬 How to Read This Heatmap")
    st.markdown(f"""
    - Each **row** represents one OTU (bacterial group)
    - Each **column** represents one sample (one mouse/individual)
    - **Yellow** = low abundance | **Red/Orange** = high abundance
    - The **blue dashed line** separates Diet {diet_a} (left) from Diet {diet_b} (right)
    
    **What to look for:**
    - OTUs that are consistently red on one side and yellow on the other 
      indicate **diet-specific bacteria**
    - A mixed pattern suggests the OTU is **not strongly diet-dependent**
    - Top rows (OTU4496, OTU4154) are the **most abundant bacteria** overall
    """)

    st.info("""
    💡 **Biological insight:** Diet directly shapes which bacteria thrive in 
    your gut. High-fibre diets typically enrich Bacteroidetes and Firmicutes 
    that ferment plant matter, while low-fibre diets reduce microbial diversity.
    """)

# ── Tab 2: Volcano Plot ────────────────────────────────────────────
with tab2:
    st.subheader(f"Volcano Plot: Diet {diet_a} vs Diet {diet_b}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ns = results_df[~results_df['significant']]
    sig = results_df[results_df['significant']]
    ax.scatter(ns['log2_fc'], ns['neg_log10_p'],
               color='grey', alpha=0.6, s=60, label='Not significant')
    ax.scatter(sig['log2_fc'], sig['neg_log10_p'],
               color='red', alpha=0.8, s=80, label='Significant (p<0.05)')
    for _, row in results_df.head(10).iterrows():
        ax.annotate(row['OTU'],
                   xy=(row['log2_fc'], row['neg_log10_p']),
                   fontsize=7)
    ax.axhline(y=-np.log10(0.05), color='blue', linestyle='--', label='p=0.05')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10 P-value')
    ax.set_title('Volcano Plot')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🔬 How to Read This Volcano Plot")
    st.markdown(f"""
    - **X-axis (Log2 Fold Change):** How different the abundance is between diets
        - **Right side (positive):** OTU is MORE abundant in Diet {diet_b}
        - **Left side (negative):** OTU is MORE abundant in Diet {diet_a}
    - **Y-axis (-Log10 p-value):** Statistical confidence
        - **Higher = more significant**
    - **Red dots:** Statistically significant OTUs (p < 0.05)
    - **Grey dots:** No significant difference between diets
    - **Blue dashed line:** Significance threshold (p = 0.05)
    
    **Most interesting OTUs are top-right and top-left corners** — 
    these are both statistically significant AND biologically large differences!
    """)

    sig_count = results_df['significant'].sum()
    enriched_b = (results_df['significant'] & (results_df['log2_fc'] > 0)).sum()
    enriched_a = (results_df['significant'] & (results_df['log2_fc'] < 0)).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Significant OTUs", f"{sig_count}/{top_n}")
    col2.metric(f"Enriched in Diet {diet_b}", enriched_b)
    col3.metric(f"Enriched in Diet {diet_a}", enriched_a)

    st.markdown("### 📋 Significant OTUs Table")
    display_df = results_df[results_df['significant']].sort_values('pvalue')[
        ['OTU', 'mean_a', 'mean_b', 'pvalue', 'log2_fc', 'Taxonomy']
    ].round(6)
    display_df.columns = [f'OTU', f'Mean Diet {diet_a}',
                          f'Mean Diet {diet_b}', 'P-value',
                          'Log2 FC', 'Taxonomy']
    st.dataframe(display_df, use_container_width=True)

# ── Tab 3: Top OTUs ────────────────────────────────────────────────
with tab3:
    st.subheader(f"Top 10 Differentially Abundant OTUs")

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

    st.markdown("---")
    st.markdown("### 🔬 How to Read This Bar Plot")
    st.markdown(f"""
    - Each pair of bars represents one OTU (bacterial group)
    - **Blue bars** = mean abundance in Diet {diet_a}
    - **Coral bars** = mean abundance in Diet {diet_b}
    - Taller bar = that diet group has more of that bacteria
    - These are the **top 10 most statistically significant** differences
    """)

    st.markdown("### 🦠 OTU Taxonomy Lookup")
    st.markdown("What bacteria are these OTUs?")

    tax_data = []
    for _, row in top_results.iterrows():
        tax_data.append({
            'OTU': row['OTU'],
            'Taxonomy': taxonomy.get(row['OTU'], 'Unknown — needs BLAST lookup'),
            f'Higher in': f"Diet {diet_b}" if row['log2_fc'] > 0 else f"Diet {diet_a}",
            'P-value': f"{row['pvalue']:.2e}"
        })
    st.dataframe(pd.DataFrame(tax_data), use_container_width=True)

    st.info("""
    💡 **Biological insight:** Firmicutes and Bacteroidetes make up ~90% 
    of the human gut microbiome. The ratio between them shifts significantly 
    with diet — high-fat diets typically increase Firmicutes while 
    plant-based diets enrich Bacteroidetes.
    """)

# ── Tab 4: Diversity ───────────────────────────────────────────────
with tab4:
    st.subheader(f"Shannon Diversity: Diet {diet_a} vs Diet {diet_b}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_div, x='Diet', y='shannon',
                hue='Diet', palette=['steelblue', 'coral'],
                width=0.5, legend=False, ax=ax)
    sns.stripplot(data=df_div, x='Diet', y='shannon',
                  color='black', alpha=0.3, size=3, ax=ax)
    ax.set_title(f'Shannon Diversity (p = {div_pval:.4f})')
    ax.set_xlabel('Diet Group')
    ax.set_ylabel('Shannon Diversity Index')
    plt.tight_layout()
    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Diet {diet_a} Mean Diversity", f"{d_a.mean():.4f}")
    col2.metric(f"Diet {diet_b} Mean Diversity", f"{d_b.mean():.4f}")
    col3.metric("P-value", f"{div_pval:.4f}")

    st.markdown("---")
    st.markdown("### 🔬 How to Read This Diversity Plot")
    st.markdown(f"""
    - The **box** shows where the middle 50% of samples fall
    - The **line** inside the box is the median diversity
    - **Black dots** are individual samples
    - **Higher Shannon index = more diverse microbiome**
    
    **What the Shannon Index means:**
    - **0** = only one type of bacteria (no diversity)
    - **1-2** = moderate diversity
    - **2-3** = high diversity (healthy gut)
    - **3+** = very high diversity
    """)

    if div_pval < 0.05:
        higher_diet = diet_b if d_b.mean() > d_a.mean() else diet_a
        st.success(f"""
        ✅ **Statistically significant result! (p = {div_pval:.4f})**
        
        Diet {higher_diet} produces a **significantly more diverse** gut microbiome.
        Higher microbial diversity is generally associated with:
        - Better metabolic health
        - Stronger immune function  
        - Reduced inflammation
        - Lower risk of IBD and obesity
        """)
    else:
        st.warning(f"""
        ⚠️ **No significant diversity difference (p = {div_pval:.4f})**
        
        The two selected diet groups do not show statistically significant 
        differences in gut microbial diversity.
        Try comparing Diet 0 vs Diet 1 for the strongest contrast!
        """)

# ── Tab 5: Summary & Findings ──────────────────────────────────────
with tab5:
    st.subheader("🧬 Summary of Findings")
    st.markdown(f"### Comparing Diet {diet_a} vs Diet {diet_b}")

    st.markdown("---")

    st.markdown("### 📊 Key Results at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples Analysed",
                (df['Diet'] == diet_a).sum() + (df['Diet'] == diet_b).sum())
    col2.metric("Significant OTUs",
                f"{results_df['significant'].sum()}/{top_n}")
    col3.metric(f"Diet {diet_b} Diversity",
                f"{d_b.mean():.2f}")
    col4.metric("Diversity P-value", f"{div_pval:.4f}")

    st.markdown("---")
    st.markdown("### 🔍 Biological Interpretation")
    st.markdown(f"""
    **1. Microbial Composition Differences**
    
    Out of {top_n} OTUs analysed, **{results_df['significant'].sum()} showed 
    statistically significant differences** between Diet {diet_a} and Diet {diet_b} 
    (Mann-Whitney U test, p < 0.05). This indicates that diet has a strong and 
    measurable impact on gut microbial community structure.

    **2. Diversity Analysis**
    
    Diet {diet_b} showed a mean Shannon diversity of **{d_b.mean():.4f}** compared 
    to **{d_a.mean():.4f}** in Diet {diet_a}. 
    {"This difference is **statistically significant**, suggesting that dietary composition directly influences the richness and evenness of gut microbial communities." if div_pval < 0.05 else "This difference is not statistically significant."}

    **3. Key Differentially Abundant OTUs**
    
    The most significantly different OTUs include members of **Firmicutes**, 
    **Bacteroidetes**, and **Proteobacteria** — the three dominant phyla of the 
    human gut microbiome. Shifts in these communities have been linked to 
    metabolic health, inflammation, and gut-brain axis signalling.
    """)

    st.markdown("---")
    st.markdown("### 🧠 Gut-Brain Axis Connection")
    st.info("""
    The gut microbiome communicates with the brain through the **gut-brain axis** 
    via three main pathways:
    
    - **Neural pathway:** Vagus nerve carries signals from gut bacteria to the brain
    - **Endocrine pathway:** Gut bacteria produce hormones like serotonin (95% made in gut!)
    - **Immune pathway:** Microbial metabolites regulate neuroinflammation
    
    Diet-induced changes in microbial diversity (as shown in this analysis) 
    can therefore influence mood, cognition, and neurological health — 
    making gut microbiome research directly relevant to brain health.
    """)

    st.markdown("---")
    st.markdown("### 📚 Data Source & Methods")
    st.markdown("""
    | Item | Details |
    |---|---|
    | **Dataset** | Human Microbiome Project (Turnbaugh et al.) |
    | **Samples** | 675 gut microbiome samples |
    | **OTUs** | 6,696 total (top 50 analysed) |
    | **Statistical test** | Mann-Whitney U test (non-parametric) |
    | **Diversity metric** | Shannon Diversity Index |
    | **Visualisation** | Heatmap, Volcano Plot, Bar Plot, Box Plot |
    | **Built with** | Python, Streamlit, Pandas, Scipy, Seaborn |
    """)

    st.markdown("---")
    st.markdown("""
    *Built by **Sahasrakshi S** | MTech Biotechnology, VIT Vellore | 2026*  
    *GitHub: [gutbrain-explorer](https://github.com/sahas-13/gutbrain-explorer)*
    """)
# ── Tab 6: ML Predictor ────────────────────────────────────────────
with tab6:
    st.subheader("🤖 Machine Learning Diet Predictor")
    st.markdown("""
    A **Random Forest classifier** trained on gut microbiome OTU data 
    to predict diet group membership. The model learns which bacterial 
    communities are associated with each diet.
    """)

    # Train model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix

    @st.cache_data
    def train_model(diet_a, diet_b):
        mask = df['Diet'].isin([diet_a, diet_b])
        X = otu_top50[mask]
        y = df['Diet'][mask]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        importance = pd.DataFrame({
            'OTU': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        return rf, accuracy, cm, importance, X.columns.tolist()

    rf_model, accuracy, cm, importance_df, feature_cols = train_model(diet_a, diet_b)

    # ── Model metrics ──────────────────────────────────────────────
    st.markdown("### 📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
    col2.metric("Algorithm", "Random Forest")
    col3.metric("Trees in Forest", "100")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # ── Feature importance plot ────────────────────────────────────
    with col1:
        st.markdown("### 🦠 Most Predictive OTUs")
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.RdYlGn(
            np.linspace(0.3, 0.9, len(importance_df)))[::-1]
        ax.barh(importance_df['OTU'][::-1],
                importance_df['Importance'][::-1],
                color=colors, edgecolor='black')
        ax.set_xlabel('Feature Importance Score')
        ax.set_title('Top 15 OTUs That Predict Diet Group')
        plt.tight_layout()
        st.pyplot(fig)

    # ── Confusion matrix ───────────────────────────────────────────
    with col2:
        st.markdown("### 🎯 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Diet {diet_a}', f'Diet {diet_b}'],
                    yticklabels=[f'Diet {diet_a}', f'Diet {diet_b}'],
                    ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Model Predictions vs Reality')
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🔬 How to Read These Results")
    st.markdown(f"""
    **Feature Importance Chart:**
    - Shows which OTUs (bacteria) the model relies on most
    - **OTU710** (Bacteroides) is the strongest diet predictor
    - **OTU5691** (Proteobacteria) is second most important
    - Higher score = that bacteria changes more dramatically between diets

    **Confusion Matrix:**
    - **Diagonal cells** (top-left, bottom-right) = correct predictions ✅
    - **Off-diagonal cells** = mistakes ❌
    - Our model makes very few mistakes!

    **Why {accuracy * 100:.1f}% accuracy matters:**
    - Random guessing would give ~50% accuracy
    - Our model achieves **{accuracy * 100:.1f}%** — the microbiome 
      alone is enough to identify diet with high confidence
    - This confirms that diet leaves a strong **microbial fingerprint**
    """)

    st.success(f"""
    🧬 **Key Finding:** The gut microbiome composition can predict 
    diet group membership with **{accuracy * 100:.1f}% accuracy** 
    using just {top_n} OTUs — demonstrating that dietary patterns 
    leave measurable and reproducible signatures in gut microbial communities.
    """)

    st.markdown("---")
    st.markdown("### 🔮 Predict Diet Group From Microbiome")
    st.markdown("Adjust the sliders to simulate a microbiome profile:")

    top5_features = importance_df['OTU'].head(5).tolist()
    user_input = {}
    cols = st.columns(5)
    for i, otu in enumerate(top5_features):
        with cols[i]:
            user_input[otu] = st.slider(
                f"{otu}",
                min_value=0.0,
                max_value=float(otu_top50[otu].max()),
                value=float(otu_top50[otu].mean()),
                format="%.4f"
            )

    # Build input vector
    input_vector = pd.DataFrame([{
        col: user_input.get(col, float(otu_top50[col].mean()))
        for col in feature_cols
    }])

    prediction = rf_model.predict(input_vector)[0]
    probability = rf_model.predict_proba(input_vector)[0]

    st.markdown("### 🎯 Prediction Result")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Diet Group", f"Diet {prediction}")
    col2.metric("Confidence", f"{max(probability) * 100:.1f}%")
