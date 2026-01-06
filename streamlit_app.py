import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import plotly.graph_objects as go
from scipy import stats

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cluster Robust SE Explorer", layout="wide")

st.title("🛡️ Clustered Robust Standard Errors")
st.markdown("""
Standard OLS assumes all observations are independent. In real-world data, observations within a **cluster** (like employees in the same store) often share unobserved traits. 
This app demonstrates how **Clustered Robust SEs** prevent "False Positive" significance.
""")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Simulation Parameters")
    n_clusters = st.slider("Number of Clusters (e.g., Stores)", 10, 100, 40)
    n_per_cluster = st.slider("Observations per Cluster", 5, 50, 20)
    
    st.subheader("The Cluster Effect")
    icc = st.slider("Intra-cluster Correlation (ICC)", 0.0, 1.0, 0.4, 
                    help="How much people in the same cluster are 'alike'. Higher = bigger need for clustered SEs.")
    
    st.info("The treatment effect is set to be statistically weak. Watch how standard SEs often 'fake' significance.")

# --- DATA GENERATION ---
@st.cache_data
def generate_clustered_data(n_clusters, n_per_cluster, icc):
    # Cluster-level random effect
    cluster_effects = np.random.normal(0, np.sqrt(icc), n_clusters)
    
    data = []
    for i in range(n_clusters):
        # Treatment is assigned at the cluster level (e.g., store-wide bonus)
        treatment = 1 if i % 2 == 0 else 0
        for j in range(n_per_cluster):
            # Individual error
            eps = np.random.normal(0, np.sqrt(1 - icc))
            # Outcome: Base + Weak Treatment + Cluster Effect + Noise
            y = 10 + (0.2 * treatment) + cluster_effects[i] + eps
            data.append({'cluster_id': i, 'treatment': treatment, 'outcome': y})
    
    return pd.DataFrame(data)

df = generate_clustered_data(n_clusters, n_per_cluster, icc)

# --- DATA VISUALIZATION SECTION ---
st.subheader("🔍 Raw Data Distribution (Colored by Cluster)")
st.write("Each point is an individual. Notice how points of the same color (cluster) tend to stay together—this is the correlation that tricks standard OLS.")

import plotly.express as px

st.subheader("🔍 Raw Data Distribution (Colored by Cluster)")
st.write("Notice how points of the same color (cluster) tend to stay together—this represents the intra-cluster correlation.")

# 1. Ensure cluster_id is treated as a string/categorical for discrete coloring
df_plot = df.copy()
df_plot['cluster_id'] = df_plot['cluster_id'].astype(str)

# 2. Use color_discrete_sequence instead of color_continuous_scale
fig_raw = px.strip(
    df_plot, 
    x="treatment", 
    y="outcome", 
    color="cluster_id", 
    stripmode="overlay",
    title="Outcome vs Treatment (Colored by Cluster ID)",
    labels={"treatment": "Group", "outcome": "Observed Value"},
    color_discrete_sequence=px.colors.qualitative.Plotly # Use a categorical palette
)

fig_raw.update_layout(
    xaxis = dict(tickmode = 'array', tickvals = [0, 1], ticktext = ['Control', 'Treated']),
    showlegend=False,
    height=500
)

st.plotly_chart(fig_raw, use_container_width=True)

# --- MODELING ---
# 1. Standard OLS (Non-Robust)
model_std = smf.ols("outcome ~ treatment", data=df).fit()

# 2. Clustered Robust OLS
# We use cov_type='cluster' and point to the cluster ID
model_clust = smf.ols("outcome ~ treatment", data=df).fit(
    cov_type='cluster', 
    cov_kwds={'groups': df['cluster_id']}
)

# --- UI LAYOUT ---
col1, col2 = st.columns(2)

def get_stats_df(model, name):
    return pd.DataFrame({
        "Model": name,
        "Estimate": model.params['treatment'],
        "Std. Error": model.bse['treatment'],
        "t-stat": model.tvalues['treatment'],
        "p-value": model.pvalues['treatment'],
        "Conf. Low": model.conf_int().loc['treatment', 0],
        "Conf. High": model.conf_int().loc['treatment', 1]
    }, index=[0])

res_df = pd.concat([
    get_stats_df(model_std, "Standard OLS (Naïve)"),
    get_stats_df(model_clust, "Clustered Robust SE")
])

with col1:
    st.subheader("📊 Regression Comparison")
    st.dataframe(res_df.style.format(subset=["p-value", "Std. Error", "Estimate"], precision=4))
    
    # Highlight the p-value danger
    p_std = res_df.iloc[0]['p-value']
    p_clust = res_df.iloc[1]['p-value']
    
    if p_std < 0.05 and p_clust >= 0.05:
        st.error(f"⚠️ **FALSE POSITIVE DETECTED!** Standard OLS says the result is significant (p={p_std:.3f}), but correcting for clusters reveals it is actually just noise (p={p_clust:.3f}).")
    elif p_clust < 0.05:
        st.success("✅ The effect is robust even after clustering.")
    else:
        st.info("The effect is not significant in either model.")

with col2:
    st.subheader("📍 Coefficient Confidence Intervals")
    
    fig = go.Figure()
    # Add Standard OLS Bar
    fig.add_trace(go.Scatter(
        x=[res_df.iloc[0]['Estimate']], y=["Standard"],
        mode='markers',
        error_x=dict(type='data', array=[res_df.iloc[0]['Conf. High'] - res_df.iloc[0]['Estimate']], visible=True),
        name="Standard", marker=dict(color='red', size=12)
    ))
    # Add Clustered Bar
    fig.add_trace(go.Scatter(
        x=[res_df.iloc[1]['Estimate']], y=["Clustered"],
        mode='markers',
        error_x=dict(type='data', array=[res_df.iloc[1]['Conf. High'] - res_df.iloc[1]['Estimate']], visible=True),
        name="Clustered", marker=dict(color='green', size=12)
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(xaxis_title="Estimated Treatment Effect", height=300)
    st.plotly_chart(fig, use_container_width=True)



st.divider()

# --- THE "WHY" SECTION ---
st.subheader("🧠 What's happening behind the scenes?")
st.write("""
The **Standard Error (SE)** measures our uncertainty. When data is clustered:
1.  **Standard OLS** thinks it has many independent observations (e.g., 800 people), making it feel 'overconfident' (Small SE).
2.  **Clustered SE** realizes that if people in a store are alike, we don't actually have 800 'bits' of information; we have something closer to the number of clusters (40 stores).
3.  The result? The **Clustered SE is wider and more honest**, preventing you from making a Type I error (False Positive).
""")
