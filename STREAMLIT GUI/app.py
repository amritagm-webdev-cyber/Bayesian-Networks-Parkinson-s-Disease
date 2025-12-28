import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination
import joblib
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import plotly.graph_objects as go
import numpy as np


# Load model and data
model = joblib.load("model.pkl")
inference = VariableElimination(model)
data = pd.read_csv("your_dataset.csv")

st.title("Bayesian Network Inference GUI")

# Sidebar inputs
st.sidebar.subheader("Select Inputs")
model_nodes = list(model.nodes())
inputs = {}

# Collect inputs only for nodes
for col in model_nodes:
    val = st.sidebar.text_input(f"{col}", value="")
    if val != "":
        try:
            inputs[col] = int(val)
        except:
            inputs[col] = val  # fallback to string if not int

# Select prediction target
possible_targets = [n for n in model_nodes if n not in inputs]
target = st.selectbox("Select Target to Predict", possible_targets)

if st.button("Run Inference"):
    try:
        result = inference.map_query(variables=[target], evidence=inputs, show_progress=False)
        st.write("Raw inference result:", result)

        if isinstance(result, dict):
            st.success(f"Prediction for {target}: {result.get(target)}")
        else:
            st.write("🔍 Full result object:", result)
            st.warning("⚠️ Unexpected result format")

    except Exception as e:
        st.error(f"❌ Inference failed: {str(e)}")
        st.info("Check that all evidence values are valid and exist in the model.")

# Optional: Correlation Heatmap
st.subheader("Correlation Heatmap (Real Data Subset)")
try:
    numeric_data = data.select_dtypes(include='number')
    fig, ax = plt.subplots()
    sns.heatmap(numeric_data.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
except:
    st.warning("⚠️ Heatmap could not be generated. Ensure your dataset contains numeric columns.")

# Feature Influence Estimation
st.subheader("Estimated Feature Influence on Selected Target")

try:
    if target and target in data.columns:
        df_clean = data[model_nodes].dropna()

        # Encode categorical variables numerically
        df_encoded = df_clean.copy()
        for col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        X = df_encoded.drop(columns=[target])
        y = df_encoded[target]

        # Calculate mutual information
        mi = mutual_info_classif(X, y, discrete_features=True)
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Estimated Influence": mi
        }).sort_values(by="Estimated Influence", ascending=False)

        st.write(importance_df)
        st.bar_chart(importance_df.set_index("Feature"))

    else:
        st.info("Select a valid target for influence analysis.")
except Exception as e:
    st.error(f"⚠️ Could not compute feature influence: {str(e)}")

# 3D Visualization of the Bayesian Network
st.subheader("3D Visualization of Bayesian Network")

try:
    # Manually convert pgmpy model to a networkx DiGraph
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())

    # Get 3D layout
    pos = nx.spring_layout(G, dim=3, seed=42)

    # Prepare edge coordinates
    edge_x = []
    edge_y = []
    edge_z = []

    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    # Prepare node coordinates
    xyz = np.array([pos[v] for v in G.nodes()])

    node_trace = go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            size=8,
            color='skyblue',
            line=dict(width=1, color='darkblue')
        )
    )

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='gray', width=2),
        hoverinfo='none'
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="3D Bayesian Network Structure",
                        showlegend=False,
                        scene=dict(
                            xaxis=dict(showbackground=False),
                            yaxis=dict(showbackground=False),
                            zaxis=dict(showbackground=False),
                        ),
                        margin=dict(l=0, r=0, b=0, t=30)
                    ))

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Could not render 3D network: {str(e)}")

