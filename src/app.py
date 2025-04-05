import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

# Set page config
st.set_page_config(
    page_title="Logistic Regression Hyperparameter Tuning",
    layout="wide"
)

# Title and description
st.title("Logistic Regression Hyperparameter Tuning Visualization")
st.markdown("""
This app helps you understand:
1. How different dataset characteristics affect Logistic Regression
2. Impact of Logistic Regression hyperparameters
3. Visualization of decision boundaries
""")

# Sidebar for dataset parameters
st.sidebar.header("Dataset Parameters")

# Dataset generation parameters
n_samples = st.sidebar.slider("Number of samples", 100, 2000, 1000)
n_features = st.sidebar.slider("Number of features", 2, 20, 2)
n_informative = st.sidebar.slider("Number of informative features", 1, n_features, min(2, n_features))

# Calculate max possible redundant features
max_redundant = max(0, n_features - n_informative)
default_redundant = min(max_redundant, 0)

# Only show redundant features slider if we can have redundant features
if max_redundant > 0:
    n_redundant = st.sidebar.slider("Number of redundant features", 0, max_redundant, default_redundant)
else:
    n_redundant = 0
    st.sidebar.text("No space for redundant features")

n_classes = st.sidebar.slider("Number of classes", 2, 5, 2)
class_sep = st.sidebar.slider("Class separation", 0.1, 5.0, 1.0)
n_clusters = st.sidebar.slider("Clusters per class", 1, 5, 1)
random_state = st.sidebar.slider("Random seed", 0, 100, 42)

# Logistic Regression parameters
st.sidebar.header("Model Parameters")

# Add Logistic Regression hyperparameters
C = st.sidebar.slider("C (Inverse of regularization strength)", 0.01, 10.0, 1.0, 0.01)
max_iter = st.sidebar.slider("Maximum iterations", 100, 1000, 100, 50)
penalty = st.sidebar.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"])

if penalty == "elasticnet":
    l1_ratio = st.sidebar.slider("L1 ratio (for elasticnet)", 0.0, 1.0, 0.5, 0.1)
else:
    l1_ratio = None

solver = st.sidebar.selectbox(
    "Solver",
    ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
    help="Some solvers are incompatible with certain penalties. The app will adjust automatically."
)

# Handle solver-penalty compatibility
if solver in ["newton-cg", "lbfgs", "sag"]:
    if penalty not in ["l2", "None"]:
        st.sidebar.warning(f"{solver} solver supports only 'l2' or 'none' penalty. Defaulting to 'l2'.")
        penalty = "l2"
elif solver == "liblinear":
    if penalty not in ["l1", "l2"]:
        st.sidebar.warning(f"liblinear solver supports only 'l1' and 'l2' penalties. Defaulting to 'l2'.")
        penalty = "l2"

# Create model with selected parameters
if penalty == "elasticnet" and solver == "saga":
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        penalty=penalty,
        l1_ratio=l1_ratio,
        solver=solver,
        random_state=random_state
    )
else:
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        penalty=penalty,
        solver=solver,
        random_state=random_state
    )

param_desc = f"C={C}, penalty={penalty}, solver={solver}"

# Ensure n_redundant is valid
max_redundant = max(0, n_features - n_informative)
n_redundant = min(n_redundant, max_redundant)

# Generate dataset
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=min(n_informative, n_features),
    n_redundant=n_redundant,
    n_classes=n_classes,
    n_clusters_per_class=n_clusters,
    class_sep=class_sep,
    random_state=random_state
)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model and get predictions
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Calculate metrics
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average='weighted'),
    "Recall": recall_score(y_test, y_pred, average='weighted'),
    "F1 Score": f1_score(y_test, y_pred, average='weighted')
}

# Display metrics
col1, col2 = st.columns(2)

with col1:
    st.header("Model Performance Metrics")
    metrics_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": list(metrics.values())
    })
    
    # Create bar chart for metrics
    fig_metrics = px.bar(
        metrics_df,
        x="Metric",
        y="Value",
        title="Logistic Regression Performance Metrics",
        text="Value"
    )
    fig_metrics.update_traces(texttemplate='%{text:.3f}')
    st.plotly_chart(fig_metrics)

# If we have 2D data, plot the decision boundary
if n_features == 2:
    with col2:
        st.header("Decision Boundary Visualization")
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # Get predictions for mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = scaler.transform(mesh_points)
        Z = model.predict(mesh_points_scaled)
        Z = Z.reshape(xx.shape)
        
        # Get probability scores for colormap intensity
        Z_prob = model.predict_proba(mesh_points_scaled)[:, 1].reshape(xx.shape)
        
        # Create figure
        fig = go.Figure()
        
        # Add decision boundary contour
        fig.add_trace(go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=Z_prob,
            colorscale='RdBu_r',
            opacity=0.4,
            name='Decision Boundary',
            showscale=True,
            contours=dict(start=0, end=1, size=0.1)
        ))
        
        # Add scatter plot for actual data points
        for i in range(n_classes):
            mask = y == i
            fig.add_trace(go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode='markers',
                name=f'Class {i}',
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=f"Decision Boundary and Data Points<br><sup>{param_desc}</sup>",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            width=700,
            height=500
        )
        
        st.plotly_chart(fig)

        # Add probability explanation
        st.markdown("""
        **Visualization Guide:**
        - The background color represents the probability of class prediction
        - Darker red indicates higher probability for class 1
        - Darker blue indicates higher probability for class 0
        - The decision boundary is where the probability is 0.5 (white region)
        """)
else:
    with col2:
        st.header("Model Coefficients")
        coef_df = pd.DataFrame({
            'Feature': [f'Feature {i}' for i in range(n_features)],
            'Coefficient': model.coef_[0]
        })
        coef_df = coef_df.sort_values('Coefficient', ascending=False)
        
        fig_coef = px.bar(
            coef_df,
            x='Feature',
            y='Coefficient',
            title='Feature Coefficients (Impact on Prediction)'
        )
        st.plotly_chart(fig_coef)
        
        st.markdown("""
        **Note:** The coefficients show the impact of each feature on the prediction:
        - Positive values increase the probability of class 1
        - Negative values increase the probability of class 0
        - Larger absolute values indicate stronger influence
        """)

# Display dataset information
st.header("Dataset Information")
st.write(f"""
- Total samples: {n_samples}
- Features: {n_features} (Informative: {n_informative}, Redundant: {n_redundant})
- Classes: {n_classes}
- Class separation: {class_sep}
- Clusters per class: {n_clusters}
""")

# Training/Test split information
st.write(f"""
- Training samples: {X_train.shape[0]}
- Test samples: {X_test.shape[0]}
""")
