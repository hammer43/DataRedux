import streamlit as st
import h2o
from h2o.estimators import H2ORandomForestEstimator
from sklearn.datasets import load_iris
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

def create_feature_scatter(df, feature1, feature2, target, title):
    """Create interactive scatter plot of two features"""
    fig = px.scatter(
        df,
        x=feature1,
        y=feature2,
        color=target,
        title=title,
        template="simple_white",
        labels={target: "Target Class"},
        hover_data=df.columns
    )
    fig.update_layout(height=500)
    return fig

def create_parallel_coordinates(df, features, target):
    """Create parallel coordinates plot for all features"""
    fig = px.parallel_coordinates(
        df,
        dimensions=features + [target],
        color=target,
        title="Parallel Coordinates Plot of All Features"
    )
    fig.update_layout(height=500)
    return fig

def main():
    st.title('Interactive H2O Random Forest Classifier')
    
    # Initialize H2O server with specific settings
    try:
        h2o.init(port=54323,  # Using different port
                 start_h2o=True,
                 nthreads=-1,
                 min_mem_size="1G",
                 max_mem_size="2G",
                 bind_to_localhost=True)
    except Exception as e:
        st.error(f"Error initializing H2O: {e}")
        st.stop()
        
    # Load dataset
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=['target'])
    df = pd.concat([X, y], axis=1)
    
    # Convert dataset to H2O format
    h2o_data = h2o.H2OFrame(df)
    
    # Split dataset into train/test
    train, test = h2o_data.split_frame(ratios=[.8])

    # Sidebar: Hyperparameters input
    st.sidebar.header('Hyperparameter Tuning')
    n_estimators = st.sidebar.slider('Number of Trees (n_estimators)', 10, 200, 100)
    max_depth = st.sidebar.slider('Maximum Depth (max_depth)', 1, 30, 10)
    min_rows = st.sidebar.slider('Minimum Samples Leaf (min_rows)', 1, 10, 1)

    # Data Visualization Section
    st.header("Data Visualization")
    
    # Feature selection for scatter plot
    st.subheader("Interactive Feature Scatter Plot")
    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.selectbox('Select first feature', X.columns, index=0)
    with col2:
        feature2 = st.selectbox('Select second feature', X.columns, index=1)
    
    # Create and display scatter plot
    scatter_fig = create_feature_scatter(df, feature1, feature2, 'target', 
                                       f'Scatter Plot: {feature1} vs {feature2}')
    st.plotly_chart(scatter_fig, use_container_width=True)
    
    # Parallel coordinates plot
    st.subheader("Parallel Coordinates Plot")
    parallel_fig = create_parallel_coordinates(df, list(X.columns), 'target')
    st.plotly_chart(parallel_fig, use_container_width=True)

    # Model Training and Prediction
    rf_model = H2ORandomForestEstimator(
        ntrees=n_estimators,
        max_depth=max_depth,
        min_rows=min_rows,
        seed=42
    )
    
    with st.spinner('Training model...'):
        rf_model.train(x=list(X.columns), y='target', training_frame=train)
    
    # Predictions
    predictions = rf_model.predict(test)
    
    # Model Performance Section
    st.header("Model Performance")
    
    # Basic metrics
    performance = rf_model.model_performance(test)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{performance.accuracy()[0][1]:.2%}")
    with col2:
        st.metric("Error Rate", f"{1 - performance.accuracy()[0][1]:.2%}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    conf_matrix = performance.confusion_matrix()
    conf_matrix_fig = go.Figure(data=go.Heatmap(
        z=conf_matrix.as_data_frame().iloc[:-1, :-1].values,
        x=['Predicted ' + str(i) for i in range(3)],
        y=['Actual ' + str(i) for i in range(3)],
        text=conf_matrix.as_data_frame().iloc[:-1, :-1].values,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    conf_matrix_fig.update_layout(title="Confusion Matrix Heatmap")
    st.plotly_chart(conf_matrix_fig, use_container_width=True)
    
    # Feature Importance
    st.subheader("Feature Importance")
    varimp = rf_model.varimp(use_pandas=True)
    if varimp is not None:
        imp_fig = go.Figure(go.Bar(
            x=varimp['percentage'],
            y=varimp['variable'],
            orientation='h',
            text=varimp['percentage'].round(2),
            texttemplate='%{text}%',
            textposition='auto',
        ))
        imp_fig.update_layout(
            title="Feature Importance (%)",
            xaxis_title="Importance",
            yaxis_title="Features",
            height=400
        )
        st.plotly_chart(imp_fig, use_container_width=True)
    
    # Cleanup
    h2o.shutdown(prompt=False)

if __name__ == "__main__":
    main()