import streamlit as st
import h2o
from h2o.estimators import H2ORandomForestEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

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

# Convert dataset to H2O format
h2o_data = h2o.H2OFrame(pd.concat([X, y], axis=1))

# Split dataset into train/test
train, test = h2o_data.split_frame(ratios=[.8])

# Sidebar: Hyperparameters input
st.sidebar.header('Hyperparameter Tuning')
n_estimators = st.sidebar.slider('Number of Trees in the Forest (n_estimators)', 10, 200, 100)
max_depth = st.sidebar.slider('Maximum Depth of the Tree (max_depth)', 1, 30, 10)
min_rows = st.sidebar.slider('Minimum Samples Leaf (min_rows)', 1, 10, 1)

# Create Random Forest model in H2O
rf_model = H2ORandomForestEstimator(
    ntrees=n_estimators,
    max_depth=max_depth,
    min_rows=min_rows,
    seed=42
)

# Train model
rf_model.train(x=list(X.columns), y='target', training_frame=train)

# Predict
predictions = rf_model.predict(test)

# Model performance
performance = rf_model.model_performance(test)

# Display accuracy
accuracy = performance.accuracy()[0][1]
st.write(f"Model accuracy: {accuracy:.2%}")

# Display chosen hyperparameters
st.write("Hyperparameters chosen:")
st.write(f"n_estimators: {n_estimators}")
st.write(f"max_depth: {max_depth}")
st.write(f"min_rows: {min_rows}")

# Shutdown H2O instance
h2o.shutdown(prompt=False)