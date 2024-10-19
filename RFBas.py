import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar: Hyperparameters input
st.sidebar.header('Hyperparameter Tuning')

n_estimators = st.sidebar.slider('Number of Trees in the Forest (n_estimators)', 10, 200, 100)
max_depth = st.sidebar.slider('Maximum Depth of the Tree (max_depth)', 1, 30, 10)
min_samples_split = st.sidebar.slider('Minimum Samples Split (min_samples_split)', 2, 10, 2)
min_samples_leaf = st.sidebar.slider('Minimum Samples Leaf (min_samples_leaf)', 1, 10, 1)

# Model creation with chosen hyperparameters
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Output
st.write(f"Model accuracy: {accuracy:.2%}")

# Show classification results
st.write("Hyperparameters chosen:")
st.write(f"n_estimators: {n_estimators}")
st.write(f"max_depth: {max_depth}")
st.write(f"min_samples_split: {min_samples_split}")
st.write(f"min_samples_leaf: {min_samples_leaf}")
