"""Lab 2, Task 2 starter: fit and visualise a decision tree on PlayGolf.

Run from the repository root with: python code/lab2/lab2_task2.py
Complete the steps below using Lab Sheet 2. The data-loading step is provided.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lab2" / "playgolf.csv"


def load_playgolf():
    """Return the four attributes as a DataFrame and the labels as an array."""
    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns="PlayGolf")
    y = data["PlayGolf"].to_numpy()
    return X, y


if __name__ == "__main__":
    X, y = load_playgolf()
    print("Attributes:", list(X.columns))
    print("Raw X shape:", X.shape, "y shape:", y.shape)
    print("Outlook values:", X["Outlook"].unique())

    # A. Try fitting DecisionTreeClassifier on X as it is. What error do you get?
    # Then one-hot encode Outlook, keeping Temp, Humidity, and Windy numeric.
    # ColumnTransformer and OneHotEncoder can do this in a Pipeline.
    # Check that the transformed X has shape (14, 6).

    ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["Outlook"])], remainder="passthrough")
    dt_pipe = Pipeline([
        ("preprocessor", ct),
        ("dt", DecisionTreeClassifier(criterion="gini", random_state=42))
    ])
    X_transformed = ct.fit_transform(X)
    print("Transformed X shape:", X_transformed.shape)
    # B. Fit a DecisionTreeClassifier to the numeric features and y.
    # Use sklearn.tree.plot_tree with feature_names and class_names.
    # If you used a Pipeline, plot its fitted tree step, not the Pipeline itself.
    # How does this tree compare with the one from the lecture?
    dt_pipe.fit(X, y)
    dt_model=dt_pipe.named_steps["dt"]
    preprocessor = dt_pipe.named_steps["preprocessor"]
    
    feature_names = preprocessor.get_feature_names_out()
    class_names = ["Don't Play", "Play"]
    
    plt.figure(figsize=(10, 6))
    plot_tree(dt_model,feature_names=feature_names,class_names=class_names,filled=True,rounded=True)
    plt.title("DecisionTreeClassifier (CART)")
    plt.show()

    # Try ExtraTreeClassifier as well. Compare the two trees and their
    # training predictions. What would you need to compare them fairly on
    # new data?
    extra_pipeline = Pipeline([("preprocessor", ct),("extra_tree", ExtraTreeClassifier(criterion="gini", random_state=42))])
    extra_pipeline.fit(X, y)
    extra_model = extra_pipeline.named_steps["extra_tree"]
    
    plt.figure(figsize=(10, 6))
    plot_tree(extra_model,feature_names=feature_names,class_names=class_names,filled=True,rounded=True,)
    plt.title("ExtraTreeClassifier")
    plt.show()

    dt_preds = dt_pipe.predict(X)
    extra_preds = extra_pipeline.predict(X)

    print("Decision Tree Training Accuracy:", np.mean(dt_preds == y))
    print("Extra Tree Training Accuracy:", np.mean(extra_preds == y))
