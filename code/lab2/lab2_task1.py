import pandas as pd
from pathlib import Path
import numpy as np


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lab2" / "playgolf.csv"

if __name__ == "__main__":    # Example usage
    data = pd.read_csv(DATA_PATH)
    print(data)
    y = data.iloc[:, 4].to_numpy()
    print(y)
    X = data.iloc[:, 0:4].to_numpy()
    print(X.shape)
    outlook = X[:, 0]
    temp = X[:, 1]
    humidity = X[:, 2]
    windy = X[:, 3]
    print(outlook)
    print(outlook.shape)

def impurity(counts: np.ndarray) -> float:
    """Return Gini impurity for one node given class counts."""
    total = np.sum(counts)
    p = counts/total
    return float(1.0 - np.sum(p ** 2))

def gini_gain(attr: np.ndarray, y: np.ndarray) -> float:
    """Compute Gini gain for splitting labels y by categorical attribute attr."""
    total_samples = len(y)
    _, counts = np.unique(y, return_counts=True)
    root_impurity = impurity(counts)
    unique_attr_values = np.unique(attr)
    weighted_child_impurity=0.0
    for value in unique_attr_values:
        sub_y=y[attr==value]
        _, child_counts = np.unique(sub_y, return_counts=True)
        child_impurity = impurity(child_counts)
        weight = len(sub_y)/total_samples
        weighted_child_impurity += weight * child_impurity
    return float(root_impurity - weighted_child_impurity)
        
_, attr_outlook = np.unique(outlook, return_inverse=True)
outlook_gini_gain = gini_gain(attr_outlook, y)
temp_gini_gain = gini_gain(temp, y)
humidity_gini_gain = gini_gain(humidity, y)
windy_gini_gain = gini_gain(windy, y)
print(f"The Gini gain for Outlook is {outlook_gini_gain}")
print(f"The Gini gain for Temperature is {temp_gini_gain}")
print(f"The Gini gain for Humidity is is {humidity_gini_gain}")
print(f"The Gini gain for Windy is {windy_gini_gain}")
