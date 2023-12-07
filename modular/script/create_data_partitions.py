import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler
from common import create_lda_partitions
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="iris")
argparser.add_argument("--num_partitions", type=int, default=3)
argparser.add_argument("--concentration", type=float, default=1e-6)
args = argparser.parse_args()

# Load breast cancer dataset
data_name = args.dataset
if data_name == "breast_cancer":
    data = load_breast_cancer()
elif data_name == "iris":
    data = load_iris()
elif data_name == "wine":
    data = load_wine()
features = data.data
labels = data.target

# Perform MinMax scaling on features
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)

# Split the data into training (80%) and testing (20%) sets
features_train, features_test, labels_train, labels_test = train_test_split(
    normalized_features, labels, test_size=0.2, random_state=42
)
data_partitions, d_d = create_lda_partitions(
    dataset=(features_train, labels_train),
    num_partitions=args.num_partitions,
    accept_imbalanced=True,
    concentration=args.concentration,
)

for idx, (partition, dis) in enumerate(zip(data_partitions, d_d)):
    # Concatenate features and labels for training and testing sets
    print(f"Partition {idx} has {dis}")
    train_data = np.column_stack((partition[0], partition[1]))

    # Create Pandas DataFrames
    train_df = pd.DataFrame(train_data, columns=np.append(data.feature_names, "label"))
    train_df.to_csv(
        f"../data/train_{data_name}_norm_{idx}.csv", header=None, index=False
    )

# Save to CSV files without headers
test_data = np.column_stack((features_test, labels_test))
test_df = pd.DataFrame(test_data, columns=np.append(data.feature_names, "label"))
test_df.to_csv(f"../data/test_{data_name}_norm.csv", header=None, index=False)
