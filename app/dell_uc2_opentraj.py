import os
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import shap
import time
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter


# Initialize TensorBoard Writer
log_dir = "logs/OpenTraj"
writer = SummaryWriter(log_dir)

# Dataset paths
datasets = {
    "L-CAS": "OpenTraj/datasets/L-CAS",
    "Town-Center": "OpenTraj/datasets/Town-Center/TownCentre-groundtruth-top.txt",
    "PETS-2009": "OpenTraj/datasets/PETS-2009/data/annotations",
    "ETH": "OpenTraj/datasets/ETH/seq_eth",
    "ETH-Person": "OpenTraj/datasets/ETH-Person/data",
    "Edinburgh": "OpenTraj/datasets/Edinburgh/annotations",
    "HERMES": "OpenTraj/datasets/HERMES/Corridor-1D",
    "Wild-Track": "OpenTraj/datasets/Wild-Track/annotations_positions",
}


# Downsampling Overrepresented Datasets for balanced dataset
def downsample_dataset(df, max_rows=10000):
    """Downsample the dataset to a maximum number of rows."""
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=42)
    return df


# Define dataset loaders
def load_lcas_dataset(lcas_path):
    images = os.listdir(os.path.join(lcas_path, "images"))
    return pd.DataFrame({"Dataset": "L-CAS", "File_Name": images})


# Town Center Dataset
def load_town_center_dataset(path):
    df = pd.read_csv(path, sep=" ", header=None, names=["ID", "Timestamp", "Position_X", "Position_Y"])
    df["Dataset"] = "Town-Center"
    return df


# ETH Dataset
def load_eth_dataset(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".txt")]
    all_data = []
    for file in files:
        try:
            # Read file dynamically
            df = pd.read_csv(file, delim_whitespace=True, header=None, on_bad_lines="skip")
            columns = ["ID", "Timestamp", "Position_X", "Position_Y"][:df.shape[1]]
            df.columns = columns + [f"Extra_Col_{i}" for i in range(df.shape[1] - len(columns))]
            df["Dataset"] = "ETH"
            all_data.append(df)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def load_generic_txt_dataset(path, dataset_name, expected_columns):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".txt")]
    all_data = []
    for file in files:
        try:
            # Use on_bad_lines="skip"
            df = pd.read_csv(file, delim_whitespace=True, header=None, on_bad_lines="skip")
            df.columns = expected_columns[:df.shape[1]]
            df["Dataset"] = dataset_name
            all_data.append(df)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# Hermes Dataset
def load_hermes_dataset(path):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    all_data = []
    for file in files:
        try:
            df = pd.read_csv(file, delim_whitespace=True, header=None, names=["Timestamp", "Position_X", "Position_Y"])
            df["Dataset"] = "HERMES-1D"
            all_data.append(df)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# Edinburgh Dataset
def load_edinburgh_dataset(path):
    annotation_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".txt")]
    all_data = []
    for file in annotation_files:
        try:
            df = pd.read_csv(file, sep="\t", header=None, names=["Timestamp", "Position_X", "Position_Y"], on_bad_lines="skip")
            df["Dataset"] = "Edinburgh"
            df["File_Name"] = os.path.basename(file)
            all_data.append(df)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# Load ETH-Pearson dataset
def load_eth_person_dataset(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".xml")]
    all_data = []
    for file in files:
        try:
            print(f"Parsing file: {file}")
            tree = ET.parse(file)
            root = tree.getroot()
            print(f"File content: {ET.tostring(root, encoding='unicode')[:500]}")  # Log first 500 characters
            for frame in root.findall(".//frame"):
                frame_number = frame.get("number")
                for obj in root.findall(".//object"):
                    obj_id = obj.get("id")
                    for box in obj.findall(".//box"):
                        x = box.get("xc")
                        y = box.get("yc")
                        print(f"Found bbox with frame: {frame}, x: {x}, y: {y}")
                        all_data.append({
                            "Dataset": "ETH-Person",
                            "ID": obj_id,
                            "Timestamp": frame_number,
                            "Position_X": float(x),
                            "Position_Y": float(y)
                        })
            print(f"Processed file: {file}, Total entries: {len(all_data)}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return pd.DataFrame(all_data)


# PETS-2009 Dataset
def load_pets_2009_dataset(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".xml")]
    all_data = []
    for file in files:
        try:
            print(f"Parsing file: {file}")
            tree = ET.parse(file)
            root = tree.getroot()
            tree.getroot()
            print(f"File content: {ET.tostring(root, encoding='unicode')[:500]}")  # Log first 500 characters
            for frame in root.findall(".//frame"):
                frame_number = frame.get("number")
                for obj in root.findall(".//object"):
                    obj_id = obj.get("id")
                    for box in obj.findall(".//box"):
                        x = box.get("xc")
                        y = box.get("yc")
                        print(f"Found bbox with frame: {frame}, x: {x}, y: {y}")
                        all_data.append({
                            "Dataset": "PETS-2009",
                            "ID": obj_id,
                            "Timestamp": frame,
                            "Position_X": float(x),
                            "Position_Y": float(y)
                        })
            print(f"Processed file: {file}, Total entries: {len(all_data)}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return pd.DataFrame(all_data)


# Wild-Track Dataset
def load_wild_track_dataset(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json")]
    all_data = []
    for file in files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
            for obj in data:
                all_data.append({
                    "Dataset": "Wild-Track",
                    "ID": obj.get("ID"),
                    "Timestamp": obj.get("Timestamp"),
                    "Position_X": obj.get("Position_X"),
                    "Position_Y": obj.get("Position_Y")
                })
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return pd.DataFrame(all_data)


# Combine Datasets Function
def combine_datasets():
    datasets_combined = [
        load_lcas_dataset(datasets["L-CAS"]),
        load_town_center_dataset(datasets["Town-Center"]),
        load_eth_dataset(datasets["ETH"]),
        load_edinburgh_dataset(datasets["Edinburgh"]),
        load_eth_person_dataset(datasets["ETH-Person"]),
        load_pets_2009_dataset(datasets["PETS-2009"]),
        load_hermes_dataset(datasets["HERMES"]),
      load_wild_track_dataset(datasets["Wild-Track"]),
    ]
    combined_df = pd.concat(datasets_combined, ignore_index=True)
    for dataset_name, count in combined_df["Dataset"].value_counts().items():
        writer.add_scalar(f"Dataset Distribution/{dataset_name}", count)
    return combined_df


def plot_dataset_distribution(dataset_counts, output_path="dataset_distribution.png"):
    dataset_counts.plot(kind="bar", color="skyblue", figsize=(10, 6))
    plt.title("Dataset Distribution", fontsize=14)
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Number of Rows", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Load combined dataset
def load_combined_dataset(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Dataset loaded with shape: {df.shape}")
    return df


# Feature Engineering and validation predictions
def preprocess_data_with_features(df):
    print("Preprocessing data and adding engineered features...")

    required_columns = ["Timestamp", "Position_X", "Position_Y"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_columns).reset_index(drop=True)
    print(f"Data shape after dropping NaNs: {df.shape}")

    df["Velocity"] = np.sqrt(df["Position_X"].diff()**2 + df["Position_Y"].diff()**2) / df["Timestamp"].diff()
    df["Acceleration"] = df["Velocity"].diff()

    df["Proximity_X"] = df["Position_X"].shift(-1) - df["Position_X"]
    df["Proximity_Y"] = df["Position_Y"].shift(-1) - df["Position_Y"]
    df["Proximity_Distance"] = np.sqrt(df["Proximity_X"]**2 + df["Proximity_Y"]**2)

    df["Direction_Angle"] = np.arctan2(df["Position_Y"].diff(), df["Position_X"].diff())

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    print("Velocity and Acceleration statistics:")
    print(df[["Velocity", "Acceleration"]].describe())

    # Log to TensorBoard
    writer.add_scalars("Preprocessing/Statistics", {
    "Velocity Mean": df["Velocity"].mean(),
    "Velocity Std": df["Velocity"].std(),
    "Acceleration Mean": df["Acceleration"].mean(),
    "Acceleration Std": df["Acceleration"].std()
    })

    df["Label"] = (df["Velocity"] > 0.1).astype(int)
    print(f"Label column distribution:\n{df['Label'].value_counts()}")

    if "Dataset" in df.columns:
        print("Encoding categorical column 'Dataset'...")
        df = pd.get_dummies(df, columns=["Dataset"], drop_first=True)

    X = df.drop(columns=["ID", "Label", "Timestamp"], errors="ignore")
    y = df["Label"]

    print("Ensuring all features are numeric...")
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"Processed dataset shape: {X.shape}")
    print("Preprocessing complete.")
    return X, y, df


# Model training and evaluation
def train_and_evaluate(X, y):
    print("Training and evaluating model...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    rf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, random_state=42)
    search.fit(X_train, y_train)
    print(f"Best hyperparameters: {search.best_params_}")

    model = search.best_estimator_
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Log to TensorBoard
    writer.add_scalar("Training/Accuracy", accuracy)
    writer.add_scalar("Training/Precision", classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["precision"])
    writer.add_scalar("Training/Recall", classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["recall"])

    return model


# Graph Visualization
def visualize_data(df):
    """Visualize preprocessed data."""
    print("Visualizing data...")

    if "Position_X" in df.columns and "Position_Y" in df.columns and "Label" in df.columns:

        plt.figure(figsize=(12, 8))
        plt.scatter(df["Position_X"], df["Position_Y"], c=df["Label"], cmap="coolwarm", s=1, alpha=0.5)
        plt.colorbar(label="Label (0: Stationary, 1: Moving)")
        plt.title("Trajectory Visualization (Stationary vs Moving)")
        plt.xlabel("Position X")
        plt.ylabel("Position Y")
        plt.savefig("OpenTraj_trajectory_visualization.png")
        plt.close()
        print("Trajectory visualization saved as 'OpenTraj_trajectory_visualization.png'.")

        df[["Velocity", "Acceleration"]].plot(figsize=(12, 6), title="Velocity and Acceleration Trends")
        plt.xlabel("Index")
        plt.ylabel("Magnitude")
        plt.savefig("OpenTraj_velocity_acceleration_trends.png")
        plt.close()
        print("Velocity and Acceleration trends saved as 'OpenTraj_velocity_acceleration_trends.png'.")
    else:
        print("Error: Missing required columns for visualization.")

    

# Adding an LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation="sigmoid")  # Binary classification (stationary vs moving)
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Train an LSTM model
def train_lstm(X_train, y_train, X_test, y_test, batch_size=64, epochs=10):

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    if False:
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)

        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"LSTM Model Accuracy: {accuracy:.4f}")

    return model


'''
# Train an LSTM model
def train_lstm(X_train, y_train, X_test, y_test, batch_size=64, epochs=10):

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"LSTM Model Accuracy: {accuracy:.4f}")

    return model
'''

#Adding Real-Time Analysis Capabilities
def real_time_analysis(model, data_stream, scaler):
    
    print("Starting real-time analysis...")
    for batch in data_stream:
        scaled_batch = scaler.transform(batch)
        predictions = model.predict(scaled_batch)

        for i, pred in enumerate(predictions):
            status = "Moving" if pred > 0.5 else "Stationary"
            print(f"Prediction {i}: {status}")
        time.sleep(1)  # Simulate real-time latency


def run_workload():
    combined_file = "OpenTraj_combined_dataset.csv"

    print("Combining datasets...")
    combined_df = combine_datasets()
    combined_df.to_csv(combined_file, index=False)
    print(f"Combined dataset saved to '{combined_file}'.")
    print(f"Number of rows: {len(combined_df)}")
    print(f"Dataset distribution:\n{combined_df['Dataset'].value_counts()}")

    if combined_df.empty:
        print("Error: Combined dataset is empty. Exiting.")
        return

    print("Loading combined dataset...")
    df = load_combined_dataset(combined_file)

    print("Preprocessing data...")
    try:
        X, y, processed_df = preprocess_data_with_features(df)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return

    if X.empty or y.empty:
        print("Error: Preprocessed data is empty. Please check the preprocessing steps.")
        return

    print("Visualizing data...")
    try:
        visualize_data(processed_df)
    except Exception as e:
        print(f"Error during visualization: {e}")
        return

    print("Training and evaluating model...")
    try:
        train_and_evaluate(X, y)
    except Exception as e:
        print(f"Error during model training and evaluation: {e}")
        return






