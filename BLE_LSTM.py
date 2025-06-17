import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os


# --- Data Generation and Helper Functions ---
# def generate_points_on_line_segment(p1, p2, num_segments):
    # """
    # Generates points along a line segment, including the start and end points.
    # Divides the line into `num_segments` equally sized segments, resulting in `num_segments + 1` points.

    # Args:
    #     p1 (list): [x1, y1] of the start point.
    #     p2 (list): [x2, y2] of the end point.
    #     num_segments (int): The number of segments to divide the line into.
    #                         This means num_segments + 1 points will be generated.
    # Returns:
    #     list: A list of [x, y] coordinates along the segment.
    # """
    # points = []
    # if num_segments <= 0:
    #     return [p1, p2]

    # for i in range(num_segments + 1):
    #     t = i / num_segments
    #     x = p1[0] * (1 - t) + p2[0] * t
    #     y = p1[1] * (1 - t) + p2[1] * t
    #     points.append([x, y])
    # return points


def preprocess_ble_data(file_path, nan_fill_method='linear'):
    """
    Reads BLE raw data, performs data processing including IQR-based outlier removal,
    robust interpolation, and optional rolling median smoothing.
    Reference polygon-based outlier removal has been removed as per user request.

    Args:
        file_path (str): The path to the CSV file containing BLE raw data.
        nan_fill_method (str): Method to fill NaN values after outlier removal.
                               Options: 'linear', 'ffill_bfill', 'mean'. Defaults to 'linear'.

    Returns:
        pandas.DataFrame: The processed DataFrame with 'x_snap', 'y_snap', 'timestamp',
                          and 'device_id' (if available). Returns None if an error occurs.
        numpy.ndarray: Original raw coordinates before any processing.
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None, None

        df = pd.read_csv(file_path)
        original_coordinates = df[['x_snap', 'y_snap']].values.copy()

        # Identify key columns
        timestamp_col = next((col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()), None)
        x_col = next((col for col in df.columns if 'x_snap' in col.lower() or 'x_coord' in col.lower() or 'x_position' in col.lower()), None)
        y_col = next((col for col in df.columns if 'y_snap' in col.lower() or 'y_coord' in col.lower() or 'y_position' in col.lower()), None)
        device_id_col = next((col for col in df.columns if 'device' in col.lower() or 'mac' in col.lower() or 'id' in col.lower()), None)

        if not all([timestamp_col, x_col, y_col]):
            print("Error: Missing required columns (Timestamp, X, Y).")
            print(f"Available columns: {df.columns.tolist()}")
            return None, None

        # Convert timestamp to proper 'datetime'
        if pd.api.types.is_numeric_dtype(df[timestamp_col]):
            if (df[timestamp_col].max() > 10**10) and (df[timestamp_col].max() < 10**13):
                 df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms')
            else:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
        else:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Keep only x_snap, y_snap, timestamp, and ID
        cols_to_keep = [x_col, y_col, timestamp_col]
        if device_id_col:
            cols_to_keep.append(device_id_col)
        df_processed = df[cols_to_keep].copy()
        df_processed.rename(columns={timestamp_col: 'timestamp', x_col: 'x_snap', y_col: 'y_snap'}, inplace=True)
        if device_id_col:
            df_processed.rename(columns={device_id_col: 'device_id'}, inplace=True)

        # IQR-based outlier removal
        def remove_outliers_iqr(df_in, column):
            Q1 = df_in[column].quantile(0.25)
            Q3 = df_in[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Replace outliers with NaN so interpolation can handle them
            df_in.loc[(df_in[column] < lower_bound) | (df_in[column] > upper_bound), column] = np.nan
            return df_in

        original_rows_iqr = len(df_processed)
        df_processed = remove_outliers_iqr(df_processed, 'x_snap')
        df_processed = remove_outliers_iqr(df_processed, 'y_snap')
        # Count actual NaNs introduced
        removed_coords_count = df_processed['x_snap'].isna().sum() + df_processed['y_snap'].isna().sum()
        if removed_coords_count > 0:
            print(f"Replaced {removed_coords_count} coordinates with NaN based on IQR outlier detection.")

        df_processed = df_processed.sort_values(by='timestamp').reset_index(drop=True)

        # Removed: Euclidean Distance-based Outlier Handling (as per user request)

        # Robust Interpolation and Filling based on specified method
        print(f"\nApplying NaN filling using method: {nan_fill_method}...")
        for col in ['x_snap', 'y_snap']:
            if df_processed[col].isnull().any(): # Only interpolate if NaNs exist
                if nan_fill_method == 'linear':
                    df_processed[col].interpolate(method='linear', limit_direction='both', inplace=True)
                    df_processed[col].fillna(method='ffill', inplace=True)
                    df_processed[col].fillna(method='bfill', inplace=True)
                elif nan_fill_method == 'ffill_bfill':
                    df_processed[col].fillna(method='ffill', inplace=True)
                    df_processed[col].fillna(method='bfill', inplace=True)
                elif nan_fill_method == 'mean':
                    mean_val = df_processed[col].mean()
                    df_processed[col].fillna(mean_val, inplace=True)
                else:
                    print(f"Warning: Unknown NaN fill method '{nan_fill_method}'. No NaN filling applied for {col}.")
        print("NaN filling complete.")

        # Optional: Apply Rolling Smoothing
        apply_rolling_smooth = True
        rolling_window_size = 5
        if apply_rolling_smooth:
            if 'device_id' in df_processed.columns and df_processed['device_id'].nunique() > 1:
                df_processed['x_snap'] = df_processed.groupby('device_id')['x_snap'].transform(
                    lambda x: x.rolling(window=rolling_window_size, min_periods=1, center=True).median().fillna(method='bfill').fillna(method='ffill')
                )
                df_processed['y_snap'] = df_processed.groupby('device_id')['y_snap'].transform(
                    lambda x: x.rolling(window=rolling_window_size, min_periods=1, center=True).median().fillna(method='bfill').fillna(method='ffill')
                )
            else:
                df_processed['x_snap'] = df_processed['x_snap'].rolling(window=rolling_window_size, min_periods=1, center=True).median().fillna(method='bfill').fillna(method='ffill')
                df_processed['y_snap'] = df_processed['y_snap'].rolling(window=rolling_window_size, min_periods=1, center=True).median().fillna(method='bfill').fillna(method='ffill')
            print(f"Rolling median smoothing applied with window size {rolling_window_size}.")

        df_processed = df_processed.sort_values(by='timestamp').reset_index(drop=True)

        return df_processed, original_coordinates

    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
        return None, None


# --- LSTM Model Components ---
class MovementDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_and_forecast_lstm(processed_df, look_back=10, forecast_horizon=10,
                            epochs=50, batch_size=16, optimizer_config=None,
                            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Prepares data for LSTM, trains the model, and generates a forecast.
    Includes train/validation/test split and RMSE calculation.

    Args:
        processed_df (pandas.DataFrame): DataFrame with processed 'x_snap' and 'y_snap' columns.
        look_back (int): Number of previous time steps to use as input for prediction.
        forecast_horizon (int): Number of future steps to forecast.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoader.
        optimizer_config (dict): Dictionary specifying optimizer 'name' and 'params'.
        train_ratio (float): Ratio of the dataset to be used for training.
        val_ratio (float): Ratio of the dataset to be used for validation.
        test_ratio (float): Ratio of the dataset to be used for testing.

    Returns:
        tuple: (numpy.ndarray, numpy.ndarray, numpy.ndarray, float, float, float)
               `coordinates_scaled`: The scaled processed data used for training, validation, and testing.
               `forecast_scaled`: The scaled forecasted future points.
               `scaler`: The MinMaxScaler used for inverse transformation.
               `train_rmse`: Root Mean Squared Error on the training set.
               `val_rmse`: Root Mean Squared Error on the validation set.
               `test_rmse`: Root Mean Squared Error on the test set.
    """
    if not (train_ratio + val_ratio + test_ratio == 1.0):
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")

    if len(processed_df) < look_back + 1:
        print(f"Insufficient processed data for LSTM. Need at least {look_back + 1} rows.")
        return None, None, None, None, None, None

    coordinates = processed_df[['x_snap', 'y_snap']].values
    scaler = MinMaxScaler()
    coordinates_scaled = scaler.fit_transform(coordinates)

    X, y = [], []
    for i in range(len(coordinates_scaled) - look_back):
        X.append(coordinates_scaled[i:i+look_back])
        y.append(coordinates_scaled[i+look_back])

    X = np.array(X)
    y = np.array(y)

    print(f"\nLSTM Data preparation:")
    print(f"Full X shape: {X.shape}, Full y shape: {y.shape}")

    # Calculate split indices
    total_samples = len(X)
    train_split_index = int(total_samples * train_ratio)
    val_split_index = int(total_samples * (train_ratio + val_ratio))

    X_train, y_train = X[:train_split_index], y[:train_split_index]
    X_val, y_val = X[train_split_index:val_split_index], y[train_split_index:val_split_index]
    X_test, y_test = X[val_split_index:], y[val_split_index:]

    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
    print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")

    train_dataset = MovementDataset(X_train, y_train)
    val_dataset = MovementDataset(X_val, y_val)
    test_dataset = MovementDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size=2, hidden_size=64, num_layers=2)
    criterion = nn.MSELoss()

    if optimizer_config is None:
        optimizer_config = {'name': 'Adam', 'params': {'lr': 0.001}}
        print("Optimizer not specified, defaulting to Adam.")

    optimizer_name = optimizer_config['name']
    optimizer_params = optimizer_config.get('params', {})

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), **optimizer_params)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Please choose from 'Adam', 'SGD', 'RMSprop', 'AdamW'.")

    print(f"Using optimizer: {optimizer_name} with params: {optimizer_params}")

    print(f"\nStarting LSTM training for {epochs} epochs...")
    val_rmse_history = []
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            output = model(batch_X)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        val_predictions = []
        val_actuals = []
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                val_output = model(batch_X_val)
                val_predictions.extend(val_output.numpy())
                val_actuals.extend(batch_y_val.numpy())

        val_predictions = np.array(val_predictions)
        val_actuals = np.array(val_actuals)
        val_predictions_original_scale = scaler.inverse_transform(val_predictions)
        val_actuals_original_scale = scaler.inverse_transform(val_actuals)
        current_val_rmse = np.sqrt(mean_squared_error(val_actuals_original_scale, val_predictions_original_scale))
        val_rmse_history.append(current_val_rmse)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.4f}, Validation RMSE: {current_val_rmse:.4f}")

    # Final evaluation on the separate test set
    model.eval()
    test_predictions = []
    test_actuals = []
    with torch.no_grad():
        for batch_X_test, batch_y_test in test_loader:
            test_output = model(batch_X_test)
            test_predictions.extend(test_output.numpy())
            test_actuals.extend(batch_y_test.numpy())

    test_predictions = np.array(test_predictions)
    test_actuals = np.array(test_actuals)

    test_predictions_original_scale = scaler.inverse_transform(test_predictions)
    test_actuals_original_scale = scaler.inverse_transform(test_actuals)

    test_rmse = np.sqrt(mean_squared_error(test_actuals_original_scale, test_predictions_original_scale))
    print(f"\nFinal Test RMSE: {test_rmse:.4f}")

    # Calculate training RMSE
    train_predictions = []
    train_actuals = []
    with torch.no_grad():
        for batch_X_train, batch_y_train in train_loader:
            train_output = model(batch_X_train)
            train_predictions.extend(train_output.numpy())
            train_actuals.extend(batch_y_train.numpy())

    train_predictions = np.array(train_predictions)
    train_actuals = np.array(train_actuals)
    train_predictions_original_scale = scaler.inverse_transform(train_predictions)
    train_actuals_original_scale = scaler.inverse_transform(train_actuals)
    train_rmse = np.sqrt(mean_squared_error(train_actuals_original_scale, train_predictions_original_scale))
    print(f"Final Training RMSE: {train_rmse:.4f}")

    final_val_rmse = val_rmse_history[-1] if val_rmse_history else None
    print(f"Final Validation RMSE: {final_val_rmse:.4f}")


    if len(X) < 1:
        print("No sequence available for prediction after training.")
        return coordinates_scaled, np.array([]), scaler, train_rmse, final_val_rmse, test_rmse

    last_sequence = torch.tensor(X[-1:], dtype=torch.float32)
    forecast_scaled = []

    with torch.no_grad():
        for _ in range(forecast_horizon):
            pred = model(last_sequence)
            forecast_scaled.append(pred.numpy()[0])
            # For autoregressive forecasting, shift the sequence and add the new prediction
            new_input = torch.cat((last_sequence[:, 1:, :], pred.unsqueeze(1)), dim=1)
            last_sequence = new_input

    forecast_scaled = np.array(forecast_scaled)
    print(f"Generated {forecast_horizon} forecast points.")

    return coordinates_scaled, forecast_scaled, scaler, train_rmse, final_val_rmse, test_rmse

# --- Plotting Functions ---

def plot_single_result(original_coordinates, processed_coordinates_scaled, forecast_scaled, scaler, optimizer_name, train_rmse=None, val_rmse=None, test_rmse=None):
    """
    Plots the original, processed, and forecasted trajectories for a single optimizer run.
    Reference polygon plotting has been removed as per user request.

    Args:
        original_coordinates (numpy.ndarray): Raw x_snap, y_snap coordinates from the initial CSV.
        processed_coordinates_scaled (numpy.ndarray): Scaled x_snap, y_snap coordinates after preprocessing.
        forecast_scaled (numpy.ndarray): Scaled forecasted x_snap, y_snap coordinates.
        scaler (MinMaxScaler): The scaler used for inverse transformation.
        optimizer_name (str): Name of the optimizer used for this run.
        train_rmse (float, optional): Training RMSE value to display on the plot. Defaults to None.
        val_rmse (float, optional): Validation RMSE value to display on the plot. Defaults to None.
        test_rmse (float, optional): Test RMSE value to display on the plot. Defaults to None.
    """

    processed_coordinates = scaler.inverse_transform(processed_coordinates_scaled)
    forecast = scaler.inverse_transform(forecast_scaled)

    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # Plot Original Raw Path
    plt.plot(original_coordinates[:, 0], original_coordinates[:, 1], 'g:', alpha=0.6, label="Original Raw Path")

    # Plot Processed Path (which was fed to LSTM)
    plt.plot(processed_coordinates[:, 0], processed_coordinates[:, 1], 'b.-', label="Processed Path (LSTM Input)")

    # Mark the start and end points of the processed data
    if len(processed_coordinates) > 0:
        plt.plot(processed_coordinates[0, 0], processed_coordinates[0, 1], 'go', markersize=10, label="Processed Start Point")
        plt.plot(processed_coordinates[-1, 0], processed_coordinates[-1, 1], 'ro', markersize=10, label="Processed End Point")

    # Prepend the last actual processed coordinate to the forecast for seamless plotting
    if len(processed_coordinates) > 0:
        connected_forecast = np.concatenate((processed_coordinates[-1:], forecast), axis=0)
    else:
        connected_forecast = forecast

    plt.plot(connected_forecast[:, 0], connected_forecast[:, 1], 'c.-', label=f"LSTM Forecast ({optimizer_name})")

    # Final legend update to ensure all labels are shown correctly and uniquely
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
    ax.legend(unique_labels.values(), unique_labels.keys(), title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

    title_text = f'BLE Movement Prediction: {optimizer_name} Optimizer'
    if train_rmse is not None:
        title_text += f'\nTraining RMSE: {train_rmse:.4f}'
    if val_rmse is not None:
        title_text += f', Validation RMSE: {val_rmse:.4f}'
    if test_rmse is not None:
        title_text += f', Test RMSE: {test_rmse:.4f}'

    plt.title(title_text, fontsize=16)
    plt.xlabel('X-coordinate', fontsize=12)
    plt.ylabel('Y-coordinate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.axis('equal')

    plot_filename = f'ble_prediction_plot_{optimizer_name.lower().replace(" ", "_")}.png'
    plt.savefig(plot_filename)
    print(f"\nPlot saved as '{plot_filename}'")
    plt.show()

def plot_combined_results(original_coordinates, processed_coordinates_scaled, scaler, all_optimizer_results):
    """
    Plots the original, processed, and all forecasted trajectories from different optimizers
    on a single combined plot, and includes grouped bar charts for RMSE comparisons.
    Reference polygon plotting has been removed as per user request.

    Args:
        original_coordinates (numpy.ndarray): Raw x_snap, y_snap coordinates from the initial CSV.
        processed_coordinates_scaled (numpy.ndarray): Scaled x_snap, y_snap coordinates after preprocessing.
        scaler (MinMaxScaler): The scaler used for inverse transformation.
        all_optimizer_results (list of dict): List containing {'optimizer_name', 'forecast_scaled', 'train_rmse', 'val_rmse', 'test_rmse'}
                                              for each optimizer run.
    """
    processed_coordinates = scaler.inverse_transform(processed_coordinates_scaled)

    # Three subplots: Trajectory, Train vs Val RMSE, Val vs Test RMSE
    fig, axes = plt.subplots(3, 1, figsize=(14, 20), gridspec_kw={'height_ratios': [3, 1, 1]})

    ax_trajectory = axes[0]

    # Removed: Plot the Reference Polygon

    ax_trajectory.plot(original_coordinates[:, 0], original_coordinates[:, 1], 'g:', alpha=0.6, label="Original Raw Path")
    ax_trajectory.plot(processed_coordinates[:, 0], processed_coordinates[:, 1], 'b.-', label="Processed Path (LSTM Input)")

    if len(processed_coordinates) > 0:
        ax_trajectory.plot(processed_coordinates[0, 0], processed_coordinates[0, 1], 'go', markersize=10, label="Processed Start Point")
        ax_trajectory.plot(processed_coordinates[-1, 0], processed_coordinates[-1, 1], 'ro', markersize=10, label="Processed End Point")

    colors = ['c', 'm', 'y', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    color_index = 0

    # Prepare data for RMSE bar charts
    optimizer_names = []
    train_rmse_values = []
    val_rmse_values = []
    test_rmse_values = []

    for result in all_optimizer_results:
        optimizer_name = result['optimizer_name']
        forecast_scaled = result['forecast_scaled']
        train_rmse = result['train_rmse']
        val_rmse = result['val_rmse']
        test_rmse = result['test_rmse']

        forecast = scaler.inverse_transform(forecast_scaled)

        if len(processed_coordinates) > 0:
            connected_forecast = np.concatenate((processed_coordinates[-1:], forecast), axis=0)
        else:
            connected_forecast = forecast

        ax_trajectory.plot(connected_forecast[:, 0], connected_forecast[:, 1], marker='.', linestyle='-',
                           color=colors[color_index % len(colors)], label=f"Forecast ({optimizer_name})")

        optimizer_names.append(optimizer_name)
        train_rmse_values.append(train_rmse)
        val_rmse_values.append(val_rmse)
        test_rmse_values.append(test_rmse)
        color_index += 1

    handles, labels = ax_trajectory.get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
    ax_trajectory.legend(unique_labels.values(), unique_labels.keys(), title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

    ax_trajectory.set_title('BLE Movement Prediction: Combined Forecasts by Optimizer', fontsize=16)
    ax_trajectory.set_xlabel('X-coordinate', fontsize=12)
    ax_trajectory.set_ylabel('Y-coordinate', fontsize=12)
    ax_trajectory.grid(True, linestyle='--', alpha=0.6)
    ax_trajectory.set_aspect('equal', adjustable='box')


    # --- Grouped Bar Chart for Training vs Validation RMSE ---
    ax_train_val_rmse = axes[1]

    x = np.arange(len(optimizer_names))
    width = 0.35

    bar_train = ax_train_val_rmse.bar(x - width/2, train_rmse_values, width, label='Training RMSE', color='lightgreen')
    bar_val = ax_train_val_rmse.bar(x + width/2, val_rmse_values, width, label='Validation RMSE', color='skyblue')

    ax_train_val_rmse.set_title('Optimizer Training vs. Validation RMSE Comparison', fontsize=14)
    ax_train_val_rmse.set_ylabel('RMSE', fontsize=12)
    ax_train_val_rmse.set_xlabel('Optimizer', fontsize=12)
    ax_train_val_rmse.set_xticks(x)
    ax_train_val_rmse.set_xticklabels(optimizer_names, rotation=45, ha='right')
    ax_train_val_rmse.legend()
    ax_train_val_rmse.grid(axis='y', linestyle='--', alpha=0.6)
    ax_train_val_rmse.set_ylim(bottom=0)

    for bars in [bar_train, bar_val]:
        for bar in bars:
            yval = bar.get_height()
            ax_train_val_rmse.text(bar.get_x() + bar.get_width()/2.0, yval + (ax_train_val_rmse.get_ylim()[1] * 0.02), f"{yval:.4f}", ha='center', va='bottom', fontsize=9)


    # --- Grouped Bar Chart for Validation vs Test RMSE ---
    ax_val_test_rmse = axes[2]

    bar_val_2 = ax_val_test_rmse.bar(x - width/2, val_rmse_values, width, label='Validation RMSE', color='skyblue')
    bar_test = ax_val_test_rmse.bar(x + width/2, test_rmse_values, width, label='Test RMSE', color='lightcoral')

    ax_val_test_rmse.set_title('Optimizer Validation vs. Test RMSE Comparison', fontsize=14)
    ax_val_test_rmse.set_ylabel('RMSE', fontsize=12)
    ax_val_test_rmse.set_xlabel('Optimizer', fontsize=12)
    ax_val_test_rmse.set_xticks(x)
    ax_val_test_rmse.set_xticklabels(optimizer_names, rotation=45, ha='right')
    ax_val_test_rmse.legend()
    ax_val_test_rmse.grid(axis='y', linestyle='--', alpha=0.6)
    ax_val_test_rmse.set_ylim(bottom=0)

    for bars in [bar_val_2, bar_test]:
        for bar in bars:
            yval = bar.get_height()
            ax_val_test_rmse.text(bar.get_x() + bar.get_width()/2.0, yval + (ax_val_test_rmse.get_ylim()[1] * 0.02), f"{yval:.4f}", ha='center', va='bottom', fontsize=9)


    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plot_filename = 'ble_prediction_plot_combined_all_rmse_comparisons_no_ref_points.png'
    plt.savefig(plot_filename)
    print(f"\nCombined plot with all RMSE comparisons saved as '{plot_filename}'")
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    csv_file_path = "data_processing.csv"

    print("--- Starting Data Preprocessing (without reference points) ---")
    # You can change 'linear' to 'ffill_bfill' or 'mean' here to test different NaN filling methods
    processed_df, original_coords = preprocess_ble_data(csv_file_path, nan_fill_method='mean')

    if processed_df is not None:
        processed_csv_file = 'after_data_processing_for_lstm_no_ref_points.csv'
        processed_df.to_csv(processed_csv_file, index=False)
        print(f"Processed data saved as '{processed_csv_file}'")

        print("\n--- Starting LSTM Training and Forecasting for Multiple Optimizers ---")
        look_back_window = 10
        forecast_steps = 10

        # Define the split ratios for train, validation, and test
        train_data_ratio = 0.7
        val_data_ratio = 0.15
        test_data_ratio = 0.15 # Make sure these sum to 1.0

        optimizer_configs = [
            {'name': 'Adam', 'params': {'lr': 0.001}},
            {'name': 'SGD', 'params': {'lr': 0.01, 'momentum': 0.9}},
            {'name': 'RMSprop', 'params': {'lr': 0.001, 'alpha': 0.99}},
            {'name': 'AdamW', 'params': {'lr': 0.001, 'weight_decay': 0.01}}
        ]

        all_optimizer_results = []

        for config in optimizer_configs:
            print(f"\n--- Running with Optimizer: {config['name']} ---")
            processed_scaled_for_lstm, forecast_scaled, data_scaler, train_rmse, val_rmse, test_rmse = train_and_forecast_lstm(
                processed_df,
                look_back=look_back_window,
                forecast_horizon=forecast_steps,
                optimizer_config=config,
                train_ratio=train_data_ratio,
                val_ratio=val_data_ratio,
                test_ratio=test_data_ratio,
                epochs=50
            )

            if processed_scaled_for_lstm is not None:
                all_optimizer_results.append({
                    'optimizer_name': config['name'],
                    'forecast_scaled': forecast_scaled,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'test_rmse': test_rmse
                })
                # Note: polygon_data is no longer passed to plot_single_result
                plot_single_result(
                    original_coords,
                    processed_scaled_for_lstm,
                    forecast_scaled,
                    data_scaler,
                    optimizer_name=config['name'],
                    train_rmse=train_rmse,
                    val_rmse=val_rmse,
                    test_rmse=test_rmse
                )
            else:
                print(f"Skipping plot for {config['name']} due to training/forecasting failure.")

        if len(all_optimizer_results) > 1:
            print("\n--- Generating Combined Plot for All Optimizers ---")
            if processed_scaled_for_lstm is not None and data_scaler is not None:
                # Note: polygon_data is no longer passed to plot_combined_results
                plot_combined_results(
                    original_coords,
                    processed_scaled_for_lstm,
                    data_scaler,
                    all_optimizer_results
                )
            else:
                print("Could not generate combined plot: No successful optimizer runs to provide necessary data.")
        elif len(all_optimizer_results) == 1:
            print("\nOnly one optimizer run completed successfully. No combined plot generated (individual plot already created).")
        else:
            print("\nNo optimizer runs completed successfully, no combined plot generated.")

    else:
        print("Data preprocessing failed, cannot proceed with LSTM.")