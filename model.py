import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data
with open("edited/Water_Parameters_2013-2025.xlsx", "rb") as file:
    data = pd.read_excel(file)

# Data Preprocessing
# Drop rows with too many missing values
data = data.dropna(thresh=len(data.columns) - 3)

# Fill missing values with column means
for col in data.columns[1:]:
    data[col] = data[col].fillna(data[col].mean())

# Calculate Water Quality Index (WQI) - using standard formula
# We'll use a simplified version of the WQI calculation
def calculate_wqi(row):
    # Weight factors for different parameters (can be adjusted based on importance)
    weights = {
        'Surface Water Temp (°C)': 0.1,
        'Middle Water Temp (°C)': 0.1,
        'Bottom Water Temp (°C)': 0.1,
        'pH Level': 0.2,
        'Ammonia (mg/L)': 0.15,
        'Nitrate-N/Nitrite-N  (mg/L)': 0.15,
        'Phosphate (mg/L)': 0.1,
        'Dissolved Oxygen (mg/L)': 0.1
    }
    
    # Normalize each parameter (0-100 scale, higher is better for most)
    # Note: Normalization depends on ideal ranges for each parameter
    normalized = {
        'Surface Water Temp (°C)': min(max(0, 100 - 5*abs(row['Surface Water Temp (°C)'] - 25)), 100),  # Ideal around 25°C
        'Middle Water Temp (°C)': min(max(0, 100 - 5*abs(row['Middle Water Temp (°C)'] - 25)), 100) if pd.notna(row['Middle Water Temp (°C)']) else 75,
        'Bottom Water Temp (°C)': min(max(0, 100 - 5*abs(row['Bottom Water Temp (°C)'] - 25)), 100) if pd.notna(row['Bottom Water Temp (°C)']) else 75,
        'pH Level': min(max(0, 100 - 20*abs(row['pH Level'] - 7.5)), 100),  # Ideal around 7.5
        'Ammonia (mg/L)': min(max(0, 100 - 500*row['Ammonia (mg/L)']), 100),  # Lower is better
        'Nitrate-N/Nitrite-N  (mg/L)': min(max(0, 100 - 500*row['Nitrate-N/Nitrite-N  (mg/L)']), 100),  # Lower is better
        'Phosphate (mg/L)': min(max(0, 100 - 1000*row['Phosphate (mg/L)']), 100) if pd.notna(row['Phosphate (mg/L)']) else 75,  # Lower is better
        'Dissolved Oxygen (mg/L)': min(max(0, 20*row['Dissolved Oxygen (mg/L)']), 100)  # Higher is better
    }
    
    # Calculate WQI
    wqi = 0
    total_weight = 0
    for param, weight in weights.items():
        if param in normalized:
            wqi += normalized[param] * weight
            total_weight += weight
    
    return wqi / total_weight

# Add WQI column to dataframe
data['WQI'] = data.apply(calculate_wqi, axis=1)

# Feature selection - use all available parameters
features = ['Surface Water Temp (°C)', 'Middle Water Temp (°C)', 'Bottom Water Temp (°C)', 
            'pH Level', 'Ammonia (mg/L)', 'Nitrate-N/Nitrite-N  (mg/L)', 
            'Phosphate (mg/L)', 'Dissolved Oxygen (mg/L)']
target = 'WQI'

# Prepare data for CNN (we'll treat the time series as 1D "images")
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Normalize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])
scaled_target = MinMaxScaler().fit_transform(data[[target]])

# Create sequences
seq_length = 6  # using 6 months of data to predict next month's WQI
X, y = create_sequences(scaled_features, seq_length)
X_target, y_target = create_sequences(scaled_target, seq_length)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension for CNN
X_test = torch.FloatTensor(X_test).unsqueeze(1)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# Create PyTorch Dataset
class WaterQualityDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

train_dataset = WaterQualityDataset(X_train, y_train)
test_dataset = WaterQualityDataset(X_test, y_test)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# CNN Model
class WQICNN(nn.Module):
    def __init__(self):
        super(WQICNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.2)
        
        # Calculate the size after convolutions and pooling
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # Initialize model, loss function, and optimizer
    model = WQICNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Test loss
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
        
        test_loss = test_loss / len(test_loader)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # Plot training history
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.show()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test).numpy()
        test_actual = y_test.numpy()

    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.plot(test_actual, label='Actual WQI')
    plt.plot(test_preds, label='Predicted WQI')
    plt.xlabel('Sample')
    plt.ylabel('WQI (Normalized)')
    plt.legend()
    plt.title('Actual vs Predicted WQI')
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'wqi_cnn_model.pth')