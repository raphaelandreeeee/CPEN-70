import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch 


with open("edited/Water_Parameters_2013-2025.xlsx", "rb") as f5, \
     open("edited/Climatological_Parameters_2013-2025.xlsx", "rb") as f6, \
     open("edited/Volcanic_Parameters_2013-2024.xlsx", "rb") as f4:
    water_parameters = pd.read_excel(f5)
    clim_paramaters = pd.read_excel(f6)
    volcanic_parameters = pd.read_excel(f4)
    # vol_param1 = pd.read_excel(f2)
    # vol_param2 = pd.read_excel(f3, index_col=0)

with open("raw/Water-Parameters_2013-2025-CpE-copy.xlsx", "rb") as f1, \
     open("raw/TV_CO2_Flux_2013-2019.xlsx", "rb") as f2, \
     open("raw/TV_SO2_Flux_2020-2024.xlsx", "rb") as f3:
    water_parameters_raw = pd.read_excel(f1)
    vol_param1 = pd.read_excel(f2)
    vol_param2 = pd.read_excel(f3)

clim_paramaters_raw = pd.read_csv("raw/Ambulong Monthly Data.csv")

#%% Editing Volcanic Parameters 2

"""
TV_SO2_Flux_2020-2024 has its "Date" column type as string. We need to change it so it corresponds with TV_CO2_Flux_2013-2019 "Date" column type, pandas datetime.
"""
# vol_param2["Date"] = pd.to_datetime(vol_param2["Date"], dayfirst=True)

# vol_param2 = vol_param2.sort_values("Date").reset_index(drop=True)

# Save the changes to another .xlsx file
# with pd.ExcelWriter("edited/SO2_Flux_2020-2024.xlsx") as writer:
#     vol_param2.to_excel(writer)

# Concatenate the two volcanic parameters.
# volcanic_parameters = pd.concat([vol_param1, vol_param2],  keys="Date")

# Save the changes to another .xlsx file
# with pd.ExcelWriter("edited/Volcanic_Parameters_2013-2024.xlsx") as writer:
#     volcanic_parameters.to_excel(writer)

# volcanic_parameters["Date"] = pd.to_datetime(volcanic_parameters["Date"])
# volcanic_parameters = volcanic_parameters.set_index("Date")

# volcanic_parameters = volcanic_parameters.set_index("Date")

# volcanic_parameters = volcanic_parameters.resample("MS").first()

# volcanic_parameters = volcanic_parameters.resample("M").mean()
# volcanic_parameters = volcanic_parameters.reset_index(names="Date")
# volcanic_parameters["Date"] = pd.to_datetime(volcanic_parameters["Date"]).apply(lambda x: x.replace(day=1))
# volcanic_parameters = volcanic_parameters.set_index("Date")

# with pd.ExcelFile("edited/Volcanic_Parameters_2013-2024.xlsx") as writer:
#     volcanic_parameters.to_excel(writer)


#%% Addressing Data Inconsistency of Water Parameters

# water_paramaters = water_paramaters.drop(columns="Location")
# water_paramaters.iloc[430, 6] = 0.09

# water_paramaters["Nitrate-N/Nitrite-N  (mg/L)"] = pd.to_numeric(water_paramaters["Nitrate-N/Nitrite-N  (mg/L)"])

# water_paramaters = water_paramaters.set_index("Date")
# water_paramaters = water_paramaters.resample("M").mean()
# water_paramaters = water_paramaters.reset_index(names="Date")
# water_paramaters["Date"] = pd.to_datetime(water_paramaters["Date"]).apply(lambda x: x.replace(day=1))
# water_paramaters = water_paramaters.set_index("Date")

# with pd.ExcelWriter("edited/Water_Parameters_2013-2025.xlsx") as writer:
#     water_paramaters.to_excel(writer)

#%% Editing Climatological Data

# clim_paramaters["Date"] = pd.to_datetime(clim_paramaters[["YEAR", "MONTH"]].assign(day=1))
# clim_paramaters = clim_paramaters.drop(columns=["YEAR", "MONTH"])
# clim_paramaters = clim_paramaters.set_index("Date")

# with pd.ExcelWriter("edited/Climatological_Parameters_2013-2025.xlsx") as writer:
#     clim_paramaters.to_excel(writer)

#%% Data Understanding
# Number of Samples
print(f"{len(water_parameters_raw)}")
print(f"{len(clim_paramaters_raw)}")
print(f"{len(vol_param1)}")
print(f"{len(vol_param2)}")

# Raw Information
print(f"{water_parameters_raw.info()}")
print(f"{clim_paramaters_raw.info()}")
print(f"{vol_param1.info()}")
print(f"{vol_param2.info()}")


#%% Data Preparation
# Null Values
print(f"{water_parameters_raw.isnull().sum()}")
print(f"{clim_paramaters_raw.isnull().sum()}")
print(f"{vol_param1.isnull().sum()}")
print(f"{vol_param2.isnull().sum()}")

#%% DATA PROCESSING
# Drop rows with too many missing values
data = water_parameters.dropna(thresh=len(water_parameters.columns) - 3)

# Drop unnecessary columns
ex_data1 = clim_paramaters.drop(columns=["RH", "WIND_SPEED", "WIND_DIRECTION"])
ex_data1["T_AVE"] = (ex_data1["TMIN"] + ex_data1["TMAX"]) / 2
ex_data1 = ex_data1.drop(columns=["TMIN", "TMAX"])

# Augment external factors to data
data["Rainfall"] = ex_data1["RAINFALL"]
data["Env_Temperature"] = ex_data1["T_AVE"]
data["CO2"] = volcanic_parameters["CO2 Flux (t/d)"]
data["SO2"] = volcanic_parameters["SO2 Flux (t/d)"]

# Fill missing values with column means
for col in data.columns[1:]:
    data[col] = data[col].fillna(data[col].mean())

# Calculate Water Quality Index (WQI) - using standard formula
# We'll use a simplified version of the WQI calculation
def calculate_wqi(row):
    # Weight factors for different parameters (can be adjusted based on importance)
    weights = {
        'Surface Water Temp (°C)': 0.03,
        'Middle Water Temp (°C)': 0.03,
        'Bottom Water Temp (°C)': 0.03,
        'pH Level': 0.2,
        'Ammonia (mg/L)': 0.2,
        'Nitrate-N/Nitrite-N  (mg/L)': 0.15,
        'Phosphate (mg/L)': 0.15,
        'Dissolved Oxygen (mg/L)': 0.2
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
            'Phosphate (mg/L)', 'Dissolved Oxygen (mg/L)', 'Rainfall', 'Env_Temperature', 'CO2', 'SO2']
target = 'WQI'

print(f"{len(data)}")
print(f"{data.info()}")
print(f"{data.isnull().sum()}")