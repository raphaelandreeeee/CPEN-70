import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with open("raw/Water-Parameters_2013-2025-CpE-copy.xlsx", "rb") as f1, \
     open("edited/Volcanic_Parameters_2013-2024.xlsx", "rb") as f4:
     # open("raw/TV_CO2_Flux_2013-2019.xlsx", "rb") as f2, \
     # open("edited/SO2_Flux_2020-2024.xlsx", "rb") as f3:
    water_paramaters = pd.read_excel(f1)
    volcanic_parameters = pd.read_excel(f4, index_col=[0, 1])
    # vol_param1 = pd.read_excel(f2)
    # vol_param2 = pd.read_excel(f3, index_col=0)

clim_paramaters = pd.read_csv("raw/Ambulong Monthly Data.csv")

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

#%% Addressing Data Inconsistency

# water_paramaters["Date"] = pd.to_datetime(water_paramaters["Date"])

# monthly_water_param = water_paramaters.set_index("Date")
# monthly_water_param = monthly_water_param.drop(columns="Location")
# monthly_water_param.iloc[430, 5] = 0.09

# print(monthly_water_param.info())

# monthly_water_param["Nitrate-N/Nitrite-N  (mg/L)"] = pd.to_numeric(monthly_water_param["Nitrate-N/Nitrite-N  (mg/L)"])
# print(monthly_water_param.dtypes)

# with pd.ExcelWriter("edited/Water_Parameters_2013-2025.xlsx") as writer:
#     monthly_water_param.to_excel(writer)

#%% Data contents
# print(f"Water Parameter Null Values: \n{water_paramaters.isna().sum()}\n")
# print(f"Climatological Parameter Null Values: \n{clim_paramaters.isna().sum()}\n")
# print(f"Volcanic Parameter Null Values: \n{volcanic_parameters.isna().sum()}")