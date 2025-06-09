import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

with open("edited/Water_Parameters_2013-2025.xlsx", "rb") as f5, \
     open("edited/Climatological_Parameters_2013-2025.xlsx", "rb") as f6, \
     open("edited/Volcanic_Parameters_2013-2024.xlsx", "rb") as f4:
     # open("raw/Water-Parameters_2013-2025-CpE-copy.xlsx", "rb") as f1:
     # open("raw/TV_CO2_Flux_2013-2019.xlsx", "rb") as f2, \
     # open("edited/SO2_Flux_2020-2024.xlsx", "rb") as f3:
    water_paramaters = pd.read_excel(f5)
    clim_paramaters = pd.read_excel(f6)
    volcanic_parameters = pd.read_excel(f4, index_col=[0, 1])
    # vol_param1 = pd.read_excel(f2)
    # vol_param2 = pd.read_excel(f3, index_col=0)

# clim_paramaters = pd.read_csv("raw/Ambulong Monthly Data.csv")

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

#%% Data contents
print(water_paramaters)
print(clim_paramaters)
print(volcanic_parameters)