import xarray as xr
import numpy as np
import pandas as pd

data = xr.open_dataset('Training_Data.nc')

v10, u10, sp, ptype, t2m, d2m = data['v10'], data['u10'], data['sp'], data['ptype'], data['t2m'], data['d2m']

# Calculate Absolute Wind Velocity at 2 meter
wv10m = np.sqrt(v10**2 + u10**2)

# Calculate Absolute Wind Velocity at 2 meter (By The Law of The Wall Equation)
k = 0.41 # von Karman constant
C = 0.075 # Constanst Value for The law of the Wall Equation
z0 = 5 # The value of roughness length
z = 2 # The hieigt interested
wv2m = np.abs(1/k * np.log(z/z0) * wv10m)
# Calculate Relative Humidity
e = 6.112 * np.exp((17.67*(d2m-273.15))/((d2m-273.15) + 243.5))
es = 6.112 * np.exp((17.67*(t2m-273.15))/((t2m-273.15) + 243.5))
rh = 100*(e/es)

# Create dataframes for each variable
df_v10m = v10.to_dataframe(name='v10m')
df_u10m = u10.to_dataframe(name='u10m')
df_sp = sp.to_dataframe(name='sp')
df_ptype = ptype.to_dataframe(name='ptype')
df_wv10m = wv10m.to_dataframe(name='wv10m')
df_t2m = t2m.to_dataframe(name='t2m')
df_d2m = d2m.to_dataframe(name='d2m')
df_rh = rh.to_dataframe(name='rh')
df_wv2m = wv2m.to_dataframe(name='wv2m')

# Create a dataframe for i
i = pd.DataFrame({'i': range(len(wv10m))}, index=wv10m.time)

df = pd.concat([df_v10m, df_u10m, df_wv2m, df_wv10m, df_sp, df_t2m, df_rh, df_ptype], axis=1).reset_index()
df.to_csv('data.csv', index=False)

df_set = pd.concat([df_sp, df_wv2m, df_t2m, df_rh, df_ptype], axis=1).reset_index()
df.to_csv('DataSet.csv', index=False)
