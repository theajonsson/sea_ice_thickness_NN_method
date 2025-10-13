import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import format_data as fd 
from cartoplot import cartoplot
from ll_xy import lonlat_to_xy
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
from scipy.spatial import KDTree
import netCDF4 as nc
from scipy.ndimage import distance_transform_edt

""" ========================================================================================== """
group_SSM_I = "scene_env"
group_SSMIS = ["scene_env1", "scene_env2"]
""" ========================================================================================== """

# Define the MLP: input -> tanh hidden -> linear output
class Model(nn.Module):
    def __init__(self, in_features=5, n_hidden=4, n_outputs=1):
        super(Model, self).__init__()
        self.hidden1 = nn.Linear(in_features, n_hidden)
        self.activation = nn.Tanh()                     
        self.output = nn.Linear(n_hidden, n_outputs)    

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.output(x)          
        return x



def synthetic_tracks():

    d = nc.Dataset("NSIDC0772_LatLon_EASE2_N3.125km_v1.1.nc", "r", format="NETCDF4")
    lats = np.array(d['latitude'])
    lons = np.array(d['longitude'])
    d.close()

    grid_dir = "EASE2_N3.125km.LOCImask_land50_coast0km.5760x5760.bin"
    land_data = np.fromfile(grid_dir, dtype=np.uint8).reshape(5760, 5760)

    land_binary = np.isin(land_data, [0, 101, 252]).astype(np.uint8)

    # Distance from land
    dist_pixels = distance_transform_edt(1 - land_binary)
    dist_km = dist_pixels * 3.125  # each pixel = 3.125 km

    # Keep only water pixels (255) that are >= min_distance_km away from land
    water_mask = (land_data == 255) & (dist_km >= 50)

    lats = lats[::10, ::10]
    lons = lons[::10, ::10]
    water_mask = water_mask[::10, ::10]

    lats_flat = lats.flatten()
    lons_flat = lons.flatten()
    water_flat = water_mask.flatten() 

    valid = (lats_flat >= 80) & water_flat
    lats_valid = lats_flat[valid]
    lons_valid = lons_flat[valid]

    x, y = lonlat_to_xy(lats_valid, lons_valid, "n")

    print(f"Synthetic track size: {x.shape}")

    return x, y




# Fill the pole hole with trained NN model using synthetic tracks
start_time = time.time()
model = Model()
NN_model = torch.load("/Users/theajonsson/Desktop/SSMIS_1wintermonth.pth") # Ändra i Model()
model.load_state_dict(NN_model["model_state_dict"])
scaler = NN_model["scaler"]

x,y = synthetic_tracks()


columns = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "X_SIT", "Y_SIT"]
df_TB_SSMIS = pd.DataFrame(columns=columns) 
index = 0
df_TB_SSMIS["X_SIT"] = x
df_TB_SSMIS["Y_SIT"] = y


lons_valid, lats_valid, land_mask_data = fd.land_mask()


folder_path_SSMIS = "/Volumes/Thea_SSD_1T/TB_SSMIS/2011/03/" #
files_SSMIS = sorted([f for f in os.listdir(folder_path_SSMIS) if f[0].isalnum()])
y_eval_all = pd.DataFrame()
day = 1
for file_SSMIS in files_SSMIS:
    index = 0
    for i in range(len(group_SSMIS)):
        if i == 0:
            vh = [1, 0, 2]  # Channel number: scene_env1 -> [V19, H19, V22]
        else:
            vh = [1, 0]     # Channel number: scene_env2 -> [V37, H37]
        
        for j in range(len(vh)):
            x_TB, y_TB, TB, TB_freq, nearest_TB_coords = fd.format_SSMIS(x, y, folder_path_SSMIS+file_SSMIS, group_SSMIS[i], vh[j], lons_valid, lats_valid, land_mask_data, debug=False)

            df_TB_SSMIS[columns[index]] = TB_freq     
            index += 1

    Test_TB = df_TB_SSMIS[["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"]].values 
    TB_xy =  df_TB_SSMIS[["X_SIT","Y_SIT"]].values 

    Test_TB = scaler.fit_transform(Test_TB)
    Test_TB = torch.FloatTensor(Test_TB)

    with torch.no_grad():
        y_eval = model.forward(Test_TB)

    y_eval_all[f"Day_{day}"] = y_eval.squeeze().numpy()
    print(f"Day_{day} done")
    day += 1

y_eval_mean = np.array(y_eval_all.mean(axis=1))


cartoplot([TB_xy[:,0]], [TB_xy[:,1]], [y_eval_mean], cbar_label="Sea ice thickness [m]", dot_size=0.1, save_name="Cart_2011mar") #

df = pd.DataFrame({
    "PredSIT": y_eval_mean,
    "X": TB_xy[:,0],
    "Y": TB_xy[:,1]
})
#df.to_csv("/Users/theajonsson/Desktop/Validate_PredSIT.csv", index=False)

# Scatter plot of CS-2 SIT (x-axis) vs pred SIT (y-axis)
file_CS2 = "/Volumes/Thea_SSD_1T/Master Thesis/Cryosat_Monthly/2011/ESACCI-SEAICE-L3C-SITHICK-SIRAL_CRYOSAT2-NH25KMEASE2-201103-fv2.0.nc" #
x_CS2, y_CS2, SIT_CS2 = fd.format_SIT(file_CS2) # SIT, lon, lat .flatten()
tree = KDTree(list(zip(x_CS2.flatten(),y_CS2.flatten())))   # Gridded CS2

X = df["X"].values
Y = df["Y"].values
SIT_pred = df["PredSIT"]
distances, indices = tree.query(list(zip(X.flatten(),Y.flatten()))) # Swath pred SIT (syntethic tracks)
SIT_matched = SIT_CS2[indices]

# Histogram
plt.figure()
plt.hist(SIT_matched, bins=100, color='blue', alpha=0.5 ,edgecolor='black', zorder=3, label="CS-2 SIT", range=(SIT_matched.min(), SIT_matched.max()))
plt.hist(SIT_pred, bins=100, color='red', edgecolor='black', zorder=1, label="Predicted SIT", range=(SIT_matched.min(), SIT_matched.max()))
plt.title("Overlay")
plt.xlabel("Sea Ice Thickness [m]")
plt.ylabel("Amount [n]")
plt.legend()
plt.savefig("/Users/theajonsson/Desktop/SIT_hist_2011mar.png", dpi=300, bbox_inches="tight") #
plt.show()

# Bias calculation
bias = np.mean(SIT_pred - SIT_matched)

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(SIT_matched, SIT_pred)
r_squared = r_value**2

# RMSE calculation
rmse = mean_squared_error(SIT_matched, SIT_pred, squared=False)

end_time = time.time()
print(f"Elapsed time: {end_time - start_time}")

plt.figure()
plt.scatter(SIT_matched, SIT_pred, s=10, color='blue', alpha=0.5, label=f"Bias: {bias} \nR-squared: {r_squared:.3f} \nRMSE={rmse:.3f}")
plt.plot(SIT_matched, intercept + slope * SIT_matched, color="red", label="Fitted line")
plt.plot([0,5],[0,5], color="black", linestyle="--", label="Optimal line")
plt.xlabel("CS-2 SIT [m]")
plt.ylabel("Mean predicted SIT [m]")
plt.legend()
plt.grid(True)
plt.ylim(0, 5)
plt.xlim(0, 5)
plt.savefig("/Users/theajonsson/Desktop/2011mar.png", dpi=300, bbox_inches="tight") #
plt.show()

