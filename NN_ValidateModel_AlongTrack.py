import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
import format_data as fd 
from cartoplot import cartoplot
import netCDF4 as nc
import numpy as np
from ll_xy import lonlat_to_xy
from scipy.spatial import KDTree
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import format_data as fd

""" ========================================================================================== """
group_SSM_I = "scene_env"
group_SSMIS = ["scene_env1", "scene_env2"]
""" ========================================================================================== """

# Define the MLP: input -> tanh hidden -> linear output
class Model(nn.Module):
    def __init__(self, in_features=5, n_hidden=40, n_outputs=1):
        super(Model, self).__init__()
        self.hidden1 = nn.Linear(in_features, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.hidden3 = nn.Linear(n_hidden, n_hidden)
        self.hidden4 = nn.Linear(n_hidden, n_hidden)
        self.activation = nn.Tanh()                     
        self.output = nn.Linear(n_hidden, n_outputs)    

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.hidden3(x)
        x = self.activation(x)
        x = self.hidden4(x)
        x = self.activation(x)
        x = self.output(x)          
        return x



def synthetic_tracks(file_name, step=np.sqrt((370**2)/2)*100):
  
  start = 1*10**6
  x_array = np.arange(-start, start, step)
  y_array = np.arange(start, -start, -step)
  x_mesh, y_mesh = np.meshgrid(x_array, y_array)

  x = x_mesh.flatten()
  y = y_mesh.flatten()

  columns = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "X_SIT", "Y_SIT"]
  df_TB_SSMIS = pd.DataFrame(columns=columns) 
  index = 0
  df_TB_SSMIS["X_SIT"] = x
  df_TB_SSMIS["Y_SIT"] = y

  index = 0
  for i in range(len(group_SSMIS)):
      if i == 0:
          vh = [1, 0, 2]  # Channel number: scene_env1 -> [V19, H19, V22]
      else:
          vh = [1, 0]     # Channel number: scene_env2 -> [V37, H37]
      
      for j in range(len(vh)):
          x_TB, y_TB, TB, TB_freq, nearest_TB_coords = fd.format_SSMIS(x, y, file_name, group_SSMIS[i], vh[j], debug=False)

          df_TB_SSMIS[columns[index]] = TB_freq     
          index += 1

  #df_TB_SSMIS.to_csv("/Users/theajonsson/Desktop/TestData.csv", index=False)
  print(df_TB_SSMIS.shape)

  return df_TB_SSMIS




# Fill the pole hole with trained NN model using synthetic tracks
model = Model()
NN_model = torch.load("/Users/theajonsson/Desktop/SSMIS_1month.pth") # Ändra i Model()
model.load_state_dict(NN_model["model_state_dict"])
scaler = NN_model["scaler"]

folder_path_SIT = "/Volumes/Thea_SSD_1T/Master Thesis/Envisat_SatSwath/2010/11/"
folder_path_SSMIS = "/Volumes/Thea_SSD_1T/TB_SSMIS/2010/11/"

files_SIT = sorted([f for f in os.listdir(folder_path_SIT) if f[0].isalnum()])
files_SSMIS = sorted([f for f in os.listdir(folder_path_SSMIS) if f[0].isalnum()])

all_x_SIT = np.array([])
all_y_SIT = np.array([])
all_SIT = np.array([])
y_eval_all = pd.DataFrame()
index = 1
for file_SIT, file_SSMIS in zip(files_SIT, files_SSMIS):

    x_SIT, y_SIT, SIT = fd.format_SIT(folder_path_SIT+file_SIT)
    all_x_SIT = np.append(all_x_SIT, x_SIT)
    all_y_SIT = np.append(all_y_SIT, y_SIT)
    all_SIT = np.append(all_SIT, SIT)

    TestData = synthetic_tracks(folder_path_SSMIS+file_SSMIS, step=370*100)

    Test_TB = TestData[["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"]].values 
    TB_xy =  TestData[["X_SIT","Y_SIT"]].values 

    Test_TB = scaler.fit_transform(Test_TB)
    Test_TB = torch.FloatTensor(Test_TB)

    with torch.no_grad():
        y_eval = model.forward(Test_TB)

    y_eval_all[f"Day_{index}"] = y_eval.squeeze().numpy()
    index += 1

y_eval_mean = np.array(y_eval_all.mean(axis=1))   

#cartoplot([TB_xy[:,0]], [TB_xy[:,1]], [y_eval_mean], cbar_label="Sea ice thickness [m]", dot_size=0.1)

df = pd.DataFrame({
    "PredSIT": y_eval_mean,
    "X": TB_xy[:,0],
    "Y": TB_xy[:,1]
})

# Scatter plot of CS-2 SIT (x-axis) vs pred SIT (y-axis)
file_CS2 = "/Volumes/Thea_SSD_1T/Master Thesis/Cryosat_Monthly/2010/ESACCI-SEAICE-L3C-SITHICK-SIRAL_CRYOSAT2-NH25KMEASE2-201011-fv2.0.nc"
x_CS2, y_CS2, SIT_CS2 = fd.format_SIT(file_CS2) # SIT, lon, lat .flatten()
tree = KDTree(list(zip(x_CS2.flatten(),y_CS2.flatten())))   # Gridded CS2

X = df["X"].values
Y = df["Y"].values
SIT_pred = df["PredSIT"]
distances, indices = tree.query(list(zip(X.flatten(),Y.flatten()))) # Swath pred SIT (syntethic tracks)
SIT_matched = SIT_CS2[indices]

# Linjär regression
slope, intercept, r_value, p_value, std_err = linregress(SIT_matched, SIT_pred)
r_squared = r_value**2

# RMSE calculation
rmse = mean_squared_error(SIT_matched, SIT_pred, squared=False)

plt.scatter(SIT_matched, SIT_pred, s=10, color='blue', alpha=0.5, label=f"R-squared: {r_squared:.3f} \nRMSE={rmse:.3f}")
plt.plot(SIT_matched, intercept + slope * SIT_matched, color="red", label="Fitted line")
plt.plot([0,5],[0,5], color="black", linestyle="--", label="Optimal line")
plt.xlabel("CS-2 SIT [m]")
plt.ylabel("Mean predicted SIT [m]")
plt.legend()
plt.grid(True)
plt.ylim(0, 5)
plt.xlim(0, 5)
plt.savefig("/Users/theajonsson/Desktop/2010nov_37km.png", dpi=300, bbox_inches="tight")
plt.show()

#cartoplot([TB_xy[:,0]], [TB_xy[:,1]], [y_eval_mean], cbar_label="Sea ice thickness [m]", title="2010-11 (model is 1 day)", dot_size=0.1, save_name="Model1day_TestData1Month_37km")
