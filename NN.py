"""
File:       NN.py
Purpose:    Neural network based on Soriot et al. (https://doi.org/10.1029/2022EA002542)
            Training on a dataset of TBs on different microwave channels and corresponding SIT value of that time period
            Different ways to evaluate the model on the test data

Function:   train_model

Other:      Created by Thea Jonsson 2025-08-19
"""

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



# Define the MLP: input -> tanh hidden -> linear output
class Model(nn.Module):
    def __init__(self, in_features=4, n_hidden=50, n_outputs=1):
        super(Model, self).__init__()
        self.hidden = nn.Linear(in_features, n_hidden)  
        self.activation = nn.Tanh()                     
        self.output = nn.Linear(n_hidden, n_outputs)    

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)          
        return x



def train_model(data, seed=100, 
                test_size=0.3, random_state=42, # x% is the test size
                lr=0.01,        # learning rate 
                epochs=200):    # number of runs through all the training data

    torch.manual_seed(seed)       # Random manual seed for randomization

    X = data.drop(["SIT", "X_SIT", "Y_SIT"], axis=1).values        # Input to model: TBs for different channels
    y = data[["SIT", "X_SIT", "Y_SIT"]].values                     # What model should predict: SIT

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train test splits
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train[:,0]).unsqueeze(1)
    y_test_xy = y_test[:,1:]
    y_test = torch.FloatTensor(y_test[:,0]).unsqueeze(1)

    # Criterion of model to measure the error
    model = Model()              # Instance for model
    criterion = nn.MSELoss()     # Mean Squared Error Loss - cont. numerical value, calc. squared diff. between predicted and target values
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)   

    # Train the model
    losses = []
    for epoch in range(epochs):

        y_pred = model.forward(X_train)   # Prediction results

        loss = criterion(y_pred, y_train)
        losses.append(loss.detach().numpy())

        if epoch % 50 == 0:
            print(f"Epoch {epoch} and loss: {loss}")
        
        # Back propogation (to fine tune the weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.plot(range(epochs), losses)
    plt.ylabel("loss/error")
    plt.xlabel("Epoch")
    plt.show()

    print("Träningen är klar")




    # Predict on test data with scatter plot
    if True:
        with torch.no_grad():
            y_pred_test = model(X_test)

        y_test_np = y_test.numpy().flatten()        # y_test 30%:an av SIT
        y_pred_np = y_pred_test.numpy().flatten()   # y_pred 30%:an av X_test

        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(y_test_np, y_pred_np)
        r_squared = r_value**2

        # RMSE calculation
        rmse = mean_squared_error(y_test_np, y_pred_np, squared=False)
        
        plt.scatter(y_test_np, y_pred_np, alpha=0.6, label="Data")
        plt.plot(y_test_np, intercept + slope * y_test_np, color='red', label=f"Fitted line (R-squared: {r_squared:.3f}, RMSE={rmse:.3f})")
        plt.xlabel("True SIT values")
        plt.ylabel("Predicted SIT values")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 8)
        plt.xlim(0, 8)
        #plt.savefig("/Users/theajonsson/Desktop/OneMonth.png", dpi=300, bbox_inches="tight")
        plt.show()
        


    # Evaluate model on test data set by plotting with cartoplot
    if False:
        # TrainingData_TB_xy: All TBs for one day, only filtered for FillValue and if one TB is NaN whole row is deleted
        # TrainingData_full_month_SSTrainingData_full_month_SSMIS_xy: All TBs for a whole month, all filters used
        TB = pd.read_csv("/Users/theajonsson/Desktop/TrainingData_TB_xy.csv")

        Test_TB = TB.drop(["x", "y"], axis=1).values
        TB_xy =  TB[["x", "y"]].values 

        Test_TB = scaler.fit_transform(Test_TB)
        Test_TB = torch.FloatTensor(Test_TB)

        with torch.no_grad():
            y_eval = model.forward(Test_TB)

        from cartoplot import multi_cartoplot
        multi_cartoplot([TB_xy[:,0]], [TB_xy[:,1]], [y_eval], cbar_label="Sea ice thickness [m]")





# -------------------- MAIN --------------------       

#data = pd.read_csv("/Users/theajonsson/Desktop/TrainingData_full_month_SSMIS_xy.csv")

data = pd.read_csv("/Users/theajonsson/Desktop/TrainingData_SSMIS_xy.csv")
train_model(data)
