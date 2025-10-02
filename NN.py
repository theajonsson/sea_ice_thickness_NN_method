"""
File:       NN.py
Purpose:    Neural network architecture Multi Layered Perceptron based on Soriot et al. (https://doi.org/10.1029/2022EA002542)
            Training on a dataset of TBs on different microwave channels and corresponding SIT value of that time period
            Different ways to evaluate the model on the test data

Function:   train_model

Other:      Created by Thea Jonsson 2025-08-19
"""
import time
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



def train_model(data, seed=100, 
                test_size=0.3, random_state=42, # x% is the test size
                lr=0.01,        # learning rate 
                epochs=200):    # number of runs through all the training data

    torch.manual_seed(seed)       # Random manual seed for randomization

    X = data[["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"]].values        # Input to model: TBs for different channels "GR_V", "GR_H", "PR_19", "PR_37"
    y = data[["SIT", "X_SIT", "Y_SIT"]].values                     # Model predict: SIT

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

    # Plot of loss/epoch
    if False:
        plt.plot(range(epochs), losses)
        plt.ylabel("loss/error")
        plt.xlabel("Epoch")
        plt.savefig("/Users/theajonsson/Desktop/Loss.png", dpi=300, bbox_inches="tight")
        plt.show()

    print("Training is done")



    # Save model
    if True:
        model_save = {
            "model_state_dict": model.state_dict(),
            "scaler": scaler
        }
        torch.save(model_save,"/Users/theajonsson/Desktop/SSMIS_1month.pth")

    

    """ ==========================================================================================
            3 different type of plots to check for different things to consider
    ========================================================================================== """
    # Plot with 4 subfigures: TB for each channel (y-axis) against predicted SIT (x-axis), with fitted line, R^2 score, RMSE value
    if False:
        fig, axes = plt.subplots(2,3, figsize=[10,5])
        axes = axes.flatten()
        channel = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"]

        with torch.no_grad():
            y_pred = (np.array(model.forward(X_test))).flatten() #Gives pred SIT value on Test TB data

        X_test_TB = scaler.inverse_transform(X_test)

        for i in range(len(X_test[1])):
            ax = axes[i]

            TB_channel= np.array(X_test_TB[:,i]).flatten()
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = linregress(y_pred,TB_channel)
            r_squared = r_value**2

            # RMSE calculation
            rmse = mean_squared_error(y_pred, TB_channel, squared=False)

            ax.hexbin(y_pred, TB_channel, gridsize = 50, mincnt=6)                  
            ax.plot(y_pred, intercept + slope * y_pred, color='red', label=f"Fitted line \nR-squared: {r_squared:.3f} \nRMSE={rmse:.3f})")

            ax.set_xlabel("Predicted test SIT [m]")
            ax.set_ylabel(f"{channel[i]} [K]")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.grid(True)
            ax.set_title(f"{channel[i]} vs predicted SIT")
        
        plt.tight_layout()
        #plt.savefig("/Users/theajonsson/Desktop/TBvspredSIT.png", dpi=300, bbox_inches="tight") 
        plt.show()

    if False:
        with torch.no_grad():
            y_pred = (np.array(model.forward(X_test))).flatten()

        X_test_TB = scaler.inverse_transform(X_test)

        # Lista på grupper av TB-kanaler
        channel_groups = [
            ["TB_V19", "TB_H19"],   # Figur 1
            ["TB_V22"],             # Figur 2
            ["TB_V37", "TB_H37"]    # Figur 3
        ]


        # Indexmappning för kanal -> kolumn i X_test_TB
        channel_order = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"]
        channel_indices = {ch: i for i, ch in enumerate(channel_order)}

        # Plot varje grupp
        for group in channel_groups:
            fig, axs = plt.subplots(1, len(group), figsize=(7 * len(group), 6), constrained_layout=True)

            if len(group) == 1:
                axs = [axs]  # Gör till lista

            for ax, channel in zip(axs, group):
                idx = channel_indices[channel]
                TB_channel = X_test_TB[:, idx]

                # Linjär regression & RMSE
                slope, intercept, r_value, p_value, std_err = linregress(y_pred, TB_channel)
                r_squared = r_value ** 2
                rmse = mean_squared_error(y_pred, TB_channel, squared=False)

                # Hexbin plot
                hb = ax.hexbin(y_pred, TB_channel, gridsize=50, mincnt=6)

                # Regressionslinje
                ax.plot(y_pred, intercept + slope * y_pred, color='red',
                        label=f"Fitted line\n$R^2$ = {r_squared:.3f}\nRMSE = {rmse:.3f}")

                ax.set_xlabel("Predicted test SIT [m]")
                ax.set_ylabel(f"{channel} [K]")
                ax.set_title(f"{channel} vs Predicted SIT")
                ax.legend()
                ax.grid(True)

            #plt.savefig("/Users/theajonsson/Desktop/TBvspredSIT_.png", dpi=300, bbox_inches="tight")
            plt.show()

    # Plot with 4 subfigures: Predicted SIT (y-axis) against true SIT (x-axis), with fitted line, R^2 score, RMSE value
    if False:
        fig, axes = plt.subplots(2,3, figsize=[10,5])
        axes = axes.flatten()
        channel = ["V19", "H19", "V22", "V37", "H37"]
        for i in range(len(X_test[1])):
            ax = axes[i]
            X_test_single = X_test.clone()           
            X_test_single[:, :] = X_test[:, i].unsqueeze(1).repeat(1, X_test.shape[1])
            X_test_single[:, i] = X_test[:, i]
            
            with torch.no_grad():
                y_pred_test = model.forward(X_test_single)

            y_test_np = y_test.flatten()       
            y_pred_np = y_pred_test.flatten() 

            # Linear regression
            slope, intercept, r_value, p_value, std_err = linregress(y_test_np, y_pred_np)
            r_squared = r_value**2

            # RMSE calculation
            rmse = mean_squared_error(y_test_np, y_pred_np, squared=False)

            #ax.scatter(y_test_np, y_pred_np, alpha=0.6, label="Data")      # Scatter plot
            ax.hexbin(y_test_np, y_pred_np, gridsize = 50, mincnt=6)         # Hexbin plot
            ax.plot(y_test_np, intercept + slope * y_test_np, color='red', label=f"Fitted line \nR-squared: {r_squared:.3f} \nRMSE={rmse:.3f})")
            
            ax.set_xlabel("True test SIT values [m]")
            ax.set_ylabel("Predicted test SIT values [m]")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.grid(True)
            ax.set_title(channel[i])
            ax.set_ylim(0, 5)
            ax.set_xlim(0, 5)
        
        plt.tight_layout()
        #plt.savefig("/Users/theajonsson/Desktop/SSMIS_tested_allcolsame.png", dpi=300, bbox_inches="tight")
        plt.show()

    # Plot (scatter or hexbin) of predicted SIT (y-axis) against true SIT (x-axis), with fitted and optimal line, R^2 score, RMSE value
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
        
        # Scatter plot
        if False:
            plt.scatter(y_test_np, y_pred_np, alpha=0.6, label="Data")
            plt.plot(y_test_np, intercept + slope * y_test_np, color='red', label=f"Fitted line (R-squared: {r_squared:.3f}, RMSE={rmse:.3f})")
            plt.plot([0,5],[0,5], color="m", linestyle="--", label="Optimal line")
            plt.xlabel("True SIT values [m]")
            plt.ylabel("Predicted SIT values [m]")
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 5)
            plt.xlim(0, 5)
            plt.show()
        
        # Hexbin plot
        if True:
            plt.hexbin(y_test_np, y_pred_np, gridsize = 50, mincnt=6)
            plt.plot(y_test_np, intercept + slope * y_test_np, color="red", label=f"Fitted line (R-squared: {r_squared:.3f}, RMSE={rmse:.3f})")
            plt.plot([0,5],[0,5], color="m", linestyle="--", label="Optimal line")
            plt.xlabel("True SIT [m]")
            plt.ylabel("Predicted SIT [m]")
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 5)
            plt.xlim(0, 5)
            plt.savefig("/Users/theajonsson/Desktop/SSMIS_1month.png", dpi=300, bbox_inches="tight")
            plt.show()

        if True: 
            plt.hist(y_test_np, bins=100, color='blue', alpha=0.5 ,edgecolor='black', zorder=3, label="True SIT", histtype='stepfilled')
            plt.hist(y_pred_np, bins=100, color='red', edgecolor='black', zorder=1, label="Predicted SIT", histtype='stepfilled')
            plt.title("Overlay")
            plt.xlabel("Sea Ice Thickness [m]")
            plt.ylabel("Amount [n]")
            plt.legend()
            plt.savefig("/Users/theajonsson/Desktop/SSMIS_1month_hist.png", dpi=300, bbox_inches="tight")
            plt.show()




# -------------------- MAIN --------------------       
# Train model on test data set
if True:
    start_time = time.time()
    try:
        data = pd.read_csv("/Users/theajonsson/Desktop/TD_SSMIS_1month_200603.csv")
    except FileNotFoundError:
        print("Error")
    train_model(data, epochs=1000, lr=0.01)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")






















""" ========================================================================================== """
# Test the trained model on a different data set
if False:
    from training_data_SSMIS import synthetic_tracks
    import os

    model = Model()
    NN_model = torch.load("/Users/theajonsson/Desktop/FillHole_test/1day/SSMIS_1day_model.pth")
    model.load_state_dict(NN_model["model_state_dict"])
    scaler = NN_model["scaler"]

    filename = "BTRin20060313000000424SSF1601GL.nc"
    TestData = synthetic_tracks(filename)
    Test_TB = TestData[["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"]].values 
    TB_xy =  TestData[["X_SIT","Y_SIT"]].values 

    Test_TB = scaler.fit_transform(Test_TB)
    Test_TB = torch.FloatTensor(Test_TB)

    with torch.no_grad():
        y_eval = model.forward(Test_TB)
   

    from format_data import format_SIT
    file = "/Volumes/Thea_SSD_1T/Master Thesis/Envisat_SatSwath/2006/03/ESACCI-SEAICE-L2P-SITHICK-RA2_ENVISAT-NH-20060331-fv2.0.nc"
    x_SIT, y_SIT, SIT = format_SIT(file)

    from cartoplot import cartoplot
    cartoplot([TB_xy[:,0], x_SIT], [TB_xy[:,1], y_SIT], [np.array(y_eval), SIT],cbar_label="Sea ice thickness [m]")




""" ==========================================================================================
            2 different type of plots to check for different things to consider
    ========================================================================================== """
# Plot of predicted SIT (y-axis) against true SIT (x-axis), with fitted and optimal line, R^2 score, RMSE value
if False:
    X_test = TestData.drop(["SIT", "X_SIT", "Y_SIT"], axis=1).values        # Input to model: TBs for different channels
    y_test = TestData[["SIT"]].values  

    X_test = scaler.fit_transform(X_test)
    X_test = torch.FloatTensor(X_test)

    with torch.no_grad():
            y_pred_test = model(X_test)

    y_test_np = y_test.flatten()        
    y_pred_np = y_pred_test.numpy().flatten()  

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(y_test_np, y_pred_np)
    r_squared = r_value**2
    
    # RMSE calculation
    rmse = mean_squared_error(y_test_np, y_pred_np, squared=False)
    
    plt.hexbin(y_test_np, y_pred_np, gridsize = 50, mincnt=6)
    plt.plot(y_test_np, intercept + slope * y_test_np, color="red", label=f"Fitted line (R-squared: {r_squared:.3f}, RMSE={rmse:.3f})")
    plt.plot([0,5],[0,5], color="m", linestyle="--", label="Optimal line")
    plt.xlabel("True SIT [m]")
    plt.ylabel("Predicted SIT [m]")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 5)
    plt.xlim(0, 5)
    #plt.savefig("/Users/theajonsson/Desktop/SSMIS_1day_hexbin.png", dpi=300, bbox_inches="tight")
    plt.show()

# Plot with 2 subfigures: Cartoplot of NN predicted SIT and true SIT with same x/y coordinates
# Plot of predicted SIT (y-axis) against true SIT (x-axis), with fitted and optimal line, R^2 score, RMSE value
if False:
    Test_TB = TestData.drop(["SIT","X_SIT","Y_SIT"], axis=1).values 
    TB_xy =  TestData[["X_SIT","Y_SIT"]].values 
    SIT = TestData[["SIT"]].values

    Test_TB = scaler.fit_transform(Test_TB)
    Test_TB = torch.FloatTensor(Test_TB)

    with torch.no_grad():
        y_eval = model.forward(Test_TB)

    from cartoplot import multi_cartoplot
    multi_cartoplot([TB_xy[:,0]], [TB_xy[:,1]], [y_eval, SIT], 
                    title=["NN predicted Value"," Real SIT values"],cbar_label="Sea ice thickness [m]")
    
    fig= plt.figure(figsize=[10,5])

    y_test_np = SIT.flatten()       
    y_pred_np = y_eval.flatten() 

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(y_test_np, y_pred_np)
    r_squared = r_value**2

    # RMSE calculation
    rmse = mean_squared_error(y_test_np, y_pred_np, squared=False)

    plt.hexbin(y_test_np, y_pred_np, gridsize = 50, mincnt=6)                  
    plt.plot(y_test_np, intercept + slope * y_test_np, color='red', label=f"Fitted line (R-squared: {r_squared:.3f}, RMSE={rmse:.3f})")
    plt.plot([0,5],[0,5], color="m", linestyle="--", label="Optimal line")
    plt.xlabel("True SIT [m]")
    plt.ylabel("Predicted SIT [m]")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 5)
    plt.xlim(0, 5)
    plt.show()
