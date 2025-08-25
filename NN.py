import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# BASIC NN MODEL
# --------------------------------------------------
class Model(nn.Module):
    # Input layer (4 features), Hidden layers (tot 2 st med x # of neuroner), Output layer (3 classes)
    def __init__(self, in_features=4, h1=8, h2=9, h3=4, out_features=3):
        super().__init__() # Instantiate nn.Module (skapa ett objekt som har de egenskaper som tillhör en viss klass)
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)
    
    def forward(self, x):
        x = F.tanh(self.fc1(x))     # Aktiveringsfunk. ReLU, tanh
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.out(x)
        
        return x

torch.manual_seed(50)       # Random manual seed for randomization
model = Model()             # Instance for model


# LOAD DATA AND TRAIN NN MODEL 
# --------------------------------------------------
url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
my_df = pd.read_csv(url)

my_df["species"] = my_df["species"].replace({
    "setosa": 0.0,
    "versicolor": 1.0,
    "virginica": 2.0
})

X = my_df.drop("species", axis=1)
y = my_df["species"]
X = X.values
y = y.values

# Train test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 30% är testset
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train) # Long tensors 64-bit int
y_test = torch.LongTensor(y_test)

# Criterion of model to measure the error (far off the prediction are from the data)
criterion = nn.CrossEntropyLoss()
# Optimizer (Adam), lr-learning rate (lower->takes longer to learn and train)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 100 # of runs through all the training data 
losses = []
for i in range(epochs):
    # Prediction result
    y_pred = model.forward(X_train)

    # Loss/error
    loss = criterion(y_pred, y_train)

    # Keep track of the losses
    losses.append(loss.detach().numpy())

    if i % 10 == 0:
        print(f"Epoch {i} and loss: {loss}")
    
    # Back propogation (to fine tune the weights)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel("Epoch")

#plt.show()


# EVALUATE TEST DATA ON NN
# --------------------------------------------------
# Validate model on test set
with torch.no_grad():   # Turn off back propogation
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        print(f"{i+1}.) {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}")  # Type of class the network think it is (highest number)

        # Correct or not
        if y_val.argmax().item() == y_test[i]:
            correct += 1
    print(f"Number of correct: {correct}")

    
# EVALUATE NEW DATA ON THE NETWORK
# --------------------------------------------------
new_iris = torch.tensor([5.9, 3.0, 5.1, 1.8])

with torch.no_grad():
    print(model(new_iris))


# SAVE AND LOAD NN MODEL
# --------------------------------------------------
torch.save(model.state_dict(), "Test_NN.pt")

new_model = Model()
new_model.load_state_dict(torch.load("Test_NN.pt"))
print(new_model.eval())