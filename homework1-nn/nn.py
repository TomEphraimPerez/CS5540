# Assignment 1 - Basic NN, arebel@calstatela.edu
#
# uses the number of fields read in via the features
# file as the input layer dimension
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_data_path', type=str )
parser.add_argument('--layers', type=int, default = 5 )
parser.add_argument('--units', type=int, default = 5 )
parser.add_argument('--learning_rate', type=float, default = 0.1 )
parser.add_argument('--epochs', type=int, default = 5 )

args = parser.parse_args()

import csv
import pandas
from torch import tensor

df = pandas.read_csv(args.input_data_path)
X = df.loc[:, df.columns[:-1]]           # TODO: fix later, output col has to be last
Y = df.loc[:, df.columns[-1]].to_frame() # TODO: fix later, output col has to be last
X = tensor( X.values ).to(float)
Y = tensor( Y.values ).to(float)

# actual inputs to our model
input_dim = X.shape[1]
data = (X, Y)

# define up the model
# here, we'll do a densely connected NN
# with reLU activation functions
# followed up with a sigmoid at the output
# layer. As basic as it gets really
from torch import nn
stackup = []

# Input Layer
stackup.append( nn.Linear( input_dim, args.units, dtype=float) )
stackup.append( nn.ReLU() )
# Hidden Layers
for _ in range( args.layers - 1 ):
    stackup.append( nn.Linear( args.units, args.units, dtype=float ) )
    stackup.append( nn.ReLU() )
# Apply the Output Layer
stackup.append( nn.Linear( args.units, 1, dtype=float ) )
stackup.append( nn.Sigmoid() )
model = nn.Sequential( *stackup )

print(model)

# We're not done yet, define out a loss function
# and an optimizer, we'll go vanilla for regression
# tasks - MSELoss (discussed in class) + SGD (with 0 batch)
from torch import optim
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = args.learning_rate )

# finally, let's train this bad boy
# for epoch, data in enumerate(data_loader): if we had a dataloader
for epoch in range(args.epochs):
    # Zero out gradients
    optimizer.zero_grad()

    # Load the data into X (input), and Y (label)
    X, Y = data

    # Inference Step (forward pass)
    y = model( X )

    # Compute loss + gradients
    loss = loss_fn( y, Y )
    loss.backward()

    # Apply parameter changes
    optimizer.step()

    # print out loss information
    print( epoch, loss.item() )

# Save the model + Parameters for later restore
from torch import save
import datetime
save(model, '{:%Y-%m-%d_%H-%M_%S}_model.pth'.format(datetime.datetime.now()))
