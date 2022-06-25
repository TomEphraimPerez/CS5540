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
parser.add_argument('--model', default = None )

args = parser.parse_args()

import csv
import pandas
from torch import tensor

df = pandas.read_csv(args.input_data_path)
X = df.loc[:, df.columns[:-1]]           # TODO: fix later, output col has to be last
Y = df.loc[:, df.columns[-1]].to_frame() # TODO: fix later, output col has to be last

# actual inputs to our model
input_dim = X.shape[1]

# split the data up using sklearn's splitting lib
from sklearn.model_selection import train_test_split
# Regular 30/70 split (30% for test, 70% for training)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

# generate the tensors using just the training set
X = tensor( X_train.values ).to(float)
Y = tensor( Y_train.values ).to(float)
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

# Now using the test-set, check how good the training was
# place model into inference only mode
from torch import inference_mode
X = tensor( X_test.values ).to(float)
Y = tensor( Y_test.values ).to(float)
with inference_mode():
    y = model( X )

# Predictor, if ( greater than median value ) == True == is_attack
import numpy
y = pandas.DataFrame( y.numpy() ) > numpy.median( y.numpy() )

# Perform the same Metrics Calculation as Other models
from sklearn import metrics
print("Accuracy of Neural Network Classifer:", metrics.accuracy_score(Y_test, y))
confusion_matrix = metrics.confusion_matrix(Y_test, y)
print( confusion_matrix )
# extract out TN, FP, FN, TP from confusion_matrix
flattened_confusion_matrix = confusion_matrix.ravel()
print("TN:{}, FP:{}, FN:{}, TP:{}".format(*flattened_confusion_matrix))
total = sum(flattened_confusion_matrix)
percents = (flattened_confusion_matrix/total) * 100
print("TN:{:.4}%, FP:{:.4}%, FN:{:.4}%, TP:{:.4}%".format(*percents))
