# ideally we figure out our features first
# so we can correctly size up the input layer
# For now, we'll make it an arguement for rapid 
# prototyping, along with some other hyperparams
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dim', type=int, required = True )
parser.add_argument('--layers', type=int, default = 5 )
parser.add_argument('--units', type=int, default = 5 )
parser.add_argument('--learning_rate', type=float, default = 0.1 )
parser.add_argument('--epochs', type=int, default = 5 )

args = parser.parse_args()

# Input data, load from a file! We'll load the
# wireshark capture, and grab the columns that
# make sense. TODO: Change to dataloader one day
from torch import tensor
# Some Junk Data to see what happens!
data = \
(
        tensor\
        ([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 1, 10]
        ]).float(),
        tensor([[1],[0]]).float() 
)
# TODO:
# This is were we'd read our featureized data
# from disk! (Including Labeled stuff)

# define up the model
# here, we'll do a densely connected NN
# with reLU activation functions
# followed up with a sigmoid at the output
# layer. As basic as it gets really
from torch import nn
stackup = []

# Input Layer
stackup.append( nn.Linear( args.input_dim, args.units ) )
stackup.append( nn.ReLU() )
# Hidden Layers
for _ in range( args.layers - 1 ):
    stackup.append( nn.Linear( args.units, args.units ) )
    stackup.append( nn.ReLU() )
# Apply the Output Layer
stackup.append( nn.Linear( args.units, 1 ) )
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
