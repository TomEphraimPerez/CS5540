# ASMT1
# SKLEARN LINEAR_MODEL 				            eg <> fr6 ppt Week 2-2

# pip3 install <lib>   EG;
  # pip3 install pandas     or
  # pip3 install -U scikit-learn    etc
# To run   
  # $ python3 ASMT1.py

#///
# stackoverflow.com/questions/61208808/how-to-use-pandas-dataframes-with-sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm                                         #†
from sklearn import metrics
from sklearn import linear_model
import matplotlib.pyplot as plt
#///

#///
# geeksforgeeks.org/python-linear-regression-using-sklearn/
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm                          #†
from sklearn.linear_model import LinearRegression
#///


df = pd.read_csv("/Users/thomasperez/Desktop/DDoSdata.csv")

X = df[['pkts', 'bytes']]
y = df['attack']

# Create a classification method. Also splitting the data into training and testing data
regr = linear_model.LinearRegression()

# Train the model portion
regr.fit(X, y)

# Predicts the response of a device from an IoT being attacked ('A'), or operating normally.  
# predictedA = regr.predict([[RST, 192.168.100.150]]) #o
predictedAttk = regr.predict([[5, 800]])

print(predictedAttk)

# test11


