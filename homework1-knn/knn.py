# Homework 1 - K nearest neighbors classifier
# using the SciKit learn KNN classifer, see how
# it performs in terms of accuracy against
# other ML methods on a DDoS dataset

# Grab the number of clusters we want to look for
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=3 )
parser.add_argument('input_data_path' )
args = parser.parse_args()

# grab the data from our features from 
# our features file + split them up!
import pandas
df = pandas.read_csv(args.input_data_path)
X = df.loc[:, df.columns[:-1]]           # TODO: fix later, output col has to be last
Y = df.loc[:, df.columns[-1]].to_frame() # TODO: fix later, output col has to be last

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

# finally we'll do the training of the classifer
from sklearn.neighbors import KNeighborsClassifier
KNN_Classifer = KNeighborsClassifier(n_neighbors=args.k)
KNN_Classifer.fit(X_train, Y_train)

# Let's do a pass using the test data
from sklearn import metrics
y = KNN_Classifer.predict(X_test)
print("Accuracy of KNN Model w/ k={}:".format(args.k), metrics.accuracy_score(Y_test, y))