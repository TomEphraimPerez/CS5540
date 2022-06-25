# Homework 1 - Random Forest Code
# using the SciKit learn random classifer, see how
# it performs in terms of accuracy against
# other ML methods on a DDoS dataset

# Grab the number of clusters we want to look for
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input_data_path' )
parser.add_argument('--model', default=None)
args = parser.parse_args()

# grab the data from our features from 
# our features file + split them up!
import pandas
df = pandas.read_csv(args.input_data_path)
X = df.loc[:, df.columns[:-1]]           # TODO: fix later, output col has to be last
Y = df.loc[:, df.columns[-1]].to_frame() # TODO: fix later, output col has to be last

from sklearn.model_selection import train_test_split
# Regular 30/70 split (30% for test, 70% for training)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

# attempt to load model if it exists
# else train the an empty model
import joblib
if args.model is not None:
    model = joblib.load(args.model)
else:
    import datetime
# finally we'll do the training of the classifer
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    # save for future use
    joblib.save(model, '{:%Y-%m-%d_%H-%M_%S}_model-k{}.pkl'.format(datetime.datetime.now(), args.k))

# Let's do a pass using the test data
from sklearn import metrics
y = model.predict(X_test)
print("Accuracy of Random Forest Classifer:", metrics.accuracy_score(Y_test, y))