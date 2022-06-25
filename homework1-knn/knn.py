# Homework 1 - K nearest neighbors classifier
# using the SciKit learn KNN classifer, see how
# it performs in terms of accuracy against
# other ML methods on a DDoS dataset

# Grab the number of clusters we want to look for
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=3 )
parser.add_argument('--model', default=None)
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

# attempt to load model if it exists
# else train the an empty model
import joblib
if args.model is not None:
    model = joblib.load(args.model)
else:
    import datetime
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=args.k)
    model.fit(X_train, Y_train)
    # save for future use
    model_file = '{:%Y-%m-%d_%H-%M_%S}_knn-k{}.pkl'.format(datetime.datetime.now(), args.k)
    joblib.dump(model, model_file)
    print('Pickled Model', model_file)

# Let's do a pass using the test data
from sklearn import metrics
y = model.predict(X_test)
print("Accuracy of KNN Model:", metrics.accuracy_score(Y_test, y))
confusion_matrix = metrics.confusion_matrix(Y_test, y)
print( confusion_matrix )
# extract out TN, FP, FN, TP from confusion_matrix
flattened_confusion_matrix = confusion_matrix.ravel()
print("TN:{}, FP:{}, FN:{}, TP:{}".format(*flattened_confusion_matrix))
total = sum(flattened_confusion_matrix)
percents = (flattened_confusion_matrix/total) * 100
print("TN:{:.4}%, FP:{:.4}%, FN:{:.4}%, TP:{:.4}%".format(*percents))
