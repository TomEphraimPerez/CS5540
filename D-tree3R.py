
# Decision tree on #5 in list		kaggle.com/preeti5607/ddos-A-prevention 
#  To compile (interpret);  $ python3 D-tree1.py
# In VS, ^cmd P -> Python tools, etc, eg to choose python interpreters
# $ bash                    # Nav to bash is best.
# pwd                       # /Users/thomasperez/5540Smr22Team/GroupProject1/DS1/CS5540
# (($ conda install graphviz # 3hr installation. conda --version  -> conda 4.12.0 in bash))



print("\n\tMERGED rendition of a decision tree for \"ddos-attacl-prevention\'.")
print('\n')

import pandas
from sklearn.tree import plot_tree
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

df = pandas.read_csv("homework1-featurizer/features.csv")       # Rebel's featureset

print(df)

X = df.loc[:, df.columns[:-1]]           # TODO: fix later, output col has to be last
y = df.loc[:, df.columns[-1]].to_frame() # TODO: fix later, output col has to be last

print('\n\t\t X ->')
print(X)
print('\n\t\tY ->')
print(y)
print('\n\n')

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)         
plt.figure(figsize=(60,30))
print('\n\t\tpy plt.figure as opossed to using graphviz ')
plot_tree(dtree, filled=True);
print('\n\t\tplot_tree(...) as opposed to using graphviz -> should have a plot below.')

plt.show()                         # Rendering takes ~75 min when merging csv's.

print('\n\n')



'''
        Future 'hash #' includes:         >>>
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
                                       <<<
'''


