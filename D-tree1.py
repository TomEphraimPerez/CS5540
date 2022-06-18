# Decision tree on #5 in list		kaggle.com/preeti5607/ddos-attack-prevention 
#  To compile (really interpret); $ python3 D-tree1.py

from tarfile import PAX_NAME_FIELDS


print("\n\t1st rendition/iteration/generation of a decision tree for \"ddos-attacl-prevention\'.")

import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg 


df = pandas.read_csv("~/desktop/Archive/dataset_attack.csv")
# print(df)   # Yields a partial DS (that's ok) then the statement: [2990062 rows x 29 columns]
print('\n')


# Using map() to convert strings to integers as required.
'''
Eg;
d = {'true':1, 'notTrue': 0}
df[someInitFeature] = df[someInitfeature].map(d)
'''


# Traffic Monitors/C-plane monitor = ['tcp.time_delta ** ', 'ip.src', 'ip.dst', 'tcp.dstport' XXXXXXXXXXXXXXXXXXXXXXXXXXXX] 
    # ...to choose from dataset_attack.csv
# DEFINITIONS ___________________________________________________________________________________________________
# ip.flags.df | df means Don't Fragment -> drop pkt and an get ICMP msg. The pkt can't be fragmented for transmission.
    # When unset, (0) it means that the pkt CAN be fragmented.
# ip.flags.mf | mf means that the pkt contains more fragments. 
    # When unset, (0) it means that no more fragments remain. 

# ** ask.wireshark.org/question/12313/what-is-the-difference-between-time-and-delta-time/
#           What's the difference between time and delta time?
# If you want to search for gaps of more than one 1 second within a TCP session, you can use the filter 
# tcp.time_delta > 1. 
# The field tcp.time_delta is calculated by calculating the difference between packets within the same tcp stream.

#      REGARDING THE DS's TCP.FLAGS. _ _   geeksforgeeks.org/tcp-flags/             
# tcp.flags.res -  ...        for all definitions, SEE "DT/NOTES.txt"
# tcp.flags.ns - 
# tcp.flags.cwr -
# tcp.flags.enc -
# tcp.flags.urg -
# tcp.flags.ack -
# tcp.flags.push -
# tcp.flags.reset -
# tcp.flags.syn -
# tcp.flags.fin -

# END definitions _______________________________________________________________________________________________



# Recall; C-plane Clustering (captures ntwk flows and records on whom-is-talking-to-whom) determined by:
# protocol (TCP or UDP)
# src IP
# dest IP
# port
#       [Consider the] fcapture tool. (Lecture Wed 6-17)
#       Each flow record contains: (as per lecture)
#           time
#           duration
#           src ip
#           dest ip
#           dest port
#           num pkts
#           pkts / byte transfered in both directons
X = df[features]
y = ['ip.flags.df']
print(X)
print(y)
print('\n')











'''
Future work >>>
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
'''


