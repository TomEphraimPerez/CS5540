# Decision tree on #5 in list		kaggle.com/preeti5607/ddos-A-prevention 
#  To compile (really interpret); $ python3 D-tree1.py

# in VS, ^ cmd P -> Python tools, etc, eg python interpreters

print("\n\t1st rendition/iteration/generation of a decision tree for \"ddos-attacl-prevention\'.")
print('\n')

import sys
import pandas
# import numpy as np                          # Not used Sat 6-18
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg 

str1 = "/library/frameworks/python.framework/versions/3.6/lib/python3.6/site-packages/graphviz/"
sys.path.append(str1)
from sklearn import tree


df = pandas.read_csv("~/Desktop/Archive/dataset_attack.csv")
df = df.reindex(columns=['frame.encap_type', 'frame.len', 'ip.hdr_len', 'ip.len', 'ip.flags.rb', 'ip.flags.df', 'ip.flags.mf', 'ip.frag_offset', 'ip.ttl', 'ip.proto', 'ipsrc', 'ipdst', 'tcp.srcport', 'tcp.dstport', 'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr', 'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push', 'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size', 'tcp.time_delta', 'A'])
print(df)   # Yields a partial DS (that's ok) then the statement: [2990062 rows x 29 columns]
# print('\n')


# Using map() to convert strings to integers as required.
'''
Eg;
d = {'true':1, 'notTrue': 0}
df[someInitFeature] = df[someInitfeature].map(d)
'''


# Traffic Monitors/C-plane monitor = ['tcp.time_delta ** ', 'ip.src', 'ip.dst', 'tcp.dstport', 'ip.len'] 
    # ...to choose from dataset_A.csv
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

#      REGARDING THE (our) DS's TCP.FLAGS. _ _   geeksforgeeks.org/tcp-flags/             
# tcp.flags.res -  ...        for all definitions, SEE "DT/NOTES.txt"
# tcp.flags.ns - 
# tcp.flags.cwr -
# tcp.flags.ecn -
# tcp.flags.urg -
# tcp.flags.ack -
# tcp.flags.push -
# - - - - - - - - ->  tcp.flags.reset -
# tcp.flags.syn -
# tcp.flags.fin -

# END definitions _______________________________________________________________________________________________



# Recall/Lecture; C-plane Clustering (captures ntwk flows & records on whom-is-talking-to-whom) determined by:
# protocol (TCP or UDP)
# src IP
# dest IP
# port
#       [Consider the] fcapture tool.       (Lecture Wed 6-17)
#       Each flow record contains: (as per lecture)
#           time
#           duration
#           src ip
#           dest ip
#           dest port
#           num pkts
#           pkts / byte transfered in both directons
# Traffic Monitors/C-plane monitor = ['tcp.time_delta ** ', 'ipsrc', 'ipdst', 'tcp.dstport', 'ip.len']  >>>
# features = ['ipsrc', 'ipdst'] # Translated from list just above. COnverted IP to long int using excel/find-replace dot w/ nothing.
features = ['ipdst', 'ip.len']     # init/orignal features
X = df[features]
# y = df['ip.flags.reset']            # Try _.reset when cvs's are combined. (See def above)
y = df['tcp.flags.ack']             # Comtains 0's and 1's           

print('\n\t\t X ->')
print(X)
print('\n\t\tY ->')
print(y)
print('\n\n')

# dtree = DecisionTreeClassifier(criterion='gini')      # O. Possibly necessry
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)         
data = tree.export_graphviz(dtree, out_file = None, feature_names = features)
graph = pydotplus.graph_from_dot_data(data)

graph.write_png('D-tree1.png')                  # Error >>>
# pydotplus.graphviz.InvocationException: GraphViz's executables not found
# Installed Anaconda sci' packages -> stil bad.
# TRY: $ svn list https://svn.code.sf.net/p/netpbm/code/stable

img = pltimg.imread('D-tree1.png')
imgplot =  plt.imshow(img)
plt.show()




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
    <<<
'''


