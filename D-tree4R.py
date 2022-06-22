
# Decision tree on #5 in list		kaggle.com/preeti5607/ddos-A-prevention 
#  To compile (interpret);  $ python3 D-tree1.py
# In VS, ^cmd P -> Python tools, etc, eg to choose python interpreters
# $ bash                    # Nav to bash is best.
# pwd                       # /Users/thomasperez/5540Smr22Team/GroupProject1/DS1/CS5540
# Github                    # https://github.com/TomEphraimPerez/CS5540
# (($ conda install graphviz # 3hr installation. conda --version  -> conda 4.12.0 in bash))

# USE/ $ python3 D-tree4R.py homework1-featurizer/monitored_ips.txt
# USE/ $ python3 D-tree4R.py homework1-featurizer/monitored_ips.txt

print("\n\tMERGED rendition of a decision tree for \"ddos-attacl-prevention\'.")
print('\n')
import pandas
from sklearn.tree import plot_tree
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# ==================== Rebel's FEATURIZER ======== (pytorch) =============================|
# Assignment 1 - Featurizer, arebel@calstatela.edu
# Some super basic code that will be used to 
# extract out eh features from the dataset given
# and produce input 'slices' for machine learning
# methods

# since our input is csv files taken from
# https://www.kaggle.com/datasets/preeti5607/ddos-attack-prevention
# we'll import CSVs and export samples
# each slice either be attack data or normal traffic data

# we'll start by including in some 'protected' ips
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('monitor_ip_file')      #o 
parser.add_argument('--output_features_file', default='features.csv')#latter in /CS5540/homework1-featurizer
parser.add_argument('--window_time', type=float, default=0.100) 
args = parser.parse_args() #o

# Pull in information on which systems we want to
# monitor! These systems are the ones we want to
# prevent from being DDOSed or becoming the DDOSer

# IP conversion stuff
import struct
import socket

monitored_ips = set()
with open(args.monitor_ip_file) as file:
    for line in file:
        ip = line
        ip_as_int = struct.unpack("!I", socket.inet_aton(ip))[0]
        monitored_ips.add( ip_as_int )

print('Loaded {} IPs to monitor'.format( len(monitored_ips) ) )

# Stuff for featurization process
# We'll read the dataset given to us from
# https://www.kaggle.com/datasets/preeti5607/ddos-attack-prevention

import csv                  # using Dictwriter/reader
import collections          # for deque, the window itself

# Settings for featurizer
output_file = open( args.output_features_file, 'w' )
# these are our features
field_names = \
[
    'num_packets',
    'num_unique_src_ips',
    'num_bytes_exchanged',
    'num_unique_protos',
    'num_packets_to_monitored',
    'num_bytes_to_monitored',
    'is_attack'
]
writer = csv.DictWriter( output_file, fieldnames=field_names )
writer.writeheader()

# our actual featureizer, assuming we've been given
# a window that meets our timing specification 
# so no need to check time deltas, unless it's a window
# related feature (ie time between retries)


def extract_features_from( window: "iterable" ):      #o      
    # the cols from the dataset are the following:
    # 'frame.encap_type', 'frame.len', 'frame.protocols', 'ip.hdr_len', 'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset', 'ip.ttl', 'ip.proto', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport', 'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr', 'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push', 'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size', 'tcp.time_delta', 'attack'

    # features we're interested in
    # to be extracted from the window
    features = \
    {
        'num_packets' : len(window),
        'num_unique_src_ips' : 0,
        'num_bytes_exchanged' : 0,
        'num_unique_protos' : 0,
        'num_packets_to_monitored' : 0,
        'num_bytes_to_monitored' : 0,
        'is_attack' : False
    }

    # supporting datastructures
    src_ips_seen = set()
    protos_seen = set()

    # now to actually process the window
    for item in window:
        src_ips_seen.add( item['ip.src'] )
        features['num_bytes_exchanged'] += int( item['frame.len'] )
        protos_seen.add( item['frame.protocols'] )
        features['is_attack'] = True if item['attack'] == 'attack' else False

        if item['ip.dst'] in monitored_ips:
            features['num_packets_to_monitored'] += 1
            features['num_bytes_to_monitored'] += int( item['frame.len'] )

    # post processing of sets
    features['num_unique_src_ips'] = len( src_ips_seen )
    features['num_unique_protos'] = len( protos_seen )

    # finally output the data to the csv file
    writer.writerow( features )

# THe abbreviated CSV's, (100 records each) are:
# datasets = ['~/desktop/Archive/abbreviated_dataset_attack.csv', '~/desktop/Archive/abbreviated_dataset_normal.csv']
# datasets = ['abbreviated_dataset_attack.csv', 'abbreviated_dataset_normal.csv']
datasets = ['features.csv']
# dataset = df.loc[1:100]               #??



# process the input CSV files
for dataset in datasets:
    with open( dataset ) as csvfile:
        # We'll use dict reader to perserve the header/col info
        dataset_reader = csv.DictReader(csvfile)

        # We need to keep track of delta time
        window_time = 0
        window = collections.deque()
        for row in dataset_reader:
            # extract interesting info from row
            time_delta = float( row['tcp.time_delta'] )
            
            # to apply time update, we should pop items
            # off until we can safely insert current item
            # if we were to drop into the following loop, this means
            # that our window is full, we should check for this
            full_window = False
            while window_time + time_delta > args.window_time and len(window) != 0:
                if full_window == False:
                    full_window = True # so we don't do another window process

                    # we'll process the window once here
                    extract_features_from( window )
                    
                popped = window.popleft()
                window_time -= float( popped['tcp.time_delta'] )

            # now we can insert into window for processing
            window.append(row)
            window_time += time_delta

        # finished processing dataset, do one final window purge
        extract_features_from( window )
    
    # File complete indicator
    print('Finished processing {}'.format(dataset))

# and we're done!
output_file.close()
# ============== END === Rebel's FEATURIZER ==============================================|




df = pandas.read_csv("homework1-featurizer/features.csv")       # Rebel's featureset
# ( abbreviated_dataset_attack.csv           abbreviated_dataset_normal.csv )

# print(df)
X = df.loc[:, df.columns[:-1]]           # TODO: fix later, output col has to be last
y = df.loc[:, df.columns[-1]].to_frame() # TODO: fix later, output col has to be last

df.head(100)


print('\n\t\t X ->')
print(X)
print('\n\t\tY ->')
print(y)
print('\n\n')
dtree = DecisionTreeClassifier()

# The two files here are from: 
# https://www.kaggle.com/datasets/preeti5607/ddos-attack-prevention
# datasets = [ 'dataset_attack.csv', 'dataset_normal.csv' ] /o
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.50)


# Let's do a pass using the test data
from sklearn import metrics
# y = KNN_Classifer.predict(X_test)
print("Accuracy of D-tree Model w/ k={}:".format(args.k), metrics.accuracy_score(Y_test, y))

dtree = dtree.fit(X, y)         
plt.figure(figsize=(60,30))
print('\n\t\tpy plt.figure as opossed to using graphviz ')
plot_tree(dtree, filled=True);
print('\n\t\tplot_tree(...) as opposed to using graphviz -> should have a plot below.')
# plt.show()                         # Rendering takes ~75 min when merging csv's.
print('\n\n')







