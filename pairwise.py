# python pairwise.py trainfile.tsv testfile.tsv

"""
performs pairwise preference ranking for a given trainfile and testfile with binary class labals (1 and not 1)

Pairwise preference ranking is commonly performed on grouped data: two items from the same group are comparable to each other (in an ordinal version: one is better than the other), while two items from different groups are not. A typical example is rank-learning in the context of an information retrieval system. Here the groups are the queries. Two documents that are retrieved for the same query have a pairwise preference; two documents that are retrieved for two different queries do not have a pairwise preference and should not be compared to each other.

The script requires 2 tab-separated files in which:
- the first column is the group id (in IR context: query id)
- the last column is the label (1 = positive, not 1 = negative)
- all the columns in between are numeric feature values

the train-test split should be on the group level: all items belonging to one group should be in the same partition.

Two example files are provided.

"""

import sys
import csv
import numpy as np
from sklearn.linear_model import SGDClassifier

trainfile = sys.argv[1]
testfile = sys.argv[2]

def get_vectors_per_groupid (filename):
    vectors_per_groupid = dict()
    tsv = open(filename, 'r')

    for line in csv.reader(tsv, delimiter="\t"):
        #print (line)
        v = np.array(line)
        vector = v.astype(np.float)
        #print (vector)

        group_id = vector[0]

        vectors_for_this_groupid = []
        if group_id in vectors_per_groupid:
            vectors_for_this_groupid = vectors_per_groupid[group_id]
        vectors_for_this_groupid.append(vector)
        vectors_per_groupid[group_id] = vectors_for_this_groupid
    return vectors_per_groupid


'''
Note that we need 2 separate function for pairwise transformation: one for the trainset that takes the labels into account
(only creating pairs of one positive and one negative example), and one for the testset that does not take the
labels into account (creates pairs of all items)
'''


def pairwise_transform_trainset (vectors_for_one_group):
    # assumes that the first column is the group id, the second column is the item id and the last column is the label
    positive_examples = []
    negative_examples = []
    #print (vectors_for_one_group)
    group_id = str(int(vectors_for_one_group[0][0]))
    #print (group_id)
    for vector in vectors_for_one_group:
        if vector[-1] == 1:
            positive_examples.append(vector[0:])
        else:
            negative_examples.append(vector[0:])
    pairwise_data = []

    for pos in positive_examples:
        for neg in negative_examples:
            #print (pos)
            if pos[1] != neg[1]:
                pair_id = str(int(pos[1]))+"-"+str(int(neg[1]))
                #print (group_id,pair_id)
                paired1 = [group_id,pair_id]
                diff1 = pos[2:-2]-neg[2:-2]
                # the last item (the label) should not be part of the vectors that are subtracted

                for i in diff1:
                    paired1.append(i)
                paired1.append(1)
                #vector1 = np.array(paired1)

                pair_id = str(int(neg[1]))+"-"+str(int(pos[1]))
                paired2 = [group_id,pair_id]
                diff2 = neg[2:-2]-pos[2:-2]
                for i in diff2:
                    paired2.append(i)
                paired2.append(0)
                #vector2 = np.array(paired2)

                pairwise_data.append(paired1)
                pairwise_data.append(paired2)
                #print (paired1)
                #print (paired2)
    return pairwise_data


def pairwise_transform_testset (vectors_for_one_group):
    group_id = str(int(vectors_for_one_group[0][0]))
    #print (group_id)
    pairwise_data = []
    for vector1 in vectors_for_one_group:
        for vector2 in vectors_for_one_group:
            if vector1[1] != vector2[1]:
                pair_id = str(int(vector1[1]))+"-"+str(int(vector2[1]))
                paired1 = [group_id,pair_id]
                diff1 = vector1[2:-2]-vector2[2:-2]
                # the last item (the label) should not be part of the vectors that are subtracted
                for i in diff1:
                    paired1.append(i)
                paired1.append(-1) # -1 is unknown label

                pair_id = str(int(vector2[1]))+"-"+str(int(vector1[1]))
                paired2 = [group_id,pair_id]
                diff2 = vector2[2:-2]-vector1[2:-2]
                for i in diff2:
                    paired2.append(i)
                paired2.append(-1) # -1 is unknown label

                pairwise_data.append(paired1)
                pairwise_data.append(paired2)
                #print (paired1)
                #print (paired2)
    return pairwise_data


def get_sum_prefs (PREF,V,v):
    sum_prefs = 0
    for u in V:
        if (v,u) in PREF:
            sum_prefs += PREF[(v,u)]
            #print ("\t",(v,u),sum_prefs)
        if (u,v) in PREF:
            sum_prefs -= PREF[(u,v)]
    return sum_prefs


def greedy_sort(X,PREF):
    V = X
    maxv = ""
    max_sum_prefs = 0
    for v in V:
        sum_prefs = get_sum_prefs(PREF,V,v)
        #print (v, "sum prefs:",sum_prefs)
        if sum_prefs > max_sum_prefs:
            maxv = v
            max_sum_prefs = sum_prefs
    sorted_X = list()
    #print ("First maxv:", maxv,"sum prefs:",max_sum_prefs,"nodes left:",len(V))

    while len(V)> 1 and not maxv is "":
        sorted_X.append(maxv)
        V.remove(maxv)
        maxv = ""
        max_sum_prefs = 0
        for v in V:
            sum_prefs = get_sum_prefs(PREF,V,v)
            #print (v, "sum prefs:",sum_prefs)

            if sum_prefs > max_sum_prefs:
                maxv = v
                max_sum_prefs = sum_prefs
        #print ("maxv:", maxv,"sum prefs:",max_sum_prefs,"nodes left:",len(V))

    # add remaining v's with sumprefs=0
    for v in V:
        sorted_X.append(v)
    return sorted_X


def compute_precision(model,reference):
    if len(model)+len(reference)>0:
        tp=len(model.intersection(reference))
        fp=len(model-reference)
        if tp > 0:
            return float(tp)/(float(fp)+float(tp))
        else:
            #print ("no true positives")
            return 0
    else:
        return 1

def compute_recall(model,reference):
    if len(model)+len(reference)>0:
        tp=len(model.intersection(reference))
        fn=len(reference-model)
        if tp > 0:
            return float(tp)/(float(fn)+float(tp))
        else:
            return 0
    else:
        return 1


print("Get data from feature files")
vectors_per_groupid_trainset = get_vectors_per_groupid(trainfile)
vectors_per_groupid_testset = get_vectors_per_groupid(testfile)


print("Do the pairwise transform for the training set")
pairwise_data_train = []
for group_id in vectors_per_groupid_trainset:
    vectors_for_this_groupid = vectors_per_groupid_trainset[group_id]
    #print (vectors_for_this_groupid)
    pairwise_data = pairwise_transform_trainset(np.array(vectors_for_this_groupid))
    for vector in pairwise_data:
        pairwise_data_train.append(vector)
    #print (pairwise_data)

print("Do the pairwise transform for the test set")
selectedposts_human = set()
pairwise_data_test = []
for group_id in vectors_per_groupid_testset:
    vectors_for_this_groupid = vectors_per_groupid_testset[group_id]
    #print (vectors_for_this_groupid)
    for vector in vectors_for_this_groupid:
        if vector[-1] == 1:
            #print (str(int(vector[1])))
            selectedposts_human.add(str(int(vector[1])))
    pairwise_data = pairwise_transform_testset(np.array(vectors_for_this_groupid))
    for vector in pairwise_data:
        pairwise_data_test.append(vector)


print("Convert to numpy arrays")

x_train = np.array([i[2:-2] for i in pairwise_data_train])
y_train = np.array([i[-1] for i in pairwise_data_train])
print("Train X dimensions:",x_train.shape)
print("Train y dimensions:",y_train.shape)
x_test = np.array([i[2:-2] for i in pairwise_data_test])
group_id_array_test = np.array([i[0] for i in pairwise_data_test])
item_pair_id_array_test = np.array([i[1] for i in pairwise_data_test])

print("Test X dimensions:",x_test.shape)
#print ("Test items:",group_id_array_test,item_pair_id_array_test)

'''
PAIRWISE PREFERENCE LEARNING
'''

print("Train classifier on pairwise data")
#clf = SVC()
#clf.fit(x_train,y_train)
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(x_train,y_train)

print ("Make predictions on testset")
predicted = clf.predict(x_test)
#print(predicted)

print ("Greedy sort parwise preferences")
'''
The binary classification on the pairwise test data gives a prediction from each pair of test items:
which of the two should be ranked higher. From these pairwise preferences a ranking can be created
using a standard greedy sort algorithm.
'''
pairwise_preferences = dict()
set_of_items_in_testset_per_group_id = dict()
k=0
for pred in predicted:
    group_id = group_id_array_test[k]
    item_pair_id = item_pair_id_array_test[k]

    (item_id1,item_id2) = str(item_pair_id).split(sep="-")
    #(item_id1,item_id2) = "-".split(item_pair_id)
    #print (group_id,item_id1,item_id2,pred)
    pairwise_preferences[(item_id1,item_id2)] = pred
    set_of_items = set()
    if group_id in set_of_items_in_testset_per_group_id:
        set_of_items = set_of_items_in_testset_per_group_id[group_id]
    if item_id1 not in set_of_items:
        set_of_items.add(item_id1)
    if item_id2 not in set_of_items:
        set_of_items.add(item_id2)
    set_of_items_in_testset_per_group_id[group_id] = set_of_items
    k += 1

ranked_postids_per_thread = dict()
for group_id in set_of_items_in_testset_per_group_id:
    set_of_items = set_of_items_in_testset_per_group_id[group_id]
    sorted_items = greedy_sort(set_of_items,pairwise_preferences)
    ranked_postids_per_thread[group_id] = sorted_items
    #print(group_id,sorted_items)

print("Evaluate: create table for Precision-Recall graph")

for cutoff in range (1,10):
    selectedposts_model = set()
    for threadid in ranked_postids_per_thread:
        ranked_postids = ranked_postids_per_thread[threadid]
        k=0
        for postid in ranked_postids:
            k +=1
            if k <= cutoff:
                selectedposts_model.add(postid)
    recall = compute_recall(selectedposts_model,selectedposts_human)
    precision = compute_precision(selectedposts_model,selectedposts_human)
    f1 = 2*(precision*recall)/(precision+recall)
    print ("pairwise_SGD","\t",cutoff,"\t",recall,"\t",precision,"\t",f1)


