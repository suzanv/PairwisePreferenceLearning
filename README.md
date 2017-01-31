# Pairwise preference ranking

```
python pairwise.py trainfile.tsv testfile.tsv
```

Performs pairwise preference ranking for a given trainfile and testfile with binary class labels (1 and not 1).

Other implementations use the same pairwise transformation function for the test set and the train set. Note however that we need 2 separate functions for the pairwise transform: one for the trainset that takes the labels into account (only creating pairs of one positive and one negative example), and one for the testset that does not take the labels into account (creates pairs of all items). 

The binary classification on the pairwise test data gives a prediction from each pair of test items: which of the two should be ranked higher. From these pairwise preferences a ranking can be created using a greedy sort algorithm.

Pairwise preference ranking is commonly performed on grouped data: two items from the same group are comparable to each other (in an ordinal version: one is better than the other), while two items from different groups are not. A typical example is rank-learning in the context of an information retrieval system. Here the groups are the queries. Two documents that are retrieved for the same query have a pairwise preference; two documents that are retrieved for two different queries do not have a pairwise preference and should not be compared to each other.

## Running the script

The script requires 2 tab-separated files (train and test) as arguments in which:
- the first column is the group id (in IR context: query id)
- the last column is the label (1 = positive, not 1 = negative)
- all the columns in between are numeric feature values

The train-test split should be on the group level: all items belonging to one group should be in the same partition.

Two example files are provided.

## Steps in the algorithm

1. Get data from feature files
2. Do the pairwise transform for the training set
3. Do the pairwise transform for the test set
4. Train classifier on pairwise data
5. Make predictions on testset
6. Greedy sort parwise preferences
7. Evaluate: create table for Precision-Recall graph
