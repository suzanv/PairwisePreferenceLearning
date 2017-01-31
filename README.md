# Pairwise preference ranking

```
python pairwise.py trainfile.tsv testfile.tsv
```

performs pairwise preference ranking for a given trainfile and testfile with binary class labals (1 and not 1)

Pairwise preference ranking is commonly performed on grouped data: two items from the same group are comparable to each other (in an ordinal version: one is better than the other), while two items from different groups are not. A typical example is rank-learning in the context of an information retrieval system. Here the groups are the queries. Two documents that are retrieved for the same query have a pairwise preference; two documents that are retrieved for two different queries do not have a pairwise preference and should not be compared to each other.

The script requires 2 tab-separated files in which:
- the first column is the group id (in IR context: query id)
- the last column is the label (1 = positive, not 1 = negative)
- all the columns in between are numeric feature values

the train-test split should be on the group level: all items belonging to one group should be in the same partition.

Two example files are provided.

## Steps in the algorithm

1. Get data from feature files
2. Do the pairwise transform for the training set
3. Do the pairwise transform for the test set
4. Train classifier on pairwise data
5. Make predictions on testset
6. Greedy sort parwise preferences
7. Evaluate: create table for Precision-Recall graph
