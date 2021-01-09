import numpy as np
import matplotlib as plt
from pathlib import Path

#The first "n" words in the vocabulary will be skipped
n = 40
#Every word after "m+n" won't be checked.
m = 400
trainDataVocab = open("aclImdb/train/labeledBow.feat","r")
trainVectors = []

reviews = trainDataVocab.readlines();
for review in reviews:
    score = review[0]
    reviewVector = []
    #print(score)
    for i in range(n,m):
        if(" "+str(i)+":" in review):
            reviewVector.append(1)
        else:
            reviewVector.append(0)
    trainVectors.append(reviewVector)