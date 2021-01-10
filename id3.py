import numpy as np

def generateVectors(reviews,n,m):

    for review in reviews:
        reviewVector = []
        score = int(review.split()[0])
        if(score>6):
            reviewVector.append(1)
        else:
            reviewVector.append(0)
        for i in range(n,m+n):
            if(" "+str(i)+":" in review):
                reviewVector.append(1)
            else:
                reviewVector.append(0)
        trainVectors.append(reviewVector)

def categoriseVectors(vectors):

    category = vectors[0][0]
    for i in range(1,len(vectors)):
        if(category!=vectors[i][0]):
            return False
    return True

def mostFrequentCategory(vectors):

    posCounter = 0
    negCounter = 0
    for i in range(len(vectors)):
        if(vectors[i][0]==1):
            posCounter+=1
        elif(vectors[i][0]==0):
            negCounter+=1
    if(posCounter>=negCounter):
        return 1
    else:
        return 0

def trainDataId3(vectors,n,m,defaultClf):

    if(vectors==None):
        return defaultClf
    elif(categoriseVectors(vectors)):
        return vectors[0][0]
    elif(m<=0):
        return mostFrequentCategory(vectors)
    else:
        bestAttribute = bestAttributeSelection(vectors,m)



#The first "n" words in the vocabulary will be skipped
n = 45
#Every word after "m+n" won't be checked.
m = 200
#Class probability
C=0.5
#Starting entropy
entropy = -C*np.log2(C) - C*np.log2(C)

trainDataVocab = open("aclImdb/train/labeledBow.feat","r")
reviews = trainDataVocab.readlines();
#Size m+1, where m vocabulary size and plus one for the review score at index 0
trainVectors = []
generateVectors(reviews,n,m)

#True equals a good review, False equals a bad review
defaultClf = True
id3Tree = trainDataId3(trainVectors,n,m,defaultClf)