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

def calculateConditionalProbability(vectors,x,value,C):
    currentProbability = 0
    scoreAppearances = 0
    totalAppearances = 0
    for vector in vectors:
        score = vector[0]
        if(vector[x]==value):
            totalAppearances+=1
            if(score==C):
                scoreAppearances+=1
    currentProbability = scoreAppearances/totalAppearances
    return currentProbability*np.log2(currentProbability)
        
def calculateEntropy(vectors,x,value):
    posReviewEntropy = calculateConditionalProbability(vectors,x,value,1)
    negReviewEntropy = calculateConditionalProbability(vectors,x,value,0)
    return (- posReviewEntropy - negReviewEntropy)

def attributeProbabilty(vectors,x,value):
    numberOfAppearances = 0
    for vector in vectors:
        if(vector[x]==value):
            numberOfAppearances+=1

    return numberOfAppearances/len(vectors)

def bestAttributeSelection(vectors,m):
    maxAttributeGain = 0
    maxInfoGain  = 0
    for x in range(1,m+1):
        infoGain = entropy - attributeProbabilty(vectors,x,1) * calculateEntropy(vectors,x,1) -  attributeProbabilty(vectors,x,0) *  calculateEntropy(vectors,x,0)
        if(infoGain>maxInfoGain):
            maxInfoGain = infoGain
            maxAttributeGain = x
    return maxAttributeGain

def trainDataId3(vectors,m,defaultClf):

    if(vectors==None):
        return defaultClf
    elif(categoriseVectors(vectors)):
        return vectors[0][0]
    elif(m<=0):
        return mostFrequentCategory(vectors)
    else:
        bestAttribute = bestAttributeSelection(vectors,m)
        print(bestAttribute)
        tree = [bestAttribute]
        leftSubtreeVector = []
        rightSubtreeVector = []
        for vector in vectors:
            if(vector[bestAttribute]==1):
                leftSubtreeVector.append(vector)
            else:
                rightSubtreeVector.append(vector)
        leftSubtree = trainDataId3(leftSubtreeVector,m,True)
        rightSubtree = trainDataId3(rightSubtreeVector,m,False)
        tree.append(leftSubtree,rightSubtree)
        return tree


#The first "n" words in the vocabulary will be skipped
n = 40
#Every word after "m+n" won't be checked.
m = 50
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
print("TRAIN START")
id3Tree = trainDataId3(trainVectors,n,m,defaultClf)