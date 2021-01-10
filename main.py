import matplotlib as plt
from pathlib import Path

def generateVectors(reviews,n,m):

    for review in reviews:
        reviewVector = []
        score = int(review.split()[0])
        for i in range(n,m+n):
            if(" "+str(i)+":" in review):
                reviewVector.append(1)
            else:
                reviewVector.append(0)
        if(score>=7):
            trainPosVectors.append(reviewVector)
        elif(score<=4):
            trainNegVectors.append(reviewVector)

def trainData(trainVector,wordProbability,n,m):

    for x in range(m):
        currProbability = 0
        numberOfAppearances = 0
        for vector in trainVector:
            if(vector[x]==1):
                numberOfAppearances +=1
        currProbability = (numberOfAppearances+1)/(numberOfPos+2)
        wordProbability.append(currProbability)

def calculateReview(testVector,wordProbability):

    probability=1
    for x in testVector:
        if(testVector[x]==1):
            probability*= wordProbability[x]
        elif(testVector[x]==0):
            probability*= 1 - wordProbability[x]
    probability *= 0.5
    return probability

#The first "n" words in the vocabulary will be skipped
n = 80
#Every word after "m+n" won't be checked.
m = 500
#Number of positive and negative reviews
numberOfNeg = 12500
numberOfPos = 12500

#Loads the training token file and splits the reviews into lines
trainDataVocab = open("aclImdb/train/labeledBow.feat","r")
reviews = trainDataVocab.readlines();

#Positive and negative review vectors
trainPosVectors = []
trainNegVectors = []

posWordProbabilities = []
negWordProbabilities = []

#Filling up the two vectors with the review tokens
generateVectors(reviews,n,m)
print("Number of vocabulary words: ",len(trainPosVectors[0]))
trainData(trainPosVectors,posWordProbabilities,n,m)
trainData(trainNegVectors,negWordProbabilities,n,m)


#TESTING
testDataVocab = open("aclImdb/test/labeledBow.feat","r")
testReviews = testDataVocab.readlines();
numberOfTests = 0
posRev = 0
negRev = 0

for test in testReviews:
    if(numberOfTests==25000):
        break
    testVector = []
    for i in range(n,m+n):
        if(" "+str(i)+":" in test):
            testVector.append(1)
        else:
            testVector.append(0)
    posReviewProbability = calculateReview(testVector,posWordProbabilities)
    negReviewProbability = calculateReview(testVector,negWordProbabilities)
    if(posReviewProbability<negReviewProbability):
        negRev+=1
    else:
        posRev+=1
    numberOfTests+=1

print(negRev)
print(posRev)
