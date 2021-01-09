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
        currProbability = (numberOfAppearances + 1)/(numberOfPos + 2)
        wordProbability.append(currProbability)

#The first "n" words in the vocabulary will be skipped
n = 40
#Every word after "m+n" won't be checked.
m = 400
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

print("Number of positive reviews probabilities: ",len(posWordProbabilities))
print("Number of negative reviews probabilities: ",len(negWordProbabilities))

#for prob in posWordProbabilities:
   # print(prob)

testDataVocab = open("aclImdb/test/labeledBow.feat","r")
testReviews = testDataVocab.readlines();
numberOfTests = 1000

#for test in numberOfTests:
