import matplotlib as plt
from pathlib import Path

def generateVectors(reviews,n,m):

    for review in reviews:
        reviewVector = []
        score = int(review.split()[0])
        for i in range(n,m):
            if(" "+str(i)+":" in review):
                reviewVector.append(1)
            else:
                reviewVector.append(0)
        if(score>=7):
            trainPosVectors.append(reviewVector)
        elif(score<=4):
            trainNegVectors.append(reviewVector)

#The first "n" words in the vocabulary will be skipped
n = 40
#Every word after "m+n" won't be checked.
m = 400
numberOfNeg = 12500
numberOfPos = 12500

trainDataVocab = open("aclImdb/train/labeledBow.feat","r")
trainPosVectors = []
trainNegVectors = []
reviews = trainDataVocab.readlines();
generateVectors(reviews,n,m)

print(len(trainNegVectors))
print(len(trainPosVectors))

#for vector in trainPosVectors:
 #   for x in vector:
  #      for i 