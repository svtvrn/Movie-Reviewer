import random as rand

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
        currProbability = (numberOfAppearances+1)/(len(trainVector)+2)
        wordProbability.append(currProbability)

def naiveBayes(testVector,wordProbability):

    probability=1
    for x in range(m):
        if(testVector[x]==1):
            probability*= wordProbability[x]
        elif(testVector[x]==0):
            probability*= (1 - wordProbability[x])
    probability *= 0.5
    return probability

def checkAccuracy(posReview,negReview,score):

    if(posReview>negReview and score>6):
        return True
    elif(posReview<negReview and score<5):
        return True
    return False

def run_tests(tests):

    accuracy = 0
    true_pos=0; true_neg=0
    false_pos = 0; false_neg = 0

    for test in tests:
        testVector = []
        score = int(test.split()[0])
        for i in range(n,m+n):
            if(" "+str(i)+":" in test):
                testVector.append(1)
            else:
                testVector.append(0)
        posReviewProbability = naiveBayes(testVector,posWordProbabilities)
        negReviewProbability = naiveBayes(testVector,negWordProbabilities)
        if(checkAccuracy(posReviewProbability,negReviewProbability,score)):
            accuracy+=1
            if(posReviewProbability>negReviewProbability):
                true_pos+=1
            else:
                true_neg+=1
        else:
            if(posReviewProbability>negReviewProbability):
                false_pos+=1
            else:
                false_neg+=1
    precision = true_pos/(true_pos + false_pos)
    recall = true_pos/(true_pos + false_neg)
    f1 = 2*(recall*precision)/(recall+precision)
    print('Precision: ', precision,)
    print('Recall:', recall)
    print('F1: ', f1)
    print('Accuracy: ', accuracy/len(tests)*100,"%")

#The first "n" words in the vocabulary will be skipped
n = 75
#Every word after "m+n" won't be checked.
m = 1000
#Loads the training token file and splits the reviews into lines
trainDataVocab = open("aclImdb/train/labeledBow.feat","r")
reviews = trainDataVocab.readlines()

#Positive and negative review vectors
trainPosVectors = []
trainNegVectors = []

posWordProbabilities = []
negWordProbabilities = []

#Filling up the two vectors with the review tokens
generateVectors(reviews,n,m)
trainData(trainPosVectors,posWordProbabilities,n,m)
trainData(trainNegVectors,negWordProbabilities,n,m)

#TESTING
testDataVocab = open("aclImdb/test/labeledBow.feat","r")
testReviews = testDataVocab.readlines()
testReviews = testReviews[0:1500] + testReviews [12500:14000]
run_tests(testReviews)