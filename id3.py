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
    for vector in vectors:
        if(vector[0]==1):
            posCounter+=1
        elif(vector[0]==0):
            negCounter+=1
    if(posCounter>=negCounter):
        return 1
    else:
        return 0

def calculateCondProbability(vectors,x,value,C):
    currentProbability = 0
    scoreAppearances = 0
    totalAppearances = 0
    for vector in vectors:
        score = vector[0]
        if(vector[x]==value):
            totalAppearances+=1
            if(score==C):
                scoreAppearances+=1
    if(totalAppearances>0):
        currentProbability = scoreAppearances/totalAppearances
    else:
        currentProbability = 0.001
    if(currentProbability<=0):
        currentProbability = 0.001
    return currentProbability*np.log2(currentProbability)
        
def calculateCondEntropy(vectors,x,value):
    posReviewEntropy = calculateCondProbability(vectors,x,value,1)
    negReviewEntropy = calculateCondProbability(vectors,x,value,0)
    return ( - posReviewEntropy - negReviewEntropy)

def calculateInfoGain(vectors,x):
    #Number of attribute showing up in reviews
    attrAppearances = 0
    #Number of positive and negative reviews
    posReviews = 0
    negReviews = 0
    #For each review
    for vector in vectors:
        if(vector[x]==1):
            attrAppearances+=1
        if(vector[0]==1):
            posReviews+=1
        else:
            negReviews += 1
    #Percentage of pos and neg reviews to calculate entropy
    posReviewPerc = posReviews/len(vectors)
    negReviewPerc = negReviews/len(vectors)
    #Calculating entropy
    entropy = -posReviewPerc*np.log2(posReviewPerc) - negReviewPerc*np.log2(negReviewPerc)
    #Probability of finding and not finding the attribute in a review
    attrProbability = attrAppearances/len(vectors)
    return (entropy - attrProbability*calculateCondEntropy(vectors,x,1) - (1-attrProbability)*calculateCondEntropy(vectors,x,0))

def bestAttributeSelection(vectors,attr):
    maxAttribute = 0
    maxInfoGain  = 0
    for x in range(1,attr):
        infoGain = calculateInfoGain(vectors,x)
        if(infoGain>maxInfoGain):
            maxInfoGain = infoGain
            maxAttribute = x
    return maxAttribute

def trainDataId3(vectors,attr,defaultCategory):
    
    if(len(vectors)==0):
        return defaultCategory
    elif(categoriseVectors(vectors)):
        return  vectors[0][0]
    elif(len(attr)==0):
        return mostFrequentCategory(vectors)
    else:
        highestIGAttr = bestAttributeSelection(vectors,len(attr))
        print('Info gain:',highestIGAttr)
        tree = [highestIGAttr]
        leftVectors = []
        rightVectors = []
        category = mostFrequentCategory(vectors)
        for vector in vectors:
            attrValue = vector.pop(highestIGAttr)
            if(attrValue == 1):
                leftVectors.append(vector)
            else:
                rightVectors.append(vector)
        attr.pop(highestIGAttr)
        leftTree = trainDataId3(leftVectors,attr,category)
        tree.append(leftTree)
        rightTree = trainDataId3(rightVectors,attr,category)
        tree.append(rightTree)
        return tree

        
#The first "n-1" words in the vocabulary will be skipped
n = 40
#Every word after "m+n" won't be checked.
m = 50
entropy = -0.5*np.log2(0.5) - 0.5*np.log2(0.5)
attr = []
for i in range(n,m+n):
    attr.append(i)
print(attr)

trainDataVocab = open("aclImdb/train/labeledBow.feat","r")
reviews = trainDataVocab.readlines()

trainVectors = []
generateVectors(reviews,n,m)

trainDataId3(trainVectors,attr,1)