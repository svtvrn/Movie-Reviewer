import numpy as np

def generateVectors(reviews,n,m):
    vectors = []
    for review in reviews:
        reviewVector = {}
        score = int(review.split()[0])
        if(score>6):
            reviewVector.update({'clf': True})
        else:
            reviewVector.update({'clf': False})
        for i in range(n,m+n):
            if(" "+str(i)+":" in review):
                reviewVector [i] = 1
            else:
                reviewVector [i] = 0
        vectors.append(reviewVector)
    return vectors

def categoriseVectors(vectors):

    category = vectors[0].get('clf')
    for i in range(1,len(vectors)):
        if(category!=vectors[i].get('clf')):
            return False
    return True

def mostFrequentCategory(vectors):

    posCounter = 0
    negCounter = 0
    for vector in vectors:
        if(vector.get('clf')==1):
            posCounter+=1
        elif(vector.get('clf')==0):
            negCounter+=1
    if(posCounter>=negCounter):
        return True
    else:
        return False

def calculateCondProbability(vectors,x,value,C):
    currentProbability = 0
    scoreAppearances = 0
    totalAppearances = 0
    for vector in vectors:
        score = vector.get('clf')
        if(vector.get(x)==value):
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
        if(vector.get(x)==1):
            attrAppearances+=1
        if(vector.get('clf')==1):
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

def bestAttributeSelection(vectors):

    attributes = list(vectors[0].keys())
    attributes.remove("clf")
    maxAttribute = 0
    maxInfoGain  = 0
    for x in attributes:
        infoGain = calculateInfoGain(vectors,x)
        if(infoGain>maxInfoGain):
            maxInfoGain = infoGain
            maxAttribute = x
    return maxAttribute

def trainDataId3(vectors,freqCategory):
    tree=[]
    if(len(vectors)==0):
        return freqCategory
    if(categoriseVectors(vectors)):
        return  vectors[0].get('clf')
    if(len(vectors[0])<=1):
        return mostFrequentCategory(vectors)
    highestIGAttr = bestAttributeSelection(vectors) 
    #print('Vocab @:',highestIGAttr)
    tree.append(highestIGAttr)
    leftVectors = []
    rightVectors = []
    category = mostFrequentCategory(vectors)
    if(highestIGAttr>0):
        for vector in vectors:
            attrValue = vector.pop(highestIGAttr)
            if(attrValue == 1):
                leftVectors.append(vector)
            else:
                rightVectors.append(vector)
            vectors.remove(vector)
    leftTree = trainDataId3(leftVectors,category)
    tree.append(leftTree)
    rightTree = trainDataId3(rightVectors,category)
    tree.append(rightTree)
    return tree
        
#The first "n-1" words in the vocabulary will be skipped
n = 1000
#Every word after "m+n" won't be checked.
m = 10
entropy = -0.5*np.log2(0.5) - 0.5*np.log2(0.5)

trainData = open("aclImdb/train/labeledBow.feat","r")
reviews = trainData.readlines()
trainVectors = generateVectors(reviews,n,m)
id3tree = trainDataId3(trainVectors,True)

testData = open("aclImdb/test/labeledBow.feat","r")
tests = generateVectors(testData.readlines(),n,m)

print(id3tree)

accuracy = 0
for test in tests:

    node = id3tree
    nodeKey = test.get(node[0])
    value = 0
    score = test.get('clf')
    depth = 1
    while(True):
        if(value==1):
            node = node[1]
            nodeKey = test.get(node[0])
        elif(value==0):
            node = node[2]
            nodeKey = test.get(node[0])
        if(nodeKey==True or nodeKey==False):
            value = nodeKey
            break
        depth+=1
    if(value==score):
        accuracy+=1

print(accuracy/250,"%")