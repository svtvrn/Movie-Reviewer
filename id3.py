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
        if(vectors[i].get('clf')!=category):
            return False
        if(i==(0.9)*len(vectors)):
            return True

def mostFrequentCategory(vectors):
    pos_counter = 0
    neg_counter = 0
    for vector in vectors:
        if(vector.get('clf')==True):
            pos_counter+=1
        elif(vector.get('clf')==False):
            neg_counter+=1
    if(pos_counter>=neg_counter):
        return True
    else:
        return False

def calculateCondProbability(vectors,x,value,C):
    current_probability = 0
    score_appearances = 0
    total_appearances = 0
    for vector in vectors:
        score = vector.get('clf')
        if(vector.get(x)==value):
            total_appearances+=1
            if(score==C):
                score_appearances+=1
    if(total_appearances>0):
        current_probability = score_appearances/total_appearances
    return current_probability*np.log2(current_probability+1)
        
def calculateCondEntropy(vectors,x,value):
    pos_review_entropy = calculateCondProbability(vectors,x,value,1)
    neg_review_entropy = calculateCondProbability(vectors,x,value,0)
    return ( - pos_review_entropy - neg_review_entropy)

def calculateInfoGain(vectors,x):
    #Number of attribute showing up in reviews
    attr_appearances = 0
    #Number of positive and negative reviews
    pos_reviews = 0
    neg_reviews = 0
    #For each review
    for vector in vectors:
        if(vector.get(x)==1):
            attr_appearances+=1
        if(vector.get('clf')==1):
            pos_reviews+=1
        else:
            neg_reviews += 1
    #Percentage of pos and neg reviews to calculate entropy
    pos_review_perc = pos_reviews/len(vectors)
    neg_review_perc = neg_reviews/len(vectors)
    #Calculating entropy
    entropy = -pos_review_perc*np.log2(pos_review_perc+1) - neg_review_perc*np.log2(neg_review_perc+1)
    #Probability of finding and not finding the attribute in a review
    attr_probability = attr_appearances/len(vectors)
    info_gain = (entropy - attr_probability*calculateCondEntropy(vectors,x,1) - (1-attr_probability)*calculateCondEntropy(vectors,x,0))
    return info_gain

def bestAttributeSelection(vectors):

    attributes = list(vectors[0].keys())
    attributes.remove("clf")
    max_attribute = 0
    max_info_gain  = 0
    for x in attributes:
        info_gain = calculateInfoGain(vectors,x)
        if(info_gain>max_info_gain):
            max_info_gain = info_gain
            max_attribute = x
    return max_attribute

def trainDataId3(vectors,freq_category):

    tree=[]
    if(len(vectors)==0):
        return freq_category
    if(categoriseVectors(vectors)):
        return  vectors[0].get('clf')
    if(len(vectors[0])==1):
        return mostFrequentCategory(vectors)
    best_attr = bestAttributeSelection(vectors) 
    tree.append(best_attr)
    left_vectors = []
    right_vectors = []
    category = mostFrequentCategory(vectors)
    if(best_attr>0):
        for vector in vectors:
            attr_value = vector.pop(best_attr)
            if(attr_value == 1):
                left_vectors.append(vector)
            else:
                right_vectors.append(vector)
            vectors.remove(vector)
    left_tree = trainDataId3(left_vectors,category)
    right_tree = trainDataId3(right_vectors,category)
    tree.append(left_tree)
    tree.append(right_tree)
    return tree
        
#The first "n-1" words in the vocabulary will be skipped
n = 40
#Every word after "m+n" won't be checked.
m = 50

train_data = open("aclImdb/train/labeledBow.feat","r")
train_vectors = generateVectors(train_data.readlines(),n,m)
id3_tree = trainDataId3(train_vectors,True)
print(id3_tree)

test_data = open("aclImdb/test/labeledBow.feat","r")
tests = generateVectors(test_data.readlines(),n,m)
pos = neg =0
accuracy = 0
for test in tests:
    node = id3_tree
    node_key = test.get(node[0])
    score = test.get('clf')
    while(True):
        if(node_key==1):
            node = node[1]
            if(node!=True and node!=False):
                node_key = test.get(node[0])
        elif(node_key==0):
            node = node[2]
            if(node!=True and node!=False):
                node_key = test.get(node[0])
        if(node_key==True or node_key==False):
            if(node_key==score):
                accuracy+=1
            if(node_key==True):
                pos+=1
            else:
                neg+=1
            break
    
print(accuracy/250,"%")
print(pos)
print(neg)

        