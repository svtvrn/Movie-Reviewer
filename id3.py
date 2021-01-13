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
        if(i==(1)*len(vectors)):
            return True

def mostFrequentCategory(vectors):
    pos_counter = 0
    neg_counter = 0
    for vector in vectors:
        if(vector.get('clf')==True):
            pos_counter+=1
        elif(vector.get('clf')==False):
            neg_counter+=1
    return pos_counter>neg_counter

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
    info_gain = ( entropy - attr_probability*calculateCondEntropy(vectors,x,1) - (1-attr_probability)*calculateCondEntropy(vectors,x,0))
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
    elif(categoriseVectors(vectors)):
        return vectors[0].get('clf')
    else:
        best_attr = bestAttributeSelection(vectors)
        if(best_attr==0):
            return mostFrequentCategory(vectors)
        else:
            freq_category = mostFrequentCategory(vectors)
        tree.append(best_attr)
        for value in range(2):
            sub_vectors = []
            for vector in vectors:
                attr_val = vector.pop(best_attr)
                if(attr_val==value):
                    sub_vectors.append(vector)
                vectors.remove(vector)
            sub_tree = trainDataId3(sub_vectors,freq_category)
            tree.append(sub_tree)
        return tree

def traverse(test,node):
    if(node==True or node==False):
        #print(node)
        return node
    else:
        answer = test.get(node[0])
        #print('Node: ',node,'Attr: ',node[0],'Direction: ',answer)
        if(answer==1):
            return traverse(test,node[1])
        elif(answer==0):
            return traverse(test,node[2])

def run_tests(tests,root):
    accuracy = 0
    pos=0
    neg=0
    for test in tests:
        clf = traverse(test,root)
        if(clf == test.get('clf')):
            accuracy+=1
            if(clf==True):
                pos+=1
            else:
                neg+=1
        #quit()
    print(pos," ",neg)
    print(accuracy/len(tests)*100,"%")

#The first "n-1" words in the vocabulary will be skipped
n = 60
#Every word after "m+n" won't be checked.
m = 60

train_data = open("aclImdb/train/labeledBow.feat","r")
train_vectors = generateVectors(train_data.readlines(),n,m)
id3_tree = trainDataId3(train_vectors,True)

test_data = open("aclImdb/test/labeledBow.feat","r")
tests = generateVectors(test_data.readlines(),n,m)
run_tests(tests,id3_tree)