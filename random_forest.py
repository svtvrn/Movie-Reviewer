import numpy as np
import random as rand

def generate_samples(reviews,n,m):
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

def check_categories(vectors):
    category = vectors[0].get('clf')
    for i in range(1,len(vectors)):
        if(vectors[i].get('clf')!=category):
            return False
    return True

def most_frequent(vectors):
    pos_counter = 0
    neg_counter = 0
    for vector in vectors:
        if(vector.get('clf')==True):
            pos_counter+=1
        elif(vector.get('clf')==False):
            neg_counter+=1
    return pos_counter>neg_counter

def attr_entropy(vectors,x,value,C):
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

def calculate_entropy(vectors,x,value):
    pos_review_entropy = attr_entropy(vectors,x,value,1)
    neg_review_entropy = attr_entropy(vectors,x,value,0)
    return ( - pos_review_entropy - neg_review_entropy)

def calculate_ig(vectors,x):
    #Number of attribute showing up in reviews.
    attr_appearances = 0
    #Number of positive and negative reviews.
    pos_reviews = 0
    neg_reviews = 0
    #For each review:
    for vector in vectors:
        if(vector.get(x)==1):
            attr_appearances+=1
        if(vector.get('clf')==1):
            pos_reviews+=1
        else:
            neg_reviews += 1
    #Percentage of pos and neg reviews to calculate entropy.
    pos_review_perc = pos_reviews/len(vectors)
    neg_review_perc = neg_reviews/len(vectors)
    #Calculating entropy
    entropy = -pos_review_perc*np.log2(pos_review_perc+1) - neg_review_perc*np.log2(neg_review_perc+1)
    #Probability of finding and not finding the attribute in a review.
    attr_probability = attr_appearances/len(vectors)
    #Calculating overall information gain.
    info_gain = ( entropy - attr_probability*calculate_entropy(vectors,x,1) - (1-attr_probability)*calculate_entropy(vectors,x,0))
    return info_gain

def select_best_attr(vectors,attributes):
    if(len(attributes)==1):
        return attributes[0]
    max_attribute = 0
    max_info_gain  = 0
    for x in attributes:
        info_gain = calculate_ig(vectors,x)
        if(info_gain>=max_info_gain):
            max_info_gain = info_gain
            max_attribute = x
    return max_attribute

def train_id3(vectors,attributes,depth,freq_category):

    if(depth==0):
        return freq_category
    elif(len(vectors)==0):
        return freq_category
    elif(check_categories(vectors)):
        return vectors[0].get('clf')
    elif(len(attributes)==0):
        return most_frequent(vectors)
    else:
        best_attr = select_best_attr(vectors,attributes)
        attributes.remove(best_attr)
        root = [best_attr]
        m = most_frequent(vectors)
        left_vectors = []
        right_vectors = []
        for vector in vectors:
            attr = vector.pop(best_attr)
            if(attr==1):
                left_vectors.append(vector)
            else: 
                right_vectors.append(vector)
        left_tree = train_id3(left_vectors,attributes,depth-1,m)
        root.append(left_tree)
        right_tree = train_id3(right_vectors,attributes,depth-1,m)
        root.append(right_tree)
        return root

#Traverses the ID3 decision tree.
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

#Runs traverse for each test sample and computes the program's accuracy,
#as well as the number of positive/negative reviews it guessed correctly.
def run_tests(tests,forest):
    accuracy = 0
    true_pos=0; true_neg=0
    false_pos = 0; false_neg = 0
    for test in tests:
        tree_predictions = []
        for tree in forest:
             tree_predictions.append(traverse(test,tree))
        pos_pred = 0
        neg_pred = 0
        for prediction in tree_predictions:
            if (prediction):
                pos_pred+=1
            else:
                neg_pred+=1
        final_prediction = pos_pred>neg_pred
        if (final_prediction==test.get('clf')):
            accuracy+=1
            if (final_prediction):
                true_pos+=1
            else:
                true_neg+=1
        else:
            if (final_prediction):
                false_pos+=1
            else:
                false_neg+=1
        #quit()
    precision = true_pos/(true_pos + false_pos)
    recall = true_pos/(true_pos + false_neg)
    f1 = 2*(recall*precision)/(recall+precision)
    print('Precision: ', precision,)
    print('Recall:', recall)
    print('F1: ', f1)
    print('Accuracy: ', accuracy/len(tests)*100,"%")


forest_range =(75,465)
#Defines the number of attributes in a sample.
m = 100
#Defines the maximum depth of a tree.
depth = 6
#Holds all the ID3 trees.
random_forest = []
#Loading the training data, converting them into a list of dictionaries. 
train_data = open("aclImdb/train/labeledBow.feat","r")
#Loading the vectors, shuffling them and choosing 1000 of them.

for i in range(30):
    train_vectors = generate_samples(train_data.readlines(),forest_range[0],forest_range[1])
    rand.shuffle(train_vectors)
    train_vectors = train_vectors[0:1200]
    #Attribute list, contains the dictionary keys.
    attributes = rand.sample(range(forest_range[0],forest_range[1]),m)
    #Training algorithm
    id3_tree = train_id3(train_vectors,attributes,depth,True)
    random_forest.append(id3_tree)
    train_data.seek(0)

print("Training complete. \n")
#Loading the testing data, converting them into a list of dictionaries. 
test_data = open("aclImdb/test/labeledBow.feat","r")
tests = generate_samples(test_data.readlines(),forest_range[0],forest_range[1])
tests = rand.sample(tests,3000)
run_tests(tests,random_forest)