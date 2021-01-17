import numpy as np
import random as rand

class Stump:

    def __init__(self,root,left,right):
        self.root = root
        self.left = left 
        self.right = right
    
    def check_sample(self,sample):
        decision = sample[0][self.root]
        sample_clf = sample[1]
        if(decision==1):
            return self.left==sample_clf
        else:
            return self.right==sample_clf

    def print_stump(self):
        print('Root: ',self.root,' Left: ',self.left, ' Right: ',self.right)

class AdaboostClf:

    def __init__(self,h,z):
        self.h = h
        self.z = z

    def test(self,sample):
        is_positive=0
        is_negative=0
        for i in range(len(self.h)):
            answer = sample[0][self.h[i].root]
            if(answer==1):
                if(self.h[i].left):
                    is_positive+= self.z[i]
                else:
                    is_negative+= self.z[i]
            else:
                if(self.h[i].right):
                    is_positive+= self.z[i]
                else:
                    is_negative+= self.z[i]
        return is_positive>is_negative

#Generating sample data in tuples
def generate_samples(samples,n,m):
    data = []
    for sample in samples:
        sample_data = []
        score = int(sample.split()[0])>5
        for i in range(n,m+n):
            if(" "+str(i)+":" in sample):
                sample_data.append(1)
            else:
                sample_data.append(0)
        tuple = (sample_data,score)
        data.append(tuple)
    return data

def normalize(weights):
    weight_sum = 0
    for weight in weights:
        weight_sum += weight
    for i in range(len(weights)):
        weights[i] /= weight_sum
    return weights

def calculate_gini(samples,attr,weights):
    #Number of positive and negative reviews that have and don't have the attribute.
    pos_has_attr = 0
    neg_has_attr = 0
    pos_hasnt_attr = 0
    neg_hasnt_attr = 0
    #Weighted indexes
    w1 = 0
    w2 = 0
    w3 = 0
    w4 = 0
    pos_weight = 0
    neg_weight = 0
    total_weight = 0
    for i in  range (len(samples)):
        if(samples[i][1]==True):
            if(samples[i][0][attr]==1):
                pos_has_attr+=1
                w1+= weights[i]
            else:
                pos_hasnt_attr+=1
                w2+= weights[i]
            pos_weight
        else:
            if(samples[i][0][attr]==1):
                neg_has_attr+=1
                w3+= weights[i]
            else:
                neg_hasnt_attr+=1
                w4+= weights[i]
            neg_weight+= weights[i]
        total_weight+= weights[i]
    pos_reviews = (pos_has_attr + pos_hasnt_attr)
    neg_reviews = (neg_has_attr + neg_hasnt_attr)

    pos_gini = 1 - pow((pos_has_attr+1)/(pos_reviews+2),2) - pow((pos_hasnt_attr+1)/(pos_reviews+2),2)
    neg_gini = 1 - pow((neg_has_attr+1)/(neg_reviews+2),2) - pow((neg_hasnt_attr+1)/(neg_reviews+2),2)

    impurity = (pos_reviews/(total_weight*len(samples)))*pos_gini + (neg_reviews/(total_weight*len(samples)))*neg_gini
    return impurity, pos_has_attr>neg_has_attr

def weak_learner(samples,weights):

    mini_gini = np.inf
    best_attr = 0 
    best__has_attr = False
    attributes = len(samples[0][0])
    for attr in range(rand.randint(0,attributes)):
        gini, has_attr = calculate_gini(samples,attr,weights)
        if(gini<mini_gini):
            mini_gini = gini
            best_attr = attr
            best__has_attr = has_attr

    return Stump(best_attr, best__has_attr, not best__has_attr)

def repopulate_samples(samples,weights):
    weight_buckets = [weights[0]]
    for i in range(1,len(weights)):
        weight_buckets.append( weights[i] + weight_buckets[i-1] )
    new_samples = []
    for i in range(len(samples)):
        rand_gen = rand.uniform(0.0,1.0)
        j=0    
        while(True):
            if rand_gen<=weight_buckets[j]:
                new_samples.append(samples[j])
                break;
            j+=1
    weights = [1/len(samples)]*len(samples)
    return new_samples,weights
    
def weighted_majority(h,z):
    return AdaboostClf(h,z)

#Adaboost training function
def adaboost(samples,iterations):
    #Number of total weights
    w = [1/len(samples)]*len(samples)
    #Number of hypotheses we've learned
    h = []
    #The ammount of say for each hypothesis
    z = []
    for i in range (iterations):
        if i!=0:
            samples,w = repopulate_samples(samples,w)
        stump = weak_learner(samples,w)
        h.append(stump)
        h[i].print_stump()
        error = 0
        #Calculating total error
        for j in range (len(samples)):
            if not h[i].check_sample(samples[j]):
                error += w[j]
        #Adding the stump weight to the list
        if error == 0.5:
            z.append(0)
        else:
            z.append(0.5*np.log(((1-error)/(error+0.001))+1))
        #Updating the sample weights
        for j in range (len(samples)):
            if h[i].check_sample(samples[j]):
                w[j] *= np.exp(-z[i])
            else:
                w[j] *= np.exp(z[i])
        #Normalizing the sample weights
        w = normalize(w)        
    return weighted_majority(h,z)

def run_tests(adaboost,tests):
    accuracy = 0
    pos=0
    neg=0
    for test in tests:
        clf = adaboost.test(test)
        if(clf == test[1]):
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
m = 100
#Number of iterations Adaboost will perfrom
iterations = 5

train_data = open("aclImdb/train/labeledBow.feat","r")
train_samples = generate_samples(train_data.readlines(),n,m)
rand.shuffle(train_samples)  
train_samples = rand.sample(range())
print(len(train_samples))

adaboost_clf = adaboost(train_samples,iterations)
test_data = open("aclImdb/test/labeledBow.feat","r")
test_samples = generate_samples(test_data.readlines(),n,m)
run_tests(adaboost_clf,test_samples)