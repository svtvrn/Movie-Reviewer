import numpy as np

class Stump:

    def __init__(self,root,left,right):
        self.root = root
        self.left = None 
        self.right = None
    
    def check_sample(self,sample):
        return sample[0][self.root]==1

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
    for weight in weights:
        weight /= weight_sum
    return weights

def attr_entropy(samples,x,value,C):
    current_probability = 0
    score_appearances = 0
    total_appearances = 0
    for sample in samples:
        score = sample[1]
        if(sample[0][x]==value):
            total_appearances+=1
            if(score==C):
                score_appearances+=1
    if(total_appearances>0):
        current_probability = score_appearances/total_appearances
    return current_probability*np.log2(current_probability+1)

def calculate_entropy(samples,x,value):
    pos_review_entropy = attr_entropy(samples,x,value,1)
    neg_review_entropy = attr_entropy(samples,x,value,0)
    return ( - pos_review_entropy - neg_review_entropy)

def calculate_ig(samples,x):
    #Number of attribute showing up in reviews.
    attr_appearances = 0
    #Number of positive and negative reviews.
    pos_reviews = 0
    neg_reviews = 0
    #For each review:
    for sample in samples:
        if(sample[0][x]==1):
            attr_appearances+=1
        if(sample[1]==1):
            pos_reviews+=1
        else:
            neg_reviews += 1
    #Percentage of pos and neg reviews to calculate entropy.
    pos_review_perc = pos_reviews/len(samples)
    neg_review_perc = neg_reviews/len(samples)
    #Calculating entropy
    entropy = -pos_review_perc*np.log2(pos_review_perc+1) - neg_review_perc*np.log2(neg_review_perc+1)
    #Probability of finding and not finding the attribute in a review.
    attr_probability = attr_appearances/len(samples)
    #Calculating overall information gain.
    info_gain = ( entropy - attr_probability*calculate_entropy(samples,x,1) - (1-attr_probability)*calculate_entropy(samples,x,0))
    return info_gain

def weak_learner(samples,weights):
    stump_clf = None
    max_info_gain = np.inf
    best_attr = 0 
    for attr in len(samples[0][0]):
        info_gain = calculate_ig(samples,attr)
        if(info_gain>max_info_gain):
            max_info_gain = info_gain
            best_attr = attr
    max_weight = -1
    weight_idx = 0
    for i in len(weights):
        if(weights[i]>max_weight):
            max_weight = weights[i]
            weight_idx = i
    sample_clf = samples[i][1]
    sample_attr = samples[i][0][best_attr]
    if(sample_attr==1):
        return Stump(best_attr,sample_clf,not sample_attr)
    else:
        return Stump(best_attr,not sample_clf,sample_attr)


def weighted_majority(h,z):
    pass

#Adaboost training function
def adaboost(samples,iterations):

    #Number of total weights
    w = [1/len(samples)]*len(samples)
    #Number of hypotheses we've learned
    h = []
    #The ammount of say for each hypothesis
    z = []
    for i in range (iterations):
        h[i] = weak_learner(samples,w)
        error = 0
        for j in range (len(samples)):
            if h[i].check_sample(j) != samples[j][1]:
                error += w[j]
        for j in range (len(samples)):
            if h[i].check_sample(j) == samples[j][1]:
                w[j] *= error/(1-error)
        w = normalize(w)
        z[i] =  np.log2((1-error)/error)
    return weighted_majority(h,z)

#The first "n-1" words in the vocabulary will be skipped
n = 60
#Every word after "m+n" won't be checked.
m = 55
#Number of iterations Adaboost will perfrom
iterations = 100

train_data = open("aclImdb/train/labeledBow.feat","r")
train_samples = generate_samples(train_data.readlines(),n,m)
#print(train_samples[0])

adaboost_clf = adaboost(train_samples,iterations)