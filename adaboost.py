import numpy as np

class Stump:

    def __init__(self,root):
        self.root = root
        self.left = True
        self.right = False
    
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

def gini(samples,attr):
    sample_appearances = 0
    for sample in samples:
        if sample[1]==True and sample[0][attr]==1:
            sample_appearances+=1
    attr_prob = sample_appearances/len(samples)
    gini = 1 - pow(attr_prob,2) - pow(1-attr_prob,2)
    return gini

def weak_learner(samples,weights):
    min_gini = np.inf
    best_attr = 0 
    for attr in len(samples[0][0]):
        gini = gini(samples,attr)
        if(gini<min_gini):
            min_gini = gini
            best_attr = attr
    clf = Stump(best_attr)
    return clf

def weighted_majority(h,z):
    pass

#Adaboost training function
def adaboost(samples,iterations):

    #Number of total weights
    w = [1/len(samples)]*len(samples)
    #Number of hypotheses we've learned
    h = []
    #The weights corresponding to each hypothesis
    z = []
    for i in range (iterations):
        h[i] = weak_learner(samples,w)
        error = 0
        for j in range (len(samples)):
            if h[i].check_sample(j) != samples[j][1]:
                error += w[j]
            else:
                w[j] *= error/(1-error)
        z[i] = np.log2((1-error)/error)
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