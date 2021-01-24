import numpy as np
import random as rand


class Stump:

    # Decision stump constructor, featuring a root
    # with the attribute from the samples.
    def __init__(self, root, left, right):
        self.root = root
        self.left = left
        self.right = right

    # Checks if the stump predicted correctly.
    def check_sample(self, sample):
        decision = sample[0][self.root]
        sample_clf = sample[1]
        if(decision == 1):
            return self.left == sample_clf
        else:
            return self.right == sample_clf

    # Stump print
    def print_stump(self):
        print('Root: ', self.root, ' Left: ',
              self.left, ' Right: ', self.right)


class AdaboostClf:

    def __init__(self, h, z):
        self.h = h
        self.z = z

    def test(self, sample):
        is_positive = 0
        is_negative = 0
        threshold = 0.8
        for i in range(len(self.h)):
            answer = sample[0][self.h[i].root]
            if(answer == 1):
                if(self.h[i].left):
                    is_positive += self.z[i]
                else:
                    is_negative += self.z[i]
            else:
                if(self.h[i].right):
                    is_positive += self.z[i]
                else:
                    is_negative += self.z[i]

            if is_positive == threshold*len(self.h) or is_negative == threshold*len(self.h):
                return is_positive > is_negative

        return is_positive > is_negative

# Generating sample data in tuples


def generate_samples(samples, n, m):
    data = []
    for sample in samples:
        sample_data = []
        score = int(sample.split()[0]) > 5
        for i in range(n, m+n):
            if(" "+str(i)+":" in sample):
                sample_data.append(1)
            else:
                sample_data.append(0)
        tuple = (sample_data, score)
        data.append(tuple)
    return data


def normalize(weights):
    weight_sum = 0
    for weight in weights:
        weight_sum += weight
    for i in range(len(weights)):
        weights[i] /= weight_sum
    return weights


def calculate_gini(samples, attr):

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for sample in samples:
        if(sample[0][attr] == 1):
            if(sample[1]):
                true_pos += 1
            else:
                false_pos += 1
        else:
            if(not sample[1]):
                true_neg += 1
            else:
                false_neg += 1
    pos_prob = 1 - pow((true_pos+1)/(true_pos+false_pos+2), 2) - \
        pow((false_pos+1)/(true_pos+false_pos+2), 2)
    neg_prob = 1 - pow((true_neg+1)/(true_neg+false_neg+2), 2) - \
        pow((false_neg+1)/(true_neg+false_neg+2), 2)
    impurity = (true_pos+false_pos)/(true_pos+false_pos+true_neg+false_neg) * \
        pos_prob + (true_neg+false_neg) / \
        (true_pos+false_pos+true_neg+false_neg)*neg_prob
    return impurity, true_pos > false_pos


def weak_learner(samples):

    min_gini = np.inf
    best_attr = 0
    best_has_attr = True
    # Calculating gini impurity for each
    # available attribute and chooosing
    # the one with the lowest one
    for attr in range(len(samples[0][0])):
        gini, has_attr = calculate_gini(samples, attr)
        if(gini < min_gini):
            min_gini = gini
            best_attr = attr
            best_has_attr = has_attr
    # We create the decision stump based
    # on the best attribute selected
    return Stump(best_attr, best_has_attr, not best_has_attr)


def repopulate_samples(samples, weights):
    # Preparing the weight buckets for the sample distribution.
    weight_buckets = [weights[0]]
    for i in range(1, len(weights)):
        weight_buckets.append(weights[i] + weight_buckets[i-1])
    # Filling the sample data again.
    new_samples = rand.choices(
        samples, cum_weights=weight_buckets, k=len(samples))
    # Balancing the sample weights.
    weights = [1/len(samples)]*len(samples)
    return new_samples, weights


def weighted_majority(h, z):
    return AdaboostClf(h, z)


def adaboost(samples, iterations):
    # Number of total weights
    w = [1/len(samples)]*len(samples)
    # Number of hypotheses we've learned.
    h = []
    # The ammount of say for each hypothesis.
    z = []
    for i in range(iterations):
        # After the first loop we have to keep repopulating the samples according to their weights.
        if i != 0:
            samples, w = repopulate_samples(samples, w)
        stump = weak_learner(samples)
        h.append(stump)
        error = 0
        # Calculating total error from every sample.
        for j in range(len(samples)):
            if not h[i].check_sample(samples[j]):
                error += w[j]
        # We add the stump weight (amount of say) to the list, if the error is 0.5 then we don't take
        # the stump into account.
        if error == 0.5:
            z.append(0)
        else:
            z.append(0.5*np.log(((1-error)/(error+0.0001))+1))
        # Updating the sample weights based on the erros we made previously.
        for j in range(len(samples)):
            if h[i].check_sample(samples[j]):
                w[j] *= np.exp(-z[i])
            else:
                w[j] *= np.exp(z[i])
        # Normalizing the sample weights so they add up to 1.
        w = normalize(w)
    return weighted_majority(h, z)


def run_tests(adaboost, tests):
    accuracy = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for test in tests:
        clf = adaboost.test(test)
        if(clf == test[1]):
            accuracy += 1
            if(clf == True):
                true_pos += 1
            else:
                true_neg += 1
        else:
            if(clf):
                false_pos += 1
            else:
                false_neg += 1
    precision = true_pos/(true_pos + false_pos)
    recall = true_pos/(true_pos + false_neg)
    f1 = 2*(recall*precision)/(recall+precision)
    print('Precision: ', precision,)
    print('Recall:', recall)
    print('F1: ', f1)
    print('Accuracy: ', accuracy/len(tests)*100, "%")


# The first "n-1" words in the vocabulary will be skipped
n = 75
# Every word after "m+n" won't be checked.
m = 400
# Number of iterations Adaboost will perfrom
iterations = 30

# Loading the training data
train_data = open("aclImdb/train/labeledBow.feat", "r")
train_samples = generate_samples(train_data.readlines(), n, m)
train_samples = train_samples[0:4000] + train_samples[12500:16500]

# Loading the test data
adaboost_clf = adaboost(train_samples, iterations)
test_data = open("aclImdb/test/labeledBow.feat", "r")
test_samples = generate_samples(test_data.readlines(), n, m)
test_samples = test_samples[0:1500] + test_samples[12500:14000]
run_tests(adaboost_clf, test_samples)
