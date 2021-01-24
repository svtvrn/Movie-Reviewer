import random as rand


def generate_samples(samples, n, m):
    # We start by turning every review into
    # a vector and splitting them into two
    # categories based on their score.
    for sample in samples:
        sample_vector = []
        score = int(sample.split()[0])
        for i in range(n, m+n):
            if(" "+str(i)+":" in sample):
                sample_vector.append(1)
            else:
                sample_vector.append(0)
        if(score >= 7):
            pos_samples.append(sample_vector)
        elif(score <= 4):
            neg_samples.append(sample_vector)


def train_bayes(samples, probabilities, m):
    # We calculate the probability of an
    # attribute showing up on each of the
    # two classes and storing that value.
    for attr in range(m):
        current_probability = 0
        appearances = 0
        for sample in samples:
            if(sample[attr] == 1):
                appearances += 1
        current_probability = (appearances+1)/(len(samples)+2)
        probabilities.append(current_probability)


def naiveBayes(sample, attr_probability, m):
    # We use our previously trained model
    # to calculate the probability of that
    # test sample being either a positive
    # or a negative review.
    probability = 1
    for attr in range(m):
        if(sample[attr] == 1):
            probability *= attr_probability[attr]
        elif(sample[attr] == 0):
            probability *= (1 - attr_probability[attr])
    probability *= 0.5
    return probability


def check_accuracy(pos_prob, neg_prob, score):
    # Checking our model's prediction
    if(pos_prob > neg_prob and score > 6):
        return True
    elif(pos_prob < neg_prob and score < 5):
        return True
    return False


def run_tests(tests):
    #threshold = 1
    # Counters used for metrics.
    accuracy = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    # For each test we convert it into a vector first
    # and then we calculate the probability of that
    # review being a positive or a negative one. In the
    # end we choose the highest probability to classify
    # the test.
    for test in tests:
        # Converting the review into a vector.
        test_sample = []
        score = int(test.split()[0])
        for i in range(n, m+n):
            if(" "+str(i)+":" in test):
                test_sample.append(1)
            else:
                test_sample.append(0)
        # Calculating both probabilities for that test sample.
        pos_review_prob = naiveBayes(test_sample, pos_probabilities, m)
        neg_review_prob = naiveBayes(test_sample, neg_probabilities, m)
        # negReviewProbability *= threshold
        # We check wether the model guessed correctly or no.
        if(check_accuracy(pos_review_prob, neg_review_prob, score)):
            accuracy += 1
            if(pos_review_prob > neg_review_prob):
                true_pos += 1
            else:
                true_neg += 1
        else:
            if(pos_review_prob > neg_review_prob):
                false_pos += 1
            else:
                false_neg += 1
    # Precision, recall, f1 and accuracy calculations.
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
m = 1000
# Loading the training data.
train_data = open("aclImdb/train/labeledBow.feat", "r")
train_samples = train_data.readlines()

# Positive and negative review samples.
pos_samples = []
neg_samples = []
# These lists keep track of the probability
# of an attribute showing up on a positive
# and respectively, a negative review.
pos_probabilities = []
neg_probabilities = []

# Generating the samples and training the model,
# thus filling up our probabilities lists.
print("Loading your data...")
generate_samples(train_samples, n, m)
print("The model is being trained...")
train_bayes(pos_samples, pos_probabilities, m)
train_bayes(neg_samples, neg_probabilities, m)

print("Training complete!")
print("Testing...\n")
# Loading the testing data.
test_data = open("aclImdb/test/labeledBow.feat", "r")
test_samples = test_data.readlines()
# Choosing a subset to run tests on.
test_samples = test_samples[0:1500] + test_samples[12500:14000]
run_tests(test_samples)
