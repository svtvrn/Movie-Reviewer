import numpy as np
import random as rand


def generate_samples(samples, n, m):
    dictionaries = []
    # We create a dictionary for each review,
    # where the first element of each dictionary
    # is the class of the review: good/bad score.
    # The rest of the elements have keys reffering
    # to the attribute number and 0 or 1 as values.
    for sample in samples:
        sample_dict = {}
        score = int(sample.split()[0])
        if score > 6:
            sample_dict.update({'clf': True})
        else:
            sample_dict.update({'clf': False})
        for i in range(n, m+n):
            if(" "+str(i)+":" in sample):
                sample_dict[i] = 1
            else:
                sample_dict[i] = 0
        dictionaries.append(sample_dict)
    return dictionaries


def check_threshold(samples):
    threshold = 1
    neg = 0
    pos = 0
    for i in range(0, len(samples)):
        if samples[i].get('clf'):
            pos += 1
        else:
            neg += 1
        if pos >= threshold*len(samples):
            return True
        elif neg >= threshold*len(samples):
            return True
    return False


def check_categories(samples):
    category = samples[0].get('clf')
    for i in range(1, len(samples)):
        if samples[i].get('clf') != category:
            return False
    return True


def most_frequent(samples):
    # Counters to keep track of the
    # number of appearances of each
    # review class: positive/negative.
    pos_counter = 0
    neg_counter = 0
    for sample in samples:
        if sample.get('clf'):
            pos_counter += 1
        else:
            neg_counter += 1
    # We return the category with
    # the most class appearances
    # as the most frequent one.
    return pos_counter > neg_counter


def attr_entropy(samples, attribute, value, C):
    # Calculating the attribute entropy
    # for the specific class.
    current_probability = 0
    score_appearances = 0
    total_appearances = 0
    for sample in samples:
        score = sample.get('clf')
        if(sample.get(attribute) == value):
            total_appearances += 1
            if(score == C):
                score_appearances += 1
    # Here we make sure we don't have a runtime error
    # when the number of total appearances is 0.
    if(total_appearances > 0):
        current_probability = score_appearances/total_appearances
    return current_probability*np.log2(current_probability+1)


def calculate_entropy(samples, attribute, value):
    # Calculating the entropy for both classes
    pos_review_entropy = attr_entropy(samples, attribute, value, 1)
    neg_review_entropy = attr_entropy(samples, attribute, value, 0)
    return (- pos_review_entropy - neg_review_entropy)


def calculate_ig(samples, attribute):
    # Number of attribute showing up in reviews.
    attr_appearances = 0
    # Number of positive and negative reviews.
    pos_reviews = 0
    neg_reviews = 0
    # For each review:
    for sample in samples:
        if sample.get(attribute) == 1:
            attr_appearances += 1
        if sample.get('clf') == 1:
            pos_reviews += 1
        else:
            neg_reviews += 1
    # Percentage of positive and negative
    # reviews to calculate the total entropy.
    pos_review_perc = pos_reviews/len(samples)
    neg_review_perc = neg_reviews/len(samples)
    # Calculating entropy.
    entropy = - pos_review_perc * \
        np.log2(pos_review_perc+1) - neg_review_perc*np.log2(neg_review_perc+1)
    # Probability of finding and not finding the attribute in a review.
    attr_probability = attr_appearances/len(samples)
    # Calculating overall information gain.
    info_gain = (entropy - attr_probability*calculate_entropy(samples,
                                                              attribute, 1) - (1-attr_probability)*calculate_entropy(samples, attribute, 0))
    return info_gain


def select_best_attr(samples, attributes):
    # If there is only one attribute left
    # we return that attribute.
    if len(attributes) == 1:
        return attributes[0]
    max_attribute = 0
    max_info_gain = 0
    # We select the attribute with
    # the highest information gain.
    for attr in attributes:
        info_gain = calculate_ig(samples, attr)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_attribute = attr
    return max_attribute


def train_id3(samples, attributes, depth, freq_category):
    # Here we check wether we've reached
    # the maximum tree depth or not.
    if(depth == 0):
        return freq_category
    elif(len(samples) == 0):
        return freq_category
    elif(check_categories(samples)):
        return most_frequent(samples)
  # elif(check_threshold(samples)):               *for metrics only, comment out*
      # return most_frequent(samples)             *check_categories() to use*
    elif(len(attributes) == 0):
        return most_frequent(samples)
    else:
        # We calculate the best word to split on.
        best_attr = select_best_attr(samples, attributes)
        # We remove that attribute since it was used.
        attributes.remove(best_attr)
        root = [best_attr]
        m = most_frequent(samples)
        left_samples = []
        right_samples = []
        # We split the data to be passed on to
        # the child nodes, based on the attribute.
        for sample in samples:
            attr = sample.get(best_attr)
            if(attr == 1):
                left_samples.append(sample)
            else:
                right_samples.append(sample)
        # We create and append the sub trees.
        left_tree = train_id3(left_samples, attributes, depth-1, m)
        root.append(left_tree)
        right_tree = train_id3(right_samples, attributes, depth-1, m)
        root.append(right_tree)
        return root


def traverse(test, node):
    # Decision tree traversal to a leaf.
    if(node == True or node == False):
        # print(node)
        return node
    else:
        answer = test.get(node[0])
        #print('Node: ',node,'Attr: ',node[0],'Direction: ',answer)
        if(answer == 1):
            return traverse(test, node[1])
        elif(answer == 0):
            return traverse(test, node[2])


# Runs traverse for each test sample and computes the program's accuracy,
# as well as the number of positive/negative reviews it guessed correctly.
def run_tests(tests, forest):
    # Counters used for metrics
    accuracy = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for test in tests:
        tree_predictions = []
        # We keep track of every prediction each tree
        # has made by storing it into a list.
        for tree in forest:
            tree_predictions.append(traverse(test, tree))
        pos_pred = 0
        neg_pred = 0
        # We then count how many trees have classified
        # the review as positive and how many have
        # classified it as negative.
        for prediction in tree_predictions:
            if (prediction):
                pos_pred += 1
            else:
                neg_pred += 1
        # For our final prediction we choose the
        # most common answer between the trees.
        final_prediction = pos_pred > neg_pred
        if (final_prediction == test.get('clf')):
            accuracy += 1
            if (final_prediction):
                true_pos += 1
            else:
                true_neg += 1
        else:
            if (final_prediction):
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


# The range of attributes used.
forest_range = (75, 465)
# Defines the number of attributes in a sample.
m = 100
# Defines the maximum depth of a tree.
depth = 6
# Holds all the decision trees.
random_forest = []
# Number of trees we want to create.
number_of_trees = 30

# Loading the training data.
print("Loading your data...")
train_data = open("aclImdb/train/labeledBow.feat", "r")
train_samples = generate_samples(
    train_data.readlines(), forest_range[0], forest_range[1])

print("The model is being trained...")
for i in range(number_of_trees):
    train_subset = rand.choices(train_samples, k=4000)
    # Attribute list, contains the dictionary keys.
    attributes = rand.sample(range(forest_range[0], forest_range[1]), m)
    # Training algorithm
    id3_tree = train_id3(train_subset, attributes, depth, True)
    random_forest.append(id3_tree)

print("Training complete!")
print("Testing...\n")
# Loading the testing data.
test_data = open("aclImdb/test/labeledBow.feat", "r")
test_samples = generate_samples(test_data.readlines(),
                                forest_range[0], forest_range[1])
test_samples = test_samples[0:1500] + test_samples[12500:14000]
run_tests(test_samples, random_forest)
