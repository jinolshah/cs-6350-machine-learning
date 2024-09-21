from collections import Counter
from statistics import median
from math import log2
from random import sample



def info_gain(examples, attribute, is_categoric_attributes, gain_parameter):
    num_examples = len(examples)
    if num_examples == 0:
        return 0

    gain = gain_parameter(examples)

    if is_categoric_attributes[attribute]:
        categoric_partitions = partition_examples_categorically(examples, attribute)

        for partition in categoric_partitions.values():
            gain -= len(partition)/num_examples * gain_parameter(partition)

        return gain
    else:
        less_subset, greater_subset, tmp = partition_examples_numerically(examples, attribute)
        return gain - len(less_subset)/num_examples * gain_parameter(less_subset) \
                    - len(greater_subset)/num_examples * gain_parameter(greater_subset)

def entropy(examples):
    if len(examples) == 0:
        return 0
    label_counts = dict()
    total_count = 0
    for sample in examples:
        total_count += 1
        if sample.label in label_counts:
            label_counts[sample.label] += 1
        else:
            label_counts[sample.label] = 1

    entropy_val = 0
    for count in label_counts.values():
        if count == 0:
            continue
        p = count / total_count
        entropy_val += -p * log2(p)

    return entropy_val

def entropy_weighted(examples):
    if len(examples) == 0:
        return 0
    counts, _, total_count = weighted_counts_and_most_common(examples)

    entropy_val = 0
    for _, count in counts.items():
        if count == 0:
            continue
        p = count / total_count
        entropy_val += -p * log2(p)

    return entropy_val

def majority_error(examples):
    if len(examples) == 0:
        return 0
    labels = (map(lambda samp: samp.label, examples))

    num_most_common = Counter(labels).most_common(1)[0][1]

    return 1 - num_most_common/len(examples)

def majority_weighted(examples):
    if len(examples) == 0:
        return 0
    counts, most_common, total_count = weighted_counts_and_most_common(examples)

    return 1 - counts[most_common]/total_count

def gini_index(examples):
    if len(examples) == 0:
        return 0
    labels = (map(lambda samp: samp.label, examples))

    gini = 1
    num_examples = len(examples)
    for _, count in Counter(labels).most_common():
        gini -= (count/num_examples)*(count/num_examples)

    return gini

def gini_weighted(examples):
    if len(examples) == 0:
        return 0
    counts, _, total_count = weighted_counts_and_most_common(examples)

    gini = 1
    for _, count in counts.items():
        portion = count / total_count
        gini -= portion*portion
    return gini



def weighted_counts_and_most_common(examples):
    if len(examples) == 0:
        return 0
    counts = dict()
    total_count = 0
    most_common_number = 0
    most_common_label = None
    for sample in examples:
        total_count += sample.weight
        if sample.label in counts:
            counts[sample.label] += sample.weight
        else:
            counts[sample.label] = sample.weight
        if counts[sample.label] > most_common_number:
            most_common_label = sample.label
            most_common_number = counts[sample.label]
    return counts, most_common_label, total_count

def partition_examples_categorically(examples, partitioning_attribute):
    categorical_partitions = dict()
    for sample in examples:
        if sample.attributes[partitioning_attribute] not in categorical_partitions.keys():
            categorical_partitions[sample.attributes[partitioning_attribute]] = [sample]
        else:
            categorical_partitions[sample.attributes[partitioning_attribute]].append(sample)
    return categorical_partitions

def partition_examples_numerically(examples, partitioning_attribute):
    less_than_examples, greater_than_examples = list(), list()
    split = median(list(map(lambda x : x.attributes[partitioning_attribute], examples)))
    for sample in examples:
        if sample.attributes[partitioning_attribute] < split:
            less_than_examples.append(sample)
        else:
            greater_than_examples.append(sample)
    return less_than_examples, greater_than_examples, split



class Example:
    def __init__(self, values, weight=1):
        self.attributes = values[0: len(values)-1]
        self.label = values[len(values)-1]
        self.weight = weight

    def __str__(self):
        return "%ds: %s : %ds" % (self.label, str(self.weight), self.attributes)
    
    def __repr__(self):
        return self.__str__()

def extract_examples(file, unknown_is_label):
    examples = []
    num_attributes = None
    attribute_values = []
    is_categoric_attribute = []
    most_common_values = {}

    with open(file, 'r') as train_data:
        for line in train_data:
            terms = line.strip().split(',')
            if num_attributes is None:
                num_attributes = len(terms) - 1
                for idx in range(num_attributes):
                    attribute_values.append(set())
                    is_categoric_attribute.append(False)

            for idx in range(num_attributes):
                try:
                    terms[idx] = float(terms[idx])
                except:
                    is_categoric_attribute[idx] = True
                attribute_values[idx].add(terms[idx])
            else:
                sample = Example(terms)
                examples.append(sample)

    for attribute in range(num_attributes):
        if is_categoric_attribute[attribute]:
            most_common = str
            count = Counter(attribute_values[attribute])
            most_common = count.most_common(1)[0][0]
            if not unknown_is_label and most_common == "unknown":
                most_common = count.most_common(2)[0][0]
            most_common_values[attribute] = most_common
            attribute_values[attribute] = set(attribute_values[attribute])
        else:
            attribute_values[attribute] = median(attribute_values[attribute])

    if not unknown_is_label:
        for sample in examples:
            for attribute in range(num_attributes):
                if sample.attributes[attribute] == "unknown":
                    sample.attributes[attribute] = most_common_values[attribute]
    return examples, attribute_values, is_categoric_attribute

def avg_err(tree, test_examples):
    incorrect = sum(1 for test in test_examples if test.label != tree.decide(test))
    correct = len(test_examples) - incorrect
    return incorrect / (incorrect + correct) if (incorrect + correct) > 0 else 0