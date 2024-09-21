from collections import Counter
from random import sample
from hyperparameter import weighted_counts_and_most_common, partition_examples_categorically, partition_examples_numerically, info_gain


class DecisionTree:
    def __init__(self, max_depth, depth, examples, gain_parameter, most_common_label, is_categoric_attribute, random_subset=False, random_subset_size=0):
        self.depth = depth
        self.max_depth = max_depth

        if len(examples) == 0:
            self.decide = lambda _ : most_common_label
            return

        _, self.most_common_label, _ = weighted_counts_and_most_common(examples)
        if depth == max_depth:
            self.decide = lambda _ : self.most_common_label
            return

        best_attribute = DecisionTree.best_attr(gain_parameter, examples, is_categoric_attribute) \
            if not random_subset else DecisionTree.best_attr_subset_rand(gain_parameter, examples, is_categoric_attribute, random_subset_size)
        self.splitting_attribute = best_attribute
        if is_categoric_attribute[best_attribute]:
            categorical_partitions = partition_examples_categorically(examples, best_attribute)

            self.categorical_children = dict()
            for value in categorical_partitions.keys():
                self.categorical_children[value] = DecisionTree(max_depth, depth + 1, categorical_partitions[value], gain_parameter, most_common_label, is_categoric_attribute, random_subset, random_subset_size)

            def categorical_decision(sample):
                if sample.attributes[best_attribute] in self.categorical_children.keys():
                    return self.categorical_children[sample.attributes[best_attribute]].decide(sample)
                else:
                    return self.most_common_label

            self.decide = categorical_decision
        else:
            less_than_examples, greater_than_examples, threshold = partition_examples_numerically(examples, best_attribute)

            self.less_child = DecisionTree(max_depth, depth + 1, less_than_examples, gain_parameter, self.most_common_label, is_categoric_attribute, random_subset, random_subset_size)
            self.greater_child = DecisionTree(max_depth, depth + 1, greater_than_examples, gain_parameter,
                                              self.most_common_label, is_categoric_attribute, random_subset, random_subset_size)

            def numeric_decision(sample):
                if isinstance(sample.attributes[best_attribute], float):
                    if threshold < sample.attributes[best_attribute]:
                        return self.greater_child.decide(sample)
                    else:
                        return self.less_child.decide(sample)
                else:
                    return self.most_common_label

            self.decide = numeric_decision

    @classmethod
    def best_attr(cls, gain_parameter, examples, attr_is_num):
        curr_best = -1.0
        curr_best_gain = float("-inf")
        for attr_i in range(len(examples[0].attributes)):
            gain = info_gain(examples, attr_i, attr_is_num, gain_parameter)
            if gain > curr_best_gain:
                curr_best = attr_i
                curr_best_gain = gain
        return curr_best

    @classmethod
    def best_attr_subset_rand(cls, gain_parameter, examples, attr_is_num, subset_size):
        curr_best = -1.0
        curr_best_gain = float("-inf")
        num_attributes = len(examples[0].attributes)
        attribute_subset = sample(range(0,num_attributes), subset_size) if num_attributes >= subset_size else range(0,num_attributes)
        for attr_i in attribute_subset:
            gain = info_gain(examples, attr_i, attr_is_num, gain_parameter)
            if gain > curr_best_gain:
                curr_best = attr_i
                curr_best_gain = gain
        return curr_best

    @classmethod
    def most_common_label(cls, examples):
        labels = list()
        for sample in examples:
            labels.append(sample.label)
        return Counter(labels).most_common(1)[0][0]
