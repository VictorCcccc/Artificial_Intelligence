# TextClassifier.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Dhruv Agarwal (dhruva2@illinois.edu) on 02/21/2019

"""
You should only modify code within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import math
class_count = [0] * 14
wordClass_p = {}

w1_w0Class_p = {}

class TextClassifier(object):

    def __init__(self):
        """Implementation of Naive Bayes for multiclass classification

        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        self.lambda_mixture = 0.0

    def fit(self, train_set, train_label):
        """
        :param train_set - List of list of words corresponding with each text
            example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
            Then train_set := [['i','like','pie'], ['i','like','cake']]

        :param train_labels - List of labels corresponding with train_set
            example: Suppose I had two texts, first one was class 0 and second one was class 1.
            Then train_labels := [0,1]
        """

        # TODO: Write your code here
        word_count = [0] * 14
        wordClass_count = {}

        w0Class_count = {}
        w0w1Class_count = {}
        unique_word = set()
        new_map = []
        for i in range(14):
            new_map.append({})
        for i in range(len(train_label)):
            class_count[train_label[i] - 1] += 1
            word_count[train_label[i] - 1] += len(train_set[i])
            # for word in train_set[i]:
            #     unique_word.add(word)
            #     if (word, train_label[i]) not in wordClass_count.keys():
            #         wordClass_count[(word, train_label[i])] = 1
            #     else:
            #         wordClass_count[(word, train_label[i])] += 1

            for j in range(len(train_set[i])):
                word = train_set[i][j]
                if word not in new_map[train_label[i] - 1].keys():
                    new_map[train_label[i] - 1][word] = 0
                new_map[train_label[i] - 1][word] += 1

                unique_word.add(train_set[i][j])
                if (train_set[i][j], train_label[i]) not in wordClass_count.keys():
                    wordClass_count[(train_set[i][j], train_label[i])] = 1
                else:
                    wordClass_count[(train_set[i][j], train_label[i])] += 1

                if j < len(train_set[i]) - 1:
                    if (train_set[i][j], train_label[i]) not in w0Class_count.keys():
                        w0Class_count[(train_set[i][j], train_label[i])] = 1
                    else:
                        w0Class_count[(train_set[i][j], train_label[i])] += 1

                    if (train_set[i][j], train_set[i][j + 1], train_label[i]) not in w0w1Class_count.keys():
                        w0w1Class_count[(train_set[i][j], train_set[i][j + 1], train_label[i])] = 1
                    else:
                        w0w1Class_count[(train_set[i][j], train_set[i][j + 1], train_label[i])] += 1

        # find the top feature words
        # for i in range(14):
        #     a = []
        #     for j in new_map[i].keys():
        #         a.append(new_map[i][j])
        #     a.sort()
        #     f = a[-20]
        #     s = 0
        #     print('class', i + 1, end = ': ')
        #     for j in new_map[i].keys():
        #         if new_map[i][j] >= f and s < 20:
        #             print(j, end = ' ')
        #             s += 1
        #     print()
        
        default_p = [0] * 14
        for c in range(14):
            default_p[c] = 1 / (word_count[c] + len(unique_word))

        for key, value in wordClass_count.items():
            if key[0] not in wordClass_p.keys():
                wordClass_p[key[0]] = default_p.copy()
            wordClass_p[key[0]][key[1] - 1] = (value + 1) / (word_count[key[1] - 1] + len(unique_word))

        for key, value in w0w1Class_count.items():
            if (key[1], key[0]) not in w1_w0Class_p.keys():
                w1_w0Class_p[(key[1], key[0])] = [0] * 14
                for p in range(len(w1_w0Class_p[(key[1], key[0])])):
                    w1_w0Class_p[(key[1], key[0])][p] = 1 / (w0Class_count[(key[0], key[2])] + len(unique_word))
            w1_w0Class_p[(key[1], key[0])][key[2] - 1] = (value + 1) / (w0Class_count[(key[0], key[2])] + len(unique_word))

    def predict(self, x_set, dev_label,lambda_mix=0.0):
        """
        :param dev_set: List of list of words corresponding with each text in dev set that we are testing on
              It follows the same format as train_set
        :param dev_label : List of class labels corresponding to each text
        :param lambda_mix : Will be supplied the value you hard code for self.lambda_mixture if you attempt extra credit

        :return:
                accuracy(float): average accuracy value for dev dataset
                result (list) : predicted class for each text
        """

        accuracy = 0.0
        result = []

        # TODO: Write your code here
        correct_count = 0
        confusion_matrix = []
        for i in range(14):
            confusion_matrix.append([0] * 14)
        fs = [0] * 14

        for i in range(len(dev_label)):
            posteriors = []
            for c in range(14):
                log_sum = math.log(class_count[c]) #MAP
                log_sum_bigram = math.log(class_count[c])#MAP
                # log_sum = 0 #uniform and ML
                # log_sum_bigram = 0 #uniform and ML

                for j in range(len(x_set[i])):
                    if x_set[i][j] in wordClass_p.keys():
                        log_sum += math.log(wordClass_p[x_set[i][j]][c])
                    if j < len(x_set[i]) - 1:
                        if (x_set[i][j + 1], x_set[i][j]) in w1_w0Class_p.keys():
                            log_sum_bigram += math.log(w1_w0Class_p[(x_set[i][j + 1], x_set[i][j])][c])

                posteriors.append((1 - lambda_mix) * log_sum + lambda_mix * log_sum_bigram)
            result.append(posteriors.index(max(posteriors)) + 1)

            if posteriors.index(max(posteriors)) + 1 == dev_label[i]:
                correct_count += 1

            confusion_matrix[dev_label[i] - 1][posteriors.index(max(posteriors))] += 1
            fs[dev_label[i] - 1] += 1

        for i in range(14):
            for j in range(14):
                confusion_matrix[i][j] /= fs[i]

        accuracy = correct_count / len(dev_label)

        #print(confusion_matrix)

        return accuracy,result
