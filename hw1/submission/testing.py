"""
    Testing script for Machine Learning - Decision Tree algorithm

    Author: Cade Parkison
"""

from decision_tree import *


# Car Data set

test_data_c = pd.read_csv('car/test.csv', header=None)
train_data_c = pd.read_csv('car/train.csv', header=None)

train_data_c.columns = ['buying', 'maint', 'doors',
                        'persons', 'lug_boot', 'safety', 'label']
test_data_c.columns = ['buying', 'maint', 'doors',
                       'persons', 'lug_boot', 'safety', 'label']

attrs_c = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']


# Entropy results
trees_entropy_car = []
train_entropy_car = []
test_entropy_car = []

for i in range(6):
    tree = id3(train_data_c, train_data_c, attrs_c, 'label',
               gain_method=entropy, parent_label=None, max_depth=i+1)
    trees_entropy_car.append(tree)
    train_entropy_car.append(evaluate(train_data_c, tree, 'y'))
    test_entropy_car.append(evaluate(test_data_c, tree, 'y'))


# Majority Error results
trees_me_car = []
train_me_car = []
test_me_car = []

for i in range(6):
    tree = id3(train_data_c, train_data_c, attrs_c, 'label',
               gain_method=maj_error, parent_label=None, max_depth=i+1)
    trees_me_car.append(tree)
    train_me_car.append(evaluate(train_data_c, tree, 'y'))
    test_me_car.append(evaluate(test_data_c, tree, 'y'))


# Gini Index results
trees_gi_car = []
train_gi_car = []
test_gi_car = []

for i in range(6):
    tree = id3(train_data_c, train_data_c, attrs_c, 'label',
               gain_method=gini, parent_label=None, max_depth=i+1)
    trees_gi_car.append(tree)
    train_gi_car.append(evaluate(train_data_c, tree, 'y'))
    test_gi_car.append(evaluate(test_data_c, tree, 'y'))


# Bank Data set
test_data_b = pd.read_csv('bank/test.csv', header=None)
train_data_b = pd.read_csv('bank/train.csv', header=None)

test_data_b.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
train_data_b.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']


# convert numerical values to binary
num_2_binary(test_data_b)
num_2_binary(train_data_b)

attrs_b = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
           'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']


# Entropy Results
trees_entropy_b = []
train_entropy_b = []
test_entropy_b = []

for i in range(16):
    tree = id3(train_data_b, train_data_b, attrs_b, 'y',
               gain_method=entropy, parent_label=None, max_depth=i+1)
    trees_entropy_b.append(tree)
    train_entropy_b.append(evaluate(train_data_b, tree, 'y'))
    test_entropy_b.append(evaluate(test_data_b, tree, 'y'))


# Majority Error Results
trees_me_b = []
train_me_b = []
test_me_b = []

for i in range(16):
    tree = id3(train_data_b, train_data_b, attrs_b, 'y',
               gain_method=maj_error, parent_label=None, max_depth=i+1)
    trees_me_b.append(tree)
    train_me_b.append(evaluate(train_data_b, tree, 'y'))
    test_me_b.append(evaluate(test_data_b, tree, 'y'))


# Gini Index Results
trees_gi_b = []
train_gi_b = []
test_gi_b = []

for i in range(16):
    tree = id3(train_data_b, train_data_b, attrs_b, 'y',
               gain_method=gini, parent_label=None, max_depth=i+1)
    trees_gi_b.append(tree)
    train_gi_b.append(evaluate(train_data_b, tree, 'y'))
    test_gi_b.append(evaluate(test_data_b, tree, 'y'))
