#!/usr/bin/env python

"""Testing functions for SVM algorithms in Machine Learning

Cade Parkison
"""

from svm import *


def test_primal_svm():
    # tuned parameters
    gamma_0 = 0.001
    d = 0.01

    # learning rate schedules
    def schedule_a(t): return gamma_0 / (1 + (gamma_0/d)*t)
    def schedule_b(t): return gamma_0 / (1 + t)

    # list of regularization parameters
    C_list = [1.0/873, 10.0/873, 50.0/873, 100.0 /
              873, 300.0/873, 500.0/873, 700.0/873]

    train_errors_a = []
    test_errors_a = []
    for c in C_list:
        svm = SVM(4, 100, c, schedule_a)
        svm.train(train_X, train_y)
        print('C {}: weights = {}'.format(c, svm.weights))
        train_errors_a.append(svm.evaluate(train_X, train_y))
        test_errors_a.append(svm.evaluate(test_X, test_y))
    print('Training Errors: {}'.format(train_errors_a))
    print('Testing Errors: {}'.format(test_errors_a))

    train_errors_b = []
    test_errors_b = []
    for c in C_list:
        svm = SVM(4, 100, c, schedule_b)
        svm.train(train_X, train_y)
        print('C {}: weights = {}'.format(c, svm.weights))
        train_errors_b.append(svm.evaluate(train_X, train_y))
        test_errors_b.append(svm.evaluate(test_X, test_y))
    print('Training Errors: {}'.format(train_errors_b))
    print('Testing Errors: {}'.format(test_errors_b))


def test_dual_svm():

    C = [100.0/873, 500.0/873, 700.0/873]

    print('Problem 3a')
    # Training and Testing errors for each C value
    train_errors = []
    test_errors = []
    for c in C:
        svm = DualSVM(c, kernel=linear_kernel)
        svm.train(train_X, train_y)
        print('C {}: weights = {}'.format(c, svm.weights))
        train_errors.append(svm.evaluate(train_X, train_y))
        test_errors.append(svm.evaluate(test_X, test_y))
    print(train_errors, test_errors)

    print('Problem 3b')
    gamma_list = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
    for g in gamma_list:
        train_errors = []
        test_errors = []
        for c in C:
            svm = DualSVM(c, kernel=gaussian_kernel(g))
            svm.train(train_X, train_y)
            train_errors.append(svm.evaluate(train_X, train_y))
            test_errors.append(svm.evaluate(test_X, test_y))
        print('Gamma {}: {}, {}'.format(g, train_errors, test_errors))

    print('Problem 3c')
    for g in gamma_list:
        supports = []
        for c in C:
            svm = DualSVM(c, kernel=gaussian_kernel(g))
            svm.train(train_X, train_y)
            supports.append(svm.n_supports)
            # train_errors.append(svm.evaluate(train_X,train_y))
            # test_errors.append(svm.evaluate(test_X,test_y))
        print('N Supports: {}'.format(supports))

    supports = []
    for g in gamma_list:
        svm = DualSVM(500/873, kernel=gaussian_kernel(g))
        svm.train(train_X, train_y)
        supports.append(svm.S)

    overlap_supports = []
    for i in range(7):
        overlap_supports.append(
            np.sum(np.logical_and(supports[i], supports[i+1])))
    print('overlap supports: {}'.format(overlap_supports))


if __name__ == "__main__":

    #
    # Data import and preprocessing
    #
    test_data = pd.read_csv('bank-note/test.csv', header=None)
    train_data = pd.read_csv('bank-note/train.csv', header=None)
    # first 7 columns are features, last column (Slump) is output
    columns = ['var', 'skew', 'curt', 'ent', 'label']
    features = columns[:-1]
    output = columns[-1]
    test_data.columns = columns
    train_data.columns = columns

    train_X = train_data.iloc[:, :-1].values
    test_X = test_data.iloc[:, :-1].values
    train_y = train_data.iloc[:, -1].values
    test_y = test_data.iloc[:, -1].values
    # Convert labels to {-1,1}
    train_y = np.array([1 if x else -1 for x in train_y])
    test_y = np.array([1 if x else -1 for x in test_y])
    # reshape to 2D array
    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    test_primal_svm()

    test_dual_svm()
