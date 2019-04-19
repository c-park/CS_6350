#!/usr/bin/env python
"""SVM Algorithms

Various SVM algorithms for Machine Learning

Author: Cade Parkison
University of Utah
Machine Learning
"""

import numpy as np
import pandas as pd
import cvxopt


class SVM(object):

    def __init__(self, no_of_inputs, epoch, C, rate_schedule):
        self.epoch = epoch
        self.C = C
        self.rate_schedule = rate_schedule
        self.weights = np.zeros(no_of_inputs + 1)  # initialize weights to zero

    def predict(self, X):
        # predicts the label of one training example input with current weights
        return np.sign(np.dot(X, self.weights[:-1]) + self.weights[-1])

    def train(self, X, y):

        N = y.shape[0]

        #labels = np.expand_dims(labels, axis=1)
        data = np.hstack((X, y))

        for e in range(self.epoch):
            #print("Epoch: "+ str(e))
            #print("Weights: " + str(self.weights))
            # print('')
            rate = self.rate_schedule(e)
            np.random.shuffle(data)
            for i, row in enumerate(data):
                x = row[:-1]
                y = row[-1]
                val = y*(np.dot(x, self.weights[:-1]) + self.weights[-1])
                if val <= 1:
                    self.weights[:-1] = (1-rate) * \
                        self.weights[:-1] + rate*self.C*N*y*x
                    self.weights[-1] = rate*self.C*N*y
                else:
                    self.weights[:-1] = (1-rate)*self.weights[:-1]

        return self.weights

    def evaluate(self, X, y):
        # calculates average prediction error on testing dataset
        errors = []
        for inputs, label in zip(X, y):
            prediction = self.predict(inputs)
            if np.sign(prediction) != label:
                errors.append(1)
            else:
                errors.append(0)

        return 100*(sum(errors) / float(X.shape[0]))


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def gaussian_kernel(gamma=0.5):
    return lambda x, y: np.exp(-np.linalg.norm(x-y)**2 / gamma)


class DualSVM(object):

    def __init__(self, C, kernel=linear_kernel):
        self.C = C
        self.kernel = kernel

    def predict(self, inputs):
        # predicts the label of one training example input with current weights
        if self.kernel == linear_kernel:
            return np.sign(np.dot(inputs, self.weights[:-1]) + self.weights[-1])
        else:
            result = 0
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                result += a * sv_y * self.kernel(inputs, sv)

            return np.sign(result).item()

    def train(self, X, y):
        n_samples, n_features = X.shape

        # Kernel Matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y)*K)
        q = cvxopt.matrix(-1*np.ones(n_samples))
        G = cvxopt.matrix(
            np.vstack((np.diag(-1*np.ones(n_samples)), np.identity(n_samples))))
        h = cvxopt.matrix(
            np.hstack((np.zeros(n_samples), self.C*np.ones(n_samples))))
        A = cvxopt.matrix(y, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)

        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['abstol'] = 1e-10
        cvxopt.solvers.options['reltol'] = 1e-10
        cvxopt.solvers.options['feastol'] = 1e-10

        # Quadratic Programming solution from cvxopt
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange Multipliers
        alphas = np.array(sol['x'])

        # weights
        w = ((y * alphas).T @ X).reshape(-1, 1)
        # non-zero alphas
        S = (alphas > 1e-4).flatten()
        self.S = S
        self.n_supports = np.sum(S)
        # intercept
        b = y[S] - np.dot(X[S], w)

        ind = np.arange(len(alphas))[S]
        self.a = alphas[S]
        self.sv = X[S]

        self.sv_y = y[S]

        self.b = 0
        for n in range(len(self.a)):
            self.b += float(self.sv_y[n])
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], S])
        self.b /= len(self.a)

        self.weights = np.zeros(n_features + 1)
        self.weights[:-1] = w.flatten()
        self.weights[-1] = b[0]

    def evaluate(self, X, y):
        # calculates average prediction error on dataset, in percentage
        errors = []
        for inputs, label in zip(X, y):
            prediction = self.predict(inputs)
            if np.sign(prediction) != label:
                errors.append(1)
            else:
                errors.append(0)

        return 100*(sum(errors) / float(X.shape[0]))
