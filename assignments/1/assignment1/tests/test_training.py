""" 			  		 			     			  	   		   	  			  	
Optimizer and MLP training Tests.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import unittest
import os
import numpy as np
import pathlib

from models.softmax_regression import SoftmaxRegression
from models.two_layer_nn import TwoLayerNet

from optimizer.sgd import SGD
from data.data_processing import load_mnist_trainval, generate_batched_data
from cs7643.solver import train, evaluate
from cs7643.mnist import MNIST


class TestTraining(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """Define the functions to be tested here."""
        self.basedir = os.path.abspath(os.path.dirname(__file__))
        datadir = pathlib.Path(__file__).parent.parent.resolve()
        self.test_ds = MNIST(str(datadir) + '/data', download=False, train=False)
        self.train_ds = MNIST(str(datadir) + '/data', download=False, train=True)

    def test_regularization(self):
        optimizer = SGD(learning_rate=1e-4, reg=100)
        model = SoftmaxRegression()

        w_grad = model.gradients['W1'].copy()
        optimizer.apply_regularization(model)
        w_grad_reg = model.gradients['W1']
        reg_diff = w_grad_reg - w_grad
        expected_diff = model.weights['W1'] * optimizer.reg

        diff = np.mean(np.abs(reg_diff - expected_diff))
        self.assertAlmostEqual(diff, 0, places=7)

    def test_sgd(self):
        optimizer = SGD(learning_rate=1e-3, reg=1e-3)
        model = SoftmaxRegression()
        np.random.seed(256)
        fake_gradients = np.random.randn(784, 10)
        model.gradients['W1'] = fake_gradients
        optimizer.update(model)
        expected_weights = np.load(self.basedir + '/sgd/sgd_updated_weights.npy')
        diff = np.abs(expected_weights - model.weights['W1'])
        diff = np.sum(diff)

        self.assertAlmostEqual(diff, 0)

    def test_one_layer_train(self):
        model = SoftmaxRegression()
        optimizer = SGD(learning_rate=0.1, reg=1e-3)
        train_data, train_label, _, _ = load_mnist_trainval(self.train_ds.train_data, self.train_ds.train_labels)

        batched_train_data, batched_train_label = generate_batched_data(train_data, train_label,
                                                                        batch_size=128, shuffle=True)
        _, train_acc = train(1, batched_train_data, batched_train_label, model, optimizer, debug=False)

        batched_test_data, batched_test_label = generate_batched_data(self.test_ds.test_data, self.test_ds.test_labels, batch_size=128)
        _, test_acc = evaluate(batched_test_data, batched_test_label, model, debug=False)
        self.assertGreater(train_acc, 0.3)
        self.assertGreater(test_acc, 0.3)

    def test_two_layer_train(self):
        model = TwoLayerNet(hidden_size=128)
        optimizer = SGD(learning_rate=0.1, reg=1e-3)
        train_data, train_label, _, _ = load_mnist_trainval(self.train_ds.train_data, self.train_ds.train_labels)

        batched_train_data, batched_train_label = generate_batched_data(train_data, train_label,
                                                                        batch_size=32, shuffle=True)
        _, train_acc = train(1, batched_train_data, batched_train_label, model, optimizer, debug=False)

        batched_test_data, batched_test_label = generate_batched_data(self.test_ds.test_data, self.test_ds.test_labels, batch_size=32)
        _, test_acc = evaluate(batched_test_data, batched_test_label, model, debug=False)
        self.assertGreater(train_acc, 0.3)
        self.assertGreater(test_acc, 0.3)
