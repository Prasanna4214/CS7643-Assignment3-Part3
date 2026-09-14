""" 			  		 			     			  	   		   	  			  	
Utiliy functions.  (c) 2021 Georgia Tech

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

import time
import numpy as np
import random
import json
import pathlib
import copy
import os

from models.softmax_regression import SoftmaxRegression
from models.two_layer_nn import TwoLayerNet
from optimizer.sgd import SGD
from data.data_processing import load_mnist_trainval, generate_batched_data
from cs7643.mnist import MNIST

import matplotlib.pyplot as plt

def train(epoch, batched_train_data, batched_train_label, model, optimizer, debug=True):
    """
    A training function that trains the model for one epoch
    :param epoch: The index of current epoch
    :param batched_train_data: A list containing batches of images
    :param batched_train_label: A list containing batches of labels
    :param model: The model to be trained
    :param optimizer: The optimizer that updates the network weights
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    """
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0
    for idx, (input, target) in enumerate(zip(batched_train_data, batched_train_label)):

        start_time = time.time()
        loss, accuracy = model.forward(input, target)

        optimizer.update(model)
        epoch_loss += loss
        hits += accuracy * input.shape[0]
        count_samples += input.shape[0]

        forward_time = time.time() - start_time
        if idx % 10 == 0 and debug:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Batch Time {batch_time:.3f} \t'
                   'Batch Loss {loss:.4f}\t'
                   'Train Accuracy ' + "{accuracy:.4f}" '\t').format(
                epoch, idx, len(batched_train_data), batch_time=forward_time,
                loss=loss, accuracy=accuracy))
    epoch_loss /= len(batched_train_data)
    epoch_acc = hits / count_samples

    if debug:
        print("* Average Accuracy of Epoch {} is: {:.4f}".format(epoch, epoch_acc))
    return epoch_loss, epoch_acc


def evaluate(batched_test_data, batched_test_label, model, debug=True):
    """
    Evaluate the model on test data
    :param batched_test_data: A list containing batches of test images
    :param batched_test_label: A list containing batches of labels
    :param model: A pre-trained model
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    """
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0
    for idx, (input, target) in enumerate(zip(batched_test_data, batched_test_label)):

        loss, accuracy = model.forward(input, target, mode='valid')

        epoch_loss += loss
        hits += accuracy * input.shape[0]
        count_samples += input.shape[0]
        if debug:
            print(('Evaluate: [{0}/{1}]\t'
                   'Batch Accuracy ' + "{accuracy:.4f}" '\t').format(
                idx, len(batched_test_data), accuracy=accuracy))
    epoch_loss /= len(batched_test_data)
    epoch_acc = hits / count_samples

    return epoch_loss, epoch_acc


def plot_curve(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, plot_type='accuracy'):
    """
    Plot learning curves with matplotlib. Make sure training loss and validation loss are plot in the same figure and
    training accuracy and validation accuracy are plot in the same figure too.
    :param train_loss_history: training loss history of epochs
    :param train_acc_history: training accuracy history of epochs
    :param valid_loss_history: validation loss history of epochs
    :param valid_acc_history: validation accuracy history of epochs
    :return: None, save two figures in the current directory
    """
    if plot_type == "loss":
      epochs = range(len(train_loss_history))
      plt.plot(epochs, train_loss_history, label='train')
      plt.plot(epochs, valid_loss_history, label='valid')
    else:
      epochs = range(len(train_acc_history))
      plt.plot(epochs, train_acc_history, label='train')
      plt.plot(epochs, valid_acc_history, label='valid')

    basedir = pathlib.Path(__file__).parent.parent.resolve()

    plt.xlabel('Epochs')
    plt.ylabel(f'{plot_type}')
    plt.legend()
    plt.title(f'{plot_type} Curve')
    plt.savefig(str(basedir) + f'/{plot_type}.png')
    plt.show()

def run_experiment(configuration_file):
  basedir = pathlib.Path(__file__).parent.parent.resolve()

  if len(configuration_file) <= 0:
    print('please select a configuration to run')
    return
  
  config_file = str(basedir) + "/configs/" + configuration_file + ".json"

  print("Training a model using configuration file " + config_file)

  with open(config_file, "r") as read_file:
    config = json.load(read_file)

  training_config = config["Train"]
  model_config = config["Model"]
  print(training_config)
  print(model_config)

  #Load MNIST data
  train_ds = MNIST(str(basedir) + '/data', download=True, train=True)
  test_ds = MNIST(str(basedir) + '/data', download=True, train=False)
  train_data, train_label, val_data, val_label = load_mnist_trainval(train_ds.train_data, train_ds.train_labels)
  
  # Create a model
  if model_config["type"] == 'SoftmaxRegression':
    model = SoftmaxRegression()
  elif model_config["type"] == 'TwoLayerNet':
    model = TwoLayerNet(hidden_size=model_config["hidden_size"])

  # Optimizer
  optimizer = SGD(learning_rate=training_config["learning_rate"], reg=training_config["reg"])

  train_loss_history = []
  train_acc_history = []
  valid_loss_history = []
  valid_acc_history = []
  best_acc = 0.0
  best_model = None
  for epoch in range(training_config["epochs"]):

      batched_train_data, batched_train_label = generate_batched_data(train_data, train_label,
                                                                      batch_size=training_config["batch_size"], shuffle=True)
      epoch_loss, epoch_acc = train(epoch, batched_train_data, batched_train_label, model, optimizer, training_config["debug"])

      train_loss_history.append(epoch_loss)
      train_acc_history.append(epoch_acc)
      # evaluate on test data
      batched_test_data, batched_test_label = generate_batched_data(val_data, val_label, batch_size=training_config["batch_size"])
      valid_loss, valid_acc = evaluate(batched_test_data, batched_test_label, model, training_config["debug"])
      if training_config["debug"]:
          print("* Validation Accuracy: {accuracy:.4f}".format(accuracy=valid_acc))

      valid_loss_history.append(valid_loss)
      valid_acc_history.append(valid_acc)

      if valid_acc > best_acc:
          best_acc = valid_acc
          best_model = copy.deepcopy(model)

  batched_test_data, batched_test_label = generate_batched_data(test_ds.test_data, test_ds.test_labels, batch_size=training_config["batch_size"])
  _, test_acc = evaluate(batched_test_data, batched_test_label, best_model)  # test the best model
  if training_config["debug"]:
      print("Final Accuracy on Test Data: {accuracy:.4f}".format(accuracy=test_acc))

  return train_loss_history, train_acc_history, valid_loss_history, valid_acc_history
