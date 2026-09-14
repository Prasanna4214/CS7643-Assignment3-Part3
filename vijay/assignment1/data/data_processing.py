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


def load_mnist_trainval(mnist_data, mnist_labels):
    """
    Load MNIST training data with labels
    :return:
        train_data: A list of list containing the training data
        train_label: A list containing the labels of training data
        val_data: A list of list containing the validation data
        val_label: A list containing the labels of validation data
    """
    print("Training data loaded with {count} images".format(count=len(mnist_data)))

    # split training/validation data
    train_data = None
    train_label = None
    val_data = None
    val_label = None
    #############################################################################
    # TODO:                                                                     #
    #    1) Split the entire training set to training data and validation       #
    #       data. Use 80% of your data for training and 20% of your data for    #
    #       validation. Note: Don't shuffle here.                               #
    #############################################################################
    
    n = len(mnist_data)
    train_size = int(0.8 * n)

    train_data = mnist_data[:train_size]
    train_label = mnist_labels[:train_size]

    val_data = mnist_data[train_size:]
    val_label = mnist_labels[train_size:]
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return train_data, train_label, val_data, val_label


def generate_batched_data(data, label, batch_size=32, shuffle=False, seed=None):
    """
    Turn raw data into batched forms
    :param data: A list of list containing the data where each inner list contains 28x28
                 elements corresponding to pixel values in images: [[pix1, ..., pix784], ..., [pix1, ..., pix784]]
    :param label: A list containing the labels of data
    :param batch_size: required batch size
    :param shuffle: Whether to shuffle the data: true for training and False for testing
    :return:
        batched_data: (List[np.ndarray]) A list whose elements are batches of images.
        batched_label: (List[np.ndarray]) A list whose elements are batches of labels.
    """
    batched_data = None
    batched_label = None
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    #############################################################################
    # TODO:                                                                     #
    #    1) Shuffle data and label if shuffle=True                              #
    #    2) Generate batches of images with the required batch size             #
    #    It's okay if the size of your last batch is smaller than the required  #
    #    batch size                                                             #
    #############################################################################
    data, label = np.asarray(data), np.asarray(label)

    data_len = len(data)

    if shuffle:
        indices = np.arange(data_len)
        np.random.shuffle(indices)
        data, label = data[indices], label[indices]

    batched_data, batched_label = [], []

    for start in range(0, data_len, batch_size):
        batched_data.append(data[start:start+batch_size])
        batched_label.append(label[start:start+batch_size])


    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return batched_data, batched_label