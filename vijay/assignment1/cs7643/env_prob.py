""" 			  		 			     			  	   		   	  			  	
Utiliy functions.  (c) 2024 Georgia Tech

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
import time, os

from models._base_network import hello_do_you_copy as bn_hello_do_you_copy
from models.softmax_regression import hello_do_you_copy as sr_hello_do_you_copy
from models.two_layer_nn import hello_do_you_copy as tl_hello_do_you_copy
from optimizer._base_optimizer import hello_do_you_copy as bo_hello_do_you_copy
from optimizer.sgd import hello_do_you_copy as sgd_hello_do_you_copy


def say_hello_do_you_copy(drive_path: str) -> None:
  print('---------- Models ------------------')
  bn_hello_do_you_copy()
  sr_hello_do_you_copy()
  tl_hello_do_you_copy()

  bn_path = os.path.join(drive_path, 'models', '_base_network.py')
  sr_path = os.path.join(drive_path, 'models', 'softmax_regression.py')
  tl_path = os.path.join(drive_path, 'models', 'two_layer_nn.py')
  base_network_edit_time = time.ctime(os.path.getmtime(bn_path))
  softmx_regression_edit_time = time.ctime(os.path.getmtime(sr_path))
  two_layer_edit_time = time.ctime(os.path.getmtime(tl_path))
  print('_base_network.py last edited on %s' % base_network_edit_time)
  print('softmax_regression.py last edited on %s' % softmx_regression_edit_time)
  print('two_layer.py last edited on %s' % two_layer_edit_time)

  print()
  print('---------- Optimizer ------------------')
  bo_hello_do_you_copy()
  sgd_hello_do_you_copy()

  bo_path = os.path.join(drive_path, 'optimizer', '_base_optimizer.py')
  sgd_path = os.path.join(drive_path, 'optimizer', 'sgd.py')
  base_optimizer_edit_time = time.ctime(os.path.getmtime(bo_path))
  sgd_edit_time = time.ctime(os.path.getmtime(sgd_path))
  print('_base_optimizer.py last edited on %s' % base_optimizer_edit_time)
  print('sgd.py last edited on %s' % sgd_edit_time)
