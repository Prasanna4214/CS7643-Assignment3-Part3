import os
import zipfile

_A1_FILES = [
    "models/softmax_regression.py",
    "models/__init__.py",
    "models/two_layer_nn.py",
    "models/_base_network.py",
    "optimizer/sgd.py",
    "optimizer/__init__.py",
    "optimizer/_base_optimizer.py",
    "data/data_processing.py"
]


def make_a1_submission(assignment_path):
  _make_submission(assignment_path, _A1_FILES, "assignment_1_submission")


def _make_submission(assignment_path, file_list, assignment_name):
  zip_path = os.path.join(assignment_path, (assignment_name + ".zip"))
  print("Writing zip file to: ", zip_path)
  with zipfile.ZipFile(zip_path, "w") as zf:
      for filename in file_list:
          in_path = os.path.join(assignment_path, filename)
          if not os.path.isfile(in_path):
              raise ValueError('Could not find file "%s"' % filename)
          zf.write(in_path, filename)
