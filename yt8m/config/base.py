import os
import subprocess


def get_max_run_id(log_dir, create_dir=True):
  max_run_id = -1
  for run_id in os.listdir(log_dir):
    run_dir = os.path.join(log_dir, run_id)
    if os.path.isdir(run_dir):
      if int(run_id) > max_run_id:
        max_run_id = int(run_id)
  max_run_id = str(max_run_id + 1)
  run_dir = os.path.join(log_dir, max_run_id)
  if create_dir:
    print("Run id: {0}".format(max_run_id))
    os.mkdir(run_dir)
  return max_run_id, run_dir

def execute_shell(cmds, wait=True):
  p = subprocess.Popen(cmds, stdout=subprocess.PIPE,
                       shell=True, preexec_fn=os.setsid)
  p.wait()


class BaseConfig(object):
  def __init__(self, stage):
    train_dir = "/data/D2DCRC/linchao/YT/log/"
    if stage == "train":
      self.phase_train = True
      data_pattern_str = "train"
      _, self.train_dir = get_max_run_id(train_dir)
      pwd = os.path.dirname(os.path.abspath(__file__))
      # execute_shell("git checkout -b {}; git commit -v -a -m 'model id: {}'".format(
          # self.run_id, self.run_id))
      execute_shell("cp -ar {0}/../../src {1}".format(pwd, os.path.join(self.train_dir)))
    elif stage == "eval" or stage == "inference":
      self.phase_train = False
      data_pattern_str = "validate" if stage == "eval" else "test"

    self.stage = stage
    self.data_pattern = "/data/uts700/linchao/yt8m/data/{0}/{0}*.tfrecord".format(data_pattern_str)

    if self.phase_train:
      self.num_readers = 8
      self.num_epochs = None
    else:
      self.num_readers = 1
      self.num_epochs = 1

    self.feature_names = "rgb"
    self.feature_sizes = "1024"
    self.use_frame_features = True
    self.model_name = "FrameLevelLogisticModel"
    self.label_loss = "CrossEntropyLoss"

    self.optimizer = "AdamOptimizer"
    self.base_learning_rate = 1e-4

    # train
    self.start_new_model = False

    # eval
    self.top_k = 20

    # inference
