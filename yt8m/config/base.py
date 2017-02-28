import os
import subprocess

def mkdir(mdir):
  if not os.path.exists(mdir):
    os.mkdir(mdir)

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
    mkdir(run_dir)
  return max_run_id, run_dir

def execute_shell(cmds, wait=True):
  p = subprocess.Popen(cmds, stdout=subprocess.PIPE,
                       shell=True, preexec_fn=os.setsid)
  p.wait()


class BaseConfig(object):
  def __init__(self, stage):
    model_names = [
        "FrameLevelLogisticModel",
        "LSTMEncoder",
        "LSTMEncDec",
        "LogisticModel",
    ]
    self.model_name = "LogisticModel"
    self.use_frame_features = False
    if self.use_frame_features:
      self.feature_names = "rgb"
      self.feature_sizes = "1024"
      # self.feature_names = "rgb, audio"
      # self.feature_sizes = "1024, 128"
    else:
      self.feature_names = "mean_rgb"
      self.feature_sizes = "1024"

    self.stage = stage
    self.input_setup()

    if self.phase_train:
      self.num_readers = 8
      self.num_epochs = None
      self.batch_size = 128
    else:
      self.num_readers = 1
      self.num_epochs = 1
      self.batch_size = 512

    self.label_loss = "CrossEntropyLoss"

    self.regularization_penalty = 1

    self.start_new_model = False
    self.top_k = 20


  def input_setup(self):
    train_dir = "/data/D2DCRC/linchao/YT/log/"
    code_saver_dir = "/data/uts221/linchao/yt8m_src_log"
    # code_saver_dir = "/data/uts711/linchao/yt8m_src_log"
    if self.stage == "train":
      self.phase_train = True
      data_pattern_str = "train"
      run_id, self.train_dir = get_max_run_id(train_dir)

      code_saver_dir = os.path.join(code_saver_dir, run_id)
      mkdir(code_saver_dir)
      pwd = os.path.dirname(os.path.abspath(__file__))
      # execute_shell("git checkout -b {}; git commit -v -a -m 'model id: {}'".format(
          # self.run_id, self.run_id))
      execute_shell("cp -ar {0}/../../../src {1}".format(
          pwd, os.path.join(code_saver_dir)))
    elif self.stage == "eval" or self.stage == "inference":
      self.phase_train = False
      data_pattern_str = "validate" if self.stage == "eval" else "test"

    if self.use_frame_features:
      self.data_pattern = "/data/state/linchao/YT/{0}/{0}*.tfrecord".format(data_pattern_str)
      # self.data_pattern = "/data/uts700/linchao/yt8m/data/{0}/{0}*.tfrecord".format(data_pattern_str)
    else:
      self.data_pattern = "/data/uts221/linchao/YT/video_level/{0}/{0}*.tfrecord".format(data_pattern_str)
