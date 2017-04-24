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
        "SkipThought",
        "Inception",
        "LSTMMemNet",
        "BinaryLogisticModel",
        "Dilation",
        "NetVLAD",
        "MoeModel",
        "MoeModel_V2",
        "NoisyLabelModel",
        "PruneCls",
        "HLSTMEncoder",
    ]
    self.label_smoothing = False

    # self.model_name = "NoisyLabelModel"
    # self.input_feat_type = "video"
    # self.use_hdfs = False

    # self.model_name = "PruneCls"
    # self.input_feat_type = "vlad"
    # self.use_hdfs = True

    # self.model_name = "MoeModel_V2"
    # self.input_feat_type = "video"
    # self.use_hdfs = False

    # self.model_name = "NetVLAD"
    # self.input_feat_type = "frame"
    # self.use_hdfs = False

    # self.model_name = "Dilation"
    # self.input_feat_type = "frame"
    # self.use_hdfs = False

    # self.model_name = "BinaryLogisticModel"
    # self.input_feat_type = "vlad"
    # self.use_hdfs = True

    # self.model_name = "MoeModel"
    # self.input_feat_type = "video"
    # self.use_hdfs = False

    # self.model_name = "LSTMMemNet"
    # self.input_feat_type = "frame"
    # self.use_hdfs = False

    # self.model_name = "SkipThought"
    # self.input_feat_type = "frame"
    # self.use_hdfs = False

    # self.model_name = "HLSTMEncoder"
    # self.input_feat_type = "frame"
    # self.use_hdfs = False

    self.model_name = "FusionModel"
    self.input_feat_type = "score"
    self.use_hdfs = False


    if self.input_feat_type == "frame":
      self.feature_names = "rgb, audio"
      self.feature_sizes = "1024, 128"
    elif self.input_feat_type == "video":
      self.feature_names = "mean_rgb, mean_audio"
      self.feature_sizes = "1024, 128"
      # self.feature_names = "mean_rgb"
      # self.feature_sizes = "1024"
    elif self.input_feat_type == "vlad":
      self.feature_names = "feas"
      self.feature_sizes = "65536"
    else:
      self.feature_names = "shit"
      self.feature_sizes = "0"

    self.stage = stage
    self.input_setup()

    if self.phase_train:
      self.num_readers = 8
      self.num_epochs = None
      if self.model_name == "LogisticModel":
        self.num_epochs = 5
      # TODO
      self.batch_size = 1024
    else:
      self.num_readers = 1
      self.num_epochs = 1
      self.batch_size = 512

    self.label_loss = "CrossEntropyLoss"

    self.regularization_penalty = 1

    self.start_new_model = False
    self.top_k = 20


  def input_setup(self):
    train_dir = "/home/linczhu/yt/log/"
    code_saver_dir = "/home/linczhu/yt/log/"
    # train_dir = "/data/D2DCRC/linchao/YT/log/"
    # code_saver_dir = "/data/D2DCRC/linchao/YT/log/"
    if self.stage == "train":
      self.phase_train = True
      data_pattern_str = "train"
      run_id, self.train_dir = get_max_run_id(train_dir)

      code_saver_dir = os.path.join(code_saver_dir, run_id)
      mkdir(code_saver_dir)
      pwd = os.path.dirname(os.path.abspath(__file__))
      # execute_shell("git checkout -b {}; git commit -v -a -m 'model id: {}'".format(
          # self.run_id, self.run_id))
      execute_shell("cd {0}/../../../ && tar cf src.tar src/ && cp src.tar {1}".format(pwd, code_saver_dir))
      # execute_shell("cp -ar {0}/../../../src {1}".format(
          # pwd, os.path.join(code_saver_dir)))
    elif self.stage == "eval" or self.stage == "inference":
      self.phase_train = False
      data_pattern_str = "validate" if self.stage == "eval" else "test"

    # data_pattern_str = "train"
    self.data_pattern = "/data/state/linchao/YT/{0}/{1}/{1}*.tfrecord".format(
        self.input_feat_type, data_pattern_str)
    # TODO
    if self.input_feat_type == "score":
      self.data_pattern = "/home/linczhu/yt/scores/fusion/*.tfrecord"
