class Config(object):
  if phase_train:
    num_readers = 8
    num_epochs = None
  else:
    num_readers = 1
    num_epochs = 1

  feature_names = ""
  feature_sizes = ""
  use_frame_features = ""
  model_name = ""
  label_loss = "CrossEntropyLoss"

  optimizer = ""
  base_learning_rate = 1e-4

  # train
  start_new_model = False

  # eval
  top_k = 20

  # inference
