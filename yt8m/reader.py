import tensorflow as tf
for example in tf.python_io.tf_record_iterator("/data/uts700/linchao/yt8m/data/train/trainzv.tfrecord"):
  print tf.train.Example.FromString(example)
