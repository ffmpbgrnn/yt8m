import h5py
import numpy as np

data = h5py.File("/data/state/linchao/YT/hdfs/video/train/mean.h5", 'r')


num_classes = 4716
classes = np.arange(num_classes)
np.random.shuffle(classes)

