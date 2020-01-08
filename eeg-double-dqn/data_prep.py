import os, sys, glob
import numpy as np
import scipy.io as sio
import tensorflow as tf
import random

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

flags.DEFINE_string('indir', '', 'data directory')
flags.DEFINE_string('outdir', '', 'output directory')
FLAGS = flags.FLAGS

# list file candidates

assert os.path.exists(FLAGS.indir), "invalid input directory: {}".format(FLAGS.indir)
assert os.path.exists(FLAGS.outdir), "invalid output directory: {}".format(FLAGS.outdir)

# index in python is zero based
num_chs = 30
num_pts = 128 * 3

def float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def prepare(filename, writer):
    print('processing {}'.format(filename))

    content = sio.loadmat(filename)
    data = content['data']
    values = content['RT']

    data = np.squeeze(data)
    values = np.squeeze(values)

    indices = list(range(len(values)))
    random.shuffle(indices)

    for i in range(len(values)):
        print("processing trial {}".format(i))

        feature = {
            'input': float_feature(np.reshape(data[:, :, indices[i]], [-1])),
            'value': float_feature([values[indices[i]]]),
        }

        example = tf.train.Example(features = tf.train.Features(feature = feature))
        writer.write(example.SerializeToString())
 
def main(unused_argv):
    filenames = glob.glob(os.path.join(FLAGS.indir, "*.mat"))

    writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.outdir, 'eeg.tfr'))

    for filename in filenames:
        prepare(filename, writer)

    writer.close()

if __name__ == '__main__':
    app.run()



