# Copyright 2017 Yurui Ming (yrming@gmail.com) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for configuring the network."""

import tensorflow as tf
from tensorflow.python.platform import flags

flags.DEFINE_float('decay', 0.25, 'The decay factor of vigilance tracing process.')

# for supervised learning
flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the optimizer.')
flags.DEFINE_boolean('lr_decay', True, 'whether or not do learning rate decay')
flags.DEFINE_float('lr_decay_factor', 0.8, 'learning rate decay factor')
flags.DEFINE_float('lr_decay_steps', 100, 'after the specified steps then learning rate decay')
flags.DEFINE_integer('num_iterations', 1000, 'number of training iterations.')

flags.DEFINE_integer('batch_size', 32, 'The mini-batch size.')

'''
# sampling rate: 250Hz
flags.DEFINE_integer('num_chans', 30, 'Number of color channels in input.')
flags.DEFINE_integer('num_points', 750, 'Screen dimensions.')
flags.DEFINE_integer('num_filters', 32, 'number of filters')
flags.DEFINE_integer('sampling_rate', 250, 'sampling rate')
flags.DEFINE_integer('pooling_stride', 2, 'pooling stride')
'''

'''
# sampling rate: 100Hz
flags.DEFINE_integer('num_chans', 30, 'Number of color channels in input.')
flags.DEFINE_integer('num_points', 300, 'Screen dimensions.')
flags.DEFINE_integer('num_filters', 32, 'number of filters')
flags.DEFINE_integer('sampling_rate', 100, 'sampling rate')
flags.DEFINE_integer('pooling_stride', 2, 'pooling stride')
'''


# sampling rate: 128Hz
flags.DEFINE_integer('num_chans', 30, 'Number of color channels in input.')
flags.DEFINE_integer('num_points', 384, 'Screen dimensions.')
flags.DEFINE_integer('num_filters', 16, 'number of filters')
flags.DEFINE_integer('sampling_rate', 128, 'sampling rate')
flags.DEFINE_integer('pooling_stride', 2, 'pooling stride')


# model parameters
flags.DEFINE_integer('num_recurs', 3, 'Number of recursion.')

flags.DEFINE_boolean('debug', False, 'debug the training procedure')

# general configurations below
flags.DEFINE_string('gpus', '', 'visible GPU list')
flags.DEFINE_string('data_dir', '', 'directory for data.')
flags.DEFINE_string('output_dir', '', 'directory for model outputs.')
flags.DEFINE_string('checkpoint_dir', '', 'directory of checkpoint files.')

flags.DEFINE_integer('summary_interval', 5, 'how often to record tensorboard summaries.')

flags.DEFINE_boolean('validation', True, 'whether do cross-validation or not')
flags.DEFINE_integer('validation_interval', 10, 'how often to run a batch through the validation model')
# flags.DEFINE_integer('save_interval', 2000, 'how often to save a model checkpoint.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


