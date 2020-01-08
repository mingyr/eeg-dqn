'''
@author: MSUser
'''
import os
import numpy as np
import tensorflow as tf
import sonnet as snt 
import scipy.io
import skimage.io
import matlab.engine as engine

class Input(snt.AbstractModule):
    def __init__(self, batch_size, data_shape = None,
                 num_epochs = -1, name = 'input'):
        '''
        Args:
            batch_size: number of tfrecords to dequeue
            data_shape: the expected shape of series of images
            num_enqueuing_threads: enqueuing threads
        '''
        super(Input, self).__init__(name = name)

        self._batch_size = batch_size
        self._data_shape = data_shape
        self._num_epochs = num_epochs

        assert data_shape, "invalid data shape" 

    def _parse_function(self, example):
        dims = np.prod(self._data_shape)

        features = { "input": tf.FixedLenFeature([dims], dtype = tf.float32),
                     "value": tf.FixedLenFeature([], dtype = tf.float32) }

        example_parsed = tf.parse_single_example(serialized = example,
                                                 features = features)
		
        data = tf.reshape(example_parsed["input"], self._data_shape)

        return data, example_parsed["value"]

    def _build(self, filenames):
        '''
        Retrieve tfrecord from files and prepare for batching dequeue
        Args:
            filenames: 
        Returns:
            wave label in batch
        '''

        assert os.path.isfile(filenames), "invalid file path: {}".format(filenames)
	
        if type(filenames) == list:
            dataset = tf.data.TFRecordDataset(filenames)
        elif type(filenames) == str:
            dataset = tf.data.TFRecordDataset([filenames])
        else:
            raise ValueError('wrong type {}'.format(type(filenames)))

        dataset = dataset.map(self._parse_function)
        dataset = dataset.shuffle(self._batch_size * 20)
        dataset = dataset.batch(self._batch_size, drop_remainder = True)
        dataset = dataset.repeat(self._num_epochs)

        iterator = dataset.make_one_shot_iterator()
        data, values = iterator.get_next()

        return data, values

def test_input():
    from config import FLAGS

    input_ = Input(32, [FLAGS.num_chans, FLAGS.num_points])
    data, values = input_('data-supervised/eeg-train.tfr')

    with tf.Session() as sess:

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        for i in range(20):
            data_val, values_val = sess.run([data, values])

            # np.set_printoptions(threshold = np.nan)

            print(data_val.shape)
            print(values_val.shape)

if __name__ == '__main__':
    test_input()


