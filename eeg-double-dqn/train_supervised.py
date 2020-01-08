import os, sys

import tensorflow as tf
import sonnet as snt

from config_supervised import FLAGS
from input_ import Input
from model import Model
from utils import LossRegression, Summaries
from optimizer import Adam

from tensorflow.python.platform import app


def train(data_path, model, summ):
    input_ = Input(FLAGS.batch_size, [FLAGS.num_chans, FLAGS.num_points])
    data, labels = input_(data_path)
  
    ''' 
    print_op = tf.print("In train procedure -> label shape: ", tf.shape(labels), output_stream = sys.stdout)
    with tf.control_dependencies([print_op]):
        logits = model(data)
    '''

    logits = model(data)
    loss = LossRegression()(logits, tf.expand_dims(labels, axis = -1))

    opt = Adam(FLAGS.learning_rate, lr_decay = FLAGS.lr_decay, lr_decay_steps = FLAGS.lr_decay_steps,
               lr_decay_factor = FLAGS.lr_decay_factor)

    train_op = opt(loss)

    summ.register('train', 'loss', loss)
    summ.register('train', 'learning_rate', opt.lr)
    train_summ_op = summ('train')

    return loss, train_op, train_summ_op

def val(data_path, model, summ):
    input_ = Input(FLAGS.batch_size, [FLAGS.num_chans, FLAGS.num_points])
    data, labels = input_(data_path)

    '''
    print_op = tf.print("In validation procedure -> label shape: ", tf.shape(labels), output_stream = sys.stdout)
    with tf.control_dependencies([print_op]):
        logits = model(data) 
    '''

    logits = model(data) 
    loss = LossRegression()(logits, tf.expand_dims(labels, axis = -1))

    summ.register('val', 'loss', loss)

    val_summ_op = summ('val')

    return val_summ_op

def main(unused_argv):
    summ = Summaries()

    if FLAGS.data_dir == '' or not os.path.exists(FLAGS.data_dir):
        raise ValueError('invalid data directory {}'.format(FLAGS.data_dir))

    # train_data_path = os.path.join(FLAGS.data_dir, 'eeg-train.tfr')
    train_data_path = os.path.join(FLAGS.data_dir, 'eeg-train.tfr')
    val_data_path = os.path.join(FLAGS.data_dir, 'eeg-val.tfr')

    if FLAGS.output_dir == '':
        raise ValueError('invalid output directory {}'.format(FLAGS.output_dir))
    elif not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)   

    event_log_dir = os.path.join(FLAGS.output_dir, '')
    
    checkpoint_path = os.path.join(FLAGS.output_dir, 'model.ckpt')

    print('Constructing models.')

    model = Model(FLAGS.batch_size, 1, FLAGS.num_chans, FLAGS.sampling_rate,\
                  FLAGS.num_filters, FLAGS.num_recurs, FLAGS.pooling_stride, name = "model")

    model = snt.Sequential([model, tf.nn.relu])

    train_loss, train_op, train_summ_op = \
        train(train_data_path, model, summ)

    if FLAGS.validation: val_summ_op = val(val_data_path, model, summ)

    print('Constructing saver.')
    saver = tf.train.Saver()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to as some of the ops do not have GPU implementations.
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)

    assert (FLAGS.gpus != ''), 'invalid GPU specification'
    config.gpu_options.visible_device_list = FLAGS.gpus

    # Build an initialization operation to run below.
    init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    with tf.Session(config = config) as sess:
        sess.run(init)

        writer = tf.summary.FileWriter(event_log_dir, graph = sess.graph)

        # Run training.
        for itr in range(FLAGS.num_iterations):
            cost, _, train_summ_str = sess.run([train_loss, train_op, train_summ_op])
            # Print info: iteration #, cost.
            print(str(itr) + ' ' + str(cost))

            writer.add_summary(train_summ_str, itr)

            if FLAGS.validation and itr % FLAGS.validation_interval == 1:
                # Run through validation set.
                val_summ_str = sess.run(val_summ_op)
                writer.add_summary(val_summ_str, itr)

        tf.logging.info('Saving model.')
        saver.save(sess, checkpoint_path)
        tf.logging.info('Training complete')

if __name__ == '__main__':
    app.run()

