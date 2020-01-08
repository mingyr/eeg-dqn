# standard libraries
import os
import numpy as np
import tensorflow as tf
import sonnet as snt
import time
import trfl
import glob


# private libraries
from config import FLAGS
from env import Env
from memory import TransitionMemory
from model import Model
from utils import Summaries
from optimizer import Adam
from scipy import io

import tfmpl

'''
greedy policy
'''
def greedy(state, target_network):
    # Turn single state into minibatch.  Needed for convolutional nets.

    assert(len(state.get_shape().as_list()) == 3)

    q = target_network(state)

    maxq = tf.reduce_max(q, axis = -1)
    besta = tf.argmax(q, axis = -1)

    return besta

'''
main function
'''
def main(unused_argv):

    '''
    check path
    '''
    if FLAGS.checkpoint_dir == '' or not os.path.exists(FLAGS.checkpoint_dir):
        raise ValueError('invalid data directory {}'.format(FLAGS.checkpoint_dir))

    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, '')

    if FLAGS.output_dir == '':
        raise ValueError('invalid output directory {}'.format(FLAGS.output_dir))
    elif not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    event_log_dir = os.path.join(FLAGS.output_dir, '')
    checkpoint_path = os.path.join(FLAGS.output_dir, 'model.ckpt')

    '''
    setup summaries
    '''
    summ = Summaries()

    '''
    setup the game environment
    '''

    filenames = glob.glob(os.path.join(FLAGS.data_dir, 'test-{}'.format(FLAGS.sampling_rate), '*.mat'))
    assert(len(filenames) > 0), "invalid file names"


    game_env = Env(FLAGS.decay)
    game_actions = list(game_env.actions.keys())

    '''
    setup agent
    '''
    stateDim = [FLAGS.num_chans, FLAGS.num_points]
    input_dim = [1] + stateDim

    s_placeholder    = tf.placeholder(tf.float32, input_dim, 's_placeholder')

    network = Model(FLAGS.batch_size, len(game_actions), FLAGS.num_chans, FLAGS.sampling_rate, \
                    FLAGS.num_filters, FLAGS.num_recurs, FLAGS.pooling_stride, name = "network") 

    state_placeholder = tf.placeholder(tf.float32, input_dim, 'state_placeholder')

    action_tensor_greedy = greedy(state_placeholder, network) 

    '''
    setup the testing process
    '''

    episode_reward_placeholder = tf.placeholder(tf.float32, [], "episode_reward_placeholder")
    # summ.register('stats', 'episode_reward', episode_reward_placeholder) 

    '''
    gathering summary operators
    '''

    test_stats_op = tf.summary.text('episode_reward', tf.as_string(episode_reward_placeholder))

    '''
    setup the testing process
    '''

    saver = tf.train.Saver(network.get_all_variables())

    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)

    assert (FLAGS.gpus != ''), 'invalid GPU specification'
    config.gpu_options.visible_device_list = FLAGS.gpus

    with tf.Session(config = config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
   
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint.
            saver.restore(sess, ckpt.model_checkpoint_path)

            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/imagenet_train/model.ckpt-0,
            # extract global_step from it.
            # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            # print('Successfully loaded model from %s at step = %s.' %
            #       (ckpt.model_checkpoint_path, global_step))

            print('Successfully loaded model from %s.' % ckpt.model_checkpoint_path)

        else:
            print('No checkpoint file found')
            return
 
        for filename in filenames:
            measured_rt = {'index': [], 'value': [], 'title': 'measured RT (original)'}
            predicted_rt = {'index': [], 'value': [], 'title': 'predicted RT (original)'}

            fb, _ = os.path.splitext(filename)
            fb = os.path.basename(fb)

            f_event_log_dir = os.path.join(event_log_dir, fb)
            if not os.path.exists(f_event_log_dir):
                os.makedirs(f_event_log_dir)

            writer = tf.summary.FileWriter(f_event_log_dir, tf.get_default_graph())
        
            print("file name: {}".format(filename))

            game_env.reset(filename)

            episode_reward = 0
            count = 0
            initial_stage = True

            while True:
                print("Evaluation step: {}".format(count))

                if initial_stage:
                    action = np.random.randint(0, len(game_actions))
                    initial_stage = False
                else:
                    action = action_index

                print("action -> {}".format(game_actions[action]))

                state, reward, terminal = game_env.step(game_actions[action])

                # game over?
                if terminal:
                    break

                episode_reward += reward

                if not terminal:
                    action_index = sess.run(action_tensor_greedy, 
                                            feed_dict = {state_placeholder: np.expand_dims(state, axis = 0)})
                    action_index = np.squeeze(action_index)

                    # print('state -> {}'.format(state))
                    # print('action_index -> {}'.format(action_index))

                else:
                    action_index = 0

                if game_env.measured_rt:
                    measured_rt['index'].append(count)
                    measured_rt['value'].append(game_env.measured_rt)
                predicted_rt['index'].append(count)
                predicted_rt['value'].append(game_env.predicted_rt)

                count += 1
                
            stats_str = sess.run(test_stats_op, feed_dict = {episode_reward_placeholder: episode_reward})
            writer.add_summary(stats_str, count)

            @tfmpl.figure_tensor
            def draw_line(measured_rt, predicted_rt):
                fig = tfmpl.create_figures(1, figsize=(16, 8))[0]
                
                ax = fig.add_subplot(1, 2, 1)
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + \
                    ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(20)

                # ax.axis('off')
                ax.plot(measured_rt['index'], measured_rt['value'], 'b')
                ax.set_title(measured_rt['title'], fontsize = 24)
                
                ax = fig.add_subplot(1, 2, 2)
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + \
                    ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(20)

                ax.plot(predicted_rt['index'], predicted_rt['value'], 'r')
                ax.set_title(predicted_rt['title'], fontsize = 24)

                fig.tight_layout()

                return fig

            image_tensor = draw_line(measured_rt, predicted_rt)
            image_summary = tf.summary.image('test', image_tensor)
            image_str = sess.run(image_summary)
            writer.add_summary(image_str, global_step = 0)

            measured_rt_ma = measured_rt.copy()
            predicted_rt_ma = predicted_rt.copy()

            import matlab.engine
            eng = matlab.engine.start_matlab("-nojvm -nodisplay")

            m_rt = matlab.double(measured_rt_ma['value'])
            p_rt = matlab.double(predicted_rt_ma['value']) 

            m_rt = eng.movmean(m_rt, 11)
            p_rt = eng.movmean(p_rt, 11)

            m_rt = np.array(m_rt)
            m_rt = np.reshape(m_rt, -1)
            measured_rt_ma['value'] = list(m_rt)
            measured_rt_ma['title'] = "measured RT (after moving average)"

            p_rt = np.array(p_rt)
            p_rt = np.reshape(p_rt, -1)
            predicted_rt_ma['value'] = list(p_rt)
            predicted_rt_ma['title'] = "predicted RT (after moving average)"

            image_tensor_ma = draw_line(measured_rt_ma, predicted_rt_ma)
            image_summary_ma = tf.summary.image('test_ma', image_tensor_ma)
            image_str_ma = sess.run(image_summary_ma)
            writer.add_summary(image_str_ma, global_step = 1)

            io.savemat(os.path.join(f_event_log_dir, "stats.mat"), {"measured_rt": measured_rt, 'predicted_rt': predicted_rt})

            eng.quit()

            writer.close()            

if __name__ == "__main__":
    tf.app.run()
    
