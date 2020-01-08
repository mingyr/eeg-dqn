# standard libraries
import os, sys
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
from utils import Summaries, Synchronizer
from optimizer import Adam

'''
greedy policy
'''
def greedy(state, network, debug = False):
    # Turn single state into minibatch.  Needed for convolutional nets.

    assert(len(state.get_shape().as_list()) == 3)

    q = network(state)

    if debug:
        print_q =  tf.print("Q(s, a): ", q, output_stream = sys.stdout)

        with tf.control_dependencies([print_q]):
            maxq = tf.reduce_max(q, axis = -1)
            besta = tf.argmax(q, axis = -1)
    else:
        maxq = tf.reduce_max(q, axis = -1)
        besta = tf.argmax(q, axis = -1)
        
    return besta

'''
behavioural policy
'''
def eGreedy(state, network, num_actions, decayed_ep, debug = False):
    besta = greedy(state, network, debug = debug)
    
    # Epsilon greedy
    action = tf.cond(tf.less(tf.random_uniform([]), decayed_ep),
        lambda: tf.random_uniform([], 0, num_actions, tf.int64), lambda: besta)

    if debug:
        print_ep =  tf.print("ep: ", decayed_ep, output_stream = sys.stdout)
        print_pre_action = tf.print("pre_action: ", besta, output_stream = sys.stdout)
        print_post_action = tf.print("post_action: ", action, output_stream = sys.stdout)

        with tf.control_dependencies([print_ep, print_pre_action, print_post_action]):
            action = tf.squeeze(action)

    else:
        action = tf.squeeze(action)

    return action

'''
main function
'''
def main(unused_argv):

    '''
    check path
    '''
    if FLAGS.data_dir == '' or not os.path.exists(FLAGS.data_dir):
        raise ValueError('invalid data directory {}'.format(FLAGS.data_dir))

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

    filenames_train = glob.glob(os.path.join(FLAGS.data_dir, 'train-{}'.format(FLAGS.sampling_rate), '*.mat'))
    filenames_val = glob.glob(os.path.join(FLAGS.data_dir, 'val-{}'.format(FLAGS.sampling_rate), '*.mat'))

    game_env_train = Env(decay = FLAGS.decay)
    game_env_val = Env(decay = FLAGS.decay)

    game_actions = list(game_env_train.actions.keys())

    '''
    setup the transition table for experience replay
    '''

    stateDim = [FLAGS.num_chans, FLAGS.num_points]

    transition_args = {
        'batchSize':   FLAGS.batch_size,
        'stateDim':    stateDim,
        'numActions':  len(game_actions),
        'maxSize':     FLAGS.replay_memory,
    }

    transitions = TransitionMemory(transition_args)

    '''
    setup agent
    '''
    s_placeholder    = tf.placeholder(tf.float32, [FLAGS.batch_size] + stateDim, 's_placeholder')
    s2_placeholder   = tf.placeholder(tf.float32, [FLAGS.batch_size] + stateDim, 's2_placeholder')
    a_placeholder    = tf.placeholder(tf.int32, [FLAGS.batch_size], 'a_placeholder')
    r_placeholder    = tf.placeholder(tf.float32, [FLAGS.batch_size], 'r_placeholder')

    pcont_t = tf.constant(FLAGS.discount, tf.float32, [FLAGS.batch_size])

    network = Model(FLAGS.batch_size, len(game_actions), FLAGS.num_chans, FLAGS.sampling_rate, \
                    FLAGS.num_filters, FLAGS.num_recurs, FLAGS.pooling_stride, name = "network") 
 
    target_network = Model(FLAGS.batch_size, len(game_actions), FLAGS.num_chans, FLAGS.sampling_rate,\
                           FLAGS.num_filters, FLAGS.num_recurs, FLAGS.pooling_stride, name = "target_n")

    q = network(s_placeholder)
    q2 = target_network(s2_placeholder)
    q_selector = network(s2_placeholder)

    loss, q_learning = trfl.double_qlearning(q, a_placeholder, r_placeholder, pcont_t, q2, q_selector)
    synchronizer = Synchronizer(network, target_network)
    sychronize_ops = synchronizer()

    training_variables = network.variables
 
    opt = Adam(FLAGS.learning_rate, lr_decay = FLAGS.lr_decay, lr_decay_steps = FLAGS.lr_decay_steps,
               lr_decay_factor = FLAGS.lr_decay_factor, clip = True)

    reduced_loss = tf.reduce_mean(loss)

    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_regularization_loss = tf.reduce_sum(graph_regularizers)

    total_loss = reduced_loss + total_regularization_loss

    update_op = opt(total_loss, var_list = training_variables)

    summ_loss_op = tf.summary.scalar('loss', total_loss)


    state_placeholder      = tf.placeholder(tf.float32, [1] + stateDim, 'state_placeholder')
    decayed_ep_placeholder = tf.placeholder(tf.float32, [], 'decayed_ep_placeholder')

    action_tensor_egreedy = eGreedy(state_placeholder, network, 
                                    len(game_actions), decayed_ep_placeholder, FLAGS.debug)

    action_tensor_greedy = greedy(state_placeholder, network)

    '''
    setup the training process
    '''
    episode_reward_placeholder = tf.placeholder(tf.float32, [], "episode_reward_placeholder")
    average_reward_placeholder = tf.placeholder(tf.float32, [], "average_reward_placeholder")
    
    summ.register('train', 'episode_reward_train', episode_reward_placeholder) 
    summ.register('train', 'average_reward_train', average_reward_placeholder)

    summ.register('val', 'episode_reward_val', episode_reward_placeholder) 
    summ.register('val', 'average_reward_val', average_reward_placeholder)

    total_reward_train = 0
    average_reward_train = 0

    total_reward_val = 0
    average_reward_val = 0

    '''
    gathering summary operators
    '''
    train_summ_op = summ('train')
    val_summ_op = summ('val')

    '''
    setup the training process
    '''
    transitions.empty()
    # print("game_actions -> {}".format(game_actions))

    writer = tf.summary.FileWriter(event_log_dir, tf.get_default_graph())

    saver = tf.train.Saver(training_variables)

    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)

    assert (FLAGS.gpus != ''), 'invalid GPU specification'
    config.gpu_options.visible_device_list = FLAGS.gpus

    with tf.Session(config = config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    
        val_step = 0

        for step in range(FLAGS.steps):
            print("Iteration: {}".format(step))

            game_env_train.reset(filenames_train[np.random.randint(0, len(filenames_train))])

            last_state = None
            last_state_assigned = False
            episode_reward = 0
            action_index = (len(game_actions) >> 2)

            for estep in range(FLAGS.eval_steps):
                # print("Evaluation step: {}".format(estep))

                # print("{} - measured RT: {}".format(estep, game_env_train.measured_rt))
                # print("{} - predicted RT: {}".format(estep, game_env_train.predicted_rt))
                # print("{} - action -> {}".format(estep, game_actions[action]))

                state, reward, terminal = game_env_train.step(game_actions[action_index])

                # game over?
                if terminal:
                    break

                episode_reward += reward

                # Store transition s, a, r, t
                # if last_state_assigned and reward:
                if last_state_assigned:
                    # print("reward -> {}".format(reward))
                    # print("action -> {}".format(game_actions[last_action]))
                    transitions.add(last_state, last_action, reward, last_terminal)


                # Select action
                # decayed_ep = FLAGS.testing_ep

                decayed_ep = max(0.1, (FLAGS.steps - step) / FLAGS.steps * FLAGS.ep)

                if not terminal:
                    action_index = sess.run(action_tensor_egreedy, feed_dict = {state_placeholder: np.expand_dims(state, axis = 0), 
                                                                                decayed_ep_placeholder: decayed_ep})
                else:
                    action_index = 0

                # Do some Q-learning updates
                if estep > FLAGS.learn_start and estep % FLAGS.update_freq == 0:
                    summ_str = None
                    for _ in range(FLAGS.n_replay):                   
                        if transitions.size > FLAGS.batch_size:
                            s, a, r, s2 = transitions.sample()

                            summ_str, _ = sess.run([summ_loss_op, update_op], feed_dict = 
                                                   {s_placeholder: s, a_placeholder: a, r_placeholder: r, s2_placeholder: s2})

                    if summ_str:
                        writer.add_summary(summ_str, step * FLAGS.eval_steps + estep)

                last_state = state
                last_state_assigned = True

                last_action = action_index
                last_terminal = terminal

                if estep > FLAGS.learn_start and estep % FLAGS.target_q == 0:
                    # print("duplicate model parameters")
                    sess.run(sychronize_ops)

            total_reward_train += episode_reward
            average_reward_train = total_reward_train / (step + 1)
            
            train_summ_str = sess.run(train_summ_op, feed_dict = {episode_reward_placeholder: episode_reward,
                                                                  average_reward_placeholder: average_reward_train})
            writer.add_summary(train_summ_str, step)

            if FLAGS.validation and step % FLAGS.validation_interval == 0:                
                game_env_val.reset(filenames_val[0])

                episode_reward = 0
                count = 0
                action_index = (len(game_actions) >> 2)

                while True:
                    # print("Evaluation step: {}".format(count))
                    # print("action -> {}".format(game_actions[action_index]))

                    state, reward, terminal = game_env_val.step(game_actions[action_index])

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

                    count += 1

                total_reward_val += episode_reward
                average_reward_val = total_reward_val / (val_step + 1)
                val_step += 1

                val_summ_str = sess.run(val_summ_op, feed_dict = {episode_reward_placeholder: episode_reward,
                                                                    average_reward_placeholder: average_reward_val})
                writer.add_summary(val_summ_str, step)
        
        tf.logging.info('Saving model.')
        saver.save(sess, checkpoint_path)
        tf.logging.info('Training complete')


    writer.close()            

    
if __name__ == "__main__":
    tf.app.run()
    
