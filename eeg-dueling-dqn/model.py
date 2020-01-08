import tensorflow as tf
import sonnet as snt
from utils import DownsampleAlongW, Activation, get_batch_size

class CNN(snt.AbstractModule):
    def __init__(self, num_chans, sampling_rate, num_filters,
                 pooling_stride, act = 'tanh', verbose = False, name = "cnn"):
        super(CNN, self).__init__(name = name)

        self._pool1 = DownsampleAlongW(pooling_stride, padding = 'VALID', verbose = verbose)
        self._pool2 = DownsampleAlongW(pooling_stride, padding = 'VALID', verbose = verbose)
        self._act = Activation(act, verbose = verbose)

        with self._enter_variable_scope():
            def clip_getter(getter, name, *args, **kwargs):
                var = getter(name, *args, **kwargs)
                clip_var = tf.clip_by_norm(var, 1)

                return clip_var

            self._l1_conv = snt.Conv2D(num_filters, [1, sampling_rate >> 1])
            self._l2_depthconv = snt.DepthwiseConv2D(1, (num_chans, 1), padding = snt.VALID,
                                                     custom_getter = {'w': clip_getter})
            self._l3_sepconv = snt.SeparableConv2D(num_filters, 1, [1, sampling_rate >> 3])

    def _build(self, inputs):
        # shape(inputs) = NxCxT

        outputs = tf.expand_dims(inputs, axis = -1)
        # shape(outputs) = NxCxTx1

        outputs = self._l1_conv(outputs)
        # shape(outputs) = NxCxTxF

        outputs = self._l2_depthconv(outputs)
        # shape(outputs) = Nx1xTx(D*F)

        outputs = self._act(outputs)
        # shape(outputs) = Nx1xTx(D*F)

        outputs = self._pool1(outputs)
        # shape(outputs) = Nx1x(T/P)x(D*F)

        outputs = self._l3_sepconv(outputs)
        # shape(outputs) = Nx1x(T/P)xF'

        outputs = self._act(outputs)
        # shape(outputs) = Nx1x(T/P)xF'

        outputs = self._pool2(outputs)
        # shape(outputs) = Nx1x(T/P^2)xF'

        return outputs

class Cell(snt.RNNCore):
    def __init__(self, num_channels, spatial_size, 
                 filter_size, forget_bias = 1.0, name = "cell"):
        """
        Args:
            num_channels: the number of output channels in the layer.
            filter_size: the shape of the each convolution filter.
            forget_bias: the initial value of the forget biases.
        """
        super(Cell, self).__init__(name = name)
        self._num_channels = num_channels
        self._spatial_size = spatial_size
        self._forget_bias = forget_bias

        self._state_size = tf.TensorShape(spatial_size + [2 * num_channels])
        self._output_size = tf.TensorShape(spatial_size + [num_channels])

        with self._enter_variable_scope():
            self._conv2d = snt.Conv2D(4 * num_channels, [1, filter_size])

    def initial_state(self, inputs, time_major, batch_size = None, dtype = tf.float32, trainable = False,
                      trainable_initializers = None, trainable_regularizers = None,
                      state_initializer = tf.zeros_initializer()):
        if not batch_size:
            batch_size = get_batch_size(inputs, time_major)

        return state_initializer([batch_size] + self._spatial_size + [2 * self._num_channels], dtype = dtype)

    def _build(self, inputs, state):
        """
        Basic LSTM recurrent network cell, with 2D convolution connctions.

        Args:
            inputs: input Tensor, 4D, batch x height x width x channels.
            state: state Tensor, 4D, batch x height x width x channels.
        Returns:
            a tuple of tensors representing output and the new state.
        """

        inputs.get_shape().assert_has_rank(4)
        state.get_shape().assert_has_rank(4)

        c, h = tf.split(axis = 3, num_or_size_splits = 2, value = state)
        inputs_h = tf.concat(axis = 3, values = [inputs, h])

        # Parameters of gates are concatenated into one conv for efficiency.
        i_j_f_o = self._conv2d(inputs_h)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(axis = 3, num_or_size_splits = 4, value = i_j_f_o)

        new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
        new_h = tf.tanh(new_c) * tf.sigmoid(o)

        return new_h, tf.concat(axis = 3, values = [new_c, new_h])

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

class Model(snt.AbstractModule):
    def __init__(self, batch_size, num_actions, num_chans, sampling_rate, num_filters, 
                 num_recurs = 3, pooling_stride = 2, act = 'tanh', verbose = True, name = "model"):
        super(Model, self).__init__(name = name)
        self._batch_size = batch_size
        self._num_recurs = num_recurs

        self._pool = DownsampleAlongW(pooling_stride, verbose = verbose)

        if verbose:
            print("configuration of the model with following specifications")
            print("input data #chan {}, #pt {}".format(num_chans, sampling_rate * num_recurs))
            print("input data sampling rate {}".format(sampling_rate))
            print("number of filters {}".format(num_filters))
            print("length of pooling stride {}".format(pooling_stride))

        with self._enter_variable_scope():
            self._cnn = CNN(num_chans, sampling_rate, num_filters,
                            pooling_stride, act = act, verbose = verbose)            

            self._rnn = Cell(num_channels = num_filters, spatial_size = [1, sampling_rate >> 2],
                             filter_size = sampling_rate >> 4)
            regularizers = {"w": tf.contrib.layers.l2_regularizer(scale = 1.0)}

            self._lin1_v = snt.Linear(512, regularizers = regularizers, name = "lin1_v")
            self._lin2_v = snt.Linear(1, name = "lin2_v")

            self._lin1_a = snt.Linear(512, regularizers = regularizers, name = "lin1_a")
            self._lin2_a = snt.Linear(num_actions, name = "lin2_a")

            self._seq_v = snt.Sequential([self._lin1_v, self._lin2_v])
            self._seq_a = snt.Sequential([self._lin1_a, self._lin2_a])

    def _build(self, inputs):
        # shape(inputs) = NxCxT
        data = tf.split(inputs, self._num_recurs, axis = -1)

        outputs = tf.stack([self._cnn(d) for d in data], axis = 0)

        initial_state = self._rnn.initial_state(outputs, time_major = True)
        output_sequence, final_state = \
            tf.nn.dynamic_rnn(self._rnn, outputs, initial_state = initial_state, time_major = True)
       
        outputs = tf.unstack(output_sequence, axis = 0)[-1] # time major

        outputs = self._pool(outputs)
        outputs = snt.BatchFlatten()(outputs)

        outputs_v = self._seq_v(outputs)
        outputs_a = self._seq_a(outputs)

        offset_a = tf.reduce_mean(outputs_a, axis = -1, keepdims = True)

        with tf.control_dependencies([tf.assert_equal(tf.rank(outputs_a), tf.rank(offset_a))]):
            outputs = outputs_v + (outputs_a - offset_a)

        return outputs

    @property
    def variables(self):
        return self._cnn.get_variables() + self._rnn.get_variables() + \
               self._lin1_v.get_variables() + self._lin2_v.get_variables() + \
               self._lin1_a.get_variables() + self._lin2_a.get_variables() 

def test_model():
    from config import FLAGS

    net = Model(32, 3, FLAGS.num_chans, FLAGS.sampling_rate, FLAGS.num_filters, FLAGS.num_recurs, FLAGS.pooling_stride)
    inputs = tf.constant(1.0, tf.float32, [32, FLAGS.num_chans, FLAGS.num_points])
    outputs = net(inputs)
    writer = tf.summary.FileWriter('model-output', tf.get_default_graph())
    with tf.Session() as (sess):
        sess.run(tf.global_variables_initializer())
        v = sess.run(outputs)
        print(v)
    writer.close()


if __name__ == '__main__':
    test_model()


 
