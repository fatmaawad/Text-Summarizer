import tensorflow as tf
from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, verbose=False):
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs

        def energy_step(decoder_state, encoder_output):
            W_a_dot_s = tf.matmul(encoder_output, self.W_a)
            U_a_dot_h = tf.expand_dims(tf.matmul(decoder_state, self.U_a), 1)
            Ws_plus_Uh = tf.tanh(W_a_dot_s + U_a_dot_h)
            e_i = tf.squeeze(tf.matmul(Ws_plus_Uh, self.V_a), axis=-1)
            e_i = tf.nn.softmax(e_i)
            return e_i, e_i

        def context_step(e_i, encoder_output):
            c_i = tf.reduce_sum(encoder_output * tf.expand_dims(e_i, -1), axis=1)
            return c_i, c_i

        # Initialize e_outputs with zeros
        e_outputs_init = tf.zeros_like(decoder_out_seq[:, :, 0:1])
        # Use tf.scan to iterate over decoder_out_seq
        e_outputs, _ = tf.scan(energy_step, decoder_out_seq, initializer=e_outputs_init, swap_memory=True)

        # Initialize c_outputs with zeros
        c_outputs_init = tf.zeros_like(encoder_out_seq[:, 0, :])
        # Use tf.scan to iterate over e_outputs
        c_outputs, _ = tf.scan(context_step, e_outputs, initializer=c_outputs_init, swap_memory=True)

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]
