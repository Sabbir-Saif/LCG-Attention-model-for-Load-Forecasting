import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout, Reshape
class MultiHeadAttention(Layer):
    def __init__(self, units, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads

    def build(self, input_shape):
        self.W = self.add_weight(name="W", shape=(input_shape[-1], self.units * self.num_heads),
                                 initializer="random_normal")
        self.b = self.add_weight(name="b", shape=(self.units * self.num_heads,), initializer="zeros")
        self.u = self.add_weight(name="u", shape=(self.units * self.num_heads, 1),
                                 initializer="random_normal")
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs):
        heads = []
        for _ in range(self.num_heads):
            score = tf.nn.tanh(tf.tensordot(inputs, self.W[:, self.units*_ : self.units*(_+1)], axes=1) + self.b[self.units*_ : self.units*(_+1)])
            attention_weights = tf.nn.softmax(tf.tensordot(score, self.u[self.units*_ : self.units*(_+1)], axes=1), axis=1)
            context_vector = attention_weights * inputs
            context_vector = tf.reduce_sum(context_vector, axis=1)
            heads.append(context_vector)
        return tf.concat(heads, axis=-1)
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1] * self.num_heads
