import tensorflow as tf

class Normalize(tf.keras.layers.Layer):
    def __init__(self, l = 2, dim = 1, **kwargs):
        super().__init__(**kwargs)
        if l == 1:
            self.norm = lambda x: x * tf.math.rsqrt(tf.math.reduce_sum(tf.math.abs(x), axis = dim, keepdims = True))
        elif l == 2:
            self.norm = lambda x: x * tf.math.rsqrt(tf.math.reduce_sum(tf.math.square(x), axis = dim, keepdims = True))
        
    def call(self, x):
        return self.norm(x)


class NT_Xent_Loss(tf.keras.losses.Loss):
    def __init__(self, temperature = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.normalize = Normalize(l = 2, dim = 1)
    
    def cosine_similarity(self, v1, v2):
        return tf.math.reduce_sum(v1*v2, axis = 1)
    
    def __call__(self, z1, z2):
        z1, z2 = self.normalize(z1), self.normalize(z2)
        z = tf.concat([z1, z2], axis = 0)
        
        # negatives
        neg_mask = tf.cast(tf.math.logical_not(tf.eye(z.shape[0], z.shape[0], dtype = tf.bool)), tf.float32)
        neg = tf.math.exp(tf.matmul(z, z, transpose_b = True) / self.temperature) * neg_mask
        neg = tf.math.reduce_sum(neg, axis = -1)
        
        # positives
        pos = tf.math.exp(tf.math.reduce_sum(z1*z2, axis = -1) / self.temperature)
        pos = tf.concat([pos, pos], axis = 0)
        
        return tf.math.reduce_mean(-tf.math.log(pos / neg))

