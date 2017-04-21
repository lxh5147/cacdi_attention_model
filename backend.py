import keras.backend as K

if K._BACKEND == 'theano':
    from theano import tensor as T
    def greater_equal(x, y):
        return T.ge(x, y)

    def clip_norm(g, c, n):
        if c > 0:
            g = K.switch(n >= c, g * c / n, g)
        return g

elif K._BACKEND == 'tensorflow':
    import tensorflow as tf
    def greater_equal(x, y):
        '''Element-wise truth value of (x >= y).
        Returns a bool tensor.
        '''
        return tf.greater_equal(x, y)


    def clip_norm(g, c, n):
        if c > 0:
            f = tf.python.control_flow_ops.cond(tf.cast(n >= c, 'bool'),
                                                lambda: c / n,
                                                lambda: tf.constant(1.0))
        return tf.scalar_mul(f, g)