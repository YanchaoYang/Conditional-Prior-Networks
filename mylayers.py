import tensorflow as tf

def lrelu(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = tf.convert_to_tensor(0.5 * (1 + leak))
        f2 = tf.convert_to_tensor(0.5 * (1 - leak))
        y = tf.add(  tf.multiply(f1,x), tf.multiply(f2,tf.abs(x))  )
        return y

def conv(inputs, name, shape, stride, reuse=None, training=True, activation=lrelu, init_w=tf.contrib.layers.xavier_initializer_conv2d(), init_b=tf.constant_initializer(0.0)):
    with tf.variable_scope(name, reuse=reuse) as scope:
        kernel = tf.contrib.framework.model_variable('weights', shape=shape, initializer=init_w, trainable=training)
        conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding='SAME')
        biases = tf.contrib.framework.model_variable('biases', shape=[shape[3]], initializer=init_b, trainable=training)
        conv = tf.nn.bias_add(conv, biases)
        conv = activation(conv)
        return conv

def dilate_conv(inputs, name, shape, rate, reuse=None, training=True, activation=lrelu, init_w=tf.contrib.layers.xavier_initializer_conv2d(), init_b=tf.constant_initializer(0.0)):
    with tf.variable_scope(name, reuse=reuse) as scope:
        kernel = tf.contrib.framework.model_variable('weights', shape=shape, initializer=init_w, trainable=training)
        conv = tf.nn.atrous_conv2d( inputs, kernel, rate, padding='SAME')
        biases = tf.contrib.framework.model_variable('biases', shape=[shape[3]], initializer=init_b, trainable=training)
        conv = tf.nn.bias_add(conv, biases)
        conv = activation(conv)
        return conv

def deconv(inputs, size, name, shape, reuse=None, training=True, activation=lrelu):
    deconv = tf.image.resize_images( inputs, size=size )
    deconv = conv( deconv, name=name, shape=shape, stride=1, reuse=reuse, training=training, activation=activation )
    return deconv

def dilate_deconv(inputs, size, name, shape, rate, reuse=None, training=True, activation=lrelu):
    deconv = tf.image.resize_images( inputs, size=size )
    deconv = dilate_conv( deconv, name=name, shape=shape, rate=rate, reuse=reuse, training=training, activation=activation )
    return deconv
