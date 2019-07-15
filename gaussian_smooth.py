import numpy as np
import scipy.stats as st
import tensorflow as tf

def gauss_kernel( kernlen=5, nsig=3, channels=1 ):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def make_gauss_var( size, nsig, c_i ):
    with tf.device("/cpu:0"):
        kernel = gauss_kernel( size, nsig, c_i )
        var = tf.Variable( tf.convert_to_tensor(kernel), trainable=False )
        return var

def gauss_conv( img, size, nsig, name, padding='SAME' ):
    c_i = img.get_shape().as_list()[3]
    convolve = lambda i, k: tf.nn.depthwise_conv2d( i, k, [1, 2, 2, 1], padding=padding )
    with tf.variable_scope(name) as scope:
        kernel = make_gauss_var( size, nsig, c_i )
        output = convolve( img, kernel )
        return output
