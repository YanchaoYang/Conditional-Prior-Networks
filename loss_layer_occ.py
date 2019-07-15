import tensorflow as tf
import numpy as np
import backwarp
import gaussian_smooth

def length_sq(x):
    return tf.reduce_sum(tf.square(x), 3, keep_dims=True)

def occlusion(flow_f, flow_b):
    outsz = flow_f.get_shape()
    outsz = tf.stack([ outsz[1], outsz[2] ])
    flow_f_ = backwarp.backwarper( flow_b, flow_f, outsz, name='flow_b_to_f' )
    flow_b_ = backwarp.backwarper( flow_f, flow_b, outsz, name='flow_f_to_b' )
    mag_sq_f = length_sq(flow_f) + length_sq(flow_f_)
    mag_sq_b = length_sq(flow_b) + length_sq(flow_b_)
    flow_diff_f = flow_f + flow_f_
    flow_diff_b = flow_b + flow_b_
    occ_thresh_f = tf.add( tf.multiply(tf.constant(0.01*1),mag_sq_f),tf.constant(0.5*10) )
    occ_thresh_b = tf.add( tf.multiply(tf.constant(0.01*1),mag_sq_b),tf.constant(0.5*10) )
    occ_f = tf.cast( length_sq(flow_diff_f)>occ_thresh_f, tf.float32 )
    occ_b = tf.cast( length_sq(flow_diff_b)>occ_thresh_b, tf.float32 )
    return occ_f, occ_b  # occ=1, covisible=0

def occ_weighted_diff( im1, im1_, occ, vmap, epsilon=0.001, cbn=0.5, occt=1.5 ):
    print(occt)
    epsilon = tf.convert_to_tensor(epsilon)
    cbn = tf.convert_to_tensor(cbn)
    occt = tf.convert_to_tensor(occt)
    diff = tf.subtract( im1, im1_ ) # [B, H, W, 3]
    diff = tf.add( tf.square(diff), tf.square(epsilon) )
    diff = tf.pow( diff, cbn, name='chamboneir' )
    diff = tf.reduce_sum(diff, axis=3, keep_dims=True) # [B, H, W, 1]
    diff = tf.multiply( vmap, diff )
    diff = tf.add( tf.multiply(diff,tf.subtract(tf.constant(1.0),occ)),tf.multiply(occt,occ) )
    return tf.reduce_sum(diff)

def get_data_occ( im1, im2, flow_f, flow_b, name, cbn, occt, vmap_f, vmap_b ):
    outsz = flow_f.get_shape()
    outsz = tf.stack([ outsz[1], outsz[2] ])
    vmap_f = tf.image.resize_images( vmap_f, size=outsz, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
    vmap_b = tf.image.resize_images( vmap_b, size=outsz, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
    im1_ = backwarp.backwarper( im2, flow_f, outsz, name=name+'forward' )
    im2_ = backwarp.backwarper( im1, flow_b, outsz, name=name+'backward' )
    occ_f, occ_b = occlusion(flow_f, flow_b)

    # get the robust difference
    loss_f = occ_weighted_diff( im1, im1_, occ_f, vmap=vmap_f, cbn=cbn, occt=occt )
    loss_b = occ_weighted_diff( im2, im2_, occ_b, vmap=vmap_b, cbn=cbn, occt=occt )
    loss = tf.add( loss_f, loss_b )
    return loss

def loss_unsup_data( im1_0, im2_0, flow_f5, flow_f4, flow_f3, flow_f2, flow_f1, flow_f0, flow_b5, flow_b4, flow_b3, flow_b2, flow_b1, flow_b0, cbn, vmap_f, vmap_b, w5, w4, w3, w2, w1, occt ):
    im1_1 = gaussian_smooth.gauss_conv( im1_0, size=5, nsig=3, name='pyr1_1' )
    im2_1 = gaussian_smooth.gauss_conv( im2_0, size=5, nsig=3, name='pyr1_2' )
    im1_2 = gaussian_smooth.gauss_conv( im1_1, size=5, nsig=3, name='pyr2_1' )
    im2_2 = gaussian_smooth.gauss_conv( im2_1, size=5, nsig=3, name='pyr2_2' )
    im1_3 = gaussian_smooth.gauss_conv( im1_2, size=5, nsig=3, name='pyr3_1' )
    im2_3 = gaussian_smooth.gauss_conv( im2_2, size=5, nsig=3, name='pyr3_2' )
    im1_4 = gaussian_smooth.gauss_conv( im1_3, size=5, nsig=3, name='pyr4_1' )
    im2_4 = gaussian_smooth.gauss_conv( im2_3, size=5, nsig=3, name='pyr4_2' )
    im1_5 = gaussian_smooth.gauss_conv( im1_4, size=5, nsig=3, name='pyr5_1' )
    im2_5 = gaussian_smooth.gauss_conv( im2_4, size=5, nsig=3, name='pyr5_2' )

    dt5 = tf.multiply( get_data_occ( im1_5, im2_5, flow_f5, flow_b5, name='data_term_5', cbn=cbn, occt=occt, vmap_f=vmap_f, vmap_b=vmap_b ), tf.constant(w5*64.0) )
    dt4 = tf.multiply( get_data_occ( im1_4, im2_4, flow_f4, flow_b4, name='data_term_4', cbn=cbn, occt=occt, vmap_f=vmap_f, vmap_b=vmap_b ), tf.constant(w4*16.0) ) #16.0
    dt3 = tf.multiply( get_data_occ( im1_3, im2_3, flow_f3, flow_b3, name='data_term_3', cbn=cbn, occt=occt, vmap_f=vmap_f, vmap_b=vmap_b ), tf.constant(w3*4.0)  ) #4.0
    dt2 = tf.multiply( get_data_occ( im1_2, im2_2, flow_f2, flow_b2, name='data_term_2', cbn=cbn, occt=occt, vmap_f=vmap_f, vmap_b=vmap_b ), tf.constant(w2*2.0)  ) #2.0
    dt1 = tf.multiply( get_data_occ( im1_1, im2_1, flow_f1, flow_b1, name='data_term_1', cbn=cbn, occt=occt, vmap_f=vmap_f, vmap_b=vmap_b ), tf.constant(w1*1.0)  ) #1.0
    dt0 = tf.multiply( get_data_occ( im1_0, im2_0, flow_f0, flow_b0, name='data_term_0', cbn=cbn, occt=occt, vmap_f=vmap_f, vmap_b=vmap_b ), tf.constant(1.0)  ) #1.0
    loss = tf.add( dt1,tf.add(tf.add(tf.add(dt5,dt4),dt3),dt2) )
    loss = tf.add( loss, dt0 )
    divideBy = tf.add(tf.reduce_sum(vmap_f), tf.reduce_sum(vmap_b))
    return tf.divide(loss, divideBy), tf.divide(dt0, divideBy)
