import tensorflow as tf
import backwarp
import numpy as np
from skimage import exposure

def backwarp_target_image( flow, imH, imW, img2, name ):
  with tf.variable_scope(name) as scope:
    flow_ = tf.image.resize_images( flow, size=[imH, imW], align_corners=True )
    ratio = tf.cast( imH/tf.shape(flow)[1], 'float32' )
    flow_ = tf.multiply( ratio, flow_ )
    img2_wp = backwarp.backwarper( img2, flow_, imsize=[imH, imW], name=name )
    return img2_wp

def contrast_change_heq( img1, img2, C=3 ):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    for i in range(C):
        r1 = img1[:,:,i]
        r2 = img2[:,:,i]
        cdf, bin_centers = exposure.cumulative_distribution(r1, nbins=256)
        rr1 = np.interp( r1.flat, bin_centers, cdf)
        rr2 = np.interp( r2.flat, bin_centers, cdf)
        r1 = rr1.reshape(r1.shape)
        r2 = rr2.reshape(r2.shape)
        xmax = np.amax(r1)
        xmin = np.amin(r1)
        dx = xmax - xmin
        r1 = (r1 - xmin)/dx
        r2 = (r2 - xmin)/dx
        img1[:,:,i] = r1
        img2[:,:,i] = r2
    return img1, img2

def cumulate_flow( flow5_1, flow4_1, flow3_1, flow2_1, flow1_1, flow5_2, flow4_2, flow3_2, flow2_2, flow1_2 ):
    flow5 = flow5_1 + flow5_2
    flow4 = flow4_1 + flow4_2
    flow3 = flow3_1 + flow3_2
    flow2 = flow2_1 + flow2_2
    flow1 = flow1_1 + flow1_2
    return flow5, flow4, flow3, flow2, flow1

def contrast_change( img1, img2, C=3 ):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    for i in range(C):
        r1 = img1[:,:,i]
        r2 = img2[:,:,i]
        xmax = np.amax(r1)
        xmin = np.amin(r1)
        dx = xmax - xmin
        r1 = (r1 - xmin)/dx
        r2 = (r2 - xmin)/dx
        img1[:,:,i] = r1
        img2[:,:,i] = r2
    return img1, img2

def upsample_flow( flowsmall, flowbig ):
    H=tf.shape(flowbig)[1]
    W=tf.shape(flowbig)[2]
    #w=tf.shape(flowsmall)[2]
    #ratio = tf.cast( W/w, 'float32' )
    flow = tf.image.resize_images( flowsmall, size=tf.stack([H,W]), align_corners=True )
    #flow = tf.multiply( ratio, flow )
    return flow

def residual_map( I1, I1_ ):
    diff = tf.subtract( I1, I1_ ) # [B, H, W, 3]
    diff = tf.square(diff)
    diff = tf.reduce_sum(diff, axis=3, keep_dims=True) / tf.constant(3.0) #[B,H,W,1]
    diff = tf.sqrt(diff)
    return diff

def sup_backwarp_target_image( flow, imH, imW, img2, ratio, name ):
  with tf.variable_scope(name) as scope:
    flow_ = tf.multiply( flow, tf.constant(20.0) )
    flow_ = tf.image.resize_images( flow_, size=[imH, imW], align_corners=True )
    flow_ = tf.multiply( flow_, tf.convert_to_tensor(ratio) )
    img2_wp = backwarp.backwarper( img2, flow_, imsize=[imH, imW], name=name )
    return img2_wp
