from __future__ import division
import tensorflow as tf
import backwarp
import gaussian_smooth

def sup_flow_loss( gtflow, flow, vmap, epsilon=0.001, cbn=0.5 ):
    epsilon = tf.convert_to_tensor(epsilon)
    outsz = flow.get_shape()
    outsz = tf.stack([ outsz[1], outsz[2] ])
    gtflow = tf.image.resize_images( gtflow, size=outsz )
    vmap = tf.image.resize_images( vmap, size=outsz )
    diff = tf.subtract( gtflow, flow )
    diff = tf.add( tf.square(diff), tf.square(epsilon) )
    diff = tf.pow( diff, tf.convert_to_tensor(cbn) ) # [B, H, W, 2]
    diff = tf.reduce_sum( diff, axis=3, keep_dims=True ) # [B, H, W, 1]
    diff = tf.multiply( vmap, diff )
    loss = tf.reduce_sum(diff)
    return loss

def sup_get_loss( gtflow, flow, vmap, weight=1.0 ):
    data_term = sup_flow_loss( gtflow, flow, vmap )
    weight = tf.convert_to_tensor(weight)
    return tf.multiply( weight, data_term )

def loss_sup( gtflow, vmap, flow5, flow4, flow3, flow2, flow1, flow0 ):
    loss5 = sup_get_loss( gtflow, flow5, vmap, weight=64.0 )
    loss4 = sup_get_loss( gtflow, flow4, vmap, weight=16.0 ) #16.0
    loss3 = sup_get_loss( gtflow, flow3, vmap, weight=4.0  ) #4.0
    loss2 = sup_get_loss( gtflow, flow2, vmap, weight=2.0  ) #2.0
    loss1 = sup_get_loss( gtflow, flow1, vmap, weight=1.0  ) #1.0
    loss0 = sup_get_loss( gtflow, flow0, vmap, weight=1.0  ) #1.0
    loss = tf.add( loss1,tf.add(tf.add(tf.add(loss5,loss4),loss3),loss2) )
    loss = tf.add( loss, loss0 )
    divideBy = tf.reduce_sum(vmap)
    return tf.divide(loss,divideBy), tf.divide(loss0,divideBy)

def training( loss, gs_all, lr_all, beta1=0.9 ):
    vars_all = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES )
    optimizer = tf.train.AdamOptimizer( learning_rate=lr_all, beta1=beta1, epsilon=1e-8 )
    grads_all = tf.gradients( loss, vars_all )
    capped_grads_all = [ ClipIfNotNone(grad,5000.0) for grad in grads_all ]
    train_op = optimizer.apply_gradients( zip(capped_grads_all, vars_all), global_step=gs_all )
    return train_op


################### BEGIN TOWER LOSSES ###################


#def training_tower( tower_loss, n_gpus, gs_all, lr_all, beta1=0.9 ):
#    vars_all = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES )
#    optimizer = tf.train.AdamOptimizer( learning_rate=lr_all, beta1=beta1, epsilon=1e-8 )
#    tower_grads = []
#    for i in range(n_gpus):
#        with tf.device('/gpu:%d' % i):
#            tower_grads.append(  tf.gradients( tower_loss[i], vars_all )  )
#    avg_gradients = average_gradients( tower_grads )
#
#    capped_grads_all = [ ClipIfNotNone(grad,5000.0) for grad in avg_gradients ]
#    train_op = optimizer.apply_gradients( zip(capped_grads_all, vars_all), global_step=gs_all )
#    return train_op

def training_tower( tower_loss, n_gpus, gs_all, lr_all, beta1=0.9 ):
    vars_all = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES )
    optimizer = tf.train.AdamOptimizer( learning_rate=lr_all, beta1=beta1, epsilon=1e-8 )
 
    print(tower_loss)
    with tf.device('/gpu:%d' % 0):
        grads_all_0 = tf.gradients( tower_loss[0], vars_all )
        capped_grads_all_0 = [ ClipIfNotNone(grad,5000.0) for grad in grads_all_0 ]
    with tf.device('/gpu:%d' % 1):
        grads_all_1 = tf.gradients( tower_loss[1], vars_all )
        capped_grads_all_1 = [ ClipIfNotNone(grad,5000.0) for grad in grads_all_1 ]

    grads_all = tf.add(capped_grads_all_0,capped_grads_all_1)
    capped_grads_all = [ ClipIfNotNone(grad,5000.0) for grad in grads_all ]
    train_op = optimizer.apply_gradients( zip(capped_grads_all, vars_all), global_step=gs_all )
    return train_op

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
  # Note that each grad_and_vars looks like the following:
  #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    #########
    grad = ClipIfNotNone(grad, 5000)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


############### END OF TOWER LOSSES #####################


def ClipIfNotNone(grad, clipvalue):
    #if grad is None:
    #    return grad
    return tf.clip_by_value(grad, -clipvalue, clipvalue)

def get_data_term( im1, im2, flow, vmap, name='data_term', epsilon=0.001, cbn=0.5 ):
    epsilon = tf.convert_to_tensor(epsilon)
    # get flow's size and resize im1, im2
    outsz = flow.get_shape()
    outsz = tf.stack([ outsz[1], outsz[2] ])
    #im1  = tf.image.resize_images( im1, size=outsz )
    #im2  = tf.image.resize_images( im2, size=outsz )
    vmap = tf.image.resize_images( vmap, size=outsz )
    im1_ = backwarp.backwarper( im2, flow, outsz, name=name )

    # get the robust difference
    diff = tf.subtract( im1, im1_ ) # [B, H, W, 3]
    diff = tf.add( tf.square(diff), tf.square(epsilon) )
    diff = tf.pow( diff, tf.convert_to_tensor(cbn), name='chamboneir' )
    diff = tf.reduce_sum(diff, axis=3, keep_dims=True) # [B, H, W, 1]
    # vmap is also                                       [B, H, W, 1]
    diff = tf.multiply( vmap, diff )
    loss = tf.reduce_sum(diff)
    return loss

def loss_unsup_data( im1_0, im2_0, flow5, flow4, flow3, flow2, flow1, flow0, vmap, cbn, w5, w4, w3, w2, w1 ):
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

    dt5 = tf.multiply( get_data_term( im1_5, im2_5, flow5, vmap, name='data_term_5', cbn=cbn ), tf.constant(w5*64.0) )
    dt4 = tf.multiply( get_data_term( im1_4, im2_4, flow4, vmap, name='data_term_4', cbn=cbn ), tf.constant(w4*16.0) ) #16.0
    dt3 = tf.multiply( get_data_term( im1_3, im2_3, flow3, vmap, name='data_term_3', cbn=cbn ), tf.constant(w3*4.0)  ) #4.0
    dt2 = tf.multiply( get_data_term( im1_2, im2_2, flow2, vmap, name='data_term_2', cbn=cbn ), tf.constant(w2*2.0)  ) #2.0
    dt1 = tf.multiply( get_data_term( im1_1, im2_1, flow1, vmap, name='data_term_1', cbn=cbn ), tf.constant(w1*1.0)  ) #1.0
    dt0 = tf.multiply( get_data_term( im1_0, im2_0, flow0, vmap, name='data_term_0', cbn=cbn ), tf.constant(1.0)  ) #1.0
    loss = tf.add( dt1,tf.add(tf.add(tf.add(dt5,dt4),dt3),dt2) )
    loss = tf.add( loss, dt0 )
    divideBy = tf.reduce_sum(vmap)
    return tf.divide(loss, divideBy), tf.divide(dt0, divideBy),     im1_5, im2_5

def loss_unsup_prior( flow5, flow4, flow3, flow2, flow1, flow0, vmap, flow5r, flow4r, flow3r, flow2r, flow1r, flow0r, w5, w4, w3, w2, w1 ):
    pt5 = tf.multiply( sup_flow_loss( flow5, flow5r, vmap ), tf.constant(w5*0.0625) ) #0.0625
    pt4 = tf.multiply( sup_flow_loss( flow4, flow4r, vmap ), tf.constant(w4*0.0625) ) #0.0625
    pt3 = tf.multiply( sup_flow_loss( flow3, flow3r, vmap ), tf.constant(w3*0.0625) ) #0.0625
    pt2 = tf.multiply( sup_flow_loss( flow2, flow2r, vmap ), tf.constant(w2*0.1250) ) #0.1250
    pt1 = tf.multiply( sup_flow_loss( flow1, flow1r, vmap ), tf.constant(w1*0.2500) ) #0.2500
    pt0 = tf.multiply( sup_flow_loss( flow0, flow0r, vmap ), tf.constant(1.0000) ) #1.0000
    loss = tf.add( pt1,tf.add(tf.add(tf.add(pt5,pt4),pt3),pt2) )
    loss = tf.add( loss, pt0 )
    divideBy = tf.reduce_sum(vmap)
    return tf.divide(loss, divideBy), tf.divide(pt0,divideBy)
