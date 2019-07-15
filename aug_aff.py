import tensorflow as tf
import backwarp

def rand_affine( minR=-17.0, maxR=17.0, minS=0.9, maxS=2.0, minT=-20.0, maxT=20.0 ):
    with tf.variable_scope('rand_affine'):
        pie = tf.constant(3.14159265)
        R = tf.random_uniform( shape=[], minval=minR, maxval=maxR, name='sample_rotation')
        S = tf.random_uniform( shape=[], minval=minS, maxval=maxS, name='sample_scale')
        tx = tf.random_uniform( shape=[], minval=minT, maxval=maxT, name='sample_translation_x')
        ty = tf.random_uniform( shape=[], minval=minT, maxval=maxT, name='sample_translation_y')
        R_ = tf.multiply(tf.divide(R,tf.constant(180.0)),pie)
        cosR = tf.cos(R_)
        sinR = tf.sin(R_)
        rotate = tf.stack( [cosR, -sinR, 0.0, sinR, cosR, 0.0, 0.0, 0.0, 1.0] )
        rotate = tf.reshape( rotate, (-1, 3, 3) )
        scale  = tf.stack( [   S,   0.0, 0.0,  0.0,    S, 0.0, 0.0, 0.0, 1.0] )
        scale  = tf.reshape( scale, (-1, 3, 3) )
        transl = tf.stack( [  1.0,  0.0,  tx,  0.0,  1.0,  ty, 0.0, 0.0, 1.0] )
        transl = tf.reshape( transl, (-1, 3, 3) )
        theta  = tf.matmul( transl,  tf.matmul(scale, rotate) )
        theta_ = tf.matrix_inverse(theta)
        return theta, theta_

def meshgrid(height, width):
    with tf.variable_scope('meshgrid'):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid( np.linspace(0, w-1, w),
        #                          np.linspace(0, h-1, h) )
        #  grid = np.vstack( [x_t.flatten(), y_t.flatten()] )
        tensor_one = tf.constant(1)
        x_t = tf.matmul( tf.ones(shape=tf.stack([height, 1])),
                         tf.transpose(tf.expand_dims(tf.linspace(0.0, tf.cast( width-tensor_one, 'float32'), width), 1), [1, 0]) )
        y_t = tf.matmul( tf.expand_dims(tf.linspace(0.0, tf.cast( height-tensor_one, 'float32'), height),1),
                         tf.ones(shape=tf.stack([1, width])) )
        x_t_flat = tf.reshape( x_t, (1, -1) )
        y_t_flat = tf.reshape( y_t, (1, -1) )
        x_t_flat = tf.subtract( x_t_flat, width*0.5 )
        y_t_flat = tf.subtract( y_t_flat, height*0.5 )
        ones = tf.ones_like(x_t_flat)
        grid = tf.concat( axis=0, values=[x_t_flat, y_t_flat, ones] ) # [ 3 , HW ]
        return grid

def aff_flow( theta, imsize, batchsize ):
    """
    theta: is the inverse of the actual transformation applied
    """
    with tf.variable_scope('aff_flow'):
        height = imsize[0]
        width  = imsize[1]
        grid = meshgrid( height, width )  # [3, HW]
        grid = tf.expand_dims( grid, 0 )  # [1, 3, HW]
        grid = tf.reshape( grid, [-1] )    # [3*HW,]
        grid = tf.tile( grid, tf.stack([batchsize]) )  # B copies of 3*HW
        grid = tf.reshape( grid, tf.stack([batchsize,3,-1]) )  # [B, 3, HW]
        theta_eye = tf.subtract( theta, tf.expand_dims(tf.eye(3),0) )
        theta_eye = tf.reshape( theta_eye, [-1] )
        theta_eye = tf.tile( theta_eye, tf.stack([batchsize]) )
        theta_eye = tf.reshape( theta_eye, tf.stack([batchsize,3,-1]) )
        T_g = tf.matmul(theta_eye, grid)
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape( x_s, [-1] )
        y_s_flat = tf.reshape( y_s, [-1] )
        x_f = tf.reshape( x_s_flat, tf.stack([batchsize, height, width, 1]) )
        y_f = tf.reshape( y_s_flat, tf.stack([batchsize, height, width, 1]) )
        flow = tf.concat( (x_f,y_f), 3 )
        return flow

def flow_transform( A0, A, gx, imsize ):
    with tf.variable_scope('flow_transform'):
        batchsize = tf.shape(gx)[0]
        height = imsize[0]
        width  = imsize[1]
        gx_u = gx[:,:,:,0]
        gx_v = gx[:,:,:,1]        
        gx_u_flat = tf.reshape( gx_u, [-1] )
        gx_v_flat = tf.reshape( gx_v, [-1] )
        gx_uu = tf.reshape( gx_u_flat, tf.stack([batchsize, 1, -1]) )
        gx_vv = tf.reshape( gx_v_flat, tf.stack([batchsize, 1, -1]) )
        zeros = tf.zeros_like(gx_uu)
        gx_aug = tf.concat( [gx_uu,gx_vv,zeros], axis=1 )    # [B, 3, HW]
        A0A = tf.matmul(A0,A)
        A0A = tf.reshape( A0A, [-1] )
        A0A = tf.tile( A0A, tf.stack([batchsize]) )
        A0A = tf.reshape( A0A, tf.stack([batchsize,3,-1]) )
        flow_out1 = tf.matmul( A0A, gx_aug ) # [B, 3, HW]
        flow_out1_x = flow_out1[:,0,:]
        flow_out1_y = flow_out1[:,1,:] # [B, HW]
        flow_out1_x = tf.reshape( flow_out1_x, [-1] )
        flow_out1_y = tf.reshape( flow_out1_y, [-1] )
        flow_out1_x = tf.reshape( flow_out1_x, tf.stack([batchsize, height, width, 1]) )
        flow_out1_y = tf.reshape( flow_out1_y, tf.stack([batchsize, height, width, 1]) )
        flow_out1 = tf.concat( (flow_out1_x,flow_out1_y), 3 )
        flow_out2 = aff_flow( A0, imsize, batchsize )
        flow_prime = tf.add( flow_out1, flow_out2 )
        return flow_prime

def random_rotate_scale_translate(img1s, img2s, edge1s, edge2s, flows, imsize, name='Rand_R_S_T', RMAX=17.0, SMIN=0.9, SMAX=2.0, TMAX=0.2, RMAX0=3.0, SMIN0=0.9, SMAX0=1.1, TMAX0=0.03 ):
    """
    flows : float, defined on img1, shape: [B, H, W, 2]
    imsize : [H, W], default in form of numpy array
    """
    with tf.variable_scope(name):
        batchsize = tf.shape(img1s)[0]
        W = imsize[1]
        A, A_   = rand_affine( minR=-RMAX, maxR=RMAX, minS=SMIN, maxS=SMAX, minT=-TMAX*W, maxT=TMAX*W )
        A0, A0_ = rand_affine( minR=-RMAX0, maxR=RMAX0, minS=SMIN0, maxS=SMAX0, minT=-TMAX0*W, maxT=TMAX0*W )

        ff1 = aff_flow( A_, imsize, batchsize )
        img1s_ = backwarp.backwarper( img1s, ff1, imsize, name='img1s_')
        edge1s_= backwarp.backwarper( edge1s,ff1, imsize, name='edge1s_')
        valid_map = tf.ones( shape=tf.stack([batchsize,imsize[0],imsize[1],1]) )
        valid_map = backwarp.backwarper( valid_map, ff1, imsize, name='valid_map' )
        gx = backwarp.backwarper( flows, ff1, imsize, name='warped_flow' )
        flows_ = flow_transform( A0, A, gx, imsize )

        A_A0_ = tf.matmul( A_, A0_ )
        ff2 = aff_flow( A_A0_, imsize, batchsize )
        img2s_ = backwarp.backwarper( img2s, ff2, imsize, name='img2s_' )
        edge2s_= backwarp.backwarper( edge2s,ff2, imsize, name='edge2s_')

        return img1s_, img2s_, edge1s_, edge2s_, flows_, valid_map

def random_crop( img1s, img2s, edge1s, edge2s, flows, imH, imW, oriH, oriW ):
    offset_height = tf.random_uniform( shape=[], minval=0, maxval=oriH-imH+1, dtype=tf.int32 )    
    offset_width  = tf.random_uniform( shape=[], minval=0, maxval=oriW-imW+1, dtype=tf.int32 )

    img1s = tf.image.crop_to_bounding_box( img1s, offset_height, offset_width, target_height=imH, target_width=imW )
    img2s = tf.image.crop_to_bounding_box( img2s, offset_height, offset_width, target_height=imH, target_width=imW )
    edge1s= tf.image.crop_to_bounding_box( edge1s,offset_height, offset_width, target_height=imH, target_width=imW )
    edge2s= tf.image.crop_to_bounding_box( edge2s,offset_height, offset_width, target_height=imH, target_width=imW )
    flows = tf.image.crop_to_bounding_box( flows, offset_height, offset_width, target_height=imH, target_width=imW )

    return img1s, img2s, edge1s, edge2s, flows

def random_rotate_scale_translate_no_edge(img1s, img2s, flows, imsize, name='Rand_R_S_T', RMAX=17.0, SMIN=0.9, SMAX=2.0, TMAX=0.2, RMAX0=3.0, SMIN0=0.9, SMAX0=1.1, TMAX0=0.03 ):
    """
    flows : float, defined on img1, shape: [B, H, W, 2]
    imsize : [H, W], default in form of numpy array
    """
    with tf.variable_scope(name):
        batchsize = tf.shape(img1s)[0]
        W = imsize[1]
        A, A_   = rand_affine( minR=-RMAX, maxR=RMAX, minS=SMIN, maxS=SMAX, minT=-TMAX*W, maxT=TMAX*W )
        A0, A0_ = rand_affine( minR=-RMAX0, maxR=RMAX0, minS=SMIN0, maxS=SMAX0, minT=-TMAX0*W, maxT=TMAX0*W )

        ff1 = aff_flow( A_, imsize, batchsize )
        img1s_ = backwarp.backwarper( img1s, ff1, imsize, name='img1s_')
        #edge1s_= backwarp.backwarper( edge1s,ff1, imsize, name='edge1s_')
        valid_map = tf.ones( shape=tf.stack([batchsize,imsize[0],imsize[1],1]) )
        valid_map = backwarp.backwarper( valid_map, ff1, imsize, name='valid_map' )
        gx = backwarp.backwarper( flows, ff1, imsize, name='warped_flow' )
        flows_ = flow_transform( A0, A, gx, imsize )

        A_A0_ = tf.matmul( A_, A0_ )
        ff2 = aff_flow( A_A0_, imsize, batchsize )
        img2s_ = backwarp.backwarper( img2s, ff2, imsize, name='img2s_' )
        #edge2s_= backwarp.backwarper( edge2s,ff2, imsize, name='edge2s_')
        valid_map_b = tf.ones( shape=tf.stack([batchsize,imsize[0],imsize[1],1]) )
        valid_map_b = backwarp.backwarper( valid_map_b, ff2, imsize, name='valid_map_b' )

        return img1s_, img2s_, flows_, valid_map, valid_map_b

def random_crop_no_edge( img1s, img2s, flows, imH, imW, oriH, oriW ):
    offset_height = tf.random_uniform( shape=[], minval=0, maxval=oriH-imH+1, dtype=tf.int32 )
    offset_width  = tf.random_uniform( shape=[], minval=0, maxval=oriW-imW+1, dtype=tf.int32 )

    img1s = tf.image.crop_to_bounding_box( img1s, offset_height, offset_width, target_height=imH, target_width=imW )
    img2s = tf.image.crop_to_bounding_box( img2s, offset_height, offset_width, target_height=imH, target_width=imW )
    #edge1s= tf.image.crop_to_bounding_box( edge1s,offset_height, offset_width, target_height=imH, target_width=imW )
    #edge2s= tf.image.crop_to_bounding_box( edge2s,offset_height, offset_width, target_height=imH, target_width=imW )
    flows = tf.image.crop_to_bounding_box( flows, offset_height, offset_width, target_height=imH, target_width=imW )

    return img1s, img2s, flows
