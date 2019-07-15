import tensorflow as tf

def backwarper( Im, flow, imsize, name='BackWarper'):
    """
    Im : float, image2,  shape: [B, H, W, C].
    flow : float, defined on image1, shape: [B, H, W, 2]
    imsize : [H, W]
    """

    def _repeat( x, n_repeats ):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose( tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])),1), [1, 0] )
            rep = tf.cast( rep, 'int32' )
            x = tf.matmul( tf.reshape(x, (-1, 1)), rep )
            return tf.reshape(x, [-1])

    def _interpolate( im, x, y, out_size ):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            channels  = tf.shape(im)[3]
            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            out_height = out_size[0]
            out_width  = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast( tf.shape(im)[1] - 1, 'int32' )
            max_x = tf.cast( tf.shape(im)[2] - 1, 'int32' )

            # do sampling
            x0 = tf.cast( tf.floor(x), 'int32' )
            x1 = x0 + 1
            y0 = tf.cast( tf.floor(y), 'int32' )
            y1 = y0 + 1
            x0 = tf.clip_by_value( x0, zero, max_x )
            x1 = tf.clip_by_value( x1, zero, max_x )
            y0 = tf.clip_by_value( y0, zero, max_y )
            y1 = tf.clip_by_value( y1, zero, max_y )
            dim2 = out_width
            dim1 = out_width*out_height
            base = _repeat( tf.range(num_batch)*dim1, out_height*out_width )
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape( im, tf.stack([-1, channels]) )
            im_flat = tf.cast( im_flat, 'float32' )
            Ia = tf.gather( im_flat, idx_a )
            Ib = tf.gather( im_flat, idx_b )
            Ic = tf.gather( im_flat, idx_c )
            Id = tf.gather( im_flat, idx_d )

            # and finally calculate interpolated values
            x0_f = tf.cast( x0, 'float32' )
            x1_f = tf.cast( x1, 'float32' )
            y0_f = tf.cast( y0, 'float32' )
            y1_f = tf.cast( y1, 'float32' )
            wa = tf.expand_dims( ((x1_f-x) * (y1_f-y)), 1 )
            wb = tf.expand_dims( ((x1_f-x) * (y-y0_f)), 1 )
            wc = tf.expand_dims( ((x-x0_f) * (y1_f-y)), 1 )
            wd = tf.expand_dims( ((x-x0_f) * (y-y0_f)), 1 )
            output = tf.add_n( [wa*Ia, wb*Ib, wc*Ic, wd*Id] )
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
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

            grid = tf.concat( axis=0,values=[x_t_flat, y_t_flat] ) # [ 2 , HW ]
            return grid

    def _transform( flow, Im, imsize ):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(Im)[0]
            num_channels = tf.shape(Im)[3]
            fu = flow[:,:,:,0]
            fv = flow[:,:,:,1]
            fu = tf.reshape( fu, tf.stack([num_batch, 1, -1]) )
            fv = tf.reshape( fv, tf.stack([num_batch, 1, -1]) )  # [B, 1, HW]
            flow_flat = tf.concat( [fu,fv], axis=1 )    # [B, 2, HW]
            flow_flat = tf.cast( flow_flat, 'float32' )

            # grid on domain of image1
            out_height = imsize[0]
            out_width  = imsize[1]
            grid = _meshgrid(out_height, out_width)  # [2, HW]
            grid = tf.expand_dims(grid, 0)   # [1, 2, HW]
            grid = tf.reshape(grid, [-1])    # [2*HW,]
            grid = tf.tile(grid, tf.stack([num_batch]))  # B copies of 2*HW
            grid = tf.reshape(grid, tf.stack([num_batch, 2, -1]))             # [B, 2, HW]

            # (x_t, y_t) -> (x_s, y_s)
            T_g = tf.add( flow_flat, grid ) # [B, 2, HW]
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape( x_s, [-1] )
            y_s_flat = tf.reshape( y_s, [-1] )

            input_transformed = _interpolate( Im, x_s_flat, y_s_flat, imsize )

            output = tf.reshape( input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]) )
            return output

    with tf.variable_scope(name):
        output = _transform( flow, Im, imsize )
        return output
