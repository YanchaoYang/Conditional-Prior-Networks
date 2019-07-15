import tensorflow as tf
import mylayers

def getNetwork( img1, img2, name, reuse=None, f=1.0, f_e=1.0, training=True ):
  with tf.variable_scope(name, reuse=reuse) as scope:
    batch_size = tf.shape(img1)[0]
    imgs = tf.concat( (img1,img2), 3 ) #[B, H, W, 6]
    C = 2

    aconv1 = mylayers.conv( imgs,    'aconv1', shape=[7,7,        6,    int(64*f_e)],  stride=2, reuse=reuse, training=training ) # h/2(192), 64
    aconv2 = mylayers.conv( aconv1,  'aconv2', shape=[5,5,int(64*f_e), int(128*f_e)],  stride=2, reuse=reuse, training=training ) # h/4(96),  128
    aconv3 = mylayers.conv( aconv2,  'aconv3', shape=[5,5,int(128*f_e),int(256*f_e)],  stride=2, reuse=reuse, training=training ) # h/8(48),  256
    aconv31= mylayers.conv( aconv3, 'aconv31', shape=[3,3,int(256*f_e),int(256*f_e)],  stride=1, reuse=reuse, training=training )
    aconv4 = mylayers.conv( aconv31, 'aconv4', shape=[3,3,int(256*f_e),int(512*f_e)],  stride=2, reuse=reuse, training=training ) # h/16(24), 512
    aconv41= mylayers.conv( aconv4, 'aconv41', shape=[3,3,int(512*f_e),int(512*f_e)],  stride=1, reuse=reuse, training=training )
    aconv5 = mylayers.conv( aconv41, 'aconv5', shape=[3,3,int(512*f_e),int(512*f_e)],  stride=2, reuse=reuse, training=training ) # h/32(12), 512
    aconv51= mylayers.conv( aconv5, 'aconv51', shape=[3,3,int(512*f_e),int(512*f_e)],  stride=1, reuse=reuse, training=training )
    aconv6 = mylayers.conv( aconv51, 'aconv6', shape=[3,3,int(512*f_e),int(1024*f_e)],  stride=2, reuse=reuse, training=training ) # h/64(6),  512*f_e
    aconv61= mylayers.conv( aconv6, 'aconv61', shape=[3,3,int(1024*f_e),int(512*f)],    stride=1, reuse=reuse, training=training ) # h/64(6),  512*f

    bconv1 = mylayers.conv( img1,    'bconv1', shape=[7,7,        3,  int(64*f)],  stride=2, reuse=reuse, training=False ) # h/2(192), 64
    bconv2 = mylayers.conv( bconv1,  'bconv2', shape=[5,5,int(64*f), int(128*f)],  stride=2, reuse=reuse, training=False ) # h/4(96),  128
    bconv3 = mylayers.conv( bconv2,  'bconv3', shape=[5,5,int(128*f),int(256*f)],  stride=2, reuse=reuse, training=False ) # h/8(48),  256
    bconv31= mylayers.conv( bconv3, 'bconv31', shape=[3,3,int(256*f),int(256*f)],  stride=1, reuse=reuse, training=False )
    bconv4 = mylayers.conv( bconv31, 'bconv4', shape=[3,3,int(256*f),int(512*f)],  stride=2, reuse=reuse, training=False ) # h/16(24), 512
    bconv41= mylayers.conv( bconv4, 'bconv41', shape=[3,3,int(512*f),int(512*f)],  stride=1, reuse=reuse, training=False )
    bconv5 = mylayers.conv( bconv41, 'bconv5', shape=[3,3,int(512*f),int(512*f)],  stride=2, reuse=reuse, training=False ) # h/32(12), 512
    bconv51= mylayers.conv( bconv5, 'bconv51', shape=[3,3,int(512*f),int(512*f)],  stride=1, reuse=reuse, training=False )
    bconv6 = mylayers.conv( bconv51, 'bconv6', shape=[3,3,int(512*f),int(512*f)],  stride=2, reuse=reuse, training=False ) # h/64(6),  512

    conv6 = tf.add( aconv61, bconv6 )
    outsz = bconv51.get_shape()                              # h/32(12), 512*f
    outsz = tf.stack([ batch_size, outsz[1], outsz[2], outsz[3] ])
    deconv5 = mylayers.deconv( conv6, size=[outsz[1],outsz[2]], name='deconv5', shape=[4,4,int(512*f),int(512*f)], reuse=reuse, training=False )
    concat5 = tf.concat( (deconv5,bconv51), 3 )              # h/32(12), 512*2*f

    flow5 = mylayers.conv( concat5, 'flow5', shape=[3,3,int(512*2*f),C], stride=1, reuse=reuse, training=False, activation=tf.identity ) # h/32(12), C

    outsz = bconv41.get_shape()                              # h/16(24), 512*f
    outsz = tf.stack([ batch_size, outsz[1], outsz[2], outsz[3] ])
    deconv4 = mylayers.deconv( concat5, size=[outsz[1],outsz[2]], name='deconv4', shape=[4,4,int(512*2*f),int(512*f)], reuse=reuse, training=False )
    upflow4 = mylayers.deconv( flow5,   size=[outsz[1],outsz[2]], name='upflow4', shape=[4,4,C,C], reuse=reuse, training=False, activation=tf.identity )
    concat4 = tf.concat( (deconv4,bconv41,upflow4), 3 )      # h/16(24), 512*2*f+C

    flow4 = mylayers.conv( concat4, 'flow4', shape=[3,3,int(512*2*f+C),C], stride=1, reuse=reuse, training=False, activation=tf.identity ) # h/16(24), C

    outsz = bconv31.get_shape()                              # h/8(48),  256*f
    outsz = tf.stack([ batch_size, outsz[1], outsz[2], outsz[3] ])
    deconv3 = mylayers.deconv( concat4, size=[outsz[1],outsz[2]], name='deconv3', shape=[4,4,int(512*2*f+C),int(256*f)], reuse=reuse, training=False )
    upflow3 = mylayers.deconv( flow4,   size=[outsz[1],outsz[2]], name='upflow3', shape=[4,4,C,C], reuse=reuse, training=False, activation=tf.identity )
    concat3 = tf.concat( (deconv3,bconv31,upflow3), 3 )      # h/8(48),  256*2*f+C
    
    flow3 = mylayers.conv( concat3, 'flow3', shape=[3,3,int(256*2*f+C),C], stride=1, reuse=reuse, training=False, activation=tf.identity ) # h/8(48), C

    outsz = bconv2.get_shape()                               # h/4(96),  128*f
    outsz = tf.stack([ batch_size, outsz[1], outsz[2], outsz[3] ])
    deconv2 = mylayers.deconv( concat3, size=[outsz[1],outsz[2]], name='deconv2', shape=[4,4,int(256*2*f+C),int(128*f)], reuse=reuse, training=False )
    upflow2 = mylayers.deconv( flow3,   size=[outsz[1],outsz[2]], name='upflow2', shape=[4,4,C,C], reuse=reuse, training=False, activation=tf.identity )
    concat2 = tf.concat( (deconv2,bconv2,upflow2), 3 )       # h/4(96),  128*2*f+C
    
    flow2 = mylayers.conv( concat2, 'flow2', shape=[3,3,int(128*2*f+C),C], stride=1, reuse=reuse, training=False, activation=tf.identity ) # h/4(96), C

    outsz = bconv1.get_shape()                               # h/2(192), 64*f
    outsz = tf.stack([ batch_size, outsz[1], outsz[2], outsz[3] ])
    deconv1 = mylayers.deconv( concat2, size=[outsz[1],outsz[2]], name='deconv1', shape=[4,4,int(128*2*f+C),int(64*f)], reuse=reuse, training=False )
    upflow1 = mylayers.deconv( flow2,   size=[outsz[1],outsz[2]], name='upflow1', shape=[4,4,C,C], reuse=reuse, training=False, activation=tf.identity )
    concat1 = tf.concat( (deconv1,bconv1,upflow1), 3 )       # h/2(192), 64*2*f+C

    flow1 = mylayers.conv( concat1, 'flow1', shape=[5,5,int(64*2*f+C),C], stride=1, reuse=reuse, training=False, activation=tf.identity ) # h/2(192), C

    outsz = img1.get_shape()
    outsz = tf.stack([ batch_size, outsz[1], outsz[2], outsz[3] ])
    flow0 = tf.image.resize_images( flow1, size=[outsz[1],outsz[2]] )

    return flow5, flow4, flow3, flow2, flow1, flow0
