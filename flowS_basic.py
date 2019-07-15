import tensorflow as tf
import mylayers

def getNetwork( img1, img2, name, orisize, reuse=None, f=1.0, training=True ):
  with tf.variable_scope(name, reuse=reuse) as scope:
    C = 2
    batch_size = tf.shape(img1)[0]
    imgs = tf.concat( (img1,img2), 3 ) #[B, H, W, 6]

    conv1 = mylayers.conv( imgs,   'conv1', shape=[7,7,        6,  int(64*f)],  stride=2, reuse=reuse, training=training ) # h/2(192), 64
    conv2 = mylayers.conv( conv1,  'conv2', shape=[5,5,int(64*f), int(128*f)],  stride=2, reuse=reuse, training=training ) # h/4(96),  128
    conv3 = mylayers.conv( conv2,  'conv3', shape=[5,5,int(128*f),int(256*f)],  stride=2, reuse=reuse, training=training ) # h/8(48),  256
    conv31= mylayers.conv( conv3, 'conv31', shape=[3,3,int(256*f),int(256*f)],  stride=1, reuse=reuse, training=training )
    conv4 = mylayers.conv( conv31, 'conv4', shape=[3,3,int(256*f),int(512*f)],  stride=2, reuse=reuse, training=training ) # h/16(24), 512
    conv41= mylayers.conv( conv4, 'conv41', shape=[3,3,int(512*f),int(512*f)],  stride=1, reuse=reuse, training=training )
    conv5 = mylayers.conv( conv41, 'conv5', shape=[3,3,int(512*f),int(512*f)],  stride=2, reuse=reuse, training=training ) # h/32(12), 512
    conv51= mylayers.conv( conv5, 'conv51', shape=[3,3,int(512*f),int(512*f)],  stride=1, reuse=reuse, training=training )
    conv6 = mylayers.conv( conv51, 'conv6', shape=[3,3,int(512*f),int(1024*f)], stride=2, reuse=reuse, training=training ) # h/64(6), 1024

    outsz = conv51.get_shape()                              # h/32(12), 512*f
    outsz = tf.stack([ batch_size, outsz[1], outsz[2], outsz[3] ])
    deconv5 = mylayers.deconv( conv6, size=[outsz[1], outsz[2]], name='deconv5', shape=[4,4,int(1024*f),int(512*f)], reuse=reuse, training=training )
    concat5 = tf.concat( (deconv5,conv51), 3 )              # h/32(12), 512*2*f

    flow5 = mylayers.conv( concat5, 'flow5', shape=[3,3,int(512*2*f),C], stride=1, reuse=reuse, training=training, activation=tf.identity ) # h/32(12), C

    outsz = conv41.get_shape()                              # h/16(24), 512*f
    outsz = tf.stack([ batch_size, outsz[1], outsz[2], outsz[3] ])
    deconv4 = mylayers.deconv( concat5, size=[outsz[1],outsz[2]], name='deconv4', shape=[4,4,int(512*2*f),int(512*f)], reuse=reuse, training=training )
    upflow4 = mylayers.deconv( tf.multiply(flow5,tf.constant(2.0)), size=[outsz[1],outsz[2]], name='upflow4', shape=[4,4,C,C], reuse=reuse, training=training, activation=tf.identity )
    concat4 = tf.concat( (deconv4,conv41,upflow4), 3 )      # h/16(24), 512*2*f+C

    flow4 = mylayers.conv( concat4, 'flow4', shape=[3,3,int(512*2*f+C),C], stride=1, reuse=reuse, training=training, activation=tf.identity ) # h/16(24), C

    outsz = conv31.get_shape()                              # h/8(48),  256*f
    outsz = tf.stack([ batch_size, outsz[1], outsz[2], outsz[3] ])
    deconv3 = mylayers.deconv( concat4, size=[outsz[1],outsz[2]], name='deconv3', shape=[4,4,int(512*2*f+C),int(256*f)], reuse=reuse, training=training )
    upflow3 = mylayers.deconv( tf.multiply(flow4,tf.constant(2.0)), size=[outsz[1],outsz[2]], name='upflow3', shape=[4,4,C,C], reuse=reuse, training=training, activation=tf.identity )
    concat3 = tf.concat( (deconv3,conv31,upflow3), 3 )      # h/8(48),  256*2*f+C
    
    flow3 = mylayers.conv( concat3, 'flow3', shape=[3,3,int(256*2*f+C),C], stride=1, reuse=reuse, training=training, activation=tf.identity ) # h/8(48), C

    outsz = conv2.get_shape()                               # h/4(96),  128*f
    outsz = tf.stack([ batch_size, outsz[1], outsz[2], outsz[3] ])
    deconv2 = mylayers.deconv( concat3, size=[outsz[1],outsz[2]], name='deconv2', shape=[4,4,int(256*2*f+C),int(128*f)], reuse=reuse, training=training )
    upflow2 = mylayers.deconv( tf.multiply(flow3,tf.constant(2.0)), size=[outsz[1],outsz[2]], name='upflow2', shape=[4,4,C,C], reuse=reuse, training=training, activation=tf.identity )
    concat2 = tf.concat( (deconv2,conv2,upflow2), 3 )       # h/4(96),  128*2*f+C
    
    flow2 = mylayers.conv( concat2, 'flow2', shape=[3,3,int(128*2*f+C),C], stride=1, reuse=reuse, training=training, activation=tf.identity ) # h/4(96), C

    outsz = conv1.get_shape()                               # h/2(192), 64*f
    outsz = tf.stack([ batch_size, outsz[1], outsz[2], outsz[3] ])
    deconv1 = mylayers.deconv( concat2, size=[outsz[1],outsz[2]], name='deconv1', shape=[4,4,int(128*2*f+C),int(64*f)], reuse=reuse, training=training )
    upflow1 = mylayers.deconv( tf.multiply(flow2,tf.constant(2.0)), size=[outsz[1],outsz[2]], name='upflow1', shape=[4,4,C,C], reuse=reuse, training=training, activation=tf.identity )
    concat1 = tf.concat( (deconv1,conv1,upflow1), 3 )       # h/2(192), 64*2*f+C

    flow1 = mylayers.conv( concat1, 'flow1', shape=[5,5,int(64*2*f+C),C], stride=1, reuse=reuse, training=training, activation=tf.identity ) # h/2(192), C

    #outsz = img1.get_shape()
    #outsz = tf.stack([ batch_size, outsz[1], outsz[2], outsz[3] ])
    flow0 = tf.image.resize_images( tf.multiply(flow1,tf.constant(2.0)), size=orisize )

    return flow5, flow4, flow3, flow2, flow1, flow0
