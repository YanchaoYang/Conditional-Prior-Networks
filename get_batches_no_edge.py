import tensorflow as tf

def read_and_decode(filename_queue, IMG_HEIGHT, IMG_WIDTH):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'img1_raw':  tf.FixedLenFeature([IMG_HEIGHT,IMG_WIDTH,3], tf.float32),
            'img2_raw':  tf.FixedLenFeature([IMG_HEIGHT,IMG_WIDTH,3], tf.float32),
            #'edge1_raw': tf.FixedLenFeature([IMG_HEIGHT,IMG_WIDTH,1], tf.float32),
            #'edge2_raw': tf.FixedLenFeature([IMG_HEIGHT,IMG_WIDTH,1], tf.float32),
            'flow_raw':  tf.FixedLenFeature([IMG_HEIGHT,IMG_WIDTH,2], tf.float32)
        })
    img1  = features['img1_raw']
    img2  = features['img2_raw']
    #edge1 = features['edge1_raw']
    #edge2 = features['edge2_raw']
    flow  = features['flow_raw']
    return img1, img2, flow

def inputs(filename, batch_size, IMG_HEIGHT, IMG_WIDTH, num_epochs=None, capp=2000):
    """Reads input data num_epochs times.
    Args:
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to train forever.
    Returns:
      A tuple (img, gtc), where:
      * image is a float tensor with shape [batch_size, H, W, 3]
        in the range [-0.5, 0.5]. Same to gtc.
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
    """
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename queue.
        img1, img2, flow = read_and_decode( filename_queue, IMG_HEIGHT, IMG_WIDTH )

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        img1s, img2s, flows = tf.train.shuffle_batch( [img1, img2, flow],
            batch_size=batch_size, num_threads=2, capacity=capp + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=capp)

    return img1s, img2s, flows
