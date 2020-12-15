import tensorflow as tf


def load():

  def read_image(image):
    image = tf.io.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)/255.0
    image = tf.reshape(image, [28,28,1])
    return image
  
  def read_label(label):
    label = tf.io.decode_raw(label, tf.uint8)
    label = tf.reshape(label, [])
    return tf.cast(label, tf.int32)

  # train set
  tr_images= tf.data.FixedLengthRecordDataset('train-images-idx3-ubyte.gz', 28*28, header_bytes=16, compression_type='GZIP').map(read_image)
  tr_labels= tf.data.FixedLengthRecordDataset('train-labels-idx1-ubyte.gz', 1, header_bytes=8, compression_type='GZIP').map(read_label)

  # test set
  te_images= tf.data.FixedLengthRecordDataset('t10k-images-idx3-ubyte.gz', 28*28, header_bytes=16, compression_type='GZIP').map(read_image)
  te_labels= tf.data.FixedLengthRecordDataset('t10k-labels-idx1-ubyte.gz', 1, header_bytes=8, compression_type='GZIP').map(read_label)

  return dict({"train": tf.data.Dataset.zip((tr_images, tr_labels)), \
    "test": tf.data.Dataset.zip((te_images, te_labels)) })
