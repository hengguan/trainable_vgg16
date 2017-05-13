########################################################################################
# Guan Heng, 2017.5.10                                                                 #
# trainable VGG16 implementation in TensorFlow                                         #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://github.com/machrisaa/tensorflow-vgg                               #
# Weights from https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM  #
########################################################################################

import tensorflow as tf
from scipy.misc import imsave
from vgg16 import Vgg16
import utils
import time

IMAGE_SIZE = 224

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_integer("max_epochs", "10000", "the number of iterate in training")
tf.flags.DEFINE_integer("num_of_classes", "1000", "the num of class for training or inference")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string('mode', "infer", "Mode train/ test/ infer")


def main(argv=None):
    if len(argv) >= 2:
        if argv[1] != "train" and argv[1] != "infer":
            print "the string of your input is not right," \
                  " you only input 'train' for training model, and 'infer' for inference"
            exit(0)
    else:
        print "please input the mode of implement, " \
              "note that 'train' for training model, and 'infer' for inference"
        exit(0)
    FLAGS.mode = argv[1]
    images = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    labels = tf.placeholder(tf.float32, shape=[None, FLAGS.num_of_classes], name="labels")
    train_mode = tf.placeholder(tf.bool)

    sess = tf.Session()

    vgg = Vgg16('vgg16.npy')
    vgg.build(images, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    opts, loss = vgg.train_opts(labels, FLAGS.learning_rate)

    sess.run(tf.global_variables_initializer())

    if FLAGS.mode == "train":
        # read training dataset

        for epoch in xrange(FLAGS.max_epochs):
            # train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            # valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {images: train_images, labels: train_labels, train_mode: True}
            sess.run(opts, feed_dict=feed_dict)

            if epoch % 10 == 0:
                train_loss = sess.run(loss, feed_dict=feed_dict)
                print("Epoch: %d, ------> loss:%g" % (epoch, train_loss))

            if epoch % (MAX_EPOCHS - 1) == 0:
                vgg.save_npy(sess, './vgg16_weights.npy')

    if FLAGS.mode == "infer":
        start_time = time.time()
        # read images for inference
        img1 = utils.load_image("./test_data/image10.jpg")
        # img1_label = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger
        batch = img1.reshape((1, 224, 224, 3))

        # test classification again, should have a higher probability about tiger
        prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
        utils.print_prob(prob[0], './synset.txt')
        print "inference time:", time.time()-start_time


if __name__ == "__main__":
    tf.app.run()
