import tensorflow as tf
import cv2
import numpy as np

sess = tf.Session()
saver = tf.train.import_meta_graph('./model/trained_model.meta')
saver.restore(sess, './model/trained_model')


def get_class_no(img):
    inp = cv2.resize(img, (32, 32))
    inp = inp.reshape(1, 32, 32, 3)

    prediction = tf.get_default_graph().get_tensor_by_name("softmax:0")
    a = sess.run(prediction, feed_dict={"x:0": inp})
    return np.argmax(a)


