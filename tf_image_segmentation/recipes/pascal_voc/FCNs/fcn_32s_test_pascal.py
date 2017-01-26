
# coding: utf-8

# In[1]:

#get_ipython().magic(u'matplotlib inline')

import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
from PIL import Image
from matplotlib import pyplot as plt
import set_paths

FLAGS = set_paths.FLAGS

slim = tf.contrib.slim

from tf_image_segmentation.models.fcn_32s import FCN_32s

from matplotlib import pyplot as plt
from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut
from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input
from tf_image_segmentation.utils.visualization import visualize_segmentation_adaptive

pascal_voc_lut = pascal_segmentation_lut()

tfrecord_filename = 'pascal_augmented_val.tfrecords'

number_of_classes = 21

filename_queue = tf.train.string_input_producer(
    [tfrecord_filename], num_epochs=1)

image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

# Fake batch for image and annotation by adding
# leading empty axis.
image_batch_tensor = tf.expand_dims(image, axis=0)
annotation_batch_tensor = tf.expand_dims(annotation, axis=0)

# Be careful: after adaptation, network returns final labels
# and not logits
FCN_32s = adapt_network_for_any_size_input(FCN_32s, 32)


pred, fcn_32s_variables_mapping = FCN_32s(image_batch_tensor=image_batch_tensor,
                                          number_of_classes=number_of_classes,
                                          is_training=False)

# Take away the masked out values from evaluation
weights = tf.to_float( tf.not_equal(annotation_batch_tensor, 255) )

# Define the accuracy metric: Mean Intersection Over Union
miou, update_op = slim.metrics.streaming_mean_iou(predictions=pred,
                                                   labels=annotation_batch_tensor,
                                                   num_classes=number_of_classes,
                                                   weights=weights)

# The op for initializing the variables.
initializer = tf.local_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    
    sess.run(initializer)

    saver.restore(sess, FLAGS.save_dir + "model_fcn32s_final.ckpt")
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    # There are 904 images in restricted validation dataset
    for i in xrange(904):
        
        image_np, annotation_np, pred_np, tmp = sess.run([image, annotation, pred, update_op])
        
        # Display the image and the segmentation result
        # upsampled_predictions = pred_np.squeeze()
        #plt.imshow(image_np)
        #plt.show()
        #visualize_segmentation_adaptive(upsampled_predictions, pascal_voc_lut)
        
    coord.request_stop()
    coord.join(threads)
    
    res = sess.run(miou)
    
    print("Pascal VOC 2012 Restricted (RV-VOC12) Mean IU: " + str(res))

