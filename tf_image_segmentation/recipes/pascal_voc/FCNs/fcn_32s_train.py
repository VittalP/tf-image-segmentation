import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
from matplotlib import pyplot as plt

lr = float(sys.argv[1])
_lambda = float(sys.argv[2])
exp = sys.argv[3]
sys.path.append(os.path.join(os.path.join(os.getcwd(), "../../../../../tf-image-segmentation")))
from tf_image_segmentation.utils import set_paths # Sets appropriate paths and provides access to log_dir and checkpoint_path via FLAGS

FLAGS = set_paths.FLAGS

checkpoints_dir = FLAGS.checkpoints_dir
log_dir = os.path.join(FLAGS.log_dir, exp, "fcn-32s/")
FLAGS.save_dir = os.path.join(FLAGS.save_dir, exp, "fcn-32s/")


slim = tf.contrib.slim
vgg_checkpoint_path = os.path.join(checkpoints_dir, 'vgg_16.ckpt')

if not os.path.isfile(vgg_checkpoint_path):
    import tf_image_segmentation.utils.download_ckpt as dl_ckpt
    dl_ckpt.download_ckpt('http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz')

from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from tf_image_segmentation.models.fcn_32s import FCN_32s, extract_vgg_16_mapping_without_fc8

from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut

from tf_image_segmentation.utils.training import get_valid_logits_and_labels

from tf_image_segmentation.utils.augmentation import (distort_randomly_image_color,
                                                      flip_randomly_left_right_image_with_annotation,
                                                      scale_randomly_image_with_annotation_with_fixed_size_output)

image_train_size = [384, 384]
number_of_classes = 21
number_of_part_classes = 195
num_epochs = 10
tfrecord_filename = os.path.join(FLAGS.data_dir, 'pascal_part_augmented_train.tfrecords')
num_training_images = 11127
pascal_voc_lut = pascal_segmentation_lut()
class_labels = pascal_voc_lut.keys()

filename_queue = tf.train.string_input_producer(
    [tfrecord_filename], num_epochs=num_epochs)

data_tuple = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue, part_=True)
image = data_tuple[0]
annotation = data_tuple[1]
if len(data_tuple) == 3:
    part_annotation = data_tuple[2]

# Various data augmentation stages
image, annotation, part_annotation = flip_randomly_left_right_image_with_annotation(image, annotation,
                                                                                    part_annotation_tensor=part_annotation)

# image = distort_randomly_image_color(image)

resized_image, resized_annotation, resized_part_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(image,
                                                                                                                         annotation,
                                                                                                                         output_shape=image_train_size,
                                                                                                                         part_annotation_tensor=part_annotation)


resized_annotation = tf.squeeze(resized_annotation)
resized_part_annotation = tf.squeeze(resized_part_annotation)

image_batch, annotation_batch, part_annotation_batch = tf.train.shuffle_batch( [resized_image, resized_annotation, resized_part_annotation],
                                             batch_size=1,
                                             capacity=3000,
                                             num_threads=2,
                                             min_after_dequeue=1000)

upsampled_logits_batch, upsampled_part_logits_batch, vgg_16_variables_mapping = FCN_32s(image_batch_tensor=image_batch,
                                                           number_of_classes=number_of_classes,
                                                           number_of_part_classes=number_of_part_classes,
                                                           is_training=True)


valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(annotation_batch_tensor=annotation_batch,
                                                                                   logits_batch_tensor=upsampled_logits_batch,
                                                                                   class_labels=class_labels)

valid_part_labels_batch_tensor, valid_part_logits_batch_tensor = get_valid_logits_and_labels(annotation_batch_tensor=part_annotation_batch,
                                                                                             logits_batch_tensor=upsampled_part_logits_batch,
                                                                                             class_labels=[ii+1 for ii in range(number_of_part_classes)] + [255])


cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                          labels=valid_labels_batch_tensor)

part_cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_part_logits_batch_tensor,
                                                          labels=valid_part_labels_batch_tensor)

# Normalize the cross entropy -- the number of elements
# is different during each step due to mask out regions

object_cross_entropy = tf.reduce_mean(cross_entropies)
part_cross_entropy = tf.reduce_mean(part_cross_entropies)

loss = object_cross_entropy + _lambda * part_cross_entropy

pred = tf.argmax(upsampled_logits_batch, dimension=3)

probabilities = tf.nn.softmax(upsampled_logits_batch)


with tf.variable_scope("adam_vars"):
    obj_lr_rate = tf.placeholder(tf.float32, shape=[])
    train_step_object = tf.train.AdamOptimizer(learning_rate=obj_lr_rate).minimize(object_cross_entropy)
    part_lr_rate = tf.placeholder(tf.float32, shape=[])
    train_step_part = tf.train.AdamOptimizer(learning_rate=part_lr_rate).minimize(part_cross_entropy)


# Variable's initialization functions
vgg_16_without_fc8_variables_mapping = extract_vgg_16_mapping_without_fc8(vgg_16_variables_mapping)


init_fn = slim.assign_from_checkpoint_fn(model_path=vgg_checkpoint_path,
                                         var_list=vgg_16_without_fc8_variables_mapping)

global_vars_init_op = tf.global_variables_initializer()

tf.summary.scalar('cross_entropy_loss', object_cross_entropy)

tf.summary.scalar('full_loss', loss)

merged_summary_op = tf.summary.merge_all()

summary_string_writer = tf.summary.FileWriter(log_dir)

# Create the log folder if doesn't exist yet
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

#The op for initializing the variables.
local_vars_init_op = tf.local_variables_initializer()

combined_op = tf.group(local_vars_init_op, global_vars_init_op)

# We need this to save only model variables and omit
# optimization-related and other variables.
model_variables = slim.get_model_variables()
saver = tf.train.Saver(model_variables)


with tf.Session()  as sess:

    sess.run(combined_op)
    init_fn(sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # 10 epochs
    for i in xrange(num_training_images * num_epochs):
	feed_dict = {obj_lr_rate: np.asarray( 0.00001 * ( (1 - i/(num_training_images*num_epochs))**0.9)),
		     part_lr_rate: np.asarray( lr * ( (1 - i/(num_training_images*num_epochs))**0.9))}
        cross_entropy, summary_string, _, _ = sess.run([loss,
                                                     merged_summary_op,
                                                     train_step_object,
						     train_step_part], feed_dict=feed_dict)

        print("Current loss: " + str(cross_entropy))

        summary_string_writer.add_summary(summary_string, i)

        if i > 0 and i % num_training_images == 0:
            save_path = saver.save(sess, FLAGS.save_dir + "model_fcn32s_epoch_" + str(i) + ".ckpt")
            print("Model saved in file: %s" % save_path)


    coord.request_stop()
    coord.join(threads)

    save_path = saver.save(sess, FLAGS.save_dir + "model_fcn32s_final.ckpt")
    print("Model saved in file: %s" % save_path)

summary_string_writer.close()
