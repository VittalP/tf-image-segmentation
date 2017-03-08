# Important: We are using PIL to read .png files later.
# This was done on purpose to read indexed png files
# in a special way -- only indexes and not map the indexes
# to actual rgb values. This is specific to PASCAL VOC
# dataset data. If you don't want thit type of behaviour
# consider using skimage.io.imread()
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Helper functions for defining tf types


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_image_annotation_pairs_to_tfrecord(filename_pairs, tfrecords_filename):
    """Writes given image/annotation pairs to the tfrecords file.
    The function reads each image/annotation pair given filenames
    of image and respective annotation and writes it to the tfrecord
    file.
    Parameters
    ----------
    filename_pairs : array of tuples (img_filepath, annotation_filepath, [part_annotation_path])
        Array of tuples of image/annotation/[part_annotation_path] filenames

        [part_annotation_path] is optional -- is present if tfrecords needs to
        be created along with part-level annotations.

    tfrecords_filename : string
        Tfrecords filename to write the image/annotation pairs
    """
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    i = 0
    for ii in range(len(filename_pairs)):
        ele = filename_pairs[ii]
        img_path = ele[0]
        annotation_path = ele[1]
        part_ = False
        if(len(ele) == 3):
            part_annotation_path = ele[2]
            part_ = True

        img = np.array(Image.open(img_path))
        if img.ndim == 2:
            img2 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img2[:, :, 1] = img
            img2[:, :, 2] = img
            img = img2

        annotation = np.array(Image.open(annotation_path))
        #print(annotation.shape)
        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,
        # convert to 1d array and convert to respective
        # shape that image used to have.
        height = img.shape[0]
        width = img.shape[1]

        img_raw = img.tostring()
        annotation_raw = annotation.tostring()

        feature = {
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw)}

        if part_ is True:
            if os.path.exists(part_annotation_path):
                part_annotation = np.array(Image.open(part_annotation_path))
            else:
                part_annotation = np.full(shape=(annotation.shape[0], annotation.shape[1]),
                                          fill_value=1102,
                                          dtype=np.int32)  # Initialize with 255 (ignored while training)
            part_annotation_raw = part_annotation.tostring()
            feature['part_mask_raw'] = _bytes_feature(part_annotation_raw)
        if(img.shape[:2] != annotation.shape or img.shape[:2] != part_annotation.shape):
            print(img_path)
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())
        if i % 1000 == 0:
            print("Processed " + str(i) + " images...")
        i = i+1

    print("Processed " + str(i) + " images...")
    print("Done!")

    writer.close()


def read_image_annotation_pairs_from_tfrecord(tfrecords_filename):
    """Return image/annotation pairs from the tfrecords file.
    The function reads the tfrecords file and returns image
    and respective annotation matrices pairs.
    Parameters
    ----------
    tfrecords_filename : string
        filename of .tfrecords file to read from

    Returns
    -------
    image_annotation_pairs : array of tuples (img, annotation)
        The image and annotation that were read from the file
    """

    image_annotation_pairs = []

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height']
                                     .int64_list
                                     .value[0])

        width = int(example.features.feature['width']
                                    .int64_list
                                    .value[0])

        img_string = (example.features.feature['image_raw']
                                      .bytes_list
                                      .value[0])

        annotation_string = (example.features.feature['mask_raw']
                                    .bytes_list
                                    .value[0])

        img_1d = np.fromstring(img_string, dtype=np.uint8)
        img = img_1d.reshape((height, width, -1))

        annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)

        # Annotations don't have depth (3rd dimension)
        # TODO: check if it works for other datasets
        annotation = annotation_1d.reshape((height, width))

        image_annotation_pairs.append((img, annotation))

    return image_annotation_pairs


def read_tfrecord_and_decode_into_image_annotation_pair_tensors(tfrecord_filenames_queue, part_=False):
    """Return image/annotation tensors that are created by reading tfrecord file.
    The function accepts tfrecord filenames queue as an input which is usually
    can be created using tf.train.string_input_producer() where filename
    is specified with desired number of epochs. This function takes queue
    produced by aforemention tf.train.string_input_producer() and defines
    tensors converted from raw binary representations into
    reshaped image/annotation tensors.
    Parameters
    ----------
    tfrecord_filenames_queue : tfrecord filename queue
        String queue object from tf.train.string_input_producer()
    part_ : a boolean flag that indicates whether the tfrecords contains
        part-level annotation.

    Returns
    -------
    image, annotation, [part_annotaion]: tuple of tf.int32 (image, annotation, [part_annotation])
        Tuple of image/annotation/[part_annotation] tensors
        [part_annotation] is returned if tfrecords contains part-level annotations
    """

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(tfrecord_filenames_queue)

    features = {
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64),
      'image_raw': tf.FixedLenFeature([], tf.string),
      'mask_raw': tf.FixedLenFeature([], tf.string)
      }

    if part_ is True:
        features['part_mask_raw'] = tf.FixedLenFeature([], tf.string)

    features = tf.parse_single_example(
      serialized_example,
      features)

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)

    if part_ is True:
        part_annotation = tf.decode_raw(features['part_mask_raw'], tf.int32)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image_shape = tf.pack([height, width, 3])

    # The last dimension was added because
    # the tf.resize_image_with_crop_or_pad() accepts tensors
    # that have depth. We need resize and crop later.
    # TODO: See if it is necessary and probably remove third
    # dimension
    annotation_shape = tf.pack([height, width, 1])

    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)

    if part_ is True:
        part_annotation = tf.reshape(part_annotation, annotation_shape)
        return image, annotation, part_annotation
    else:
        return image, annotation
