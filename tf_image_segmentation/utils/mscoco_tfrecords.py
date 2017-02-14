
# coding: utf-8

# In[1]:

# Update the following four paths
images_dir = '/mnt/disk2/mscoco/train2014/'
seg_map_dir = '/mnt/disk2/mscoco/train2014/seg_map/'
# This file contains the list of file names with segmentation maps
# Note that there is no extension.
# e.g.
# COCO_val2014_000000000042
# COCO_val2014_000000000073
# COCO_val2014_000000000074
# COCO_val2014_000000000133
# COCO_val2014_000000000136
# ...

list_file = '/mnt/disk2/mscoco/train2014/seg_map/list.txt' 

tf_records_filename = '/mnt/disk2/mscoco/mscoco_train2014.tfrecords'


# In[ ]:

# Get some image/annotation pairs for example 
filename_pairs = []
with open(list_file, 'r') as ff:
    fnames = ff.readlines()
    for fname in fnames:
        fname = fname.rstrip('\n')
        pair = (os.path.join(images_dir, fname + '.jpg'), os.path.join(seg_map_dir, fname + '.png'))
        filename_pairs.append(pair)


# In[ ]:

# %matplotlib inline

# Important: We are using PIL to read .png files later.
# This was done on purpose to read indexed png files
# in a special way -- only indexes and not map the indexes
# to actual rgb values. This is specific to PASCAL VOC
# dataset data. If you don't want thit type of behaviour
# consider using skimage.io.imread()
from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# Let's collect the real images to later on compare
# to the reconstructed ones
original_images = []
i = 0

for img_path, annotation_path in filename_pairs:
    
    img = np.array(Image.open(img_path))
    if img.ndim == 2:
        img2 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8 )
        img2[:,:,1] = img
        img2[:,:,2] = img
        img = img2
    annotation = np.array(Image.open(annotation_path))
    
    # The reason to store image sizes was demonstrated
    # in the previous example -- we have to know sizes
    # of images to later read raw serialized string,
    # convert to 1d array and convert to respective
    # shape that image used to have.
    height = img.shape[0]
    width = img.shape[1]
    
    # Put in the original images into array
    # Just for future check for correctness
    #original_images.append((img, annotation))
    
    img_raw = img.tostring()
    annotation_raw = annotation.tostring()
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw),
        'mask_raw': _bytes_feature(annotation_raw)}))
    
    writer.write(example.SerializeToString())
    
    if i%1000 == 0:
        print("Processed " + str(i) + " images...")
    i = i+1
print("Processed " + str(i) + " images...")
print("Done!")

writer.close()


# In[34]:

reconstructed_images = []

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
    reconstructed_img = img_1d.reshape((height, width, -1))
    
    annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
    
    # Annotations don't have depth (3rd dimension)
    reconstructed_annotation = annotation_1d.reshape((height, width))
    
    reconstructed_images.append((reconstructed_img, reconstructed_annotation))
    


# In[9]:

# Let's check if the reconstructed images match
# the original images

for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
    
    img_pair_to_compare, annotation_pair_to_compare = zip(original_pair,
                                                          reconstructed_pair)
    print(np.allclose(*img_pair_to_compare))
    print(np.allclose(*annotation_pair_to_compare))


