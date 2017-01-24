import tensorflow as tf
import sys
import os

# Use second GPU -- change if you want to use a first one
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

flags = tf.app.flags

machine = '4GPU'

if machine == '4GPU':
    # Add a path to a custom fork of TF-Slim
    # Get it from here:
    # https://github.com/warmspringwinds/models/tree/fully_conv_vgg


    flags.DEFINE_string("slim_path", "/home/vittal/work/tf-slim-models/models/slim", "The path to tf slim repo")

    # Add path to the cloned library
    flags.DEFINE_string("tf_image_seg_dir", "/home/vittal/work/segmentation/tf-image-segmentation/", "Dir for tf-image-segmentation repo")
    flags.DEFINE_string("checkpoints_dir", "/home/vittal/work/ckpts/", "Directory where checkpoints are saved")
    flags.DEFINE_string("log_dir", "/home/vittal/work/segmentation/tf-image-segmentation/log_dir/", "Directory to save TF logs")
    flags.DEFINE_string("save_dir", "/home/vittal/work/segmentation/tf-image-segmentation/save_dir/", "Directory to save checkpoint models")

if machine == 'daniil':
    # Add a path to a custom fork of TF-Slim
    # Get it from here:
    # https://github.com/warmspringwinds/models/tree/fully_conv_vgg


    flags.DEFINE_string("slim_path", "/home/dpakhom1/workspace/my_models/slim/", "The path to tf slim repo")

    # Add path to the cloned library
    flags.DEFINE_string("tf_image_seg_dir", "/home/dpakhom1/tf_projects/segmentation/tf-image-segmentation/", "Dir for tf-image-segmentation repo")
    flags.DEFINE_string("checkpoints_dir", "/home/dpakhom1/checkpoints/", "Directory where checkpoints are saved")
    flags.DEFINE_string("log_dir", "/home/dpakhom1/tf_projects/segmentation/log_folder/", "Directory to save TF logs")
    flags.DEFINE_string("save_dir", "/home/dpakhom1/tf_projects/segmentation/", "Directory to save checkpoint models")

FLAGS = flags.FLAGS

sys.path.append(FLAGS.slim_path)
sys.path.append(FLAGS.tf_image_seg_dir)