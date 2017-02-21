import tensorflow as tf
import sys
import os
import socket

flags = tf.app.flags

exp = 'part'
machine = socket.gethostname()

if machine == 'ccvl-4gpu':
    # Add a path to a custom fork of TF-Slim
    # Get it from here:
    # https://github.com/warmspringwinds/models/tree/fully_conv_vgg

    # Use second GPU -- change if you want to use a first one
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    flags.DEFINE_string("slim_path", "/home/vittal/work/tf-slim-models/models/slim", "The path to tf slim repo")

    # Add path to the cloned library
    flags.DEFINE_string("tf_image_seg_dir", "/home/vittal/work/segmentation/tf-image-segmentation/", "Dir for tf-image-segmentation repo")
    flags.DEFINE_string("checkpoints_dir", "/home/vittal/work/ckpts/", "Directory where ImageNet pretrained checkpoints are saved")
    flags.DEFINE_string("log_dir", os.path.join("/home/vittal/work/segmentation/tf-image-segmentation/log_dir/", exp), "Directory to save TF logs")
    flags.DEFINE_string("save_dir", os.path.join("/home/vittal/work/segmentation/tf-image-segmentation/save_dir/", exp), "Directory to save trained models")
    flags.DEFINE_string("data_dir", "/home/vittal/work/segmentation/tf-image-segmentation/data/", "Directory which hosts datasets")

elif 'login' or 'gpu' in machine:
    # Add a path to a custom fork of TF-Slim
    # Get it from here:
    # https://github.com/warmspringwinds/models/tree/fully_conv_vgg

    flags.DEFINE_string("slim_path", "/home-4/vpremac1@jhu.edu/projects/tf-models/models/slim", "The path to tf slim repo")

    # Add path to the cloned library
    flags.DEFINE_string("tf_image_seg_dir", "/home-4/vpremac1@jhu.edu/projects/tf-image-segmentation/", "Dir for tf-image-segmentation repo")
    flags.DEFINE_string("checkpoints_dir", "/home-4/vpremac1@jhu.edu/scratch/ckpts/", "Directory where ImageNet pretrained checkpoints are saved")
    flags.DEFINE_string("log_dir", os.path.join("/home-4/vpremac1@jhu.edu/projects/tf-image-segmentation/log_dir/", exp), "Directory to save TF logs")
    flags.DEFINE_string("save_dir", os.path.join("/home-4/vpremac1@jhu.edu/projects/tf-image-segmentation/save_dir/", exp), "Directory to save trained models")
    flags.DEFINE_string("data_dir", "/home-4/vpremac1@jhu.edu/projects/tf-image-segmentation/data/", "Directory which hosts datasets")

elif "thin6" in machine:
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

if not os.path.exists(FLAGS.data_dir):
    print('Could not find path to datasets. Exiting...')
    sys.exit()
