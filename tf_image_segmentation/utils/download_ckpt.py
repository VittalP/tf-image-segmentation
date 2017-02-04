import wget
import tarfile
import shutil
import set_paths
import os

FLAGS = set_paths.FLAGS

def download_ckpt(url):
    fname = wget.download(url)
    with tarfile.open(fname) as tar:
        tar.extractall()
        ckpt_name = tar.getnames()[0]
    
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)

    shutil.move(ckpt_name, FLAGS.checkpoints_dir)
    os.remove(fname)

