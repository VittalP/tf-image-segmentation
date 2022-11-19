import wget
import tarfile
import shutil
import set_paths
import os

FLAGS = set_paths.FLAGS

def download_ckpt(url):
    fname = wget.download(url)
    with tarfile.open(fname) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar)
        ckpt_name = tar.getnames()[0]
    
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)

    shutil.move(ckpt_name, FLAGS.checkpoints_dir)
    os.remove(fname)

