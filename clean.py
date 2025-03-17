import os
import sys
import shutil
import argparse

folders = ['detections', 'labels', 'mask_scales', 'rgb_masks', 'sam_masks']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clean argument')
    parser.add_argument('--source_path', '-s', type=str, required=True)
    args = parser.parse_args(sys.argv[1:])
    for folder in folders:
        path = os.path.join(args.source_path, folder)
        if os.path.exists(path):
            shutil.rmtree(path)