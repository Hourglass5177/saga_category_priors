import os
import sys
import cv2
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('preprocess parser')
    parser.add_argument('--source_path', '-s', type=str, required=True)
    parser.add_argument('--images', type=str, default='images')
    args = parser.parse_args(sys.argv[1:])
    images_path = os.path.join(args.source_path, args.images)
    for image_name in tqdm([e for e in os.listdir(images_path) if os.path.splitext(e)[1]=='.jpg']):
        image_path = os.path.join(images_path, image_name)
        image = cv2.imread(image_path)
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        assert cv2.imwrite(image_path, rotated_image), 'rotate fail'
