import os
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
from groundingdino.util.inference import Model

if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM segment everything masks extracting params")
    
    parser.add_argument("--source_path", '-s', type=str, required=True)
    parser.add_argument('--images', type=str, default='images')
    parser.add_argument("--sam_checkpoint_path", default='third_party/segment-anything/weights/sam_vit_h_4b8939.pth', type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--groundingdino_checkpoint_path", default='third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth', type=str)
    parser.add_argument("--groundingdino_config_path", default='third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', type=str)
    parser.add_argument("--downsample", default=1, type=int)
    parser.add_argument("--downsample_type", default='image', type=str, choices=['image', 'mask'], help="Downsample then segment, or segment then downsample.")
    parser.add_argument('--classes', nargs='+', type=str, default=['chair', 'desk', 'plant'])
    parser.add_argument('--box_threshold', type=float, default=0.25)
    parser.add_argument('--text_threshold', type=float, default=0.25)
    parser.add_argument('--nms_threshold', type=float, default=0.8)

    args = parser.parse_args()
    
    print("Initializing Grounded SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)
    grounding_dino_model = Model(model_config_path=args.groundingdino_config_path, model_checkpoint_path=args.groundingdino_checkpoint_path)

    downsample_manually = False
    if args.downsample == "1" or args.downsample_type == 'mask':
        images_path = os.path.join(args.source_path, args.images)
    else:
        images_path = os.path.join(args.source_path, f'{args.images}_{str(args.downsample)}')
        if not os.path.exists(images_path):
            images_path = os.path.join(args.source_path, args.images)
            downsample_manually = True
            print("No downsampled images, do it manually.")

    assert os.path.exists(images_path) and "Please specify a valid image root"
    masks_path = os.path.join(args.source_path, 'sam_masks')
    os.makedirs(masks_path, exist_ok=True)
    
    print("Extracting Grounded SAM masks...")
    
    for path in tqdm(sorted(os.listdir(images_path))):
        image_name = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(os.path.join(images_path, path))
        if downsample_manually:
            img = cv2.resize(img,dsize=(img.shape[1] // args.downsample, img.shape[0] // args.downsample),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        masks = mask_generator.generate(img)
        # print(len(masks))
        mask_list = []
        for m in masks:
            m_score = torch.from_numpy(m['segmentation']).float().to('cuda')

            if args.downsample_type == 'mask':
                m_score = torch.nn.functional.interpolate(m_score.unsqueeze(0).unsqueeze(0), size=(img.shape[0] // args.downsample, img.shape[1] // args.downsample) , mode='bilinear', align_corners=False).squeeze()
                m_score[m_score >= 0.5] = 1
                m_score[m_score != 1] = 0
                m_score = m_score.bool()

            if len(m_score.unique()) < 2:
                continue
            else:
                mask_list.append(m_score.bool())
        masks = torch.stack(mask_list, dim=0)

        torch.save(masks, os.path.join(masks_path, image_name+'.pt')) # bool[masks, h, w]